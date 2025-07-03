import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    # 位置编码
    def __init__(self, d_model: int, max_len: int=256 ):
        super().__init__()

        # 位置编码矩阵[max_len, d_model]
        pe = torch.zeros(max_len, d_model) # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 计算分母项
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)  # non-parameter buffer

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        将位置编码加到输入 embedding 上
        :param x: Tensor [B, L, d_model]
        :return: x + positional encoding [B, L, d_model]
        """
        L = x.size(1)
        # pe: [1, max_len, d_model], 取前 L
        x = x + self.pe[:, :L, :].to(x.device)
        return x

class CaptionDecoder(nn.Module):
    """
    image_local_features 投影后形状为 [B, hidden_size, H, W]，其中 hidden_size 与 decoder hidden size 一致。
    decoder_input_ids: [B, L]，以 BOS 开头、pad 部分为 pad_token_id。
    decoder_attention_mask: [B, L]，1 表示有效 token，0 表示 pad。
    在 collate stage，decoder_labels pad 为 -100，用于 loss 计算
    """
    def __init__(self,
                 hidden_size: int,
                 vocab_size: int,
                 pad_token_id: int,
                 bos_token_id: int,
                 eos_token_id: int,
                 image_feat_dim: int = None,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 ff_size: int = 2048,
                 dropout: float = 0.1,
                 max_position_embeddings: int = 512
                 ):

        """
        hidden_size: Transformer hidden size，也应等于 image_local_embedder 输出通道数
        vocab_size: 词表大小
        pad_token_id: pad token id，用于 padding mask
        bos_token_id: BOS token id
        eos_token_id: EOS token id
        image_feat_dim: image 特征原始维度。如果为 None，则假定 image_feats 传入时已经是 hidden_size；否则若传入 [B, N, image_feat_dim] 或 [B, image_feat_dim, H, W]，会先投影到 hidden_size。
        num_layers: TransformerDecoder 层数
        num_heads: multi-head attention 头数
        ff_size: feed-forward 隐藏层大小
        dropout: dropout 比例
        max_position_embeddings: 位置编码最大长度，应 >= 数据集中最大 decoder length
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.image_feat_dim = image_feat_dim

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        # 位置编码
        self.pos_encoding = PositionalEncoding(hidden_size, max_len=max_position_embeddings)

        # 如果 image_feat_dim != hidden_size，需要投影层
        if self.image_feat_dim != self.hidden_size:
            self.img_feat_proj = nn.Linear(self.image_feat_dim, self.hidden_size)
        else:
            # 恒等映射
            self.img_feat_proj = nn.Identity()

        # Transformer Decoder层
        # shape 要为 [T, B, H]
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # 输出投影到词表
        self.output_proj = nn.Linear(hidden_size, vocab_size)

    def forward(self,
                decoder_input_ids: torch.LongTensor,
                decoder_attention_mask: torch.LongTensor,
                image_local_feat: torch.Tensor) -> torch.Tensor:
        """
                前向计算 logits。

                decoder_input_ids: [B, L], 包含 BOS；pad 部分为 pad_token_id
                decoder_attention_mask: [B, L], 1 表示有效 token, 0 表示 pad
                image_local_feat:
                    - 如果形状为 [B, interm_feature_dim, H, W]，需先由 ImageEncoder.local_embedder 投影到 [B, hidden_size, H, W] 后再传入；
                    - 或已在外部投影后传入，形状 [B, hidden_size, H, W]；
                    - 或已 flatten 为 [B, N, hidden_size]，其中 N = H*W。
                :return: logits: [B, L, vocab_size]
        """

        B, L = decoder_input_ids.size()
        device = decoder_input_ids.device

        assert decoder_input_ids.min() >= 0, \
            f"Negative token ID found: min={decoder_input_ids.min().item()}"
        assert decoder_input_ids.max() < self.vocab_size, \
            f"Token ID too large: max={decoder_input_ids.max().item()}, vocab_size={self.vocab_size}"

        # Embedding + PositionalEncoding
        emb = self.embedding(decoder_input_ids) * math.sqrt(self.hidden_size)  # [B, L, H]
        emb = self.pos_encoding(emb)  # [B, L, H]
        tgt = emb # batch_first=True, 直接[B, L, hidden_size]

        # 处理 image_local_feat
        mem = image_local_feat

        if mem.dim() == 4:
            # Flatten: [B, hidden_size, H, W] -> [B, N, hidden_size]
            Bf, C, H, W = mem.shape

            # 检查C是否等于hidden_size
            if C != self.hidden_size:
                raise ValueError(f"传入 image_local_feat 通道数 {C} 与 hidden_size {self.hidden_size} 不匹配")
            mem_flat = mem.view(Bf, C, H*W).permute(0, 2, 1)  # [B, N, hidden_size]
        elif mem.dim() == 3:
            # 已经flat: [B, N, hidden_size]
            mem_flat = mem
            if mem_flat.size(2) != self.hidden_size:
                raise ValueError(
                    f"传入 image_local_feat 尺寸 {mem_flat.shape} 与 hidden_size {self.hidden_size} 不匹配")
        else:
            raise ValueError(f"image_local_feat 维度应为 3 或 4，得到 {mem.dim()}")

        # 投影到 hidden_size
        # mem_proj: [B, N, hidden_size]
        mem_proj = self.img_feat_proj(mem_flat)

        # mask
        # torch.triu with diagonal=1 得到上三角 True 表示 mask未来
        tgt_mask = torch.triu(torch.ones((L, L), device=device, dtype=torch.bool), diagonal=1)  # [L, L]

        # key padding mask: [B, L], True 表示 pad 位置要 mask
        tgt_key_padding_mask = (decoder_input_ids == self.pad_token_id) # [B, L], bool

        # memory_key_padding_mask: 若 image memory 有 padding，需要传入；通常 image_feats 全有效，传 None
        memory_key_padding_mask = None

        # TransformerDecoder
        # Outshape : [L, B, hidden_size]
        out = self.transformer_decoder(
            tgt,
            memory=mem_proj,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask # 一般不对 image pos 做 mask
        )

        # 输出logits
        logits = self.output_proj(out) # [B, L, vocab_size]
        return logits

    @torch.no_grad()
    def generate_greedy(self,
                        image_local_raw: torch.Tensor,
                        max_len: int,
                        tokenizer,
                        image_encoder,
                        device: torch.device = None) -> str:
        """
                Greedy 解码生成单张图像的 caption。

                image_local_raw:
                    - 如果形状为 [1, interm_feature_dim, H, W]，需要先投影：image_encoder.local_embedder(raw)；
                    - 如果已是 [1, hidden_size, H, W]，可直接传入；
                max_len: 最大生成长度（包含 BOS/EOS）
                tokenizer: 用于 decode token ids
                image_encoder: 用于将 raw_local 投影到 hidden_size，如 image_encoder.local_embedder
                device: 运行设备
                生成的文本字符串
            """
        self.eval()
        if device is None:
            device = next(self.parameters()).device

        # 处理image_local_raw
        mem = image_local_raw.to(device)
        # 如果raw, 需要投影
        if mem.dim() == 4:
            if mem.shape[1] == image_encoder.interm_feature_dim:  # e.g., 1024
                mem = image_encoder.local_embedder(mem)
            elif mem.shape[1] == self.hidden_size:  # already projected
                pass  # skip local_embedder
            else:
                raise ValueError(f"Unknown image_feat channel size: {mem.shape[1]}")

        # flatten
        if mem.dim() == 3:
            mem_flat = mem
        elif mem.dim() == 4:
            Bf, C, H, W = mem.shape
            mem_flat = mem.view(Bf, C, H*W).permute(0, 2, 1)  # [1, N, hidden_size]
        else:
            raise ValueError(f"image_local_raw 维度应为 3 或 4，得到 {mem.dim()}")

        # 如果 image_feat_dim != hidden_size，需要投影
        if self.image_feat_dim != self.hidden_size:
            mem_proj = self.img_feat_proj(mem_flat)  # [1, N, hidden_size]
        else:
            mem_proj = mem_flat  # [1, N, hidden_size]

        # 生成循环
        generated = [tokenizer.bos_token_id]
        for _ in range(max_len - 1):
            cur_ids = torch.tensor(generated, device=device).unsqueeze(0)  # [1, cur_len]
            attn_mask = (cur_ids != tokenizer.pad_token_id).long()  # [1, cur_len]

            logits = self.forward(cur_ids, attn_mask, mem_flat)
            next_logits = logits[0, -1, :]  # [vocab_size]
            next_id = next_logits.argmax(dim=-1).item()
            generated.append(next_id)
            if next_id == tokenizer.eos_token_id:
                break
        # decoder, 跳过特殊token
        text = tokenizer.decode(generated, skip_special_tokens=True)
        return text


