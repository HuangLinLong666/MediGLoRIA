import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from .CaptionDecoder import CaptionDecoder
from gloria.models.text_model import BertEncoder
from gloria.models.vision_model import ImageEncoder


class ImageCaptioningLightningModule(pl.LightningModule):
    """
        LightningModule for image-to-text captioning, 可选地包含 image-text matching 对比学习任务。
        依赖：
          - ImageEncoder(cfg)，其 output_dim 应与 BertEncoder.embedding_dim 对齐。
          - BertEncoder(cfg)，用于 tokenizer 和（可选）对比学习、复用 embedding 权重。
          - CaptionDecoder(hidden_size, vocab_size, pad_token_id, bos_token_id, eos_token_id,
                           image_feat_dim, num_layers, num_heads, ff_size, dropout, max_position_embeddings)
          - MatchingDataModule 提供 batch，包含:
                'images': Tensor[B,3,224,224],
                'decoder_input_ids': Tensor[B,L],
                'decoder_attention_mask': Tensor[B,L],
                'labels': Tensor[B,L] (pad 部分为 -100),
                可选:
                'class_labels': Tensor[B, num_classes],
                'match_input_ids': Tensor[B, M]（对比学习用文本输入 ids）,
                'match_attn_mask': Tensor[B, M]
        Config (cfg) 需要至少含以下字段:
          cfg.train.lr: float, 学习率
          cfg.train.do_matching: bool, 是否做对比学习
          cfg.train.match_weight: float, 对比学习损失权重
          cfg.train.match_temperature: float, 对比学习温度
          cfg.train.max_seq_length: int, decoder 最大序列长度
          cfg.model.text.embedding_dim: int, BERT embedding 维度，应等于 image encoder output_dim
          cfg.model.decoder.num_layers, num_heads, ff_size, dropout (可选，如无可使用默认)
    """

    def __init__(self, cfg):
        super().__init__()
        # 保存超参数
        self.val_outputs = None
        self.test_output = [] # 存储测试输出
        self.save_hyperparameters(ignore=['bert_encoder', 'image_encoder', 'decoder'])
        # 配置
        self.cfg = cfg
        # 初始化 BertEncoder、ImageEncoder
        self.do_matching = getattr(cfg.train, 'do_matching', False)
        if self.do_matching:
            self.bert_encoder = BertEncoder(cfg)
        else:
            self.bert_encoder = None  # 需确保输出 embedding_dim 与 image output_dim 对齐
        self.image_encoder = ImageEncoder(cfg)  # 其 local_embedder 输出通道数应与 cfg.model.text.embedding_dim 对齐
        # Tokenizer
        self.tokenizer = self.bert_encoder.tokenizer
        # 为 tokenizer 设置 bos_token 和 eos_token
        if self.tokenizer.bos_token is None:
            self.tokenizer.bos_token = self.tokenizer.cls_token
            self.tokenizer.bos_token_id = self.tokenizer.cls_token_id

        if self.tokenizer.eos_token is None:
            self.tokenizer.eos_token = self.tokenizer.sep_token
            self.tokenizer.eos_token_id = self.tokenizer.sep_token_id
        # 初始化 CaptionDecoder，hidden_size 与 embedding_dim 对齐
        embedding_dim = cfg.model.text.embedding_dim
        # decoder 超参数，可从 cfg 读取或使用默认
        num_layers = getattr(cfg.model.decoder, 'num_layers', 6)
        num_heads = getattr(cfg.model.decoder, 'num_heads', 8)
        ff_size = getattr(cfg.model.decoder, 'ff_size', embedding_dim * 4)
        dropout = getattr(cfg.model.decoder, 'dropout', 0.1)
        max_pos = cfg.train.max_seq_length if hasattr(cfg.train, 'max_seq_length') else 512

        self.decoder = CaptionDecoder(
            hidden_size=embedding_dim,
            vocab_size=self.tokenizer.vocab_size,
            pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            image_feat_dim=embedding_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_size=ff_size,
            dropout=dropout,
            max_position_embeddings=max_pos
        )

        # 复用 BERT embedding 权重给 decoder.embedding
        with torch.no_grad():
            try:
                bert_word_emb = self.bert_encoder.model.embeddings.word_embeddings.weight  # [vocab_size, dim]
                # 直接拷贝
                self.decoder.embedding.weight.copy_(bert_word_emb)
                # 可选复用 position embedding
                if hasattr(self.bert_encoder.model.embeddings, "position_embeddings"):
                    pe = self.bert_encoder.model.embeddings.position_embeddings.weight  # [max_pos_bert, dim]
                    max_len = self.decoder.pos_encoding.pe.size(1)
                    # 仅 copy 前 max_len
                    self.decoder.pos_encoding.pe[:, :max_len, :].copy_(pe[:max_len, :])
            except Exception as e:
                print(f"Warning: failed to copy BERT embeddings to decoder: {e}")

        # 损失与对比学习配置
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        if self.do_matching:
            # 检查维度对齐
            # image_encoder output_dim 应等于 embedding_dim
            with torch.no_grad():
                bert_emb = self.bert_encoder.model.embeddings.word_embeddings.weight
                self.decoder.embedding.weight.copy_(bert_emb)
                self.match_weight = cfg.train.match_weight
                self.match_temp = cfg.train.match_temperature

        # 学习率
        self.lr = cfg.train.lr

    def forward(self, images, decoder_input_ids, decoder_attention_mask):
        """
                images: [B, 3, H, W]
                decoder_input_ids: [B, L]
                decoder_attention_mask: [B, L], 1 为有效 token, 0 为 pad
                :return:
                    logits: [B, L, vocab_size]
                    global_emb: [B, D]，用于可选对比学习
        """
        # 提取 image 特征
        # 支持 get_local=True
        global_emb_local, local_emb = self.image_encoder(images, get_local=True)
        # local_emb: [B, embedding_dim, H', W']
        B, C, Hf, Wf = local_emb.shape
        assert C == self.decoder.image_feat_dim
        image_feats = local_emb  # 传给 decoder，会在内部 flatten

        global_emb_full = self.image_encoder(images)
        _ = (global_emb_full * 0).sum()

        # 不支持 get_local
        global_emb = global_emb_local  # [B, embedding_dim]

        # 调用 decoder
        logits = self.decoder(decoder_input_ids, decoder_attention_mask, image_feats)
        return logits, global_emb

    def on_after_backward(self) -> None:
        # 在反向之后，检查哪些参数没有梯度
        unused = []
        for name, param in self.named_parameters():
            if param.requires_grad and param.grad is None:
                unused.append(name)
        if unused:
            print(f"[Warning] The following parameters were unused in this step:\n" +
                  "\n".join(f"  - {n}" for n in unused))

    def training_step(self, batch, batch_idx):
        """
                batch 来自 MatchingDataModule，字段:
                  'images', 'decoder_input_ids', 'decoder_attention_mask', 'labels'
                  可选 'class_labels'
                  如果 do_matching=True，还需 'match_input_ids', 'match_attn_mask'
        """
        images = batch['images']
        decoder_input_ids = batch['decoder_input_ids']
        decoder_attention_mask = batch['decoder_attention_mask']
        labels = batch['labels']

        # logits: [B, L, V], labels: [B, L]
        logits, global_emb = self(images, decoder_input_ids, decoder_attention_mask)
        loss_gen = self.loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        # 再跑一次全局分支，激活 layer4 + global_embedder 参数
        global_emb_full = self.image_encoder(images)
        dummy_loss = global_emb_full.sum() - global_emb_full.sum()

        loss = loss_gen + dummy_loss
        self.log("train/loss_gen", loss_gen, prog_bar=True, on_step=True, on_epoch=True)

        # 可选对比学习损失
        if self.do_matching:
            # 确保 batch 中包含 match_input_ids, match_attn_mask
            text_ids = batch.get('match_input_ids', None)
            text_mask = batch.get('match_attn_mask', None)
            if text_ids is not None and text_mask is not None:
                # BertEncoder.forward 返回 (word_embs, sent_embs, sents)
                word_embs, sent_embs, _ = self.bert_encoder(text_ids, text_mask)
                # sent_embs: [B, embedding_dim]
                # L2 normalize
                img_emb_norm = F.normalize(global_emb, dim=-1)
                text_emb_norm = F.normalize(sent_embs, dim=-1)
                logits_i2t = torch.matmul(img_emb_norm, text_emb_norm.T) / self.match_temp  # [B, B]
                labels_match = torch.arange(images.size(0), device=self.device)
                loss_i2t = F.cross_entropy(logits_i2t, labels_match)
                loss_t2i = F.cross_entropy(logits_i2t.T, labels_match)
                loss_match = (loss_i2t + loss_t2i) / 2
                loss = loss + self.match_weight * loss_match
                self.log("train/loss_match", loss_match, prog_bar=True, on_step=True, on_epoch=True)
            else:
                # 如果未提供 match fields，可抛警告或跳过
                self.log("train/warning", 0.0, prog_bar=False)
        return loss

    def on_validation_epoch_start(self):
        self.val_outputs = []

    def validation_step(self, batch, batch_idx):
        """
        Validation: 计算生成 loss，并可少量示例生成
        """
        images = batch['images']
        decoder_input_ids = batch['decoder_input_ids']
        decoder_attention_mask = batch['decoder_attention_mask']
        labels = batch['labels']

        logits, global_emb = self(images, decoder_input_ids, decoder_attention_mask)
        loss_gen = self.loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        self.log("val/loss_gen", loss_gen, prog_bar=True, on_step=False, on_epoch=True)

        # 返回部分信息供 validation_epoch_end 使用
        self.val_outputs.append({"images": images})
        return {"val_loss": loss_gen, "images": images}

    def on_validation_epoch_end(self):
        """
        在 epoch 结束时做少量生成示例并记录。
        outputs: list of dict from validation_step
        """
        # 取前几个 batch 的 images
        if hasattr(self, "val_outputs") and self.val_outputs:
            outputs = self.val_outputs
            num_show = 2
            imgs = []
            for out in outputs[:num_show]:
                imgs.append(out["images"])
            if imgs:
                imgs = torch.cat(imgs, dim=0)[:num_show]  # [num_show, 3, H, W]
                captions = self.generate_captions(imgs)
                # 通过 logger 记录文本；若使用 TensorBoardLogger:
                if self.logger and hasattr(self.logger, "experiment"):
                    # 将多条 caption 合并为一个大字符串，或分条记录
                    # 这里示例一次性记录为一个 tag
                    all_caps = "\n".join(f"{idx}: {cap}" for idx, cap in enumerate(captions))
                    # current_epoch 作为 step
                    try:
                        # 对于 TensorBoardLogger，experiment 是 SummaryWriter
                        self.logger.experiment.add_text("val/generated_examples", all_caps,
                                                        global_step=self.current_epoch)
                    except Exception as e:
                        # 如果 logger 不是 TensorBoard 或发生错误，fallback 打印
                        print(f"[VAL GEN] 无法通过 logger 记录文本，改为打印: {e}")
                        for idx, cap in enumerate(captions):
                            print(f"[VAL GEN epoch {self.current_epoch}][{idx}]: {cap}")
                else:
                    # 没有 logger 时直接打印
                    for idx, cap in enumerate(captions):
                        print(f"[VAL GEN epoch {self.current_epoch}][{idx}]: {cap}")
        # 清空以备下一个 epoch
        self.val_outputs = []

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def test_step(self, batch, batch_idx):
        """
        测试阶段对每个 batch 执行。本示例中，我们主要进行生成，并可选计算 loss（如果 labels 存在）。
        :param batch: 包含 'images', 可能还有 'decoder_input_ids','decoder_attention_mask','labels'
        :return: dict，包含生成结果和可选 loss
        """
        images = batch['images']
        device = images.device
        # 如果有 labels，则可计算 loss；否则只生成
        has_labels = 'decoder_input_ids' in batch and 'decoder_attention_mask' in batch and 'labels' in batch

        result = {}
        if has_labels:
            decoder_input_ids = batch['decoder_input_ids']
            decoder_attention_mask = batch['decoder_attention_mask']
            labels = batch['labels']
            logits, global_emb = self(images, decoder_input_ids, decoder_attention_mask)
            loss_gen = self.loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            # 记录测试 loss（可选）
            self.log("test/loss_gen", loss_gen, prog_bar=True, on_step=False, on_epoch=True)
            result['loss_gen'] = loss_gen.detach()

        # 生成 captions
        # 使用已有的 generate_captions 方法；注意 generate_captions 会返回 list of str
        captions = self.generate_captions(images)
        result['predictions'] = captions

        # 保存结果用于epoch处理
        self.test_output.append(result)

        return captions, result

    def on_test_epoch_end(self):
        """
        测试集所有 batch 完成后调用。outputs 是 list of dicts，为每个 batch 返回的 test_step 结果。
        常见操作：汇总生成结果、将结果保存到文件、打印部分示例、计算整体指标等。
        """
        # 汇总所有生成结果
        all_preds = []
        all_loss = []
        avg_loss = None
        for out in self.test_output:
            # out['predictions'] 是当前 batch 的 list of str
            if 'predictions' in out:
                all_preds.extend(out['predictions'])
            if 'loss_gen' in out:
                if isinstance(out['loss_gen'], torch.Tensor):
                    all_loss.append(out['loss_gen'].item())  # 直接取 item() 避免 .cpu()
                else:
                    all_loss.append(out['loss_gen'])

        num_show = min(5, len(all_preds))
        print(f"=== Test Epoch End: 显示前 {num_show} 条生成示例 ===")
        for i in range(num_show):
            print(f"Sample {i}: {all_preds[i]}")
        if all_loss:
            avg_loss = sum(all_loss) / len(all_loss)
            print(f"Test average loss_gen: {avg_loss:.4f}")
        else:
            all_loss = None

        # 清空测试输出
        self.test_output = []
        return {
            "test_predictions": all_preds,
            "test_loss": avg_loss
        }

    @torch.no_grad()
    def generate_captions(self, images, max_len=None):
        """
        对一批图像进行 greedy 生成
        :param images: Tensor[B, 3, H, W]
        :param max_len: 最大生成长度，若 None，则使用 cfg.train.max_seq_length
        :return: list of str, 长度 B
        """
        self.eval()
        device = next(self.parameters()).device
        images = images.to(device)
        if max_len is None:
            max_len = getattr(self.cfg.train, 'max_seq_length', 128)

        # 提取 image_feats
        try:
            global_emb, local_emb = self.image_encoder(images, get_local=True)
            # local_emb: [B, embedding_dim, H', W']
            image_local = local_emb  # 传给 decoder.generate_greedy
        except TypeError:
            global_emb = self.image_encoder(images)
            # single global: 视作 [B, 1, embedding_dim]
            image_local = global_emb.unsqueeze(1)  # 但 CaptionDecoder.generate_greedy 期望 raw local; 对单位置情形可特殊处理
        captions = []
        for i in range(images.size(0)):
            # 单张生成
            img_feat = image_local[i:i + 1]  # [1, embedding_dim, H', W'] 或 [1, 1, embedding_dim]
            # 如果是 global only case ([1,1,embedding_dim])，需要转为适配: 传入 [1, embedding_dim, 1, 1]?
            # 这里假设总是有 local_emb 情形
            cap = self.decoder.generate_greedy(img_feat,
                                               max_len,
                                               self.tokenizer,
                                               self.image_encoder,
                                               device)
            captions.append(cap)
        return captions
