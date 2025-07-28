import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from .CaptionDecoder import CaptionDecoder
from gloria.models.text_model import BertEncoder
from gloria.models.vision_model import ImageEncoder

class ImageCaptioningLightningModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(ignore=['bert_encoder', 'image_encoder', 'decoder'])
        self.cfg = cfg
        self.do_matching = getattr(cfg.train, 'do_matching', False)
        if self.do_matching:
            self.bert_encoder = BertEncoder(cfg)
        else:
            self.bert_encoder = None
        self.image_encoder = ImageEncoder(cfg)
        self.tokenizer = self.bert_encoder.tokenizer
        # ensure BOS/EOS
        if self.tokenizer.bos_token is None:
            self.tokenizer.bos_token = self.tokenizer.cls_token
            self.tokenizer.bos_token_id = self.tokenizer.cls_token_id
        if self.tokenizer.eos_token is None:
            self.tokenizer.eos_token = self.tokenizer.sep_token
            self.tokenizer.eos_token_id = self.tokenizer.sep_token_id

        emb_dim = cfg.model.text.embedding_dim
        num_layers = getattr(cfg.model.decoder, 'num_layers', 6)
        num_heads  = getattr(cfg.model.decoder, 'num_heads', 8)
        ff_size    = getattr(cfg.model.decoder, 'ff_size', emb_dim * 4)
        dropout    = getattr(cfg.model.decoder, 'dropout', 0.1)
        max_pos    = cfg.train.max_seq_length

        self.decoder = CaptionDecoder(
            hidden_size=emb_dim,
            vocab_size=self.tokenizer.vocab_size,
            pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            image_feat_dim=emb_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_size=ff_size,
            dropout=dropout,
            max_position_embeddings=max_pos,
        )

        # copy BERT embeddings
        with torch.no_grad():
            try:
                bert_emb = self.bert_encoder.model.embeddings.word_embeddings.weight
                self.decoder.embedding.weight.copy_(bert_emb)
                if hasattr(self.bert_encoder.model.embeddings, "position_embeddings"):
                    pe = self.bert_encoder.model.embeddings.position_embeddings.weight
                    L = self.decoder.pos_encoding.pe.size(1)
                    self.decoder.pos_encoding.pe[:, :L, :].copy_(pe[:L, :])
            except:
                pass

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        self.lr = cfg.train.lr
        if self.do_matching:
            self.match_weight = cfg.train.match_weight
            self.match_temp   = cfg.train.match_temperature

        # 用于 validation_step 收集
        self.val_outputs = []

    def forward(self, images, decoder_input_ids, decoder_attention_mask):
        global_emb_local, local_emb = self.image_encoder(images, get_local=True)
        logits = self.decoder(decoder_input_ids, decoder_attention_mask, local_emb)
        return logits, global_emb_local

    def training_step(self, batch, batch_idx):
        images = batch['images']
        ids    = batch['decoder_input_ids']
        mask   = batch['decoder_attention_mask']
        labels = batch['labels']
        logits, img_emb = self(images, ids, mask)
        loss = self.loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        # dummy to touch visual encoder
        dummy = self.image_encoder(images).sum() * 0
        loss = loss + dummy
        self.log("train/loss_gen", loss, prog_bar=True, on_step=True, on_epoch=True)
        # matching loss if needed
        if self.do_matching:
            txt_ids  = batch.get('match_input_ids', None)
            txt_mask = batch.get('match_attn_mask', None)
            if txt_ids is not None:
                _, sent_emb, _ = self.bert_encoder(txt_ids, txt_mask)
                i2t = F.normalize(img_emb, dim=-1) @ F.normalize(sent_emb, dim=-1).T / self.match_temp
                labels_m = torch.arange(img_emb.size(0), device=self.device)
                lm = F.cross_entropy(i2t, labels_m) + F.cross_entropy(i2t.T, labels_m)
                loss = loss + self.match_weight * (lm / 2)
                self.log("train/loss_match", lm/2, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def on_validation_epoch_start(self):
        self.val_outputs = []

    def validation_step(self, batch, batch_idx):
        images = batch['images']
        ids    = batch['decoder_input_ids']
        mask   = batch['decoder_attention_mask']
        labels = batch['labels']
        logits, _ = self(images, ids, mask)
        loss = self.loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        self.log("val/loss_gen", loss, prog_bar=True, on_epoch=True)
        # 不再收集图像和生成示例
        return loss

    # 注释掉原 on_validation_epoch_end，避免任何额外生成
    # def on_validation_epoch_end(self):
    #     self.val_outputs = []

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    @torch.no_grad()
    def generate_captions(self, images, max_len=None):
        self.eval()
        device = next(self.parameters()).device
        images = images.to(device)
        max_len = max_len or self.cfg.train.max_seq_length
        global_emb, local_emb = self.image_encoder(images, get_local=True)
        captions = []
        for img in local_emb:
            cap = self.decoder.generate_beam(
                img.unsqueeze(0), max_len, self.tokenizer,
                self.image_encoder, device=device,
                beam_width=5, length_penalty=1.2, no_repeat_ngram_size=2
            )
            captions.append(cap)
        return captions
