import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer



class BertEncoder(nn.Module):
    def __init__(self, cfg):
        super(BertEncoder, self).__init__()

        self.bert_type = cfg.model.text.bert_type
        self.last_n_layers = cfg.model.text.last_n_layers
        self.aggregate_method = cfg.model.text.aggregate_method
        self.norm = cfg.model.text.norm
        self.embedding_dim = cfg.model.text.embedding_dim
        self.freeze_bert = cfg.model.text.freeze_bert
        self.agg_tokens = cfg.model.text.agg_tokens

        self.model = AutoModel.from_pretrained(
            self.bert_type, output_hidden_states=True, use_safetensors=True
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_type)
        self.idxtoword = {v: k for k, v in self.tokenizer.get_vocab().items()}

        self.emb_global, self.emb_local = None, None

        if self.freeze_bert is True:
            print("Freezing BERT model")
            for param in self.model.parameters():
                param.requires_grad = False

    def aggregate_tokens(self, embeddings, caption_ids):

        batch_size, num_layers, num_words, dim = embeddings.shape
        embeddings = embeddings.permute(0, 2, 1, 3)
        agg_embs_batch = []
        sentences = []

        # loop over batch
        for embs, caption_id in zip(embeddings, caption_ids):

            agg_embs = []
            token_bank = []
            words = []
            word_bank = []

            # loop over sentence
            for word_emb, word_id in zip(embs, caption_id):

                word = self.idxtoword.get(word_id.item(), "[UNK]")

                if word == "[SEP]":
                    if token_bank:
                        new_emb = torch.stack(token_bank)
                        new_emb = new_emb.sum(axis=0)
                        agg_embs.append(new_emb)
                        words.append("".join(word_bank))

                        agg_embs.append(word_emb)
                        words.append(word)
                        break

                if not word.startswith("##"):
                    if token_bank:
                        new_emb = torch.stack(token_bank).sum(dim=0)
                        agg_embs.append(new_emb)
                        words.append("".join(word_bank))
                    token_bank = [word_emb]
                    word_bank = [word]
                else:
                    token_bank.append(word_emb)
                    word_bank.append(word[2:])

            if token_bank:
                new_emb = torch.stack(token_bank).sum(dim=0)
                agg_embs.append(new_emb)
                words.append("".join(word_bank))

            agg_embs = torch.stack(agg_embs)
            padding_size = num_words - len(agg_embs)
            paddings = torch.zeros(padding_size, num_layers, dim, device=agg_embs.device)
            agg_embs = torch.cat([agg_embs, paddings])
            words = words + ["[PAD]"] * padding_size

            agg_embs_batch.append(agg_embs)
            sentences.append(words)

        agg_embs_batch = torch.stack(agg_embs_batch)
        agg_embs_batch = agg_embs_batch.permute(0, 2, 1, 3)
        return agg_embs_batch, sentences

    def forward(self, input_ids, attn_mask):

        outputs = self.model(input_ids, attn_mask)

        # aggregate intermetidate layers
        if self.last_n_layers > 1:
            all_hidden = outputs.hidden_states # tuple of (layer, batch, seq_len, dim)
            embeddings = torch.stack(
                all_hidden[-self.last_n_layers :]
            )  # num_layers, batch, sent_len, embedding size

            embeddings = embeddings.permute(1, 0, 2, 3) # shape: (batch, num_layers, sent_len, embedding_size)

            if self.agg_tokens:
                embeddings, sents = self.aggregate_tokens(embeddings, input_ids)
            else:
                sents = [[self.idxtoword.get(tok.item(), "[UNK]") for tok in sent] for sent in input_ids]

            sent_embeddings = embeddings.mean(dim=2)

            if self.aggregate_method == "sum":
                word_embeddings = embeddings.sum(dim=1)
                sent_embeddings = sent_embeddings.sum(dim=1)
            elif self.aggregate_method == "mean":
                word_embeddings = embeddings.mean(dim=1)
                sent_embeddings = sent_embeddings.mean(dim=1)
            else:
                print(self.aggregate_method)
                raise Exception("Aggregation method not implemented")

        # use last layer
        else:
            word_embeddings = outputs.last_hidden_state  # shape: (b, s, d)
            sent_embeddings = outputs.pooler_output  # shape: (b, d)

        # Optional projection layers
        batch_size, seq_len, feat_dim = word_embeddings.shape
        word_embeddings = word_embeddings.view(batch_size * seq_len, feat_dim)
        if self.emb_local is not None:
            word_embeddings = self.emb_local(word_embeddings)
        word_embeddings = word_embeddings.view(batch_size, seq_len, self.embedding_dim)
        word_embeddings = word_embeddings.permute(0, 2, 1)

        if self.emb_global is not None:
            sent_embeddings = self.emb_global(sent_embeddings)

        # L2 normalization
        if self.norm is True:
            word_embeddings = word_embeddings / torch.norm(
                word_embeddings, 2, dim=1, keepdim=True
            ).clamp(min=1e-6)
            sent_embeddings = sent_embeddings / torch.norm(
                sent_embeddings, 2, dim=1, keepdim=True
            ).clamp(min=1e-6)

        return word_embeddings, sent_embeddings, sents
