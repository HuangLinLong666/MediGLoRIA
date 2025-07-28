#!/usr/bin/env python
import os
import argparse
import json
import logging

# —— 显存碎片化缓解 ——
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64,expandable_segments:True"

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LambdaCallback
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import KFold
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Subset

from omegaconf import OmegaConf
from dataset_managing.Matching_Dataset_Module import MatchingDataModule
from dataset_managing.dataset_reader import MatchingDataset
from Decoder.LightningModule import ImageCaptioningLightningModule

def collate_fn(batch, pad_token_id):
    batch = [b for b in batch if b is not None]
    images = torch.stack([b["image"] for b in batch], dim=0)
    input_ids = [b["decoder_input_ids"] for b in batch]
    labels    = [b["decoder_labels"]     for b in batch]

    decoder_input_ids     = pad_sequence(input_ids,  batch_first=True, padding_value=pad_token_id)
    decoder_labels        = pad_sequence(labels,     batch_first=True, padding_value=-100)
    decoder_attention_mask= (decoder_input_ids != pad_token_id).long()

    return {
        "images":               images,
        "decoder_input_ids":    decoder_input_ids,
        "decoder_attention_mask": decoder_attention_mask,
        "labels":               decoder_labels
    }

def train_validate_fold(fold_idx, full_dataset, cfg, trainer_kwargs):
    """对单折数据做训练+验证，返回该折的 validate 结果 dict。"""
    pl.seed_everything(cfg.data.seed + fold_idx, workers=True)

    # KFold 划分
    indices = list(range(len(full_dataset)))
    kf = KFold(n_splits=cfg.train.cv, shuffle=True, random_state=cfg.data.seed)
    for i, (tr_idx, val_idx) in enumerate(kf.split(indices), start=1):
        if i == fold_idx:
            break

    pad_id = full_dataset.pad_token_id
    train_ds = Subset(full_dataset, tr_idx)
    val_ds   = Subset(full_dataset, val_idx)

    train_loader = DataLoader(train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        collate_fn=lambda b: collate_fn(b, pad_id)
    )
    val_loader = DataLoader(val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        collate_fn=lambda b: collate_fn(b, pad_id)
    )

    # 模型、Logger 和 Checkpoint
    model   = ImageCaptioningLightningModule(cfg)
    logger  = TensorBoardLogger("logs_cv", name=f"fold{fold_idx}")
    ckpt_cb = ModelCheckpoint(
        monitor="val/loss_gen", mode="min", save_top_k=1,
        dirpath=f"logs_cv/fold{fold_idx}", filename="best"
    )

    # 更新当前 fold 的回调与 logger
    kwargs = trainer_kwargs.copy()
    kwargs.update({
        "logger":   logger,
        "callbacks":[ckpt_cb, trainer_kwargs["callbacks"][0]]  # 保留 clear_cache_cb
    })

    trainer = Trainer(**kwargs)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    res = trainer.validate(model, dataloaders=val_loader, verbose=False)[0]
    print(f"Fold {fold_idx} result:", res)
    return res

def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    parser = argparse.ArgumentParser(description="Train Image Captioning with Lightning")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="YAML config 文件路径")
    parser.add_argument("--resume", type=str, default=None,
                        help="Checkpoint 路径（可选）")
    parser.add_argument("--cv", type=int, default=1,
                        help="K‑Fold 折数 (>1 启用 CV)")
    args = parser.parse_args()

    print("========== Device Info ==========")
    print(f"CUDA available: {torch.cuda.is_available()}")
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs detected: {num_gpus}")
    for i in range(num_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print("=================================")

    # 加载 cfg 和种子
    cfg = OmegaConf.load(args.config)

    logger.info(f"Loaded max_epochs = {cfg.train.max_epochs}")


    seed = getattr(cfg.data, "seed", getattr(cfg.train, "seed", None))
    if seed is not None:
        pl.seed_everything(seed, workers=True)
    cfg.train.cv = args.cv

    # 公共 Trainer 参数
    clear_cache_cb = LambdaCallback(on_validation_epoch_start=lambda *a: torch.cuda.empty_cache())
    ngpu = torch.cuda.device_count()
    trainer_kwargs = {
        "accelerator":             "gpu",
        "devices":                 ngpu,
        "precision":               16,
        "accumulate_grad_batches": getattr(cfg.train, "accumulate_grad_batches", 1),
        "log_every_n_steps":       50,
        "callbacks":               [clear_cache_cb],
        "max_epochs":              cfg.train.max_epochs
    }

    # CV 分支
    if args.cv > 1:
        full_dataset = MatchingDataset(
            annotation_files=cfg.data.train_files,
            image_root=cfg.data.image_root,
            scibert_path=cfg.model.text.bert_type,
            max_sequence_length=cfg.train.max_seq_length,
            use_refs=cfg.data.use_refs,
            caption_fields=list(cfg.data.caption_fields),
            image_ext=cfg.data.image_ext,
            ensure_unique=cfg.data.ensure_unique,
            return_class_labels=cfg.data.return_class_labels,
            limit=getattr(cfg.data, "limit", None)
        )
        all_res = []
        for fold in range(1, args.cv + 1):
            res = train_validate_fold(fold, full_dataset, cfg, trainer_kwargs)
            all_res.append(res)
        avg = {k: sum(r[k] for r in all_res) / len(all_res) for k in all_res[0]}
        print("\n=== CV Average ===")
        print(json.dumps(avg, indent=2))
        return

    # 单次训练分支
    data_module = MatchingDataModule(
        annotation_files={"train": cfg.data.train_files},
        image_root=cfg.data.image_root,
        scibert_path=cfg.model.text.bert_type,
        batch_size=cfg.train.batch_size,
        max_sequence_length=cfg.train.max_seq_length,
        use_refs=cfg.data.use_refs,
        caption_fields=list(cfg.data.caption_fields),
        image_ext=cfg.data.image_ext,
        ensure_unique=cfg.data.ensure_unique,
        return_class_labels=cfg.data.return_class_labels,
        val_split=getattr(cfg.data, "val_split", 0),
        test_split=getattr(cfg.data, "test_split", 0.2),
        seed=seed
    )
    model = ImageCaptioningLightningModule(cfg)

    # 单次训练的 checkpoint 回调
    ckpt_cb = ModelCheckpoint(
        monitor="val/loss_gen", mode="min", save_top_k=3,
        filename="epoch{epoch:02d}-val{val_loss_gen:.4f}"
    )
    trainer_kwargs["callbacks"] = [clear_cache_cb, ckpt_cb]
    trainer_kwargs["logger"] = TensorBoardLogger("logs", name="image_caption")

    trainer = Trainer(
        **trainer_kwargs,
        limit_val_batches=0
    )
    if args.resume:
        trainer.fit(model, datamodule=data_module, ckpt_path=args.resume)
    else:
        trainer.fit(model, datamodule=data_module)

    print("训练结束，最佳模型：", ckpt_cb.best_model_path)

    if getattr(cfg.data, "test_files", None) or getattr(cfg.train, "test_split", 0) > 0:
        trainer.test(model, datamodule=data_module)

if __name__ == "__main__":
    main()
