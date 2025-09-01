#!/usr/bin/env python
import os
import argparse
import json
import logging
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# —— 显存碎片化缓解 ——
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64,expandable_segments:True"

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LambdaCallback, Callback
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

class MetricHistoryCallback(Callback):
    """
    收集训练/验证/test 指标并在训练结束后绘图保存。
    兼容命名: train_loss_gen, val_loss_gen, test_* 以及任意以 'val_' 开头的指标。
    """
    def __init__(self, save_dir: str):
        super().__init__()
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        # history containers
        self.epochs = []
        self.train_losses = []   # per-epoch aggregated training loss
        self.val_metrics = {}    # key -> list aligned with epochs

        # keys priority
        self.train_loss_keys = ["train_loss_gen", "train_loss", "loss"]
        self.tracked_val_keys = ["val_loss_gen", "val_loss"]

    def _to_float(self, v):
        try:
            return float(v.cpu().detach()) if hasattr(v, "cpu") else float(v)
        except Exception:
            return None

    def on_train_epoch_end(self, trainer, pl_module, outputs=None):
        epoch = int(trainer.current_epoch)
        self.epochs.append(epoch + 1)  # store 1-based epoch

        cbm = trainer.callback_metrics  # dict-like

        # training loss priority
        t_loss = None
        for k in self.train_loss_keys:
            if k in cbm:
                t_loss = self._to_float(cbm[k])
                if t_loss is not None:
                    break
        self.train_losses.append(t_loss)

        # gather validation metrics (keys starting with val_ and tracked keys)
        keys_to_check = set(k for k in cbm.keys() if str(k).startswith("val_"))
        keys_to_check.update(self.tracked_val_keys)
        for key in sorted(keys_to_check):
            val = None
            if key in cbm:
                val = self._to_float(cbm[key])
            if key not in self.val_metrics:
                self.val_metrics[key] = [None] * (len(self.epochs) - 1)
            self.val_metrics[key].append(val)

        # pad other val_metrics that didn't get updated this epoch
        for k, lst in self.val_metrics.items():
            if len(lst) < len(self.epochs):
                lst.append(None)

    def on_test_end(self, trainer, pl_module):
        cbm = trainer.callback_metrics
        test_metrics = {}
        for k, v in dict(cbm).items():
            ks = str(k)
            if ks.startswith("test_"):
                fv = self._to_float(v)
                if fv is not None:
                    test_metrics[ks] = fv
        if test_metrics:
            try:
                with open(os.path.join(self.save_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
                    json.dump(test_metrics, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
            try:
                fig = plt.figure()
                keys = list(test_metrics.keys())
                vals = [test_metrics[k] for k in keys]
                plt.bar(range(len(vals)), vals)
                plt.xticks(range(len(vals)), keys, rotation=45, ha="right")
                plt.title("Test metrics")
                plt.tight_layout()
                plt.savefig(os.path.join(self.save_dir, "test_metrics.png"))
                plt.close(fig)
            except Exception:
                pass

    def on_fit_end(self, trainer, pl_module):
        # debug print to confirm callback invoked
        print(f"[MetricHistoryCallback] on_fit_end called, epochs collected: {len(self.epochs)}")

        # training loss plot
        try:
            if any(x is not None for x in self.train_losses):
                fig = plt.figure()
                plt.plot(self.epochs, self.train_losses)
                plt.xlabel("epoch")
                plt.ylabel("train_loss")
                plt.title("Training Loss")
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(os.path.join(self.save_dir, "training_loss.png"))
                plt.close(fig)
        except Exception:
            pass

        # validation metrics plot (plot all tracked val keys)
        try:
            if self.val_metrics:
                fig = plt.figure()
                keys = sorted(self.val_metrics.keys())
                plotted = False
                for key in keys:
                    vals = self.val_metrics.get(key, [])
                    if not any(v is not None for v in vals):
                        continue
                    plt.plot(self.epochs, vals, label=key)
                    plotted = True
                if plotted:
                    plt.xlabel("epoch")
                    plt.ylabel("value")
                    plt.title("Validation Metrics")
                    plt.legend()
                    plt.grid(True)
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.save_dir, "validation_metrics.png"))
                plt.close(fig)
        except Exception:
            pass

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
        monitor="val_loss_gen", mode="min", save_top_k=1,
        dirpath=f"logs_cv/fold{fold_idx}", filename="best"
    )

    # 合并 callbacks：保留 trainer_kwargs 中已有的 callbacks，并在前端添加本 fold 的 ckpt_cb
    kwargs = trainer_kwargs.copy()
    existing_cbs = list(trainer_kwargs.get("callbacks", []))
    kwargs.update({
        "logger": logger,
        "callbacks": [ckpt_cb] + existing_cbs
    })

    trainer = Trainer(**kwargs)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    best = ckpt_cb.best_model_path
    print(f"[Fold {fold_idx}] best ckpt: {best}")

    res = trainer.validate(model, dataloaders=val_loader, verbose=False)[0]
    print(f"Fold {fold_idx} result:", res)
    return res, best

def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Train Image Captioning with Lightning")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--cv", type=int, default=1)
    args = parser.parse_args()

    print("Device Info")
    print(f"CUDA available: {torch.cuda.is_available()}")
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs detected: {num_gpus}")
    for i in range(num_gpus):
        try:
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        except Exception:
            pass
    print(" ")

    # 加载 cfg 和种子
    cfg = OmegaConf.load(args.config)
    logger.info(f"Loaded max_epochs = {cfg.train.max_epochs}")

    seed = getattr(cfg.data, "seed", getattr(cfg.train, "seed", None))
    if seed is not None:
        pl.seed_everything(seed, workers=True)
    cfg.train.cv = args.cv

    # 公共 Trainer 参数
    clear_cache_cb = LambdaCallback(on_validation_epoch_start=lambda *a: torch.cuda.empty_cache())

    # prepare metric callback (single instance; will be passed into each Trainer via trainer_kwargs)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    metric_cb = MetricHistoryCallback(save_dir=f"logs/metrics-{timestamp}")

    # devices / accelerator decision
    ngpu = torch.cuda.device_count()
    accelerator = "gpu" if ngpu > 0 else "cpu"
    devices = ngpu if ngpu > 0 else 1
    precision = 16 if (ngpu > 0 and getattr(cfg.train, "use_amp", True)) else 32

    trainer_kwargs = {
        "accelerator":             accelerator,
        "devices":                 devices,
        "precision":               precision,
        "accumulate_grad_batches": getattr(cfg.train, "accumulate_grad_batches", 1),
        "log_every_n_steps":       50,
        "callbacks":               [clear_cache_cb, metric_cb],
        "max_epochs":              cfg.train.max_epochs
    }

    all_res = []
    all_ckpts = []

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

        for fold in range(1, args.cv + 1):
            # 直接传 trainer_kwargs（包含 metric_cb），train_validate_fold 内部会在 callbacks 前面添加 fold 的 ckpt_cb
            res, best = train_validate_fold(fold, full_dataset, cfg, trainer_kwargs)
            all_res.append(res)
            all_ckpts.append(best)

        print("\n=== CV best ckpts ===")
        for i, p in enumerate(all_ckpts, 1):
            print(f" Fold {i} -> {p}")
    else:
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
            monitor="val_loss_gen", mode="min", save_top_k=3,
            filename=f"{timestamp}-epoch{{epoch:02d}}-val{{val_loss_gen:.4f}}"
        )
        # Make sure callbacks include clear_cache_cb, ckpt_cb, metric_cb (and any others)
        trainer_kwargs_local = trainer_kwargs.copy()
        existing = list(trainer_kwargs_local.get("callbacks", []))
        # put ckpt_cb at front so it gets called; keep others (including metric_cb)
        trainer_kwargs_local["callbacks"] = [clear_cache_cb, ckpt_cb] + [c for c in existing if c not in (clear_cache_cb, ckpt_cb)]

        tb_logger = TensorBoardLogger("logs", name=f"timestamp-{timestamp}")
        trainer_kwargs_local["logger"] = tb_logger

        trainer = Trainer(**trainer_kwargs_local)
        if args.resume:
            trainer.fit(model, datamodule=data_module, ckpt_path=args.resume)
        else:
            trainer.fit(model, datamodule=data_module)

        print("训练结束，最佳模型：", ckpt_cb.best_model_path)
        all_ckpts.append(ckpt_cb.best_model_path)

        if getattr(cfg.data, "test_split", 0) > 0 or cfg.data.get("test_files"):
            trainer.test(model, datamodule=data_module)

if __name__ == "__main__":
    main()
