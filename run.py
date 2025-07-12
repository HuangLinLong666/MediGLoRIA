import argparse, json
from omegaconf import OmegaConf
import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import KFold
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Subset

from dataset_managing.Matching_Dataset_Module import MatchingDataModule
from Decoder.LightningModule import ImageCaptioningLightningModule
from dataset_managing.dataset_reader import MatchingDataset


def collate_fn(batch, pad_token_id):
    batch = [b for b in batch if b is not None]
    images =        torch.stack([b['image'] for b in batch], dim=0)
    input_ids =     [b['decoder_input_ids'] for b in batch]
    labels =        [b['decoder_labels']    for b in batch]
    decoder_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    decoder_labels    = pad_sequence(labels,    batch_first=True, padding_value=-100)
    decoder_attention_mask = (decoder_input_ids != pad_token_id).long()
    return {
        'images': images,
        'decoder_input_ids': decoder_input_ids,
        'decoder_attention_mask': decoder_attention_mask,
        'labels': decoder_labels
    }


def train_validate_fold(fold_idx, full_dataset, cfg):
    """对单折数据做训练+验证，返回该折的 validate 结果 dict。"""
    pl.seed_everything(cfg.data.seed + fold_idx)

    # 拆 indices
    N = len(full_dataset)
    indices = list(range(N))
    kf = KFold(n_splits=cfg.train.cv, shuffle=True, random_state=cfg.data.seed)
    # 取第 fold_idx‑1 折的 train/val 索引
    for i, (tr_idx, val_idx) in enumerate(kf.split(indices), start=1):
        if i == fold_idx:
            break

    # 子集 & DataLoader
    pad_id = full_dataset.pad_token_id
    train_ds = Subset(full_dataset, tr_idx)
    val_ds   = Subset(full_dataset, val_idx)
    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True,
                              num_workers=cfg.data.num_workers, collate_fn=lambda b: collate_fn(b, pad_id))
    val_loader   = DataLoader(val_ds,   batch_size=cfg.train.batch_size, shuffle=False,
                              num_workers=cfg.data.num_workers, collate_fn=lambda b: collate_fn(b, pad_id))

    # LightningModule + Trainer
    model = ImageCaptioningLightningModule(cfg)
    logger = TensorBoardLogger("logs_cv", name=f"fold{fold_idx}")
    ckpt_cb = pl.callbacks.ModelCheckpoint(
        monitor="val/loss_gen", mode="min", save_top_k=1,
        dirpath=f"logs_cv/fold{fold_idx}", filename="best"
    )
    trainer = Trainer(
        max_epochs=cfg.train.max_epochs,
        logger=logger,
        callbacks=[ckpt_cb],
        accelerator="auto"
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    res = trainer.validate(model, dataloaders=val_loader, verbose=False)[0]
    print(f"Fold {fold_idx} result:", res)
    return res

def main():
    parser = argparse.ArgumentParser(description="Train Image Captioning with Lightning")
    parser.add_argument(
        "--config", type=str,
        default="/Users/vegeta/PycharmProjects/MediGLoRIA/config/config.yaml",
        help="Path to YAML config file, e.g. config.yaml"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from (optional)"
    )
    parser.add_argument(
        "--cv", type=int, default=1,
        help="Number of folds for k‑fold cross validation (default: 1, i.e. no CV)"
    )

    args = parser.parse_args()
    # 加载配置
    cfg = OmegaConf.load(args.config)

    # 设置种子
    seed = None
    if hasattr(cfg, "data") and "seed" in cfg.data:
        seed = cfg.data.seed
    elif hasattr(cfg.train, "seed"):
        seed = cfg.train.seed
    if seed is not None:
        pl.seed_everything(seed, workers=True)

    cfg.train.cv = args.cv

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
            res = train_validate_fold(fold, full_dataset, cfg)
            all_res.append(res)
        # 3) 平均
        avg = {}
        for r in all_res:
            for k, v in r.items():
                avg.setdefault(k, []).append(v)
        avg = {k: sum(vs) / len(vs) for k, vs in avg.items()}
        print("\n=== CV Average ===")
        print(json.dumps(avg, indent=2))
        return

    # 构建DataModule
    if not hasattr(cfg, "data") or not hasattr(cfg.data, "train_files") or not hasattr(cfg.data, "image_root"):
        raise ValueError("配置文件中必须包含 data.train_files 和 data.image_root")

    data_args = {
        "annotation_files": {"train": cfg.data.train_files},
        "image_root": cfg.data.image_root,
        "scibert_path": cfg.model.text.bert_type,
        "batch_size": cfg.train.batch_size,
        "max_sequence_length": cfg.train.max_seq_length,
        "use_refs": getattr(cfg.data, "use_refs", True),
        "caption_fields": tuple(getattr(cfg.data, "caption_fields", ["s2_caption", "s2orc_caption"])),
        "image_ext": getattr(cfg.data, "image_ext", "png"),
        "ensure_unique": getattr(cfg.data, "ensure_unique", True),
        "return_class_labels": getattr(cfg.data, "return_class_labels", False),
        "val_split": getattr(cfg.data, "val_split", getattr(cfg.train, "val_split", 0.1)),
        "test_split": getattr(cfg.data, "test_split", getattr(cfg.train, "test_split", 0.0)),
        "seed": seed if seed is not None else getattr(cfg.train, "seed", 42)
    }

    data_module = MatchingDataModule(**data_args)

    model_module = ImageCaptioningLightningModule(cfg)

    # 配置trainer
    trainer_kwargs = {}

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("mps")
    else:
        device = torch.device("cpu")
        print("MPS not available, using CPU.")

    # 最大 epoch
    trainer_kwargs["max_epochs"] = cfg.train.max_epochs

    # Logger
    tb_logger = TensorBoardLogger("logs/", name="image_caption")
    trainer_kwargs["logger"] = tb_logger

    # 监控 val loss（LightningModule 中 log 的 key 为 "val/loss_gen"）
    monitor_metric = getattr(cfg.trainer, "checkpoint_monitor", "val/loss_gen") if hasattr(cfg,
                                                                                           "trainer") else "val/loss_gen"
    mode = getattr(cfg.trainer, "checkpoint_mode", "min") if hasattr(cfg, "trainer") else "min"
    save_top_k = getattr(cfg.trainer, "save_top_k", 3) if hasattr(cfg, "trainer") else 3
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor=monitor_metric,
        mode=mode,
        save_top_k=save_top_k,
        filename="{epoch:02d}-{" + monitor_metric.replace("/", "_") + ":.4f}",
        verbose=True
    )
    trainer_kwargs["callbacks"] = [checkpoint_callback]

    trainer = pl.Trainer(**trainer_kwargs)

    # 启动训练
    if args.resume:
        print(f"Resume training from checkpoint: {args.resume}")
        trainer.fit(model_module, datamodule=data_module, ckpt_path=args.resume)
    else:
        trainer.fit(model_module, datamodule=data_module)

    # 训练结束后输出最优模型路径
    best_path = checkpoint_callback.best_model_path
    print("Training finished. Best model path:", best_path)

    # 测试阶段
    do_test = False
    if hasattr(cfg.data, "test_files") and cfg.data.test_files:
        do_test = True
    elif getattr(cfg.data, "test_split", 0.0) > 0:
        do_test = True
    if do_test:
        print("Running test set evaluation")
        trainer.test(model_module, datamodule=data_module)

if __name__ == "__main__":
    main()
