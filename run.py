import argparse
import os
from omegaconf import OmegaConf
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger

from dataset_managing.Matching_Dataset_Module import MatchingDataModule
from Decoder.LightningModule import ImageCaptioningLightningModule

def main():
    parser = argparse.ArgumentParser(description="Train Image Captioning with Lightning")
    parser.add_argument(
        "--config", type=str,
        default="/Users/vegeta/PycharmProjects/MediGloria/config/config.yaml",
        help="Path to YAML config file, e.g. config.yaml"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from (optional)"
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
        pl.seed_everything(seed)

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
        print("Running test set evaluation...")
        # LightningModule 需实现 test_step/test_epoch_end，如无可简单调用:
        trainer.test(model_module, datamodule=data_module)

if __name__ == "__main__":
    main()