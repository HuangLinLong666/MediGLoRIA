# run_eval.py
import json
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from evaluate.evaluate_caption import EvaluateCaption
from dataset_managing.Matching_Dataset_Module import MatchingDataModule
from Decoder.LightningModule import ImageCaptioningLightningModule


def main():
    cfg = OmegaConf.load("config.yaml")
    seed_everything(cfg.data.seed)

    # 1) 构建 DataModule，仅做 test_split
    dm = MatchingDataModule(
        annotation_files={"train": cfg.data.train_files, "test": cfg.data.test_files},
        image_root=cfg.data.image_root,
        scibert_path=cfg.model.text.bert_type,
        batch_size=cfg.train.batch_size,
        max_sequence_length=cfg.train.max_seq_length,
        use_refs=cfg.data.use_refs,
        caption_fields=tuple(cfg.data.caption_fields),
        image_ext=cfg.data.image_ext,
        ensure_unique=cfg.data.ensure_unique,
        return_class_labels=False,
        val_split=0.0, test_split=cfg.data.test_split,
        seed=cfg.data.seed
    )
    dm.prepare_data()
    dm.setup(stage="test")
    test_loader = dm.test_dataloader()

    # 2) 加载模型
    model = ImageCaptioningLightningModule.load_from_checkpoint(
        "logs/image_caption/version_0/checkpoints/best.ckpt",
        cfg=cfg
    ).eval().to("cuda")

    # 3) 评估
    evaluator = EvaluateCaption(
        medical_dict_path="data/medical_dict.json",
        expert_feedback_path="data/expert_feedback.json"
    )
    modality_labels = dm.get_modality_labels() if hasattr(dm, "get_modality_labels") else None

    results = evaluator.evaluate_medical_captioning_model(
        model, test_loader, modality_labels
    )
    print("=== Final Evaluation Results ===")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
