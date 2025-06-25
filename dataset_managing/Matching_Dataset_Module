import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from .dataset_reader import MatchingDataset

import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch, pad_token_id):
    """
    batch: list of sample dict, 每个含 'image', 'decoder_input_ids', 'decoder_labels', 可选 'class_labels'
    """
    # 过滤 None
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    images = torch.stack([item['image'] for item in batch], dim=0)

    # pad decoder_input_ids 和 decoder_labels
    input_id_list = [item['decoder_input_ids'] for item in batch]  # list of [Li]
    label_list = [item['decoder_labels'] for item in batch]        # list of [Li]

    # pad_sequence 默认 pad 到最长 seq，用 pad_token_id；batch_first=True 得到 [B, L_max]
    decoder_input_ids = pad_sequence(input_id_list, batch_first=True, padding_value=pad_token_id)
    # labels pad 时用 -100
    labels = pad_sequence(label_list, batch_first=True, padding_value=-100)

    # decoder_attention_mask: 1 for real tokens, 0 for pad
    decoder_attention_mask = (decoder_input_ids != pad_token_id).long()

    out = {
        'images': images,
        'decoder_input_ids': decoder_input_ids,
        'decoder_attention_mask': decoder_attention_mask,
        'labels': labels
    }
    # 如果有 class_labels
    if 'class_labels' in batch[0]:
        class_labels = torch.stack([item['class_labels'] for item in batch], dim=0)
        out['class_labels'] = class_labels
    return out

class MatchingDataModule(pl.LightningDataModule):
    def __init__(
        self,
        annotation_files: dict,  # {'train': [...], 'val': [...], 'test': [...]}
        image_root: str,
        scibert_path: str,
        batch_size: int = 16,
        num_workers: int = 0,
        max_sequence_length: int = 256,
        use_refs: bool = True,
        caption_fields=("s2_caption", "s2orc_caption"),
        image_ext: str = "png",
        ensure_unique: bool = True,
        return_class_labels: bool = False,
        val_split: float = 0.1,
        test_split: float = 0.0,
        seed: int = 42,
        limit: int = 1000
    ):
        super().__init__()
        self.annotation_files = annotation_files
        self.image_root = image_root
        self.scibert_path = scibert_path

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_sequence_length = max_sequence_length
        self.use_refs = use_refs
        self.caption_fields = caption_fields
        self.image_ext = image_ext
        self.ensure_unique = ensure_unique
        self.return_class_labels = return_class_labels

        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed
        self.limit = limit

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):

        # 按阶段加载数据集
        train_files = self.annotation_files.get('train')
        if not train_files:
            raise ValueError('请提供 train 文件列表')
        full_dataset = MatchingDataset(
            annotation_files=train_files,
            image_root=self.image_root,
            scibert_path=self.scibert_path,
            max_sequence_length=self.max_sequence_length,
            use_refs=self.use_refs,
            caption_fields=self.caption_fields,
            image_ext=self.image_ext,
            ensure_unique=self.ensure_unique,
            return_class_labels=self.return_class_labels,
            limit=self.limit
        )
        total = len(full_dataset)

        # External val/test files
        val_files = self.annotation_files.get('val')
        test_files = self.annotation_files.get('test')

        val_size = int(total * self.val_split)
        test_size = int(total * self.test_split)
        train_size = total - val_size - test_size
        if train_size <= 0:
            raise ValueError('val_split/test_split 配置过大，导致 train_size <= 0')
        lengths = [train_size]
        lengths.append(val_size)
        if test_size > 0:
            lengths.append(test_size)
        generator = torch.Generator().manual_seed(self.seed)
        splits = random_split(full_dataset, lengths, generator=generator)
        self.train_dataset = splits[0]
        self.val_dataset = splits[1] if val_size > 0 else None
        self.test_dataset = splits[2] if test_size > 0 else None

        if test_files and self.test_dataset is None:
            self.test_dataset = MatchingDataset(
                annotation_files=test_files,
                image_root=self.image_root,
                scibert_path=self.scibert_path,
                max_sequence_length=self.max_sequence_length,
                use_refs=self.use_refs,
                caption_fields=self.caption_fields,
                image_ext=self.image_ext,
                ensure_unique=self.ensure_unique,
                return_class_labels=self.return_class_labels,
                limit=self.limit
            )

            # For test stage fallback
        if stage == 'test' and self.test_dataset is None:
            self.test_dataset = self.train_dataset

    def train_dataloader(self):
        underlying = self.train_dataset.dataset if isinstance(self.train_dataset, torch.utils.data.Subset) else self.train_dataset
        pad_token_id = underlying.pad_token_id
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True,
            collate_fn=lambda b: collate_fn(b, pad_token_id=pad_token_id)
        )

    def val_dataloader(self):
        underlying = self.train_dataset.dataset if isinstance(self.train_dataset, torch.utils.data.Subset) else self.train_dataset
        pad_token_id = underlying.pad_token_id
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            collate_fn=lambda b: collate_fn(b, pad_token_id=pad_token_id)
        )

    def test_dataloader(self):
        underlying = self.train_dataset.dataset if isinstance(self.train_dataset, torch.utils.data.Subset) else self.train_dataset
        pad_token_id = underlying.pad_token_id
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            collate_fn=lambda b: collate_fn(b, pad_token_id=pad_token_id)
        )
