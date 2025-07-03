import hashlib
import json
import logging
import os.path
import textwrap
import time
from pathlib import Path
from typing import List, Dict, Union, Optional
import torch

import matplotlib
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import BertTokenizer
import matplotlib.pyplot as plt
import random

# 配置日志
logger = logging.getLogger(__name__)
LABEL_MAPPING = {
    'radiology': 0,      # 影像学图像
    'scope': 1,          # 内窥镜图像
    'graph': 2,          # 图表
    'photo': 3,          # 照片
    'medicalimage': 4,   # 医学图像（通用）
}
NUM_CLASSES = len(LABEL_MAPPING)

class MatchingDataset(Dataset):
    def __init__(
            self,
            annotation_files: Union[str, List[str]],
            image_root: str,
            scibert_path: str,
            limit: Optional[int] = None,
            max_sequence_length: int = 256,
            use_refs: bool = True,
            caption_fields: List[str] = ("s2_caption", "s2orc_caption"),
            image_ext: str = "png",
            ensure_unique: bool = True,
            return_class_labels: bool = False
    ):
        """
        annotation_files: JSONL 文件路径或列表，每行一条 JSON 记录
        image_root: 图像根目录，image_path = image_root / f"{pdf_hash}_{fig_uri}.{image_ext}"
        scibert_path: 用于初始化 BertTokenizer 的预训练模型路径或名称
        limit: 可选，限制读取样本总数
        max_sequence_length: 最大 token 数（包括 BOS/EOS），实际 tokenize 时会扣除两位给 BOS/EOS
        use_refs: 是否使用 reference 字段生成样本
        caption_fields: 优先使用的 caption 字段名列表
        image_ext: 图像扩展名（如 "png"）
        ensure_unique: 是否根据 md5 去重
        return_class_labels: 是否返回多标签分类向量，用于多任务 learning
        """

        super().__init__()
        self.image_root = Path(image_root)
        if not self.image_root.exists():
            raise FileNotFoundError(f"图像目录不存在: {self.image_root}")

        self.max_sequence_length = max_sequence_length
        self.caption_fields = caption_fields
        self.use_refs = use_refs
        self.image_ext = image_ext
        self.ensure_unique = ensure_unique
        self.return_class_labels = return_class_labels

        # 初始化BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(scibert_path)
        # 增加 special tokens：<bos>, <eos>
        special_tokens = {}
        if self.tokenizer.bos_token is None:
            special_tokens['bos_token'] = '<bos>'
        if self.tokenizer.eos_token is None:
            special_tokens['eos_token'] = '<eos>'
        if special_tokens:
            self.tokenizer.add_special_tokens(special_tokens)
            logger.info(f"Added special tokens: {special_tokens}, new vocab size: {self.tokenizer.vocab_size}")
        # 保存特殊 token id
        if self.tokenizer.bos_token is not None:
            self.bos_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.bos_token)
        else:
            self.bos_token_id = None
        if self.tokenizer.eos_token is not None:
            self.eos_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)
        else:
            self.eos_token_id = None
        # pad_token_id 一般 BertTokenizer 已有 [PAD]
        self.pad_token_id = self.tokenizer.pad_token_id

        # 图像预处理流水线
        self.image_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 加载数据
        ann_files = [annotation_files] if isinstance(annotation_files, str) else annotation_files
        self.examples = self._load_examples(
            annotation_files=[annotation_files] if isinstance(annotation_files, str) else annotation_files,
            limit=limit,
            ensure_unique=ensure_unique
        )

        if not self.examples:
            raise ValueError("数据集为空，请检查路径和文件格式")

        # 初始化后统计 caption 长度分布，帮助设定 max_sequence_length
        logger.info(f"Loaded {len(self.examples)} examples from {len(ann_files)} annotation files")

    def _load_examples(self, annotation_files: List[str], limit: Optional[int], ensure_unique: bool) -> List[Dict]:
        """数据加载方法"""
        examples = []
        seen = set() if ensure_unique else None

        for file_path in annotation_files:
            file_path = Path(file_path)
            if not file_path.exists():
                logging.warning(f"跳过不存在的文件: {file_path}")
                continue

            with open(file_path, 'r', encoding='utf-8-sig') as f:
                for line in f:
                    if limit and len(examples) >= limit:
                        break
                    line = line.strip()

                    try:
                        record = json.loads(line)
                        samples = self._validate_record(record)
                        for sample, unique_key in samples:  # 解包样本和唯一键
                            if ensure_unique:
                                if unique_key in seen:
                                    continue
                                seen.add(unique_key)
                            examples.append(sample)
                    except Exception as e:
                        logging.error(f"解析错误 {file_path.name}: {str(e)}")
        return examples

    def _validate_record(self, record: Dict) -> List[tuple]:
        """
        验证单条数据记录并返回样本列表
        同时将 record 中的 s2_caption、s2orc_caption 放到 sample 顶层，以便 __getitem__ 取用。
        sample dict 包含至少：
          - 'image_path': str
          - 'captions': dict （保留，若需可用于可视化）
          - 顶层 's2_caption', 's2orc_caption' 字段，用于 __getitem__ 获取
          - 'metadata'
        """
        samples = []
        try:
            # 构建基础 metadata
            pdf_hash = record.get('pdf_hash', '').strip()
            fig_uri_str = record.get('fig_uri', '')
            fig_uri = Path(fig_uri_str).stem  # 去掉扩展名部分

            # 校验 pdf_hash 格式
            if not (isinstance(pdf_hash, str) and len(pdf_hash) == 40 and pdf_hash.isalnum()):
                raise ValueError(f"无效pdf_hash格式: {pdf_hash}")

            # 校验 fig_uri 格式
            if not (isinstance(fig_uri_str, str) and fig_uri_str.lower().endswith(f'.{self.image_ext}')):
                raise ValueError(f"无效fig_uri格式: {fig_uri_str}")

            # 图像路径验证
            image_path = self.image_root / f"{pdf_hash}_{fig_uri}.{self.image_ext}"

            metadata = {
                'pdf_hash': pdf_hash,
                'fig_uri': fig_uri,
                'radiology': record.get('radiology'),
                'scope': record.get('scope'),
                'predicted_type': record.get('predicted_type')
            }

            # Caption处理
            captions = {}
            valid_captions = []
            for field in self.caption_fields:
                value = record.get(field)
                if isinstance(value, str) and value.strip():
                    captions[field] = value.strip()
                    valid_captions.append(value.strip())
                else:
                    captions[field] = "字段缺失"

            # 验证至少有一个有效caption（非标记值）
            if not valid_captions:
                if not self.use_refs:
                    raise ValueError(f"所有caption字段均无效: {self.caption_fields}")

            # 获取record中原始的s2_caption、s2orc_caption,无论是否有效，都放到 sample 顶层
            s2_caption_text = record.get('s2_caption') if isinstance(record.get('s2_caption'), str) else ''
            s2orc_caption_text = record.get('s2orc_caption') if isinstance(record.get('s2orc_caption'), str) else ''
            # 构建主样本
            main_sample = {
                'image_path': str(image_path),
                'captions': captions,
                'metadata': metadata,
                's2_caption': s2_caption_text,
                's2orc_caption': s2orc_caption_text
            }
            # 生成主样本唯一键
            main_caption = valid_captions[0]
            text_hash = hashlib.md5(main_caption.encode()).hexdigest()[:8]
            unique_key = f"{pdf_hash}_{fig_uri}_{text_hash}"
            samples.append((main_sample, unique_key))

            # 参考文献处理
            if self.use_refs:
                # 安全获取并强制类型转换
                refs = record.get('s2orc_references')
                refs = refs if isinstance(refs, list) else []

                cleaned_refs = set()
                for ref in refs:
                    if isinstance(ref, str) and (clean_ref := ref.strip()):
                        cleaned_refs.add(clean_ref)

                for ref in cleaned_refs:
                    ref_sample = {
                        'image_path': str(image_path),
                        'captions': {'reference': ref},
                        'metadata': {
                            **metadata,
                            'is_reference': True,
                            'original_refs': refs
                        },
                        's2_caption': s2_caption_text,
                        's2orc_caption': s2orc_caption_text
                    }
                    # 生成参考文献唯一键
                    ref_hash = hashlib.md5(ref.encode()).hexdigest()[:8]
                    ref_unique_key = f"{pdf_hash}_{fig_uri}_{ref_hash}"
                    samples.append((ref_sample, ref_unique_key))

            return samples

        except Exception as e:
            logging.error(
                f"记录验证失败: {e}\n"
                f"记录内容: {json.dumps(record, ensure_ascii=False, indent=2)}"
            )
            return []

    def visualize_samples(self, num_samples=5, seed=None):
        """可视化方法"""
        output_dir = Path("visualization")
        output_dir.mkdir(parents=True, exist_ok=True)
        current_seed = seed or int(time.time() * 1000) % 2 ** 32
        random.seed(current_seed)

        indices = random.sample(range(len(self)), min(num_samples, len(self)))

        for i, idx in enumerate(indices):
            try:
                example = self.examples[idx]
                img = Image.open(example['image_path']).convert('RGB')

                # 显示所有caption字段
                caption_lines = []
                for field, text in example['captions'].items():
                    wrapped = textwrap.fill(f"{field}: {text}", width=60)
                    caption_lines.append(wrapped)

                full_caption = "\n\n".join(caption_lines)

                # 添加元数据信息
                meta_info = "\n".join([f"{k}: {v}" for k,v in example['metadata'].items()
                                      if k not in ['is_reference', 'original_refs']])
                full_caption = f"{full_caption}\n\n---Metadata---\n{meta_info}"

                fig, ax = plt.subplots(figsize=(10, 10))
                ax.imshow(img)
                ax.set_title(full_caption, fontsize=8, pad=5, loc='left')
                ax.axis('off')

                matplotlib.rcParams['font.family'] = 'Arial Unicode MS'
                save_path = output_dir / f"sample_{i + 1}.png"
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                logging.info(f"成功保存可视化样本: {save_path}")

            except Exception as e:
                logging.error(f"可视化失败索引 {idx}: {str(e)}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        example = self.examples[idx]
        metadata = example.get('metadata', {}).copy()

        # 图像加载
        image_path = example.get('image_path', '')
        print(f"[__getitem__] 加载路径: {image_path} | exists={os.path.exists(image_path)} | repr={repr(image_path)}")
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"{image_path} 不存在 (exists=False)")
            with Image.open(image_path) as img:
                img = img.convert('RGB')
                image = self.image_transform(img)
                metadata['image_error'] = False
        except Exception as e:
            logging.error(f"图像加载失败: {image_path} - {repr(e)}")
            image = torch.zeros(3, 224, 224)
            metadata['image_error'] = True

        # 获取主caption（优先使用第一个可用字段）
        caption = example.get('s2_caption', '')
        if not caption:
            caption = example.get('s2orc_caption', '')

        # Tokenize, 不添加padding, 仅truncation
        # 留出空位给BOS/EOS
        # 如果tokenizer未定义bos/eos, 则下面的self.bos_token_id/self.eos_token_id 可能为 None，应保证已添加
        max_len_no_special = max(0, self.max_sequence_length - 2)
        encoding = self.tokenizer(
            caption,
            add_special_tokens=False,
            truncation=True,
            max_length=max_len_no_special,
            return_attention_mask=False,
            return_tensors=None
        )
        token_ids = encoding.get('input_ids', [])  # list[int]
        # 构造 decoder_input_ids 和 decoder_labels
        if self.bos_token_id is not None:
            decoder_input_ids = [self.bos_token_id] + token_ids
        else:
            decoder_input_ids = token_ids.copy()  # 如果无 BOS，则直接用 token_ids
        if self.eos_token_id is not None:
            decoder_labels = token_ids + [self.eos_token_id]
        else:
            decoder_labels = token_ids.copy()  # 如果无 EOS，则直接用 token_ids

        # 转为 Tensor
        decoder_input_ids = torch.tensor(decoder_input_ids, dtype=torch.long)
        decoder_labels = torch.tensor(decoder_labels, dtype=torch.long)

        # 保证所有 id 落在 [0, vocab_size-1] 范围
        max_id = self.tokenizer.vocab_size - 1
        # 超出部分变成 max_id，即最后一个 token
        decoder_input_ids = decoder_input_ids.clamp(min=0, max=max_id)
        decoder_labels = decoder_labels.clamp(min=0, max=max_id)


        sample: Dict = {
            'image': image,
            'decoder_input_ids': decoder_input_ids,
            'decoder_labels': decoder_labels,
            'metadata': metadata,
            'image_path': image_path,
        }

        # 可选多标签分类
        if self.return_class_labels:
            # 根据 metadata 生成多标签向量
            label_vector = [0] * NUM_CLASSES
            # radiology, scope
            if metadata.get('radiology', False):
                label_vector[LABEL_MAPPING['radiology']] = 1
            if metadata.get('scope', False):
                label_vector[LABEL_MAPPING['scope']] = 1
            # predicted_type
            predicted_type = metadata.get('predicted_type', '').strip().lower()
            predicted_type = predicted_type.replace(" ", "").replace("-", "")
            if 'medicalimage' in predicted_type:
                label_vector[LABEL_MAPPING['radiology']] = 1
            for label_name, index in LABEL_MAPPING.items():
                if label_name != 'medicalimage' and label_name in predicted_type:
                    label_vector[index] = 1
            if sum(label_vector) == 0:
                label_vector[LABEL_MAPPING['medicalimage']] = 1
                logger.info(f"样本 idx={idx} 无明确分类标签，标记为 medicalimage")
            sample['class_labels'] = torch.tensor(label_vector, dtype=torch.float)

        return sample
