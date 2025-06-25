import logging
import shutil
import textwrap
from pathlib import Path
from torchvision.transforms.functional import to_pil_image
import torch
from dataset_managing.dataset_reader import MatchingDataset
from dataset_managing.unet import UnetEnhancer
from torchvision import transforms
import random
from PIL import Image


def main():
    try:
        # 初始化数据集
        dataset = MatchingDataset(
            annotation_files=["/Users/vegeta/s2_full_figures_oa_nonroco_combined_medical_top4_public.jsonl"],
            image_root="/Users/vegeta/release/figures",
            scibert_path="/Users/vegeta/JupterLab/scibert",
            limit=1000,
            ensure_unique=True,
            return_class_labels=True
        )
        logging.info(f"成功加载数据集，样本数: {len(dataset)}")

        for i in range(20):
            sample = dataset[i]
            print("image:", sample['image'].shape)
            print("decoder_input_ids:", sample['decoder_input_ids'])
            print("decoder_labels:", sample['decoder_labels'])
            print("class_labels:", sample.get('class_labels'))

        # 创建可视化目录并清空旧文件
        vis_dir = Path("visualization")
        if vis_dir.exists():
            shutil.rmtree(vis_dir)
        vis_dir.mkdir(parents=True)

        # 初始化模型
        device = torch.device("mps" if torch.cuda.is_available() else "cpu")
        model = UnetEnhancer().to(device)
        model.load_state_dict(torch.load("best_model_epoch3.pth", map_location=device))
        model.eval()
        logging.info("模型权重加载成功")

        # 随机选择样本
        selected_indices = random.sample(range(len(dataset)), 5)

        # 反归一化参数
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        with torch.no_grad():
            for sample_idx in selected_indices:
                sample = dataset[sample_idx]
                input_tensor = sample['image'].unsqueeze(0).to(device)
                metadata = sample['metadata']
                captions = sample.get('captions', {})

                # 生成文件名
                pdf_hash = metadata['pdf_hash']
                fig_uri = metadata['fig_uri']
                base_name = f"{pdf_hash}_{fig_uri.replace('/', '_')}"

                # 使用Path对象构建完整路径
                orig_filename = vis_dir / f"{base_name}_original.png"
                enh_filename = vis_dir / f"{base_name}_enhanced.png"
                txt_filename = vis_dir / f"{base_name}_original.txt"

                # 图像处理
                # 检查图像是否有效
                if metadata.get('image_error') or metadata.get('invalid_image'):
                    logging.warning(f"图像标记为无效: {base_name}")
                    # 创建黑色占位图像
                    placeholder = torch.zeros(3, 224, 224)
                    orig_pil = to_pil_image(placeholder)
                    enh_pil = to_pil_image(placeholder)
                else:
                    # 直接加载原始图像文件
                    try:
                        orig_img = Image.open(sample['image_path']).convert('RGB')
                        orig_pil = orig_img.copy()
                        orig_img.save(orig_filename)

                        # 模型处理
                        input_tensor = sample['image'].unsqueeze(0).to(device)
                        enhanced_tensor = model(input_tensor)

                        # 反归一化处理
                        enh_img = (enhanced_tensor * std.to(device) + mean.to(device)).squeeze().cpu()
                        enh_pil = transforms.ToPILImage()(enh_img)
                    except Exception as e:
                        logging.error(f"图像处理失败: {str(e)}")
                        continue

                # 保存增强后的图像
                enh_pil.save(enh_filename)

                # 保存文本
                with open(txt_filename, "w", encoding="utf-8") as f:
                    # 使用Path对象的name属性
                    f.write(f"图片名称: {orig_filename.name}\n")
                    f.write(f"PDF哈希值: {pdf_hash}\n")
                    f.write(f"图表URI: {metadata['fig_uri']}\n")

                    # 图像状态
                    if metadata.get('image_error'):
                        f.write("\n 图像状态: 加载失败\n")
                    elif metadata.get('invalid_image'):
                        f.write("\n 图像状态: 标记为无效\n")
                    else:
                        f.write("\n 图像状态: 有效\n")

                    # 样本类型
                    if metadata.get('is_reference'):
                        f.write("样本类型: 参考文献\n")
                    else:
                        f.write("样本类型: 主样本\n")

                    # 元数据
                    f.write("\n 元数据 \n")
                    meta_mapping = [
                        ('radiology', 'Radiological image'),
                        ('scope', 'Microscopic image'),
                        ('predicted_type', 'Image type')
                    ]
                    for field, display in meta_mapping:
                        value = metadata.get(field, '未知字段')
                        f.write(f"{display}: {value}\n")

                    # Caption信息
                    if metadata.get('is_reference'):
                        # 根据 pdf_hash 和 fig_uri 找主样本
                        pdf = metadata['pdf_hash']
                        uri = metadata['fig_uri']
                        for ex in dataset.examples:
                            meta = ex['metadata']
                            if (meta.get('pdf_hash') == pdf
                                    and meta.get('fig_uri') == uri
                                    and not meta.get('is_reference', False)):
                                main_caps = ex['captions']
                                break
                        else:
                            main_caps = {}

                        # 写入主样本中的 s2_caption 和 s2orc_caption
                        f.write("\n—— 对应主样本的 Caption ——\n")
                        for field in ['s2_caption', 's2orc_caption']:
                            text = main_caps.get(field)
                            if isinstance(text, str) and text.strip() and text not in ["字段缺失", ""]:
                                f.write(f"\n{field}:\n")
                                f.write(textwrap.fill(text, width=80) + "\n")

                    f.write("\n 描述信息 \n")
                    valid_descriptions = False
                    for field, text in captions.items():
                        # 跳过无效描述
                        if text in ["字段缺失", "空字符串", "无有效内容"]:
                            continue

                        valid_descriptions = True
                        f.write(f"\n{field}:\n")
                        wrapped_text = textwrap.fill(text.strip(), width=80)
                        f.write(wrapped_text + "\n")

                    if not valid_descriptions:
                        f.write("\n 无有效描述信息\n")

                logging.info(f"已保存: {base_name}*")

    except Exception as e:
        logging.error(f"程序执行失败: {str(e)}")
        raise


if __name__ == "__main__":
    main()
