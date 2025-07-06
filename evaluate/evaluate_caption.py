# evaluation.py
import json
import numpy as np
import torch
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu
from rouge_score import rouge_scorer
from pycocoevalcap.meteor.meteor import Meteor


class EvaluateCaption:
    """
    封装通用指标计算和医学领域完整评估流程的类。

    Methods
    -------
    compute_generic_metrics(pred_captions, ref_captions)
        计算 BLEU‑1/4, ROUGE‑1/2/L, METEOR。

    evaluate_medical_captioning_model(model, eval_loader, modality_labels,
                                      medical_dict_path=None, expert_feedback_path=None)
        运行生成、通用指标、医学术语评估、错误分析、模态对比与专家反馈，保存报告和可视化。
    """

    @staticmethod
    def compute_generic_metrics(pred_captions, ref_captions):
        results = {}
        # BLEU
        refs_for_bleu = [[r.split() for r in refs] for refs in ref_captions]
        hyps_for_bleu = [p.split() for p in pred_captions]
        results['bleu-1'] = corpus_bleu(refs_for_bleu, hyps_for_bleu, weights=(1, 0, 0, 0))
        results['bleu-4'] = corpus_bleu(refs_for_bleu, hyps_for_bleu, weights=(0, 0, 0, 1))
        # ROUGE
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge1, rouge2, rougeL = [], [], []
        for pred, refs in zip(pred_captions, ref_captions):
            sc = scorer.score(pred, refs[0])
            rouge1.append(sc['rouge1'].fmeasure)
            rouge2.append(sc['rouge2'].fmeasure)
            rougeL.append(sc['rougeL'].fmeasure)
        results['rouge-1'] = float(np.mean(rouge1))
        results['rouge-2'] = float(np.mean(rouge2))
        results['rouge-L'] = float(np.mean(rougeL))
        # METEOR
        coco_ref = {'images': [], 'annotations': []}
        coco_pred = {'images': [], 'annotations': []}
        for i, refs in enumerate(ref_captions):
            coco_ref['images'].append({'id': i})
            for j, r in enumerate(refs):
                coco_ref['annotations'].append({
                    'image_id': i, 'id': i * 100 + j, 'caption': r
                })
        for i, p in enumerate(pred_captions):
            coco_pred['images'].append({'id': i})
            coco_pred['annotations'].append({
                'image_id': i, 'id': i, 'caption': p
            })
        meteor = Meteor()
        meteor_score, _ = meteor.compute_score(coco_ref, coco_pred)
        results['meteor'] = float(meteor_score)
        return results

    def evaluate_medical_captioning_model(
            self, model, eval_loader, modality_labels,
            medical_dict_path=None, expert_feedback_path=None
    ):
        # 1. 生成预测与收集参考
        pred_captions, ref_captions = [], []
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Generating captions"):
                # 假设 batch 包含 'images' 和 'raw_captions'
                imgs = batch['images']
                preds = model.generate_captions(imgs)
                pred_captions.extend(preds)
                ref_captions.extend(batch['raw_captions'])
        # 2. 通用指标
        generic_metrics = self.compute_generic_metrics(pred_captions, ref_captions)
        # 3. 医学术语评估
        med_metrics, errors = {}, {}
        if medical_dict_path:
            from medical_evaluator import MedicalTermEvaluator
            mte = MedicalTermEvaluator(medical_dict_path)
            med_metrics, term_count = mte.compute_medical_accuracy(pred_captions)
            errors = mte.analyze_term_errors(pred_captions, ref_captions, term_count)
        # 4. 模态 & 专家反馈评估
        modality_cmp, expert_fb = {}, {}
        if modality_labels is not None:
            from manual_eval import manual_evaluation
            manual = manual_evaluation(
                pred_captions, ref_captions, modality_labels,
                expert_feedback=expert_feedback_path and json.load(open(expert_feedback_path))
            )
            modality_cmp = manual.get('modality_performance', {})
            expert_fb = manual.get('expert_feedback', {})
        # 5. 汇总 & 保存报告
        final_results = {
            "generic_metrics": generic_metrics,
            "medical_specific_metrics": med_metrics,
            "error_analysis": errors,
            "modality_comparison": modality_cmp,
            "expert_feedback": expert_fb
        }
        with open("evaluation_report.json", "w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=4, ensure_ascii=False)
        # 6. 可视化示例：术语错误分布
        try:
            import matplotlib.pyplot as plt
            plt.bar(['missing', 'incorrect'],
                    [len(errors.get('missing_terms', [])),
                     len(errors.get('incorrect_terms', []))])
            plt.title("Medical Term Errors")
            plt.savefig("term_errors.png", bbox_inches="tight")
        except:
            pass
        return final_results
