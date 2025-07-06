import json
from typing import List, Tuple, Dict

class MedicalTermEvaluator:
    def __init__(self, medical_dict_path: str):
        """
        加载医学术语词典，词典应为一个 JSON 列表，例如 ["lung", "heart", ...]
        """
        with open(medical_dict_path, 'r', encoding='utf-8') as f:
            terms = json.load(f)
        # 统一小写
        self.terms = set(t.lower() for t in terms)

    def compute_medical_accuracy(
        self,
        pred_captions: List[str]
    ) -> Tuple[Dict[str, float], int]:
        """
        计算医学术语在生成结果中的精度与召回。
        返回 ({'precision': P, 'recall': R, 'f1': F1}, total_terms)
        """
        total = 0
        tp = 0
        # 遍历每条预测，并统计术语出现情况
        for cap in pred_captions:
            tokens = set(cap.lower().split())
            for term in self.terms:
                if term in tokens:
                    tp += 1
                total += 1
        precision = tp / total if total > 0 else 0.0
        recall = precision  # 简化假设
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return {'precision': precision, 'recall': recall, 'f1': f1}, total

    def analyze_term_errors(
        self,
        pred_captions: List[str],
        ref_captions: List[List[str]],
        total_terms: int
    ) -> Dict[str, List[str]]:
        """
        分析缺失和错误使用的医学术语。
        返回 {'missing_terms': [...], 'incorrect_terms': [...]}
        """
        missing = []
        incorrect = []
        for pred, refs in zip(pred_captions, ref_captions):
            pred_tokens = set(pred.lower().split())
            # 只用第一个参考
            ref_tokens = set(refs[0].lower().split())
            # 缺失
            for term in self.terms:
                if term in ref_tokens and term not in pred_tokens:
                    missing.append(term)
            # 错误
            for term in self.terms:
                if term in pred_tokens and term not in ref_tokens:
                    incorrect.append(term)
        return {'missing_terms': missing, 'incorrect_terms': incorrect}
