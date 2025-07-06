from typing import List, Dict, Any
from nltk.translate.bleu_score import corpus_bleu


def manual_evaluation(
        preds: List[str],
        refs: List[List[str]],
        modality_labels: List[Any],
        expert_feedback: Dict = None
) -> Dict[str, Any]:
    """
    简单做按模态分组的 BLEU-1 对比，以及整合专家反馈。

    :param preds: 预测字幕列表
    :param refs: 每张图对应的参考字幕列表
    :param modality_labels: 与 preds/refs 对齐的模态标签列表
    :param expert_feedback: 可选专家反馈字典
    :return:
        {
          'modality_performance': {modality: bleu1_score, ...},
          'expert_feedback': expert_feedback or {}
        }
    """
    # 按模态分组
    mod2preds = {}
    mod2refs = {}
    for p, r, m in zip(preds, refs, modality_labels):
        mod2preds.setdefault(m, []).append(p.split())
        mod2refs.setdefault(m, []).append([ref.split() for ref in r])

    perf = {}
    for m in mod2preds:
        try:
            score = corpus_bleu(mod2refs[m], mod2preds[m], weights=(1, 0, 0, 0))
        except Exception:
            score = 0.0
        perf[m] = float(score)

    return {
        'modality_performance': perf,
        'expert_feedback': expert_feedback or {}
    }
