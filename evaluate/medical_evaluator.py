# medical_evaluator.py
import json
import os
import re
from typing import List, Dict, Any, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv

class MedicalTermEvaluator:
    """
    加载医学术语（json 或 jsonl），并计算 / 绘制每个术语的混淆（TP/FP/FN/TN）。
    支持 JSON 列表或 JSONL（每行为 dict 或 string）。
    """

    def __init__(self, medical_dict_path: str):
        if not os.path.exists(medical_dict_path):
            raise FileNotFoundError(f"medical_dict_path not found: {medical_dict_path}")
        self.terms = self._load_terms_any_format(medical_dict_path)
        # 归一化并去重
        uniq = []
        seen = set()
        for t in self.terms:
            if not isinstance(t, str): continue
            s = t.strip().lower()
            if not s: continue
            if s not in seen:
                seen.add(s); uniq.append(s)
        self.terms = uniq
        # 标记 multiword
        self._is_multi = {t: (" " in t) for t in self.terms}

    def _load_terms_any_format(self, path: str) -> List[str]:
        """
        优先尝试 json.load，然后退回到逐行 json.loads（jsonl）。
        dict 类型会尝试从常见键提取值。
        """
        def extract_from_item(item):
            if item is None:
                return None
            if isinstance(item, str):
                return item
            if isinstance(item, (int, float)):
                return str(item)
            if isinstance(item, dict):
                for k in ("term","text","label","name","entity","concept","keyword","keywords","caption"):
                    if k in item:
                        v = item[k]
                        if isinstance(v, str): return v
                        if isinstance(v, (list,tuple)) and v and isinstance(v[0], str): return v[0]
                for v in item.values():
                    if isinstance(v, str):
                        return v
                try:
                    return json.dumps(item, ensure_ascii=False)
                except Exception:
                    return None
            if isinstance(item, (list,tuple)) and item:
                return extract_from_item(item[0])
            return None

        items = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for it in data:
                    ex = extract_from_item(it)
                    if ex: items.append(ex)
            elif isinstance(data, dict):
                for v in data.values():
                    if isinstance(v, list):
                        for it in v:
                            ex = extract_from_item(it)
                            if ex: items.append(ex)
                if not items:
                    items.append(json.dumps(data, ensure_ascii=False))
            else:
                items.append(str(data))
            return items
        except Exception:
            # fallback jsonl
            items = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line=line.strip()
                    if not line: continue
                    try:
                        obj = json.loads(line)
                        ex = extract_from_item(obj)
                        if ex: items.append(ex)
                    except Exception:
                        items.append(line)
            return items

    # ---------------- 归一化 / 匹配 ----------------
    def normalize_text(self, text: Optional[str]) -> str:
        if not text:
            return ""
        txt = str(text).lower()
        txt = re.sub(r"[-_/\\]+", " ", txt)
        txt = re.sub(r"[^\w\s]", " ", txt, flags=re.UNICODE)
        txt = re.sub(r"\s+", " ", txt).strip()
        return txt

    def _word_variants(self, word: str):
        # 简单生成复数/单数变体
        yield word
        if word.endswith("ies"):
            yield word[:-3] + "y"
        if word.endswith("ves"):
            yield word[:-3] + "f"
        if word.endswith("s") and len(word) > 3:
            yield word[:-1]
        if word.endswith("es") and len(word) > 3:
            yield word[:-2]

    def _term_in_text(self, term: str, text: str) -> bool:
        """
        匹配策略：
         - 多词短语：在归一化文本中做 substring（宽松匹配）
         - 单词：做 token membership，并尝试简单变体（处理复数）
        """
        if not term: return False
        norm_text = self.normalize_text(text)
        norm_term = self.normalize_text(term)
        if not norm_text or not norm_term:
            return False
        if " " in norm_term:
            return norm_term in norm_text
        toks = set(norm_text.split())
        if norm_term in toks:
            return True
        for v in self._word_variants(norm_term):
            if v in toks:
                return True
        return False

    # ---------------- 混淆矩阵计算 ----------------

    def compute_term_confusion(self, pred_captions: List[str], ref_captions: List[List[str]]) -> Dict[str, Dict[str, int]]:
        counts = {t: {"TP":0, "FP":0, "FN":0, "TN":0} for t in self.terms}
        n = min(len(pred_captions), len(ref_captions))
        for i in range(n):
            pred = pred_captions[i] or ""
            refs = ref_captions[i] if i < len(ref_captions) else []
            ref = refs[0] if isinstance(refs, (list,tuple)) and refs else (refs if isinstance(refs, str) else "")
            for t in self.terms:
                p = self._term_in_text(t, pred)
                r = self._term_in_text(t, ref)
                if r and p:
                    counts[t]["TP"] += 1
                elif (not r) and p:
                    counts[t]["FP"] += 1
                elif r and (not p):
                    counts[t]["FN"] += 1
                else:
                    counts[t]["TN"] += 1
        return counts

    # ---------------- 绘图与保存 ----------------

    def plot_term_confusion_matrix(
        self,
        pred_captions: List[str],
        ref_captions: List[List[str]],
        out_png: str = "outputs/term_confusion.png",
        out_json: Optional[str] = "outputs/term_confusion.json",
        out_csv: Optional[str] = "outputs/term_confusion.csv",
        normalize: bool = False,
        top_n: Optional[int] = None,
        figsize: tuple = (10, 8)
    ) -> Dict[str, Any]:
        os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
        counts = self.compute_term_confusion(pred_captions, ref_captions)
        terms = list(counts.keys())
        TP, FP, FN, TN = [], [], [], []
        for t in terms:
            TP.append(counts[t]["TP"])
            FP.append(counts[t]["FP"])
            FN.append(counts[t]["FN"])
            TN.append(counts[t]["TN"])
        mat = np.stack([TP, FP, FN, TN], axis=1).astype(np.float64)
        support = (mat[:,0] + mat[:,2]).astype(int)
        total = int(mat.sum())

        # 按 support 选取 top_n
        if top_n and 0 < top_n < len(terms):
            idx = np.argsort(-support)[:top_n]
            mat_show = mat[idx]
            terms_show = [terms[i] for i in idx]
            support_show = support[idx]
        else:
            mat_show = mat
            terms_show = terms
            support_show = support

        display_mat = mat_show.copy()
        if normalize:
            row_sums = display_mat.sum(axis=1, keepdims=True)
            row_sums[row_sums==0] = 1.0
            display_mat = display_mat / row_sums

        if total == 0:
            fig, ax = plt.subplots(figsize=(8,4))
            ax.axis("off")
            ax.text(0.5, 0.6, "No medical-term hits detected.\nPer-term confusion matrix is all zeros.",
                    ha="center", va="center", fontsize=12)
            ax.text(0.5, 0.35, f"num_terms={len(terms)}, num_samples={len(pred_captions)}",
                    ha="center", va="center", fontsize=10, color="gray")
            fig.savefig(out_png, dpi=150, bbox_inches="tight")
            plt.close(fig)
        else:
            vmax = max(1.0, np.percentile(display_mat, 99))
            fig, ax = plt.subplots(figsize=figsize)
            im = ax.imshow(display_mat, aspect='auto', interpolation='nearest', cmap='viridis', vmin=0, vmax=vmax)
            col_labels = ["TP","FP","FN","TN"]
            ax.set_xticks(np.arange(len(col_labels))); ax.set_xticklabels(col_labels, rotation=45, ha="right")
            ax.set_yticks(np.arange(len(terms_show)))
            yticks = [t if len(t)<=40 else t[:37]+"..." for t in terms_show]
            ax.set_yticklabels(yticks)
            for i in range(display_mat.shape[0]):
                for j in range(display_mat.shape[1]):
                    val = display_mat[i,j]
                    s = f"{val:.2f}" if normalize else f"{int(mat_show[i,j])}"
                    color = "white" if val > (vmax*0.5) else "black"
                    ax.text(j, i, s, ha="center", va="center", color=color, fontsize=8)
            ax.set_title("Per-term confusion (rows=terms, cols=TP/FP/FN/TN)")
            fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
            plt.tight_layout()
            fig.savefig(out_png, dpi=150)
            plt.close(fig)

        # 保存 json/csv
        if out_json:
            out = {}
            for i, t in enumerate(terms_show):
                out[t] = {"TP": int(mat_show[i,0]), "FP": int(mat_show[i,1]), "FN": int(mat_show[i,2]), "TN": int(mat_show[i,3]), "support": int(support_show[i])}
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
        if out_csv:
            with open(out_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f); w.writerow(["term","TP","FP","FN","TN","support"])
                for i,t in enumerate(terms_show):
                    w.writerow([t, int(mat_show[i,0]), int(mat_show[i,1]), int(mat_show[i,2]), int(mat_show[i,3]), int(support_show[i])])

        return {"png": out_png, "json": out_json, "csv": out_csv, "num_terms": len(terms), "total_counts": int(total)}
