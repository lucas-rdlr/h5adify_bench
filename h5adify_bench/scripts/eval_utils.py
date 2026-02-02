# scripts/eval_utils.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import re

def exact_match(a: str, b: str) -> float:
    return 1.0 if a == b else 0.0

def _tokenize(s: str) -> List[str]:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.split() if s else []

def token_f1(pred: str, gold: str) -> float:
    """SQuAD-style token F1 on strings."""
    p = _tokenize(pred)
    g = _tokenize(gold)
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    common = {}
    for t in p:
        common[t] = common.get(t, 0) + 1
    overlap = 0
    for t in g:
        if common.get(t, 0) > 0:
            overlap += 1
            common[t] -= 1
    prec = overlap / len(p)
    rec = overlap / len(g)
    return 0.0 if (prec + rec) == 0 else (2 * prec * rec) / (prec + rec)

def rouge_l_f1(pred: str, gold: str) -> float:
    """ROUGE-L F1 using LCS length (string tokens)."""
    p = _tokenize(pred)
    g = _tokenize(gold)
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    # LCS DP
    dp = [[0] * (len(g) + 1) for _ in range(len(p) + 1)]
    for i in range(1, len(p) + 1):
        for j in range(1, len(g) + 1):
            if p[i - 1] == g[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[-1][-1]
    prec = lcs / len(p)
    rec = lcs / len(g)
    return 0.0 if (prec + rec) == 0 else (2 * prec * rec) / (prec + rec)

def set_prf(pred: Set[str], gold: Set[str]) -> Tuple[float, float, float]:
    if not pred and not gold:
        return 1.0, 1.0, 1.0
    if not pred and gold:
        return 1.0, 0.0, 0.0
    if pred and not gold:
        return 0.0, 1.0, 0.0
    tp = len(pred & gold)
    fp = len(pred - gold)
    fn = len(gold - pred)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 0.0 if (p + r) == 0 else (2 * p * r) / (p + r)
    return p, r, f1

def jaccard(pred: Set[str], gold: Set[str]) -> float:
    if not pred and not gold:
        return 1.0
    u = len(pred | gold)
    return len(pred & gold) / u if u else 0.0

def slot_error_rate(pred: Set[str], gold: Set[str]) -> float:
    """
    SER used in structured extraction tasks:
      SER = (S + D + I) / N
    With set comparison:
      I = |pred - gold|
      D = |gold - pred|
      S ~ 0 (no alignment), so SER = (I + D)/|gold|
    """
    if not gold and not pred:
        return 0.0
    if not gold and pred:
        return 1.0
    ins = len(pred - gold)
    dele = len(gold - pred)
    return (ins + dele) / max(1, len(gold))

def coverage(pred_field_is_filled: bool) -> float:
    return 1.0 if pred_field_is_filled else 0.0
