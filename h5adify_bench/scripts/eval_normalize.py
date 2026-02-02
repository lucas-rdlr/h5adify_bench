# scripts/eval_normalize.py
from __future__ import annotations
import re
from typing import Any, Dict, List, Optional, Set, Tuple

def _norm_str(x: str) -> str:
    x = x.strip().lower()
    x = re.sub(r"\s+", " ", x)
    x = x.replace("–", "-").replace("—", "-")
    return x

def load_synonyms(norm_cfg: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    """
    Build inverse maps:
      category -> {synonym_norm -> canonical}
    """
    inv: Dict[str, Dict[str, str]] = {}
    for cat, canon_map in norm_cfg.items():
        inv[cat] = {}
        for canon, syns in canon_map.items():
            inv[cat][_norm_str(canon)] = canon
            for s in syns:
                inv[cat][_norm_str(str(s))] = canon
    return inv

def canonize_value(cat: str, v: str, inv: Dict[str, Dict[str, str]]) -> str:
    vv = _norm_str(v)
    if cat in inv and vv in inv[cat]:
        return inv[cat][vv]
    return vv

def to_str_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    if isinstance(x, (list, tuple, set)):
        out = []
        for t in x:
            if t is None:
                continue
            if isinstance(t, (list, tuple, set)):
                out.extend([str(z) for z in t if z is not None])
            else:
                out.append(str(t))
        return out
    if isinstance(x, dict):
        # common pattern: list of objects with db/id
        return [str(x)]
    return [str(x)]

def canonize_set(cat: str, values: Any, inv: Dict[str, Dict[str, str]]) -> Set[str]:
    items = to_str_list(values)
    return set(canonize_value(cat, it, inv) for it in items if str(it).strip())
