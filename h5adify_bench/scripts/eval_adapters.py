# scripts/eval_adapters.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Set, Tuple
import json
from pathlib import Path

def _get_by_path(obj: Any, path: str) -> Any:
    """
    Supports dotted paths: "schema.title", "paper.title".
    """
    cur = obj
    for part in path.split("."):
        if cur is None:
            return None
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur

def pick_root(pred_obj: Dict[str, Any], root_candidates: List[str]) -> Dict[str, Any]:
    for r in root_candidates:
        v = _get_by_path(pred_obj, r)
        if isinstance(v, dict):
            return v
    return pred_obj

def pick_first(pred_obj: Dict[str, Any], candidates: List[str]) -> Any:
    for c in candidates:
        v = _get_by_path(pred_obj, c)
        if v is not None:
            return v
    return None

def gold_to_canonical(gold_item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Expects gold JSON entries similar to prior response:
      gold_standard.species, gold_standard.assay_modalities, gold_standard.technology
      data_availability.repositories (list with type/accession)
      paper.title (optional)
    """
    out: Dict[str, Any] = {}
    gs = gold_item.get("gold_standard", {}) or {}
    out["title"] = (gold_item.get("paper", {}) or {}).get("title") or gold_item.get("study_title") or None
    out["species"] = gs.get("species", [])
    out["assay_modalities"] = gs.get("assay_modalities", [])
    out["technology"] = gs.get("technology", [])
    out["tissue_scope"] = gs.get("tissue_scope", None)
    out["disease_or_context"] = gs.get("disease_or_context", None)

    repos = []
    da = gold_item.get("data_availability", {}) or {}
    for r in (da.get("repositories") or []):
        if isinstance(r, dict):
            t = r.get("type")
            acc = r.get("accession")
            if t and acc:
                repos.append({"type": t, "accession": acc})
    out["repositories"] = repos
    return out

def pred_to_canonical(pred_obj: Dict[str, Any], field_map: Dict[str, Any], method: str) -> Dict[str, Any]:
    """
    field_map has:
      method.root: candidates for root object (e.g. ["schema"])
      method.<field>: list of candidate paths
    """
    m = field_map.get(method, {})
    root = pick_root(pred_obj, m.get("root", field_map.get("default", {}).get("root", [])))
    dflt = field_map.get("default", {})

    def get_field(name: str) -> Any:
        cands = m.get(name, dflt.get(name, []))
        return pick_first(root, cands) if cands else None

    out = {
        "title": get_field("title"),
        "species": get_field("species"),
        "assay_modalities": get_field("assay_modalities"),
        "technology": get_field("technology"),
        "tissue_scope": get_field("tissue_scope"),
        "disease_or_context": get_field("disease_or_context"),
        "repositories": get_field("repositories"),
    }
    # If repositories is not already list of dicts, keep as-is; evaluator will normalize.
    return out
