#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import anndata as ad
from sklearn.metrics import adjusted_mutual_info_score

from common import read_json, read_yaml, ensure_dir, write_json

# Fields that map to .obs columns
MAP_FIELDS = ["batch", "sample", "donor", "domain", "sex", "species", "technology"]

# Fields that can be implicit dataset-wide values
CANON_FIELDS = ["species", "technology", "sex"]

def load_all_preds(results_dir: Path) -> List[Path]:
    return list(results_dir.glob("**/pred.json"))

def extract_pred_mapping(pred: Dict[str, Any]) -> Dict[str, Optional[str]]:
    rep = pred.get("report", {})
    chosen = rep.get("chosen_keys", {}) if isinstance(rep, dict) else {}
    out = {}
    for f in MAP_FIELDS:
        v = chosen.get(f, None)
        out[f] = v if (isinstance(v, str) or v is None) else None
    return out

def extract_pred_canon_list(pred: Dict[str, Any]) -> Dict[str, List[str]]:
    cp = pred.get("canon_preview", {}) if isinstance(pred.get("canon_preview", {}), dict) else {}
    out = {}
    for f in CANON_FIELDS:
        vals = []
        try:
            ex_list = cp.get(f, {}).get("examples", [])
            if ex_list:
                vals = [str(x).strip().lower() for x in ex_list]
        except Exception:
            vals = []
        out[f] = vals
    return out

def compute_soft_score(adata: ad.AnnData, gold_col: str, pred_col: str) -> int:
    """Returns 1 if columns have identical information (AMI > 0.95), else 0."""
    if gold_col not in adata.obs.columns or pred_col not in adata.obs.columns:
        return 0
    try:
        y_true = adata.obs[gold_col].astype(str)
        y_pred = adata.obs[pred_col].astype(str)
        # Drop NaNs for fair comparison
        mask = ~(y_true.isna() | y_pred.isna() | (y_true == "nan") | (y_pred == "nan"))
        if mask.sum() < 2:
            return 0
        score = adjusted_mutual_info_score(y_true[mask], y_pred[mask])
        return 1 if score > 0.95 else 0
    except Exception:
        return 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="experiments/gold/doi20_gold.json")
    ap.add_argument("--results", default="experiments/results/part1")
    ap.add_argument("--outdir", default="experiments/results/part1_complet_scores")
    args = ap.parse_args()

    ensure_dir(args.outdir)

    gold = read_json(args.gold)
    gold_index = {(it["doi"], it["dataset_id"]): it for it in gold["items"]}

    preds = load_all_preds(Path(args.results))
    rows = []

    # Cache anndata objects to speed up soft scoring
    adata_cache = {}

    for pj in preds:
        pred = read_json(pj)
        doi = pred.get("doi")
        dsid = pred.get("dataset_id")
        key = (doi, dsid)
        if key not in gold_index:
            continue
        g = gold_index[key]

        pred_map = extract_pred_mapping(pred)
        pred_canon_lists = extract_pred_canon_list(pred)
        h5ad_path = g.get("h5ad_path")

        # 1. SCORING: Column Mapping (Hard & Soft)
        acc_hard = {}
        acc_soft = {}
        halluc_fields = {}

        # Lazy load adata only if we need it for soft scoring
        adata = None 

        for f in MAP_FIELDS:
            gk = g["gold_key"].get(f, None)
            pk = pred_map.get(f, None)

            # Hallucination: predicted key not in obs_columns at all
            halluc = False
            if pk is not None and pk not in g["obs_columns"]:
                halluc = True
            halluc_fields[f] = int(halluc)

            if gk is None:
                # If gold says NO column exists, predicting anything is wrong (Hard=0, Soft=0)
                # unless they correctly predicted None (Hard=1, Soft=1)
                is_correct = int(pk is None)
                acc_hard[f] = is_correct
                acc_soft[f] = is_correct
            else:
                # Gold says a column exists
                if pk is None:
                    # Missed it
                    acc_hard[f] = 0
                    acc_soft[f] = 0
                elif pk == gk:
                    # Exact match
                    acc_hard[f] = 1
                    acc_soft[f] = 1
                else:
                    # Mismatch name. Hard=0. Let's check Soft score.
                    acc_hard[f] = 0
                    
                    if not halluc:
                        # Load data if not already loaded
                        if adata is None:
                            if h5ad_path not in adata_cache:
                                try:
                                    # Load backed to save RAM, we only need .obs
                                    adata_cache[h5ad_path] = ad.read_h5ad(h5ad_path, backed="r")
                                except Exception:
                                    pass
                            adata = adata_cache.get(h5ad_path)
                        
                        if adata is not None:
                            acc_soft[f] = compute_soft_score(adata, gk, pk)
                        else:
                            acc_soft[f] = 0
                    else:
                        acc_soft[f] = 0

        # 2. SCORING: Canonical Values (Implicit vs Explicit)
        canon_scores = {}
        for f in CANON_FIELDS:
            gk = g["gold_key"].get(f, None)
            gold_val = str(g.get(f"gold_{f}_canon", "")).strip().lower()
            pred_vals = pred_canon_lists.get(f, [])

            if gk is not None:
                # EXPLICIT CASE: The data is in a column. 
                # We do not score the "global inference" here, because the answer varies per cell.
                # The performance is entirely captured by the Column Mapping score above.
                canon_scores[f] = np.nan
            else:
                # IMPLICIT CASE: No column exists.
                if gold_val == "" or gold_val == "null":
                    # We have no ground truth for this implicit value
                    canon_scores[f] = np.nan
                else:
                    # We know the truth (e.g. "female"). Did the model find it?
                    # Check if the exact string is in the predicted list
                    canon_scores[f] = int(gold_val in pred_vals)

        # Completeness
        completeness = float(np.mean([pred_map[f] is not None for f in MAP_FIELDS]))

        rows.append({
            "model": pred.get("model", ""),
            "prompt_name": pred.get("prompt_name", ""),
            "use_llm": bool(pred.get("use_llm", False)),
            "doi": doi,
            "dataset_id": dsid,
            "status": pred.get("status", ""),
            "elapsed_sec": float(pred.get("elapsed_sec", np.nan)),
            "completeness": completeness,
            **{f"acc_hard_{f}": acc_hard[f] for f in MAP_FIELDS},
            **{f"acc_soft_{f}": acc_soft[f] for f in MAP_FIELDS},
            **{f"halluc_{f}": halluc_fields[f] for f in MAP_FIELDS},
            **{f"acc_val_{f}": canon_scores[f] for f in CANON_FIELDS},
        })

    df = pd.DataFrame(rows)
    out_csv = Path(args.outdir) / "scores_long.csv"
    df.to_csv(out_csv, index=False)

    # Summary by model/prompt
    grp = df[df["status"] == "ok"].groupby(["model", "prompt_name", "use_llm"], dropna=False)
    summ = grp.mean(numeric_only=True).reset_index()
    out_sum = Path(args.outdir) / "scores_summary.csv"
    summ.to_csv(out_sum, index=False)

    print(f"[ok] wrote {out_csv}")
    print(f"[ok] wrote {out_sum}")

if __name__ == "__main__":
    main()