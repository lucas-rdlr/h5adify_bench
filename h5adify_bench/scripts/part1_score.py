#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from common import read_json, read_yaml, ensure_dir, write_json

# Fields to check for Column Mapping (finding the right .obs column)
MAP_FIELDS = ["batch", "sample", "donor", "domain", "sex", "species", "technology"]

# Fields to check for Value Inference (extracting "Female" or "Human" from text/data)
# You can now add "sex", "age", "disease" here without changing more code.
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

# MODIFIED: Returns a LIST of values found in the data, not just one string
def extract_pred_canon(pred: Dict[str, Any]) -> Dict[str, List[str]]:
    cp = pred.get("canon_preview", {}) if isinstance(pred.get("canon_preview", {}), dict) else {}
    out = {}
    for f in CANON_FIELDS:
        vals = []
        try:
            # Get the list of unique examples found in the column
            ex_list = cp.get(f, {}).get("examples", [])
            if ex_list:
                # Normalize everything to lowercase string for fair comparison
                vals = [str(x).strip().lower() for x in ex_list]
        except Exception:
            vals = []
        out[f] = vals
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="experiments/gold/doi20_gold.json")
    ap.add_argument("--results", default="experiments/results/part1")
    ap.add_argument("--outdir", default="experiments/results/part1_scores")
    args = ap.parse_args()

    ensure_dir(args.outdir)

    gold = read_json(args.gold)
    gold_index = {(it["doi"], it["dataset_id"]): it for it in gold["items"]}

    preds = load_all_preds(Path(args.results))
    rows = []

    for pj in preds:
        pred = read_json(pj)
        doi = pred.get("doi")
        dsid = pred.get("dataset_id")
        key = (doi, dsid)
        if key not in gold_index:
            continue
        g = gold_index[key]

        pred_map = extract_pred_mapping(pred)
        pred_canon = extract_pred_canon(pred)

        # 1. SCORING: Column Mapping Accuracy
        acc_fields = {}
        halluc_fields = {}
        for f in MAP_FIELDS:
            gk = g["gold_key"].get(f, None)
            pk = pred_map.get(f, None)

            # Hallucination: predicted key not in obs_columns
            halluc = False
            if pk is not None and pk not in g["obs_columns"]:
                halluc = True
            halluc_fields[f] = int(halluc)

            # Accuracy
            if gk is None:
                acc_fields[f] = int(pk is None)
            else:
                acc_fields[f] = int(pk == gk)

        # 2. SCORING: Canonical Value Accuracy (Global Attributes)
        val_acc_fields = {}
        for f in CANON_FIELDS:
            # Look for keys like "gold_species_canon", "gold_sex_canon"
            g_val = str(g.get(f"gold_{f}_canon", "")).strip().lower()
            # p_val = str(pred_canon.get(f, "")).strip().lower()
            p_val = pred_canon.get(f, [])
            
            # Score is 1 only if BOTH exist and match
            # If gold is missing, we don't score it (or score as correct? usually ignore)
            if g_val == "":
                score = np.nan # Not applicable
            else:
                # score = int(p_val != "" and p_val == g_val)
                # IS THE GOLD VALUE IN THE PREDICTED LIST?
                score = int(g_val in p_val)
            
            val_acc_fields[f] = score

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
            **{f"acc_map_{f}": acc_fields[f] for f in MAP_FIELDS},
            **{f"halluc_{f}": halluc_fields[f] for f in MAP_FIELDS},
            **{f"acc_val_{f}": val_acc_fields[f] for f in CANON_FIELDS},
        })

    df = pd.DataFrame(rows)
    out_csv = Path(args.outdir) / "scores_long.csv"
    df.to_csv(out_csv, index=False)

    # Summary
    grp = df[df["status"] == "ok"].groupby(["model", "prompt_name", "use_llm"], dropna=False)
    summ = grp.mean(numeric_only=True).reset_index()
    out_sum = Path(args.outdir) / "scores_summary.csv"
    summ.to_csv(out_sum, index=False)

    print(f"[ok] wrote {out_csv}")
    print(f"[ok] wrote {out_sum}")

if __name__ == "__main__":
    main()
