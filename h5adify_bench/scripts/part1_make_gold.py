#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from common import (
    read_json, write_json, ensure_dir, canonicalize_species_to_h5adify,
    detect_technology_from_string, pick_best_gold_key, doi_slug, now_iso
)

from h5adify.core import ensure_user_metadata_vocab
from h5adify.core.metadata_harmonize import load_metadata_vocab


FIELD_SYNONYMS = {
    "batch":   ["batch", "batch_id", "library", "library_id", "run", "lane", "seq_batch", "chemistry_batch"],
    "sample":  ["sample", "sample_id", "sample_name", "biosample_id", "specimen", "specimen_id"],
    "donor":   ["donor", "donor_id", "patient", "patient_id", "individual", "individual_id", "subject", "subject_id"],
    "domain":  ["domain", "region", "anatomical_region", "area", "cluster", "subclass", "compartment"],
    "sex":     ["sex", "gender", "donor_sex", "biological_sex"],
    # species/technology are often dataset-level or free text; keep for completeness
    "species": ["species", "organism"],
    "technology": ["technology", "assay", "platform", "method"],
}


def candidates_from_obs(obs_columns: List[str], field: str) -> List[str]:
    # stable candidate list: exact synonyms + fuzzy contains
    syns = FIELD_SYNONYMS.get(field, [field])
    syns_norm = {s.lower().replace("-", "_") for s in syns}
    out = []
    for k in obs_columns:
        kn = k.lower().replace("-", "_")
        if kn in syns_norm:
            out.append(k)
    if not out:
        for k in obs_columns:
            kn = k.lower().replace("-", "_")
            for s in syns_norm:
                if s in kn or kn in s:
                    out.append(k); break
    # stable unique
    seen = set()
    out2 = []
    for x in out:
        if x not in seen:
            seen.add(x); out2.append(x)
    return out2

def is_valid_column(adata, col_name: str) -> bool:
    """
    Returns True if the column has valid data (not all NaNs, not single value).
    """
    try:
        # We need to access the data. Since adata is backed, this reads from disk.
        series = adata.obs[col_name]
        
        # Check 1: Is it empty or all NaNs?
        if series.isnull().all():
            return False
            
        # Check 2: Does it have 0 or 1 unique values? (e.g. all "unknown")
        # We allow 1 unique value ONLY if it's the 'species' or 'technology' 
        # which might be constant for a dataset. But for batch/sex, we usually want variation.
        # For safety in this generic function, we'll just check for emptiness.
        if series.nunique() == 0:
            return False
            
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="experiments/data/doi20/manifest.json")
    ap.add_argument("--out", default="experiments/gold/doi20_gold.json")
    ap.add_argument("--use-small", action="store_true", help="Prefer .small.h5ad when available")
    args = ap.parse_args()

    ensure_dir(Path(args.out).parent)

    man = read_json(args.manifest)
    ensure_user_metadata_vocab(overwrite=False)
    vocab = load_metadata_vocab(None)
    tech_keywords = vocab.get("technology_keywords", {})

    gold: Dict[str, Any] = {
        "created_at": now_iso(),
        "manifest": args.manifest,
        "fields": ["batch", "sample", "donor", "domain", "sex", "species", "technology"],
        "items": [],
    }

    import anndata as ad

    for it in man["items"]:
        path = it.get("small_h5ad") if args.use_small and it.get("small_h5ad") else it.get("source_h5ad")
        path = Path(path)
        adata = ad.read_h5ad(path, backed="r")  # fast header-only
        obs_cols = list(adata.obs.columns)

        # gold keys for obs-mapped fields
        chosen = {}
        for f in gold["fields"]:
            candidates = candidates_from_obs(obs_cols, f)
            valid_candidates = []
            for cand in candidates:
                if is_valid_column(adata, cand):
                    valid_candidates.append(cand)
            
            if not valid_candidates:
                chosen[f] = None
            else:
                # If we have multiple valid options (e.g. 'batch' and 'batch_id'),
                # we prefer the one that is NOT numeric if possible (usually more descriptive),
                # or simply fall back to the shortest name if both are strings.
                
                # Sort by:
                # 1. Is it an exact match to our synonym list? (Highest Priority)
                # 2. Length (shorter is usually 'cleaner' ID, e.g. "sex" vs "donor_sex")
                
                exact_matches = FIELD_SYNONYMS.get(f, [])
                def sort_score(c):
                    is_exact = 0 if c in exact_matches else 1
                    return (is_exact, len(c), c)
                
                # Pick the top one
                chosen[f] = sorted(valid_candidates, key=sort_score)[0]

            # chosen[f] = pick_best_gold_key(obs_cols, cand)

        # gold canonical species/technology from Census metadata
        species_canon = canonicalize_species_to_h5adify(it.get("organism", ""))
        tech_raw = it.get("assay", "") or ""
        tech_canon = detect_technology_from_string(tech_raw, tech_keywords)

        gold["items"].append({
            "doi": it["doi"],
            "dataset_id": it["dataset_id"],
            "h5ad_path": str(path),
            "obs_columns": obs_cols,
            "gold_key": chosen,                    # obs key expected (or None)
            "gold_species_canon": species_canon,   # "human"/"mouse"/"rat"/...
            "gold_technology_canon": tech_canon,   # e.g. "Visium"/"MERFISH"/...
            "meta": {
                "organism_raw": it.get("organism", ""),
                "assay_raw": tech_raw,
                "dataset_title": it.get("dataset_title", ""),
                "collection_name": it.get("collection_name", ""),
            }
        })

    write_json(args.out, gold)
    print(f"[ok] wrote {args.out} with {len(gold['items'])} items")


if __name__ == "__main__":
    main()
