#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from common import read_yaml, read_json, write_json, ensure_dir, doi_slug, now_iso, humanize_exception

import anndata as ad
from h5adify.core import harmonize_metadata, DEFAULT_METADATA_FIELDS


def preview_canon(adata: Any, fields: List[str]) -> Dict[str, Any]:
    obs = adata.obs
    out: Dict[str, Any] = {}
    for f in fields:
        col = f"h5adify_{f}"
        if col in obs.columns:
            s = obs[col]
            try:
                ex = s.dropna().astype(str).unique().tolist()[:10]
                out[f] = {
                    "col": col,
                    "n_unique": int(s.dropna().astype(str).nunique()),
                    "examples": ex,
                }
            except Exception:
                out[f] = {"col": col}
        else:
            out[f] = {"col": col, "missing": True}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="configs/models.yaml")
    ap.add_argument("--gold", default="gold/doi20_gold.json")
    ap.add_argument("--results", default="results/part1")
    ap.add_argument("--prompt-name", default="metadata_harmonize_v1_default")
    ap.add_argument("--use-llm", action="store_true")
    ap.add_argument("--no-sex-from-expression", action="store_true")
    ap.add_argument("--save-h5ad", action="store_true", help="Save harmonized h5ad outputs")
    ap.add_argument("--prefer-small", action="store_true", help="Use the .small.h5ad path from gold if present")
    ap.add_argument("--limit", type=int, default=0, help="Debug: only run first N datasets")
    args = ap.parse_args()

    cfg = read_yaml(args.models)
    base_url = cfg.get("ollama_base_url", "http://localhost:11434")
    models = [m["name"] for m in cfg["models"]]

    gold = read_json(args.gold)
    items = gold["items"]
    if args.limit and args.limit > 0:
        items = items[: args.limit]

    out_root = ensure_dir(args.results)
    fields = list(DEFAULT_METADATA_FIELDS)
    
    if not args.use_llm: models = ["deterministic"]

    for model in models:
        model_dir = ensure_dir(out_root / model.replace(":", "_"))
        runlog = {"created_at": now_iso(), "model": model, "prompt_name": args.prompt_name, "items": []}

        for it in tqdm(items, desc=f"model={model}"):
            doi = it["doi"]
            dsid = it["dataset_id"]
            h5ad_path = Path(it["h5ad_path"])
            # if prefer-small, try to locate matching .small.h5ad near it
            if args.prefer_small:
                small = h5ad_path.with_suffix("").with_suffix(".small.h5ad")
                if small.exists():
                    h5ad_path = small

            one_out_dir = ensure_dir(model_dir / doi_slug(doi) / dsid)
            pred_json = one_out_dir / "pred.json"
            pred_h5ad = one_out_dir / "harmonized.h5ad"

            t0 = time.time()
            rec: Dict[str, Any] = {
                "doi": doi,
                "dataset_id": dsid,
                "h5ad_path": str(h5ad_path),
                "model": model,
                "prompt_name": args.prompt_name,
                "use_llm": bool(args.use_llm),
                "started_at": now_iso(),
            }

            try:
                print("Trying adata harmonization...")
                adata = ad.read_h5ad(h5ad_path)
                adata2, report = harmonize_metadata(
                    adata,
                    fields=fields,
                    use_llm=bool(args.use_llm),
                    llm_prompt_name=args.prompt_name,
                    sex_from_expression=(not args.no_sex_from_expression),
                    ollama_base_url=base_url,
                    ollama_model=model,
                    inplace=False,
                )
                print("Finished adata harmonization...")
                rec["report"] = report
                rec["canon_preview"] = preview_canon(adata2, fields)

                if args.save_h5ad:
                    adata2.write_h5ad(pred_h5ad)
                    rec["harmonized_h5ad"] = str(pred_h5ad)

                rec["status"] = "ok"

            except Exception as e:
                rec["status"] = "error"
                rec["error"] = humanize_exception(e)

            rec["elapsed_sec"] = round(time.time() - t0, 3)
            write_json(pred_json, rec)
            runlog["items"].append({"doi": doi, "dataset_id": dsid, "pred_json": str(pred_json), "status": rec["status"]})

        write_json(model_dir / "runlog.json", runlog)
        print(f"[ok] model runlog: {model_dir/'runlog.json'}")


if __name__ == "__main__":
    main()
