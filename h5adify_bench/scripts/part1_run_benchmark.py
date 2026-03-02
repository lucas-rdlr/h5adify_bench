#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

# Import the new class
from h5adify.core.harmonizer4 import H5adHarmonizer 
# We might need the OllamaClient wrapper if H5adHarmonizer expects a client object
from h5adify.annotation.llm_extractor import OllamaClient, LLMExtractor
from h5adify.annotation.prompt_store import PromptStore

from common import read_yaml, read_json, write_json, ensure_dir, doi_slug, now_iso, humanize_exception, setup_logging
import anndata as ad
from h5adify.core import DEFAULT_METADATA_FIELDS

import logging

def preview_canon(adata: Any, fields: List[str]) -> Dict[str, Any]:
    """
    Extracts the actual values from the harmonized columns.
    """
    obs = adata.obs
    out: Dict[str, Any] = {}
    for f in fields:
        col = f"h5adify_{f}"
        if col in obs.columns:
            s = obs[col]
            try: 
                ex = s.dropna().astype(str).unique().tolist()[:5]
                out[f] = {
                    "n_unique": int(s.dropna().astype(str).nunique()),
                    "examples": ex,
                }
            except Exception:
                out[f] = {"col": col}
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="configs/models.yaml")
    ap.add_argument("--gold", default="experiments/gold/doi20_gold_all.json")
    ap.add_argument("--results", default="experiments/results/part1_5")
    # ap.add_argument("--prompt-name", default="metadata_harmonize_v1_default")
    ap.add_argument("--prompt-name", default="extraction_v2_default_no_examples")
    ap.add_argument("--use-llm", action="store_true")
    ap.add_argument("--level", default="semantic", choices=["semantic", "paper_aware"])
    ap.add_argument("--save-h5ad", action="store_true", help="Save harmonized h5ad outputs")
    ap.add_argument("--prefer-small", action="store_true", help="Use the .small.h5ad path from gold if present")
    ap.add_argument("--limit", type=int, default=0, help="Debug: only run first N datasets")
    args = ap.parse_args()

    # 1. SETUP LOGGING
    out_root = ensure_dir(args.results)
    # log_path = out_root / "benchmark_run.log"

    cfg = read_yaml(args.models)
    base_url = cfg.get("ollama_base_url", "http://localhost:11434")
    
    # 1. Select Models
    if args.use_llm:
        models = [m["name"] for m in cfg["models"]]
    else:
        models = ["deterministic"]
    
    gold = read_json(args.gold)
    items = gold["items"]
    if args.limit and args.limit > 0:
        items = items[: args.limit]
    
    # fields = list(DEFAULT_METADATA_FIELDS)
    fields = (
        "batch",
        "donor",
        "disease",
        "sex",
        "species",
        "technology",
    )

    pbar_model = tqdm(models, desc="LLM", position=0)
    for model_name in pbar_model:
        pbar_model.set_postfix_str(f"{model_name}")

        # Setup output directories
        if model_name == "deterministic":
            model_dir = ensure_dir(out_root / "deterministic")
            ollama_model = None
            level = "deterministic"
        else:
            model_dir = ensure_dir(out_root / model_name.replace(":", "_"))
            ollama_model = model_name
            level = args.level
        
        # Set to DEBUG to see all the inner workings of h5adify
        logger = setup_logging(log_file=model_dir / "benchmark_run.log", level=logging.DEBUG) 
        
        logger.info("Started Benchmark Run")

        logger.info("Arguments:")
        vars_args = vars(args)
        for k, v in vars_args.items():
            logger.info(f"\t{k}: {v}")

        logger.debug(f"Using models: {models}")

        runlog = {"created_at": now_iso(), "model": model_name, "prompt_name": args.prompt_name, "items": []}

        # --- SETUP HARMONIZER ---
        client = None
        if args.use_llm and ollama_model:
            client = OllamaClient(base_url=base_url, model=ollama_model)

        logger.debug(f"Using model from Ollama: {client.model}")

        extractor = None
        if client and args.use_llm:
            store = PromptStore(store_dir="/home/user/Documents/h5adify_release/h5adify") # Loads prompts from h5adify/assets/prompts or similar
            extractor = LLMExtractor(
                client=client,
                prompt_store=store,
                prompt_name=args.prompt_name # MUST match a valid prompt file in your library!
            )
        
        # Instantiate the unified class
        harmonizer = H5adHarmonizer(
            fields=fields, 
            llm_client=client,
            llm_extractor=extractor
            # You can pass vocabulary here if needed
        )

        pbar = tqdm(items, desc="DOI", position=1)
        for it in pbar:
            pbar.set_postfix_str(f"{it['doi']}")

        # for it in tqdm(items, desc=f"model={model_name}"):
            doi = it["doi"]
            dsid = it["dataset_id"]
            h5ad_path = Path(it["h5ad_path"])
            
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
                "model": model_name,
                "prompt_name": args.prompt_name,
                "use_llm": bool(args.use_llm),
                "started_at": now_iso(),
            }

            try:
                adata = ad.read_h5ad(h5ad_path)

                # Load PDF text if level is paper_aware
                pdf_path = None
                if level == "paper_aware":
                    # Assuming your structure: papers/slug/paper_fulltext.txt
                    slug = doi_slug(doi)
                    paper_dir = Path("experiments/data/doi20") / slug
            
                    # Simple check: take the first PDF found in the folder
                    pdfs = list(paper_dir.glob("*.pdf"))
                    if pdfs:
                        pdf_path = pdfs[0]

                # Instantiate the unified class
                harmonizer = H5adHarmonizer(
                    fields=fields, 
                    llm_client=client,
                    llm_extractor=extractor
                    # You can pass vocabulary here if needed
                )

                # Run Pipeline
                adata2 = harmonizer.process(
                    adata, 
                    level=level,
                    pdf_path=pdf_path,
                    inplace=False
                )

                if args.save_h5ad:
                    adata2.write_h5ad(pred_h5ad)
                    rec["harmonized_h5ad"] = str(pred_h5ad)
                
                # --- EXTRACT RESULTS FOR SCORING ---
                # provenance is the new 'report'
                provenance = adata2.uns.get("h5adify_sources", {})
                
                # Reconstruct the 'report' structure expected by pred.json for logging
                rec["report"] = {
                    "columns_selection": {}, 
                    "global_inference": {} ,
                    "thumbnail": {}
                }

                for f in fields:
                    # Parse the provenance string "source_type:detail"
                    field_column = provenance.get("columns_selection", {}).get(f, None)
                    rec["report"]["columns_selection"][f] = field_column

                    field_global = provenance.get("global_inference", {}).get(f, {}).get("value", None)
                    field_source = provenance.get("global_inference", {}).get(f, {}).get("source", [])
                    field_evidence = provenance.get("global_inference", {}).get(f, {}).get("evidence", [])
                    rec["report"]["global_inference"][f] = {
                        "value": field_global,
                        "source": field_source,
                        "evidence": field_evidence,
                }

                rec["report"]["thumbnail"] = preview_canon(adata2, fields)


                rec["status"] = "ok"

            except Exception as e:
                rec["status"] = "error"
                rec["error"] = humanize_exception(e)

            rec["elapsed_sec"] = round(time.time() - t0, 3)
            write_json(pred_json, rec)
            runlog["items"].append({"doi": doi, "dataset_id": dsid, "pred_json": str(pred_json), "status": rec["status"]})

            del adata
            del adata2
            del harmonizer
            if 'pdf_text' in locals(): del pdf_text
            
            # 2. Force Garbage Collection
            gc.collect()

        write_json(model_dir / "runlog.json", runlog)
        print(f"[ok] model runlog: {model_dir/'runlog.json'}")


if __name__ == "__main__":
    main()