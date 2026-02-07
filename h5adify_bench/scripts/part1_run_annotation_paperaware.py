#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
part1_run_annotation_paperaware.py (Enhanced)
"""

from __future__ import annotations

import os
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
import anndata as ad  # Moved to top level

from h5adify.annotation.evidence_store import EvidenceStore, SourceType
from h5adify.annotation.deterministic import DeterministicExtractor
from h5adify.annotation.llm_extractor import LLMExtractor, OllamaClient
from h5adify.annotation.prompt_store import PromptStore
from h5adify.annotation.verifier import AnnotationVerifier

# NEW: Import the logic from the core module
from h5adify.core.metadata_harmonize import infer_sex_from_expression, _maybe_infer_species

def doi_slug(doi: str) -> str:
    doi = doi.strip().lower()
    doi = re.sub(r"^https?://doi\.org/", "", doi)
    doi = re.sub(r"^doi:", "", doi)
    return re.sub(r"[^a-z0-9._-]+", "_", doi)

def read_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def add_long_text_in_chunks(
    store: EvidenceStore,
    text: str,
    source_type: SourceType,
    source_name: str,
    section: str,
    chunk_chars: int = 2500,
) -> None:
    text = (text or "").strip()
    if not text:
        return
    for i in range(0, len(text), chunk_chars):
        sub = text[i : i + chunk_chars]
        store.add_text(sub, source_type=source_type, source_name=source_name, section=section, chunk_index=i // chunk_chars)

def pick_items(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    if isinstance(cfg, dict) and "papers" in cfg:
        return cfg["papers"]
    if isinstance(cfg, list):
        return cfg
    raise ValueError("Config must contain 'papers' list or be a list.")

def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--doi-config", required=True, help="configs/doi20.yaml")
    ap.add_argument("--papers-dir", default="papers")
    ap.add_argument("--outdir", default="results_part1_paperaware")
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--ollama-model", default=os.environ.get("H5ADIFY_MODEL", "qwen2.5:3b"))
    ap.add_argument("--prompt-name", default="extraction_v2_default")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = read_yaml(args.doi_config)
    items = pick_items(cfg)

    det = DeterministicExtractor()
    client = OllamaClient(base_url=args.ollama_url, model=args.ollama_model)
    store = PromptStore()
    extractor = LLMExtractor(client=client, prompt_store=store, prompt_name=args.prompt_name)

    for p in items:
        doi = str(p.get("doi") or "").strip()
        if not doi:
            continue

        datasets = p.get("datasets") or []
        for ds in datasets:
            h5ad_path = ds.get("h5ad_path") or ds.get("local_h5ad") or ds.get("path")
            ds_id = ds.get("id") or ds.get("name") or Path(h5ad_path).stem

            if not h5ad_path:
                continue

            # Load data
            adata = ad.read_h5ad(h5ad_path)
            ev = EvidenceStore()

            # Stage A: Metadata Headers (Deterministic)
            facts, ev = det.extract_from_h5ad(adata, ev)

            # Stage B: Biological Content (NEW: Sex/Species from Expression)
            # This bridges the gap with harmonize_metadata()
            try:
                # 1. Infer Species from Gene Names (if missing)
                inferred_species = _maybe_infer_species(adata)
                if inferred_species:
                    # Add as a high-confidence fact
                    ev.add_text(
                        f"Inferred species from gene naming conventions: {inferred_species}",
                        SourceType.H5AD_CONTENT, "gene_names", "inference"
                    )

                # 2. Infer Sex from XIST/Y Markers
                sex, details = infer_sex_from_expression(adata, species=inferred_species)
                if sex != "unknown":
                    # We format this as a clear "Evidence Statement" for the LLM
                    sex_evidence = (
                        f"Biological sex inferred from gene expression markers: {sex}. "
                        f"Markers detected: {details.get('x_gene')} (X-linked), "
                        f"{details.get('y_genes_present')} (Y-linked)."
                    )
                    ev.add_text(
                        sex_evidence,
                        SourceType.H5AD_CONTENT, "expression_markers", "inference"
                    )
            except Exception as e:
                print(f"Warning: biological inference failed for {ds_id}: {e}")

            # Stage C: DOI/Paper Context
            doi_facts, ev = det.extract_from_doi(doi, ev)
            facts.doi = doi_facts.doi
            facts.title = doi_facts.title
            
            # Add PDF text
            slug = doi_slug(doi)
            txt_path = Path(args.papers_dir) / slug / "paper_fulltext.txt"
            if txt_path.exists():
                paper_text = txt_path.read_text(encoding="utf-8", errors="ignore")
                add_long_text_in_chunks(ev, paper_text, SourceType.PDF, "paper_fulltext", "fulltext")

            # Stage D: LLM Extraction
            extraction = extractor.extract(ev, facts)
            schema = extractor.build_schema(extraction, facts)

            if args.verify and client.available:
                verifier = AnnotationVerifier(ev, client)
                schema = verifier.verify_schema(schema)

            result = {
                "dataset_id": ds_id,
                "model": client.model,
                "prompt_name": args.prompt_name,
                "schema": schema.to_dict(),
                "extraction_raw": extraction,
            }

            out_path = outdir / f"{ds_id}.paperaware.json"
            out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"✅ Done. Results in: {outdir}")

if __name__ == "__main__":
    main()
