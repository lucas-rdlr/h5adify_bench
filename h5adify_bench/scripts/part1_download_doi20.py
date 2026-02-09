#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from common import ensure_dir, read_yaml, read_json, write_json, doi_slug, now_iso, humanize_exception

# external
import cellxgene_census


def load_datasets_table(census_version: str = "stable") -> pd.DataFrame:
    with cellxgene_census.open_soma(census_version=census_version) as census:
        tbl = census["census_info"]["datasets"].read().concat().to_pandas()
    return tbl


def find_best_dataset_for_doi(df: pd.DataFrame, doi: str) -> Optional[pd.Series]:
    doi_l = doi.lower()
    # primary match: collection_doi
    cols = set(df.columns)
    cand = df
    if "collection_doi" in cols:
        cand = df[df["collection_doi"].astype(str).str.lower() == doi_l]
    # fallback: citation contains DOI
    if cand.shape[0] == 0 and "citation" in cols:
        cand = df[df["citation"].astype(str).str.lower().str.contains(doi_l, na=False)]
    if cand.shape[0] == 0:
        return None

    # choose largest dataset by available count column
    for size_col in ["dataset_total_cell_count", "dataset_total_cell_count_est", "n_obs", "dataset_n_obs"]:
        if size_col in cand.columns:
            idx = cand[size_col].astype(float).fillna(0).idxmax()
            return cand.loc[idx]
    # fallback: first
    return cand.iloc[0]


def download_source_h5ad(dataset_id: str, out_path: Path, census_version: str = "stable") -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        return
    cellxgene_census.download_source_h5ad(dataset_id, to_path=str(out_path), census_version=census_version)
    
def clear_in_progress(manifest, doi: str):
    manifest["in_progress"] = [
        x for x in manifest["in_progress"]
        if x.get("doi") != doi
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doi-config", required=True, help="configs/doi20.yaml")
    args = ap.parse_args()

    cfg = read_yaml(args.doi_config)
    out_dir = Path(cfg["download"]["out_dir"])
    ensure_dir(out_dir)

    census_version = cfg["download"].get("census_version", "stable")
    make_small = bool(cfg["download"].get("make_small_copy", True))
    small_max_obs = int(cfg["download"].get("small_max_obs", 20000))
    small_max_vars = int(cfg["download"].get("small_max_vars", 8000))
    seed = int(cfg["download"].get("seed", 42))

    print(f"[i] Loading Census datasets table ({census_version}) ...")
    df = load_datasets_table(census_version=census_version)
    
    manifest_path = out_dir / "manifest.json"

    if manifest_path.exists():
        print(f"[i] Resuming from existing manifest: {manifest_path}")
        manifest = read_json(manifest_path)
    else:
        manifest = {
            "created_at": now_iso(),
            "census_version": census_version,
            "items": [],
            "missing": [],
            "errors": [],
        }
        manifest.setdefault("in_progress", [])

    processed_dois = {
        entry["doi"]
        for section in ("items", "missing", "errors")
        for entry in manifest.get(section, [])
        if "doi" in entry
    }

    pbar = tqdm(cfg["dois"], desc="DOIs")

    for doi in pbar:
        pbar.set_postfix_str(f"DOI: {doi}")
        if doi in processed_dois: continue
    	
        manifest["in_progress"].append({
            "doi": doi,
            "started_at": now_iso(),
        })
        write_json(manifest_path, manifest)
		
        try:
            row = find_best_dataset_for_doi(df, doi)
            if row is None:
                manifest["missing"].append({"doi": doi})
                continue

            dataset_id = str(row["dataset_id"])
            title = str(row.get("dataset_title", row.get("title", "")))
            collection = str(row.get("collection_name", ""))
            organism = str(row.get("organism", row.get("organism_name", "")))
            assay = str(row.get("assay", row.get("assay_ontology_term_id", "")))

            doi_dir = out_dir / doi_slug(doi)
            ensure_dir(doi_dir)

            source_path = doi_dir / f"{dataset_id}.source.h5ad"
            download_source_h5ad(dataset_id, source_path, census_version=census_version)

            small_path = None
            if make_small:
                try:
                    import anndata as ad
                    from common import subsample_adata, subset_vars_by_hvg_or_random
                    adata = ad.read_h5ad(source_path)
                    adata = subsample_adata(adata, max_obs=small_max_obs, seed=seed)
                    adata = subset_vars_by_hvg_or_random(adata, max_vars=small_max_vars, seed=seed)
                    small_path = doi_dir / f"{dataset_id}.small.h5ad"
                    adata.write_h5ad(small_path)
                except Exception as e:
                    manifest["errors"].append({"doi": doi, "dataset_id": dataset_id, "stage": "small_copy", "error": humanize_exception(e)})

            manifest["items"].append({
                "doi": doi,
                "dataset_id": dataset_id,
                "dataset_title": title,
                "collection_name": collection,
                "organism": organism,
                "assay": assay,
                "source_h5ad": str(source_path),
                "small_h5ad": str(small_path) if small_path else None,
            })

        except Exception as e:
            manifest["errors"].append({"doi": doi, "stage": "download", "error": humanize_exception(e)})
            
        finally:
            clear_in_progress(manifest, doi)
            write_json(manifest_path, manifest)

    write_json(out_dir / "manifest.json", manifest)
    print(f"[ok] Wrote manifest: {out_dir/'manifest.json'}")
    print(f"[ok] Downloaded: {len(manifest['items'])}, missing: {len(manifest['missing'])}, errors: {len(manifest['errors'])}")


if __name__ == "__main__":
    main()