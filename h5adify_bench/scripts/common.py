#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

import logging
import sys


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_yaml(path: str | Path) -> Dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def doi_slug(doi: str) -> str:
    s = doi.strip()
    s = s.replace("/", "_")
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s


def humanize_exception(e: Exception) -> str:
    return f"{type(e).__name__}: {e}"


def subsample_adata(adata: Any, max_obs: int, seed: int = 0) -> Any:
    if max_obs is None or max_obs <= 0:
        return adata
    n = int(getattr(adata, "n_obs", 0))
    if n <= max_obs:
        return adata
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_obs, replace=False)
    return adata[idx].copy()


def subset_vars_by_hvg_or_random(adata: Any, max_vars: int, seed: int = 0) -> Any:
    if max_vars is None or max_vars <= 0:
        return adata
    m = int(getattr(adata, "n_vars", 0))
    if m <= max_vars:
        return adata
    # try HVG if scanpy available, else random
    try:
        import scanpy as sc
        ad = adata.copy()
        sc.pp.highly_variable_genes(ad, n_top_genes=max_vars, flavor="seurat_v3")
        keep = np.asarray(ad.var["highly_variable"].values, dtype=bool)
        if keep.sum() >= 10:
            return adata[:, keep].copy()
    except Exception:
        pass
    rng = np.random.default_rng(seed)
    j = rng.choice(m, size=max_vars, replace=False)
    return adata[:, j].copy()


def batch_entropy(nei_batches: np.ndarray) -> float:
    # nei_batches: array of batch labels for neighbors of one cell
    vals, cnts = np.unique(nei_batches, return_counts=True)
    p = cnts / cnts.sum()
    return float(-(p * np.log(p + 1e-12)).sum())


def knn_batch_entropy(emb: np.ndarray, batches: np.ndarray, k: int = 30) -> float:
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=min(k + 1, emb.shape[0]), metric="euclidean")
    nn.fit(emb)
    idx = nn.kneighbors(return_distance=False)
    # drop self neighbor
    idx = idx[:, 1:]
    ent = []
    for i in range(idx.shape[0]):
        ent.append(batch_entropy(batches[idx[i]]))
    return float(np.mean(ent))


def canonicalize_species_to_h5adify(name: str) -> str:
    s = (name or "").lower()
    if "homo sapiens" in s or s == "human":
        return "human"
    if "mus musculus" in s or s == "mouse":
        return "mouse"
    if "rattus norvegicus" in s or s == "rat":
        return "rat"
    return name or None


def detect_technology_from_string(s: str, tech_keywords: Dict[str, List[str]]) -> str:
    x = (s or "").lower()
    for canon, kws in tech_keywords.items():
        for kw in kws:
            if kw.lower() in x:
                return canon
    return s or None


def pick_best_gold_key(obs_columns: List[str], candidates: List[str]) -> Optional[str]:
    # stable gold heuristic: prefer exact matches, then smallest name, then None
    inter = [c for c in candidates if c in obs_columns]
    if not inter:
        return None
    # prefer deterministic stable selection
    return sorted(inter, key=lambda z: (len(z), z.lower()))[0]


from tqdm import tqdm

class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            self.handleError(record)

def setup_logging(log_file: str = None, level: int = logging.INFO):

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S"
    )

    logger = logging.getLogger()
    logger.setLevel(level)

    if logger.hasHandlers():
        logger.handlers.clear()

    # ✅ Replace StreamHandler with TqdmLoggingHandler
    ch = TqdmLoggingHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler remains unchanged
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, mode='w')
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    return logger


# def setup_logging(log_file: str = None, level: int = logging.INFO):
#     """
#     Configures logging for the benchmark script AND the h5adify library.
#     """
#     # 1. format the log
#     fmt = logging.Formatter(
#         fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
#         datefmt="%H:%M:%S"
#     )

#     # 2. Get the root logger (or specifically 'h5adify' if you want to isolate it)
#     # Using root logger captures logs from your script, h5adify, and requests/urllib3
#     logger = logging.getLogger()
#     logger.setLevel(level)

#     # 3. Reset handlers (prevents duplicate logs if run in notebooks/repeatedly)
#     if logger.hasHandlers():
#         logger.handlers.clear()

#     # 4. Console Handler (Standard Output)
#     ch = logging.StreamHandler(sys.stdout)
#     ch.setFormatter(fmt)
#     logger.addHandler(ch)

#     # 5. File Handler (Optional)
#     if log_file:
#         log_path = Path(log_file)
#         log_path.parent.mkdir(parents=True, exist_ok=True)
#         fh = logging.FileHandler(log_path, mode='w') # 'w' overwrites, 'a' appends
#         fh.setFormatter(fmt)
#         logger.addHandler(fh)
        
#     # 6. Quiet down noisy libraries (optional)
#     logging.getLogger("urllib3").setLevel(logging.WARNING)
#     logging.getLogger("httpx").setLevel(logging.WARNING)
    
#     return logger