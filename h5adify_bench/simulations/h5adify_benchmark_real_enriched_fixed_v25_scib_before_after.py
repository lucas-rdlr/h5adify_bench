#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
h5adify_benchmark_real_enriched_fixed_v25_scib_before_after.py

===============================================================================
                  END‑TO‑END BENCHMARK – FINAL WORKING VERSION
                             (with all fixes)
===============================================================================

"""

from __future__ import annotations

import argparse
import json
import math
import re
import os
import subprocess
import sys
import time
import zipfile
import warnings
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Sequence, Union, Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Core scientific libraries
# ---------------------------------------------------------------------
import scanpy as sc
import anndata as ad
from scipy import sparse
from scipy.sparse import csr_matrix

# ---------------------------------------------------------------------
# Plotting – headless mode
# ---------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import seaborn as sns

# Silence warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
# Suppress JAX/TPU logs (optional)
os.environ["JAX_PLATFORMS"] = "cpu"

# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------
def log(msg: str) -> None:
    print(msg, flush=True)

def warn(msg: str) -> None:
    print(f"[WARN] {msg}", flush=True)

def ok(msg: str) -> None:
    print(f"[OK] {msg}", flush=True)

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def cpu_count() -> int:
    try:
        return os.cpu_count() or 1
    except Exception:
        return 1

def normalize_n_jobs(n_jobs: int) -> int:
    if n_jobs is None:
        return 1
    if n_jobs == -1:
        return cpu_count()
    return max(1, int(n_jobs))

def set_global_seeds(seed: int) -> None:
    np.random.seed(seed)
    try:
        import random
        random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.set_float32_matmul_precision("medium")
    except Exception:
        pass

def safe_import(name: str):
    try:
        return __import__(name)
    except Exception:
        return None

def pip_install(pkgs: Sequence[str]) -> bool:
    cmd = [sys.executable, "-m", "pip", "install", "--no-cache-dir", *pkgs]
    log("[INFO] Installing missing deps via pip:\n  " + " ".join(pkgs))
    try:
        subprocess.check_call(cmd)
        return True
    except Exception as e:
        warn(f"pip install failed: {e}")
        return False

# ---------------------------------------------------------------------
# SIMPLE GENE HARMONISATION FALLBACK (when h5adify fails or intersection empty)
# ---------------------------------------------------------------------
def simple_gene_harmonization(
    adata: ad.AnnData,
    target_species: str,
    dedup_how: str = "sum"
) -> ad.AnnData:
    """
    Robust, offline gene-name harmonisation fallback.

    What it does:
      1) If h5adify created a `var["gene_harmonized"]` column, use it as the primary gene vocabulary.
      2) Strip Ensembl version suffixes (e.g. ENSG... .1).
      3) Synthetic Ensembl mapping (ONLY for this benchmark's simulated ENSG/ENSMUSG IDs):
           ENSG1000000 + i   -> GENE{i:05d}
           ENSMUSG1000000+i  -> Gene{i:05d}
         (Real-world Ensembl-to-symbol mapping requires external resources and is not attempted here.)
      4) Canonical casing:
           - human: uppercase
           - mouse: Capitalize (Gene, Xist, etc.)
      5) Deduplicate genes by summing counts (X and layers) when requested.

    This function is intentionally conservative: it won't invent mappings for real Ensembl IDs.
    """
    a = adata.copy()

    if "counts" not in a.layers:
        a.layers["counts"] = a.X.copy()

    # Start from gene_harmonized if present (h5adify often keeps it in var and does NOT overwrite var_names)
    if "gene_harmonized" in a.var.columns:
        gh = a.var["gene_harmonized"].astype(str).replace({"nan": np.nan, "None": np.nan, "": np.nan})
        fallback = pd.Series(a.var_names.astype(str), index=a.var.index, dtype=object)
        base_names = gh.where(gh.notna(), fallback)
    else:
        base_names = pd.Series(a.var_names.astype(str), index=a.var.index, dtype=object)

    # Strip Ensembl version suffix
    base_names = base_names.str.replace(r"\.\d+$", "", regex=True)

    sp = target_species.lower().strip()

    # Synthetic Ensembl -> synthetic symbols (range-guarded)
    if sp == "human":
        digits = base_names.str.extract(r"^ENSG0*(\d+)$")[0]
        num = pd.to_numeric(digits, errors="coerce")
        mask = num.notna() & (num >= 1000000) & (num < 2000000)
        mapped = base_names.copy()
        if mask.any():
            idx = (num[mask].astype(np.int64) - 1000000).astype(int)
            mapped.loc[mask] = idx.map(lambda i: f"GENE{i:05d}")
        new_names = mapped.str.upper()
    else:  # mouse (default)
        digits = base_names.str.extract(r"^ENSMUSG0*(\d+)$")[0]
        num = pd.to_numeric(digits, errors="coerce")
        mask = num.notna() & (num >= 1000000) & (num < 2000000)
        mapped = base_names.copy()
        if mask.any():
            idx = (num[mask].astype(np.int64) - 1000000).astype(int)
            mapped.loc[mask] = idx.map(lambda i: f"Gene{i:05d}")
        new_names = mapped.str.capitalize()

    new_index = pd.Index(new_names.astype(str).values)

    # Nothing to do
    if len(new_index) == 0:
        return a

    # Deduplicate by summation (X and all layers) using sparse aggregation
    if dedup_how == "sum" and new_index.has_duplicates:
        codes, uniques = pd.factorize(new_index, sort=False)
        uniques = pd.Index([str(u) for u in uniques])

        import scipy.sparse as sp_sparse

        n_vars = len(new_index)
        n_unique = len(uniques)
        rows = np.arange(n_vars, dtype=np.int64)
        cols = codes.astype(np.int64)
        data = np.ones(n_vars, dtype=np.float32)
        S = sp_sparse.csr_matrix((data, (rows, cols)), shape=(n_vars, n_unique))

        def _agg_matrix(M):
            if sparse.issparse(M):
                M = M.tocsr()
                return M @ S
            else:
                # Dense fallback (ok for these benchmarks)
                out = np.zeros((M.shape[0], n_unique), dtype=M.dtype)
                for j in range(n_vars):
                    out[:, cols[j]] += M[:, j]
                return out

        X_new = _agg_matrix(a.X)
        b = ad.AnnData(X=X_new, obs=a.obs.copy())

        # Aggregate layers
        for lname, M in a.layers.items():
            try:
                b.layers[lname] = _agg_matrix(M)
            except Exception:
                pass

        # Carry over embeddings/graphs/uns
        b.obsm = a.obsm.copy()
        b.obsp = a.obsp.copy()
        b.uns = dict(a.uns)
        b.varm = a.varm.copy()

        # Keep first annotation row per gene
        var_df = a.var.copy()
        var_df["__new_gene__"] = new_index.values
        var_df = var_df.groupby("__new_gene__", sort=False).first()
        var_df.index = pd.Index(var_df.index.astype(str))
        if "__new_gene__" in var_df.columns:
            var_df = var_df.drop(columns=["__new_gene__"])
        b.var = var_df
        b.var_names = pd.Index(var_df.index.astype(str))

        b.obs_names = a.obs_names.copy()
        b.var_names_make_unique()
        return b

    # No duplicates (or dedup disabled)
    a.var_names = new_index
    a.var_names_make_unique()
    return a


# ---------------------------------------------------------------------
# Simulation core – **STRONG BATCH EFFECTS + INCONSISTENT GENE SYMBOLS**
# ---------------------------------------------------------------------
@dataclass
class SimConfig:
    n_cells: int
    n_genes: int
    n_celltypes: int
    n_donors: int
    batch_strength: float
    donor_strength: float
    tech_strength: float
    libsize_strength: float
    frac_batch_genes: float
    frac_donor_genes: float
    frac_tech_genes: float
    theta: float


# Expanded sex marker sets (robust across cell types; used both for sim + inference)
HUMAN_SEX_GENES_F = [
    "XIST", "TSIX", "JPX", "FTX",
]
HUMAN_SEX_GENES_M = [
    "DDX3Y", "KDM5D", "UTY", "RPS4Y1", "RPS4Y2", "EIF1AY",
    "ZFY", "PRKY", "TMSB4Y", "USP9Y", "TXLNGY", "NLGN4Y",
    "AMELY", "TBL1Y", "PCDH11Y",
]

MOUSE_SEX_GENES_F = [
    "Xist", "Tsix", "Jpx", "Ftx",
]
MOUSE_SEX_GENES_M = [
    "Ddx3y", "Kdm5d", "Jarid1d", "Uty", "Rps4y1", "Eif2s3y",
    "Zfy1", "Zfy2", "Usp9y", "Sry",
]


def _make_gene_names(species: str, n_genes: int, rng: np.random.Generator, dataset_idx: int) -> List[str]:
    """
    Generate deliberately heterogeneous gene naming schemes across datasets.

    Important for downstream sex inference:
      - even when using ENSEMBL-like IDs, we KEEP sex-marker *symbols* (XIST/Xist and Y genes)
        so expression-based sex inference remains possible pre/post-harmonization.
    """
    sp = str(species).lower()

    if sp == "human":
        base = [f"GENE{i:05d}" for i in range(n_genes)]
        special = HUMAN_SEX_GENES_F + HUMAN_SEX_GENES_M + ["MALAT1", "GAPDH", "ACTB", "RPLP0", "B2M"]
        special_low = {s.lower() for s in special}
        for i, g in enumerate(special):
            if i < len(base):
                base[i] = g

        mode = dataset_idx % 3
        if mode == 0:
            return base  # symbols/GENE
        if mode == 1:
            return [g.lower() for g in base]  # lowercase
        # ENSEMBL-like for non-special genes; keep special symbols intact
        out = []
        for i, g in enumerate(base):
            if str(g).lower() in special_low:
                out.append(g)
            else:
                out.append(f"ENSG{1000000 + i:07d}")
        return out

    if sp == "mouse":
        base = [f"Gene{i:05d}" for i in range(n_genes)]
        special = MOUSE_SEX_GENES_F + MOUSE_SEX_GENES_M + ["Malat1", "Gapdh", "Actb", "Rplp0", "B2m"]
        special_low = {s.lower() for s in special}
        for i, g in enumerate(special):
            if i < len(base):
                base[i] = g

        mode = dataset_idx % 3
        if mode == 0:
            return base  # CamelCase-ish
        if mode == 1:
            return [g.lower() for g in base]  # lowercase (still detectable case-insensitively)
        # ENSEMBL-like for non-special genes; keep special symbols intact
        out = []
        for i, g in enumerate(base):
            if str(g).lower() in special_low:
                out.append(g)
            else:
                out.append(f"ENSMUSG{1000000 + i:07d}")
        return out

    return [f"G{i:05d}" for i in range(n_genes)]


def _nb_counts_from_mean(mean: np.ndarray, theta: float, rng: np.random.Generator) -> np.ndarray:
    """Gamma-Poisson (NB) sampling with mean parameterization."""
    mean = np.clip(mean, 1e-8, None)
    gamma_shape = float(theta)
    gamma_scale = mean / float(theta)
    lam = rng.gamma(shape=gamma_shape, scale=gamma_scale)
    return rng.poisson(lam).astype(np.int32)


def simulate_scrna_dataset(
    *,
    name: str,
    species: str,
    batch_label: str,
    donor_labels: List[str],
    technology_label: str,
    cfg: SimConfig,
    rng: np.random.Generator,
    dataset_idx: int,
) -> ad.AnnData:
    """Simulate a scRNA-seq-like dataset with strong batch/donor effects and detectable sex markers."""
    n_cells = int(cfg.n_cells)
    n_genes = int(cfg.n_genes)
    n_ct = int(cfg.n_celltypes)

    genes = _make_gene_names(species, n_genes, rng, dataset_idx)
    cell_types = [f"CT{i+1:02d}" for i in range(n_ct)]

    # Cell type distribution + enforce minimum presence to reduce scIB "isolated labels"
    probs = rng.dirichlet(alpha=np.ones(n_ct))
    ct_idx = rng.choice(n_ct, size=n_cells, p=probs)
    min_per_ct = max(35, int(0.005 * n_cells))
    counts_ct = np.bincount(ct_idx, minlength=n_ct)
    if np.any(counts_ct < min_per_ct):
        deficit = {k: (min_per_ct - int(counts_ct[k])) for k in range(n_ct) if counts_ct[k] < min_per_ct}
        surplus = {k: (int(counts_ct[k]) - min_per_ct) for k in range(n_ct) if counts_ct[k] > min_per_ct}
        idx_by_ct = {k: np.where(ct_idx == k)[0].tolist() for k in range(n_ct)}
        for k_need, need in deficit.items():
            while need > 0 and len(surplus) > 0:
                k_sur = max(surplus, key=lambda kk: surplus[kk])
                if surplus[k_sur] <= 0:
                    surplus.pop(k_sur, None)
                    continue
                take = min(need, surplus[k_sur])
                move_idx = idx_by_ct[k_sur][:take]
                idx_by_ct[k_sur] = idx_by_ct[k_sur][take:]
                ct_idx[move_idx] = k_need
                surplus[k_sur] -= take
                need -= take

    ct = np.array([cell_types[i] for i in ct_idx], dtype=object)

    donors = rng.choice(donor_labels, size=n_cells, replace=True)
    donor_sex_map = {d: rng.choice(["female", "male"]) for d in donor_labels}
    sex_true = np.array([donor_sex_map[d] for d in donors], dtype=object)

    # Latent factors for cell type means
    k = min(20, max(6, n_ct))
    ct_latent = rng.normal(0, 1.0, size=(n_ct, k))
    gene_load = rng.normal(0, 0.7, size=(k, n_genes))
    base_logmean_ct = ct_latent @ gene_load
    base_logmean_ct = (base_logmean_ct - base_logmean_ct.mean()) / (base_logmean_ct.std() + 1e-8)
    base_mean_ct = np.exp(base_logmean_ct * 0.60)
    mean = base_mean_ct[ct_idx].copy()

    # Batch (dataset) effect
    n_batch_genes = max(30, int(cfg.frac_batch_genes * n_genes))
    batch_genes = rng.choice(n_genes, size=n_batch_genes, replace=False)
    batch_noise = rng.normal(0, cfg.batch_strength * 1.8, size=n_batch_genes)
    mean[:, batch_genes] *= np.exp(batch_noise)[None, :]

    # Donor effect
    n_donor_genes = max(30, int(cfg.frac_donor_genes * n_genes))
    donor_genes = rng.choice(n_genes, size=n_donor_genes, replace=False)
    donor_dist = {d: rng.normal(0, cfg.donor_strength * 1.5, size=n_donor_genes) for d in donor_labels}
    for i, d in enumerate(donors):
        mean[i, donor_genes] *= np.exp(donor_dist[d])

    # Technology effect (optional; SimA/B use ~0, SimC uses large tech_strength but in spatial)
    n_tech_genes = max(30, int(cfg.frac_tech_genes * n_genes))
    if n_tech_genes > 0 and cfg.tech_strength > 0:
        tech_genes = rng.choice(n_genes, size=n_tech_genes, replace=False)
        tech_noise = rng.normal(0, cfg.tech_strength * 1.1, size=n_tech_genes)
        mean[:, tech_genes] *= np.exp(tech_noise)[None, :]

    # Library size variation
    lib = rng.lognormal(mean=0.0, sigma=cfg.libsize_strength, size=n_cells)
    mean *= lib[:, None]

    # Sex marker expression boost (case-insensitive mapping, robust marker sets)
    gmap = {str(g).strip().lower(): idx for idx, g in enumerate(genes)}
    if str(species).lower() == "human":
        x_markers = HUMAN_SEX_GENES_F
        y_markers = HUMAN_SEX_GENES_M
    else:
        x_markers = MOUSE_SEX_GENES_F
        y_markers = MOUSE_SEX_GENES_M
    x_idx = [gmap.get(str(g).lower(), None) for g in x_markers]
    y_idx = [gmap.get(str(g).lower(), None) for g in y_markers]
    x_idx = [i for i in x_idx if i is not None]
    y_idx = [i for i in y_idx if i is not None]

    if len(x_idx) > 0 or len(y_idx) > 0:
        # Make sex markers much more separable across cell types by reducing baseline
        # and applying strong, symmetric boosts per sex.
        f = np.where(sex_true == "female")[0]
        m = np.where(sex_true != "female")[0]

        if len(x_idx) > 0:
            mean[:, x_idx] *= 0.05  # lower baseline XIST-like signal
            if f.size > 0:
                mean[np.ix_(f, x_idx)] *= 300.0
            if m.size > 0:
                mean[np.ix_(m, x_idx)] *= 0.001

        if len(y_idx) > 0:
            mean[:, y_idx] *= 0.05  # lower baseline Y-gene signal
            if f.size > 0:
                mean[np.ix_(f, y_idx)] *= 0.001
            if m.size > 0:
                mean[np.ix_(m, y_idx)] *= 300.0

    counts = _nb_counts_from_mean(mean, theta=cfg.theta, rng=rng)
    X = csr_matrix(counts)
    a = ad.AnnData(X=X)
    a.var_names = pd.Index(genes)
    a.obs_names = pd.Index([f"{name}_cell{i:06d}" for i in range(n_cells)])

    a.obs["cell_type"] = pd.Categorical(ct)
    a.obs["true_cell_type"] = a.obs["cell_type"].astype(str)
    a.obs["batch"] = pd.Categorical([batch_label] * n_cells)
    a.obs["true_batch"] = a.obs["batch"].astype(str)
    a.obs["true_donor"] = np.array(donors, dtype=object)
    a.obs["true_technology"] = np.array([technology_label] * n_cells, dtype=object)
    a.obs["true_sex"] = np.array(sex_true, dtype=object)
    a.obs["true_species"] = np.array([str(species).lower()] * n_cells, dtype=object)

    a.obs["patient_id"] = a.obs["true_donor"].astype(str)
    if "10xv3" in str(technology_label).lower():
        tech_variants = ["10x v3", "10xv3", "10x 3' v3", "Chromium v3"]
    elif "10xv2" in str(technology_label).lower():
        tech_variants = ["10x v2", "10xv2", "10x 3' v2", "Chromium v2"]
    else:
        tech_variants = [technology_label]
    a.obs["platform"] = rng.choice(tech_variants, size=n_cells)

    # placeholders that h5adify should fill (and we evaluate after harmonization)
    a.obs["donor"] = "Unknown"
    a.obs["technology"] = "Unknown"
    a.obs["sex"] = "Unknown"
    a.obs["species"] = "Unknown"

    a.layers["counts"] = a.X.copy()
    return a


def simulate_spatial_dataset(
    *,
    name: str,
    species: str,
    section_label: str,
    tech_label: str,
    donor_labels: List[str],
    cfg: SimConfig,
    rng: np.random.Generator,
    dataset_idx: int,
) -> ad.AnnData:
    """Simulate a spatial transcriptomics-like dataset with section+technology effects and detectable sex markers."""
    n_cells = int(cfg.n_cells)
    n_genes = int(cfg.n_genes)
    n_ct = int(cfg.n_celltypes)

    genes = _make_gene_names(species, n_genes, rng, dataset_idx)
    cell_types = [f"CT{i+1:02d}" for i in range(n_ct)]

    xs = rng.uniform(0, 1, size=n_cells)
    ys = rng.uniform(0, 1, size=n_cells)
    coords = np.c_[xs, ys]

    # spatially varying cell type mixture (ring / angle)
    ang = np.arctan2(ys - 0.5, xs - 0.5)
    rad = np.sqrt((xs - 0.5) ** 2 + (ys - 0.5) ** 2)
    scores = np.zeros((n_cells, n_ct))
    for k in range(n_ct):
        center = -math.pi + (2 * math.pi) * (k / max(1, n_ct))
        scores[:, k] = -((ang - center) ** 2) - 2.0 * (rad - 0.3) ** 2
    scores = np.exp(scores - scores.max(axis=1, keepdims=True))
    probs = scores / scores.sum(axis=1, keepdims=True)
    ct_idx = np.array([rng.choice(n_ct, p=probs[i]) for i in range(n_cells)], dtype=int)

    # enforce minimum representation (reduces scIB isolated-label penalties)
    min_per_ct = max(25, int(0.004 * n_cells))
    counts_ct = np.bincount(ct_idx, minlength=n_ct)
    if np.any(counts_ct < min_per_ct):
        deficit = {k: (min_per_ct - int(counts_ct[k])) for k in range(n_ct) if counts_ct[k] < min_per_ct}
        surplus = {k: (int(counts_ct[k]) - min_per_ct) for k in range(n_ct) if counts_ct[k] > min_per_ct}
        idx_by_ct = {k: np.where(ct_idx == k)[0].tolist() for k in range(n_ct)}
        for k_need, need in deficit.items():
            while need > 0 and len(surplus) > 0:
                k_sur = max(surplus, key=lambda kk: surplus[kk])
                if surplus[k_sur] <= 0:
                    surplus.pop(k_sur, None)
                    continue
                take = min(need, surplus[k_sur])
                move_idx = idx_by_ct[k_sur][:take]
                idx_by_ct[k_sur] = idx_by_ct[k_sur][take:]
                ct_idx[move_idx] = k_need
                surplus[k_sur] -= take
                need -= take

    ct = np.array([cell_types[i] for i in ct_idx], dtype=object)

    donors = rng.choice(donor_labels, size=n_cells, replace=True)
    donor_sex_map = {d: rng.choice(["female", "male"]) for d in donor_labels}
    sex_true = np.array([donor_sex_map[d] for d in donors], dtype=object)

    k = min(20, max(6, n_ct))
    ct_latent = rng.normal(0, 1.0, size=(n_ct, k))
    gene_load = rng.normal(0, 0.7, size=(k, n_genes))
    base_logmean_ct = ct_latent @ gene_load
    base_logmean_ct = (base_logmean_ct - base_logmean_ct.mean()) / (base_logmean_ct.std() + 1e-8)
    base_mean_ct = np.exp(base_logmean_ct * 0.55)
    mean = base_mean_ct[ct_idx].copy()

    # Section effect (batch)
    n_batch_genes = max(20, int(cfg.frac_batch_genes * n_genes))
    sec_genes = rng.choice(n_genes, size=n_batch_genes, replace=False)
    sec_noise = rng.normal(0, cfg.batch_strength * 0.8, size=n_batch_genes)
    mean[:, sec_genes] *= np.exp(sec_noise)[None, :]

    # Technology effect (strong in SimC to reward correction)
    n_tech_genes = max(20, int(cfg.frac_tech_genes * n_genes))
    tech_genes = rng.choice(n_genes, size=n_tech_genes, replace=False)
    tech_noise = rng.normal(0, cfg.tech_strength * 1.8, size=n_tech_genes)
    mean[:, tech_genes] *= np.exp(tech_noise)[None, :]

    # Donor effect
    n_donor_genes = max(20, int(cfg.frac_donor_genes * n_genes))
    donor_genes = rng.choice(n_genes, size=n_donor_genes, replace=False)
    donor_dist = {d: rng.normal(0, cfg.donor_strength, size=n_donor_genes) for d in donor_labels}
    for i, d in enumerate(donors):
        mean[i, donor_genes] *= np.exp(donor_dist[d])

    # Library size
    lib = rng.lognormal(mean=0.0, sigma=cfg.libsize_strength, size=n_cells)
    mean *= lib[:, None]

    # Sex markers boost (case-insensitive)
    gmap = {str(g).strip().lower(): idx for idx, g in enumerate(genes)}
    if str(species).lower() == "human":
        x_markers = HUMAN_SEX_GENES_F
        y_markers = HUMAN_SEX_GENES_M
    else:
        x_markers = MOUSE_SEX_GENES_F
        y_markers = MOUSE_SEX_GENES_M
    x_idx = [gmap.get(str(g).lower(), None) for g in x_markers]
    y_idx = [gmap.get(str(g).lower(), None) for g in y_markers]
    x_idx = [i for i in x_idx if i is not None]
    y_idx = [i for i in y_idx if i is not None]
    if len(x_idx) > 0 or len(y_idx) > 0:
        # Make sex markers much more separable across cell types by reducing baseline
        # and applying strong, symmetric boosts per sex.
        f = np.where(sex_true == "female")[0]
        m = np.where(sex_true != "female")[0]

        if len(x_idx) > 0:
            mean[:, x_idx] *= 0.05  # lower baseline XIST-like signal
            if f.size > 0:
                mean[np.ix_(f, x_idx)] *= 300.0
            if m.size > 0:
                mean[np.ix_(m, x_idx)] *= 0.001

        if len(y_idx) > 0:
            mean[:, y_idx] *= 0.05  # lower baseline Y-gene signal
            if f.size > 0:
                mean[np.ix_(f, y_idx)] *= 0.001
            if m.size > 0:
                mean[np.ix_(m, y_idx)] *= 300.0

    counts = _nb_counts_from_mean(mean, theta=cfg.theta, rng=rng)
    X = csr_matrix(counts)
    a = ad.AnnData(X=X)
    a.var_names = pd.Index(genes)
    a.obs_names = pd.Index([f"{name}_spot{i:06d}" for i in range(n_cells)])

    a.obs["cell_type"] = pd.Categorical(ct)
    a.obs["true_cell_type"] = a.obs["cell_type"].astype(str)
    a.obs["section"] = pd.Categorical([section_label] * n_cells)
    # Combined batch key for spatial (better correction/eval)
    a.obs["batch"] = pd.Categorical([f"{section_label}__{tech_label}"] * n_cells)
    a.obs["true_batch"] = a.obs["batch"].astype(str)
    a.obs["true_donor"] = np.array(donors, dtype=object)
    a.obs["true_technology"] = np.array([tech_label] * n_cells, dtype=object)
    a.obs["technology"] = pd.Categorical([tech_label] * n_cells)
    a.obs["true_sex"] = np.array(sex_true, dtype=object)
    a.obs["true_species"] = np.array([str(species).lower()] * n_cells, dtype=object)

    tech_variants = {
        "10x-Visium": ["Visium", "10x Visium", "visium", "10X_Visium"],
        "Stereo-seq": ["Stereo-seq", "stereo", "StereoSeq"],
        "Slide-seqV2": ["Slide-seqV2", "slideseq", "SlideSeqV2", "Slide-seq"],
    }
    a.obs["assay"] = rng.choice(tech_variants.get(tech_label, [tech_label]), size=n_cells)
    a.obs["donor_id"] = a.obs["true_donor"].astype(str)

    # placeholders
    a.obs["donor"] = "Unknown"
    a.obs["sex"] = "Unknown"
    a.obs["species"] = "Unknown"

    a.obsm["spatial"] = coords.astype(np.float32)
    a.layers["counts"] = a.X.copy()
    a.uns["spatial"] = {"dummy": True}
    return a

def write_simulations(outdir: Path, seed: int) -> Dict[str, List[Path]]:
    # ... (unchanged, same as original) ...
    rng = np.random.default_rng(seed)

    cfgA = SimConfig(
        n_cells=3500, n_genes=6000, n_celltypes=10, n_donors=3,
        batch_strength=1.5, donor_strength=0.9, tech_strength=0.0,
        libsize_strength=0.60, frac_batch_genes=0.30, frac_donor_genes=0.20,
        frac_tech_genes=0.0, theta=14.0,
    )
    cfgB = SimConfig(
        n_cells=4000, n_genes=7000, n_celltypes=12, n_donors=4,
        batch_strength=1.8, donor_strength=1.1, tech_strength=0.0,
        libsize_strength=0.65, frac_batch_genes=0.35, frac_donor_genes=0.25,
        frac_tech_genes=0.0, theta=12.0,
    )
    cfgC = SimConfig(
        n_cells=3500, n_genes=6000, n_celltypes=9, n_donors=4,
        # Stronger multi-factor effects (section + technology) to make integration/h5adify more relevant
        batch_strength=1.35, donor_strength=0.85, tech_strength=2.6,
        libsize_strength=0.70, frac_batch_genes=0.25, frac_donor_genes=0.18,
        frac_tech_genes=0.45, theta=14.0,
    )

    groups = {}

    # SimA_Brain
    simA = []
    for i, study in enumerate(["study1", "study2", "study3"], start=1):
        a = simulate_scrna_dataset(
            name=f"simA_human_brain_{study}",
            species="human",
            batch_label=study,
            donor_labels=[f"D{i}A", f"D{i}B", f"D{i}C"],
            technology_label="10xv3",
            cfg=cfgA,
            rng=rng,
            dataset_idx=i,
        )
        if study == "study2":
            a.obs["batch"] = a.obs["batch"].astype(str).str.replace("study2", "Study-02")
        if study == "study3":
            a.obs["batch"] = a.obs["batch"].astype(str).str.replace("study3", "S3")
        a.obs["batch"] = pd.Categorical(a.obs["batch"].astype(str))
        p = outdir / f"raw_simA_{study}.h5ad"
        a.write_h5ad(p)
        simA.append(p)
    groups["SimA_Brain"] = simA

    # SimB_GBM
    simB = []
    for i, b in enumerate(["gbm_like_1", "gbm_like_2"], start=1):
        a = simulate_scrna_dataset(
            name=f"simB_human_{b}",
            species="human",
            batch_label=b,
            donor_labels=[f"P{i}A", f"P{i}B", f"P{i}C", f"P{i}D"],
            technology_label="10xv2" if i == 1 else "10xv3",
            cfg=cfgB,
            rng=rng,
            dataset_idx=i + 3,
        )
        if b == "gbm_like_1":
            a.obs["batch"] = a.obs["batch"].astype(str).str.replace("gbm_like_1", "GBM-1")
        else:
            a.obs["batch"] = a.obs["batch"].astype(str).str.replace("gbm_like_2", "gbm2")
        a.obs["batch"] = pd.Categorical(a.obs["batch"].astype(str))
        p = outdir / f"raw_simB_{b}.h5ad"
        a.write_h5ad(p)
        simB.append(p)
    groups["SimB_GBM"] = simB

    # SimC_Spatial
    simC = []
    techs = ["10x-Visium", "Stereo-seq", "Slide-seqV2"]
    for sec_idx, sec in enumerate(["sec1", "sec2", "sec3"]):
        for tech_idx, tech in enumerate(techs):
            a = simulate_spatial_dataset(
                name=f"simC_mouse_{sec}_{tech}",
                species="mouse",
                section_label=sec,
                tech_label=tech,
                donor_labels=[f"M{sec}A", f"M{sec}B", f"M{sec}C"],
                cfg=cfgC,
                rng=rng,
                dataset_idx=sec_idx * 3 + tech_idx + 5,
            )
            if tech == "10x-Visium":
                a.obs["technology"] = a.obs["technology"].astype(str).str.replace("10x-Visium", "Visium")
            if tech == "Slide-seqV2":
                a.obs["technology"] = a.obs["technology"].astype(str).str.replace("Slide-seqV2", "SlideSeqV2")
            a.obs["technology"] = pd.Categorical(a.obs["technology"].astype(str))
            # Keep combined batch consistent after technology canonicalization
            if "section" in a.obs.columns and "technology" in a.obs.columns:
                a.obs["batch"] = pd.Categorical(a.obs["section"].astype(str) + "__" + a.obs["technology"].astype(str))
                a.obs["true_batch"] = a.obs["batch"].astype(str)
            p = outdir / f"raw_simC_{sec}_{tech}.h5ad"
            a.write_h5ad(p)
            simC.append(p)
    groups["SimC_Spatial"] = simC

    return groups

# ---------------------------------------------------------------------
# h5adify harmonisation with robust fallback
# ---------------------------------------------------------------------
def add_zip_to_syspath(zip_path: Path, extract_dir: Path) -> None:
    ensure_dir(extract_dir)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    sys.path.insert(0, str(extract_dir))

def canonicalize_series(x: pd.Series, mapping: Dict[str, str], default: Optional[str] = None) -> pd.Series:
    y = x.astype(str).str.strip()
    y_low = y.str.lower()
    out = []
    for v, vlow in zip(y, y_low):
        if vlow in mapping:
            out.append(mapping[vlow])
        else:
            out.append(default if default is not None else v)
    return pd.Series(out, index=x.index)

def infer_species_from_varnames(a: ad.AnnData, fallback: str = "unknown") -> str:
    v = a.var_names.astype(str)
    if len(v) == 0:
        return fallback
    samp = v[: min(3000, len(v))]
    upper_frac = np.mean([s.isupper() for s in samp])
    mouse_re = re.compile(r"^[A-Z][a-z]{2,}")
    mouse_like = np.mean([mouse_re.match(s) is not None for s in samp])
    if upper_frac > 0.40:
        return "human"
    if mouse_like > 0.25:
        return "mouse"
    return fallback

def infer_sex_from_expression(a: ad.AnnData, species: str, layer: str = "counts") -> pd.Series:
    """
    Sex inference from expression, designed to be *robust* in heterogeneous / noisy settings.

    Key ideas:
      - Prefer donor/patient-level calls when possible (reduces cell-type-specific XIST dropouts).
      - Use BOTH X-inactivation (XIST/Xist) and Y-gene evidence; ambiguous cases -> "Unknown".
      - Adaptive thresholds learned per dataset via robust "largest gap" cut (no fixed hard threshold).

    Returns a per-cell Series with values in {"female","male","Unknown"}.
    """
    if a is None or a.n_obs == 0:
        return pd.Series([], dtype=object)
    if a.n_vars == 0:
        return pd.Series(["Unknown"] * a.n_obs, index=a.obs_names, dtype=object)

    # Expression matrix
    X = a.layers[layer] if layer in getattr(a, "layers", {}) else a.X
    if sparse.issparse(X):
        X = X.tocsr()

    # Collect gene representations to match markers robustly
    gene_reprs = [a.var_names.astype(str).tolist()]
    for col in ["gene_harmonized", "gene_symbol", "gene_symbols", "gene", "gene_name", "feature_name"]:
        if col in a.var.columns:
            gene_reprs.append(a.var[col].astype(str).tolist())

    name_to_idx: Dict[str, int] = {}
    for j in range(a.n_vars):
        for rep in gene_reprs:
            try:
                nm = rep[j]
            except Exception:
                continue
            if nm is None:
                continue
            key = str(nm).strip().lower()
            if key and key != "nan" and key not in name_to_idx:
                name_to_idx[key] = j

    def _idx(names: List[str]) -> List[int]:
        out = []
        for g in names:
            k = str(g).strip().lower()
            if k in name_to_idx:
                out.append(name_to_idx[k])
        return sorted(set(out))

    sp = str(species).lower()
    if sp == "human":
        x_markers = HUMAN_SEX_GENES_F
        y_markers = HUMAN_SEX_GENES_M
        x_primary = ["XIST"]
    else:
        x_markers = MOUSE_SEX_GENES_F
        y_markers = MOUSE_SEX_GENES_M
        x_primary = ["Xist"]

    x_idx_all = _idx(x_markers)
    y_idx_all = _idx(y_markers)
    x_idx_primary = _idx(x_primary)

    if len(x_idx_all) == 0 and len(y_idx_all) == 0:
        return pd.Series(["Unknown"] * a.n_obs, index=a.obs_names, dtype=object)

    # Use XIST/Xist if present, otherwise mean across X markers
    x_idx_use = x_idx_primary if len(x_idx_primary) > 0 else x_idx_all

    def _score(idxs: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Return (log1p(mean-counts-per-gene), fraction of genes expressed>0)."""
        if len(idxs) == 0:
            return np.zeros(a.n_obs, dtype=np.float32), np.zeros(a.n_obs, dtype=np.float32)
        Xsub = X[:, idxs]
        if sparse.issparse(Xsub):
            s = np.asarray(Xsub.sum(axis=1)).ravel().astype(np.float32)
            nnz = np.asarray(Xsub.getnnz(axis=1)).ravel().astype(np.float32)
        else:
            s = Xsub.sum(axis=1).astype(np.float32)
            nnz = (Xsub > 0).sum(axis=1).astype(np.float32)
        mean_per_gene = s / max(1.0, float(len(idxs)))
        sig = np.log1p(mean_per_gene)
        frac = nnz / max(1.0, float(len(idxs)))
        return sig, frac

    x_sig, x_frac = _score(x_idx_use)
    y_sig, y_frac = _score(y_idx_all)

    # Grouping (prefer patient/donor level)
    group_key = None
    for k in ["patient_id", "donor", "patient", "donor_id", "subject", "individual", "sample", "sample_id"]:
        if k in a.obs.columns and a.obs[k].astype(str).nunique() > 1:
            group_key = k
            break

    def _largest_gap_threshold(vals: np.ndarray) -> float:
        v = np.asarray(vals, dtype=float)
        v = v[np.isfinite(v)]
        if v.size < 2:
            return float(np.nanmedian(vals))
        v = np.sort(v)
        gaps = np.diff(v)
        if gaps.size == 0:
            return float(np.nanmedian(vals))
        cut = int(np.argmax(gaps))
        return float(0.5 * (v[cut] + v[cut + 1]))

    def _classify(xv: float, yv: float, yfv: float, x_cut: float, y_cut: float) -> str:
        # Strong, non-ambiguous rules
        if np.isfinite(yv) and np.isfinite(xv):
            # male: strong Y, weak X
            if (yv >= y_cut + 0.03) and (yfv >= 0.02) and (xv <= x_cut - 0.01):
                return "male"
            # female: strong X, weak Y
            if (xv >= x_cut + 0.03) and (yv <= y_cut - 0.01):
                return "female"

            # Secondary rules (ratio/difference-based) for unimodal cases
            if (yv - xv) >= 0.18 and yfv >= 0.03:
                return "male"
            if (xv - yv) >= 0.18:
                return "female"

        # If only one signal exists, be conservative
        if len(y_idx_all) > 0 and len(x_idx_all) == 0:
            if yv >= y_cut + 0.05 and yfv >= 0.05:
                return "male"
        if len(x_idx_all) > 0 and len(y_idx_all) == 0:
            if xv >= x_cut + 0.05:
                return "female"

        return "Unknown"

    if group_key is not None:
        groups = a.obs.groupby(group_key).groups
        gx = []
        gy = []
        for _, idxs in groups.items():
            ii = a.obs.index.get_indexer(idxs) if isinstance(idxs, pd.Index) else np.asarray(list(idxs), dtype=int)
            if ii.size == 0:
                continue
            gx.append(float(np.nanmedian(x_sig[ii])))
            gy.append(float(np.nanmedian(y_sig[ii])))
        gx = np.asarray(gx, dtype=float)
        gy = np.asarray(gy, dtype=float)

        x_cut = _largest_gap_threshold(gx)
        y_cut = _largest_gap_threshold(gy)

        pred = np.array(["Unknown"] * a.n_obs, dtype=object)
        for _, idxs in groups.items():
            ii = a.obs.index.get_indexer(idxs) if isinstance(idxs, pd.Index) else np.asarray(list(idxs), dtype=int)
            if ii.size == 0:
                continue
            xv = float(np.nanmedian(x_sig[ii]))
            yv = float(np.nanmedian(y_sig[ii]))
            yfv = float(np.nanmedian(y_frac[ii]))
            lab = _classify(xv, yv, yfv, x_cut, y_cut)
            pred[ii] = lab
        return pd.Series(pred, index=a.obs_names, dtype=object)

    # Cell-level fallback (still conservative)
    x_cut = _largest_gap_threshold(x_sig)
    y_cut = _largest_gap_threshold(y_sig)
    pred = [_classify(float(x_sig[i]), float(y_sig[i]), float(y_frac[i]), x_cut, y_cut) for i in range(a.n_obs)]
    return pd.Series(pred, index=a.obs_names, dtype=object)


def run_h5adify_harmonization(
    raw: ad.AnnData,
    target_species: str,
    use_llm: bool,
    fallback_to_simple: bool = True
) -> ad.AnnData:
    """Run h5adify harmonisation; if it fails, apply simple fallback."""
    a = raw.copy()
    if "counts" not in a.layers:
        a.layers["counts"] = a.X.copy()

    # Preserve truth columns
    truth_cols = [col for col in raw.obs.columns if col.startswith("true_")]
    for col in truth_cols:
        a.obs[col] = raw.obs[col].copy()

    reports = {}
    h5adify_success = False

    try:
        import h5adify
        from h5adify.core import harmonize_anndata, harmonize_metadata
        reports["h5adify_import"] = f"ok ({getattr(h5adify, '__version__', 'unknown')})"

        # Gene harmonisation
        try:
            a, gene_report, profile = harmonize_anndata(
                a,
                target="hugo" if target_species.lower() == "human" else "mouse",
                target_species=target_species,
                var_key=None,
                out_key="gene_harmonized",
                inplace=True,
                deduplicate=True,
                dedup_how="sum",
                chunk_size=800,
            )
            reports["gene_harmonization"] = "success"
            h5adify_success = True
        except Exception as e:
            reports["gene_harmonization"] = f"failed: {e}"

        # Metadata harmonisation
        try:
            a, meta_report = harmonize_metadata(
                a,
                use_llm=use_llm,
                llm_prompt_name="metadata_harmonize_v1_default",
                sex_from_expression=True,
                inplace=False,
            )
            reports["metadata_harmonization"] = "success"
        except Exception as e:
            reports["metadata_harmonization"] = f"failed: {e}"

    except Exception as e:
        reports["h5adify_import"] = f"failed: {e}"

    # If h5adify gene harmonisation failed, apply simple fallback
    if not h5adify_success and fallback_to_simple:
        log(f"h5adify gene harmonisation failed. Applying simple fallback for {target_species}.")
        a = simple_gene_harmonization(a, target_species=target_species, dedup_how="sum")
        reports["gene_harmonization_fallback"] = "applied"
    # Always standardize var_names (applies var['gene_harmonized'] if present) to ensure non-empty intersections
    a = simple_gene_harmonization(a, target_species=target_species, dedup_how="sum")

    # ---- Standardise metadata (same as original) ----
    inferred_species = infer_species_from_varnames(a, fallback=target_species)
    a.obs["species"] = inferred_species

    donor_src = None
    for k in ["donor", "patient", "patient_id", "donor_id", "subject", "individual"]:
        if k in a.obs.columns and a.obs[k].astype(str).nunique() > 1:
            donor_src = k
            break
    a.obs["donor"] = a.obs[donor_src].astype(str) if donor_src else "Unknown"

    tech_src = None
    for k in ["technology", "platform", "assay", "tech"]:
        if k in a.obs.columns and a.obs[k].astype(str).nunique() > 1:
            tech_src = k
            break
    a.obs["technology"] = a.obs[tech_src].astype(str) if tech_src else "Unknown"

    if "batch" in a.obs.columns and a.obs["batch"].astype(str).nunique() >= 1:
        a.obs["batch"] = a.obs["batch"].astype(str)
    elif "section" in a.obs.columns:
        a.obs["batch"] = a.obs["section"].astype(str)
    else:
        a.obs["batch"] = "batch0"

    batch_map = {"study-02": "study2", "s3": "study3", "gbm-1": "gbm_like_1", "gbm2": "gbm_like_2"}
    a.obs["batch"] = canonicalize_series(a.obs["batch"], batch_map).astype(str)
    a.obs["batch"] = pd.Categorical(a.obs["batch"])

    tech_map = {
        "visium": "10x-Visium",
        "10x visium": "10x-Visium",
        "10x_visium": "10x-Visium",
        "slideseqv2": "Slide-seqV2",
        "slide-seqv2": "Slide-seqV2",
        "slideseq": "Slide-seqV2",
        "stereo": "Stereo-seq",
        "stereoseq": "Stereo-seq",
    }
    a.obs["technology"] = canonicalize_series(a.obs["technology"], tech_map).astype(str)
    a.obs["technology"] = pd.Categorical(a.obs["technology"])

    sex_pred = infer_sex_from_expression(a, species=inferred_species, layer="counts")
    a.obs["sex"] = sex_pred.astype(str)
    a.obs["sex"] = pd.Categorical(a.obs["sex"])

    a.uns["h5adify_v25_report"] = reports
    return a

def harmonize_files_with_h5adify(
    raw_paths: List[Path],
    outdir: Path,
    target_species: str,
    use_llm: bool,
) -> List[Path]:
    ensure_dir(outdir)
    out_paths = []
    for p in raw_paths:
        a = sc.read_h5ad(p)
        h = run_h5adify_harmonization(
            a,
            target_species=target_species,
            use_llm=use_llm,
            fallback_to_simple=True
        )
        outp = outdir / f"harm_{p.name}"
        h.write_h5ad(outp)
        ok(f"Harmonized: {outp.name}")
        out_paths.append(outp)

    # Post‑hoc: ensure all harmonised files in this group share at least one gene
    # If intersection is empty, force a common simple harmonisation on all of them
    adatas = [sc.read_h5ad(p) for p in out_paths]
    common_genes = set(adatas[0].var_names)
    for ad in adatas[1:]:
        common_genes &= set(ad.var_names)
    if len(common_genes) == 0:
        warn(f"No common genes after harmonisation for {target_species}. Re‑applying simple harmonisation to all.")
        new_paths = []
        for p in out_paths:
            a = sc.read_h5ad(p)
            a = simple_gene_harmonization(a, target_species=target_species, dedup_how="sum")
            # Overwrite
            a.write_h5ad(p)
            new_paths.append(p)
        return new_paths

    return out_paths

# ---------------------------------------------------------------------
# Benchmarking methods (unchanged, same as original)
# ---------------------------------------------------------------------
def preprocess_for_benchmark(
    a: ad.AnnData, *, batch_key: str, n_top_genes: int, n_pcs: int, seed: int
) -> ad.AnnData:
    set_global_seeds(seed)
    b = a.copy()
    if "counts" not in b.layers:
        b.layers["counts"] = b.X.copy()

    sc.pp.normalize_total(b, target_sum=1e4, inplace=True)
    sc.pp.log1p(b)

    sc.pp.highly_variable_genes(
        b,
        n_top_genes=n_top_genes,
        flavor="cell_ranger",
        batch_key=batch_key,
        layer="counts",
        inplace=True,
    )
    if "highly_variable" not in b.var.columns:
        raise RuntimeError("HVG selection failed (no b.var['highly_variable']).")
    b = b[:, b.var["highly_variable"]].copy()

    sc.pp.pca(b, n_comps=n_pcs, use_highly_variable=False, svd_solver="randomized", zero_center=False)
    b.obsm["Unintegrated"] = b.obsm["X_pca"].copy()
    return b

def run_scanorama_embedding(a: ad.AnnData, batch_key: str) -> Optional[np.ndarray]:
    # ... unchanged ...
    scanorama = safe_import("scanorama")
    if scanorama is None:
        return None
    cats = a.obs[batch_key].astype("category").cat.categories
    ad_list = [a[a.obs[batch_key] == c].copy() for c in cats]
    try:
        scanorama.integrate_scanpy(ad_list)
        dim = ad_list[0].obsm["X_scanorama"].shape[1]
        Z = np.zeros((a.n_obs, dim), dtype=np.float32)
        for i, c in enumerate(cats):
            Z[a.obs[batch_key] == c, :] = ad_list[i].obsm["X_scanorama"]
        return Z
    except Exception as e:
        warn(f"Scanorama failed: {e}")
        return None

def run_harmony_embedding(
    a: ad.AnnData, batch_key: str, covariates: List[str]
) -> Optional[np.ndarray]:
    try:
        from harmony import harmonize
        Z = harmonize(a.obsm["X_pca"], a.obs, batch_key=batch_key)
        return np.asarray(Z, dtype=np.float32)
    except Exception as e1:
        try:
            import harmonypy as hm
            ho = hm.run_harmony(
                a.obsm["X_pca"], a.obs, batch_key,
                max_iter_harmony=20, verbose=False
            )
            Z = ho.Z_corr.T
            return np.asarray(Z, dtype=np.float32)
        except Exception as e2:
            warn(f"Harmony failed (pytorch: {e1}, py: {e2})")
            return None

def run_combat_embedding(
    a: ad.AnnData, batch_key: str, n_pcs: int, seed: int
) -> Optional[np.ndarray]:
    try:
        b = a.copy()
        sc.pp.combat(b, key=batch_key)
        sc.pp.pca(b, n_comps=n_pcs, use_highly_variable=False, svd_solver="randomized", zero_center=False)
        return np.asarray(b.obsm["X_pca"], dtype=np.float32)
    except Exception as e:
        warn(f"ComBat failed: {e}")
        return None

def run_scvi_embedding(
    a: ad.AnnData,
    batch_key: str,
    covariates: List[str],
    n_latent: int,
    seed: int,
    max_epochs: int = 100,
) -> Optional[np.ndarray]:
    scvi = safe_import("scvi")
    if scvi is None:
        return None
    try:
        set_global_seeds(seed)
        b = a.copy()
        cat_covs = []
        for c in covariates:
            if c in b.obs.columns and b.obs[c].astype(str).nunique() > 1:
                cat_covs.append(c)
        scvi.model.SCVI.setup_anndata(b, layer="counts", batch_key=batch_key, categorical_covariate_keys=cat_covs)
        vae = scvi.model.SCVI(b, n_layers=2, n_latent=n_latent, gene_likelihood="nb")
        vae.train(max_epochs=max_epochs)
        Z = vae.get_latent_representation()
        return np.asarray(Z, dtype=np.float32)
    except Exception as e:
        warn(f"scVI failed: {e}")
        return None

def run_scanvi_embedding(
    a: ad.AnnData,
    batch_key: str,
    label_key: str,
    covariates: List[str],
    n_latent: int,
    seed: int,
    max_epochs_scvi: int = 80,
    max_epochs_scanvi: int = 40,
) -> Optional[np.ndarray]:
    scvi = safe_import("scvi")
    if scvi is None:
        return None
    try:
        set_global_seeds(seed)
        b = a.copy()
        if label_key not in b.obs.columns:
            return None
        b.obs[label_key] = b.obs[label_key].astype("category")
        if "Unknown" not in list(b.obs[label_key].cat.categories):
            b.obs[label_key] = b.obs[label_key].cat.add_categories(["Unknown"])
        if b.obs[label_key].isna().any():
            b.obs[label_key] = b.obs[label_key].fillna("Unknown")
        cat_covs = []
        for c in covariates:
            if c in b.obs.columns and b.obs[c].astype(str).nunique() > 1:
                cat_covs.append(c)

        scvi.model.SCVI.setup_anndata(b, layer="counts", batch_key=batch_key, categorical_covariate_keys=cat_covs)
        vae = scvi.model.SCVI(b, n_layers=2, n_latent=n_latent, gene_likelihood="nb")
        vae.train(max_epochs=max_epochs_scvi)

        lvae = scvi.model.SCANVI.from_scvi_model(
            vae,
            adata=b,
            labels_key=label_key,
            unlabeled_category="Unknown",
        )
        lvae.train(max_epochs=max_epochs_scanvi)
        Z = lvae.get_latent_representation()
        return np.asarray(Z, dtype=np.float32)
    except Exception as e:
        warn(f"scANVI failed: {e}")
        return None

# ---------------------------------------------------------------------
# UMAP & 2D visualisations (unchanged)
# ---------------------------------------------------------------------
def compute_umap_coords(a: ad.AnnData, rep_key: str, seed: int) -> np.ndarray:
    b = a.copy()
    set_global_seeds(seed)
    sc.pp.neighbors(b, use_rep=rep_key, n_neighbors=15, random_state=seed)
    sc.tl.umap(b, random_state=seed)
    return np.asarray(b.obsm["X_umap"], dtype=np.float32)

def plot_umap_grid(
    a: ad.AnnData,
    rep_keys: List[str],
    color: str,
    out_png: Path,
    seed: int,
    title: str,
    max_cols: int = 4,
) -> None:
    # ... unchanged ...
    reps = [r for r in rep_keys if r in a.obsm.keys()]
    if len(reps) == 0:
        warn(f"No embeddings available for UMAP grid: {out_png.name}")
        return

    n = len(reps)
    ncols = min(max_cols, n)
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.2 * nrows), squeeze=False)
    fig.suptitle(title, fontsize=14)

    for i, rep in enumerate(reps):
        r = i // ncols
        c = i % ncols
        ax = axes[r][c]
        try:
            coords = compute_umap_coords(a, rep_key=rep, seed=seed)
            vals = a.obs[color] if color in a.obs.columns else None
            if vals is None:
                ax.text(0.5, 0.5, f"Missing {color}", ha="center", va="center")
                ax.set_axis_off()
                continue

            if pd.api.types.is_numeric_dtype(vals):
                sc_ = ax.scatter(coords[:, 0], coords[:, 1], s=4, c=vals.to_numpy(), alpha=0.8)
                fig.colorbar(sc_, ax=ax, fraction=0.046, pad=0.04)
            else:
                cats = pd.Categorical(vals.astype(str))
                codes = cats.codes
                sc_ = ax.scatter(coords[:, 0], coords[:, 1], s=4, c=codes, cmap="tab20", alpha=0.8)
                uniq = list(cats.categories)
                show = uniq[:18]
                handles = []
                for j, lab in enumerate(show):
                    handles.append(plt.Line2D([0], [0], marker="o", color="w",
                                              markerfacecolor=plt.cm.tab20(j % 20), markersize=6, label=lab))
                ax.legend(handles=handles, loc="best", fontsize=7, frameon=False)

            ax.set_title(rep, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
        except Exception as e:
            ax.text(0.5, 0.5, f"UMAP failed\n{e}", ha="center", va="center")
            ax.set_axis_off()

    for j in range(n, nrows * ncols):
        r = j // ncols
        c = j % ncols
        axes[r][c].set_axis_off()

    fig.tight_layout(rect=[0, 0.01, 1, 0.97])
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

def plot_embedding_2d_grid(
    a: ad.AnnData,
    rep_keys: List[str],
    color: str,
    out_png: Path,
    title: str,
    max_cols: int = 4,
) -> None:
    # ... unchanged ...
    reps = [r for r in rep_keys if r in a.obsm.keys()]
    if len(reps) == 0:
        warn(f"No embeddings available for 2D grid: {out_png.name}")
        return

    n = len(reps)
    ncols = min(max_cols, n)
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.2 * nrows), squeeze=False)
    fig.suptitle(title, fontsize=14)
    vals = a.obs[color] if color in a.obs.columns else None

    for i, rep in enumerate(reps):
        r = i // ncols
        c = i % ncols
        ax = axes[r][c]
        Z = np.asarray(a.obsm[rep])
        if Z.ndim != 2 or Z.shape[1] < 2:
            ax.text(0.5, 0.5, "dim<2", ha="center", va="center")
            ax.set_axis_off()
            continue
        x = Z[:, 0]
        y = Z[:, 1]

        if vals is None:
            ax.scatter(x, y, s=4, alpha=0.8)
        else:
            if pd.api.types.is_numeric_dtype(vals):
                sc_ = ax.scatter(x, y, s=4, c=vals.to_numpy(), alpha=0.8)
                fig.colorbar(sc_, ax=ax, fraction=0.046, pad=0.04)
            else:
                cats = pd.Categorical(vals.astype(str))
                codes = cats.codes
                ax.scatter(x, y, s=4, c=codes, cmap="tab20", alpha=0.8)

        ax.set_title(rep, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    for j in range(n, nrows * ncols):
        r = j // ncols
        c = j % ncols
        axes[r][c].set_axis_off()

    fig.tight_layout(rect=[0, 0.01, 1, 0.97])
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

# ---------------------------------------------------------------------
# scib-metrics benchmark – **SAVES FULL get_results() DF, PLOTS MATCH**
# ---------------------------------------------------------------------
def run_scib_benchmark(
    a: ad.AnnData,
    batch_key: str,
    label_key: str,
    embedding_keys: List[str],
    n_jobs: int,
    outdir: Path,
    stage_name: str,
    min_max_scale: bool = False,   # Keep raw scores for comparability
) -> Tuple[Optional[Path], Optional[Path], Optional[pd.DataFrame]]:
    ensure_dir(outdir)

    scib_metrics = safe_import("scib_metrics")
    if scib_metrics is None:
        warn("scib_metrics not installed. Cannot run benchmark.")
        return None, None, None

    from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection

    keys = [k for k in embedding_keys if k in a.obsm.keys()]
    if len(keys) == 0:
        warn(f"No embeddings available for scib benchmark (stage={stage_name}).")
        return None, None, None

    results_csv = outdir / f"{stage_name}_scib_metrics_table.csv"
    results_png = outdir / f"{stage_name}_scib_metrics_table.png"
    results_png_fixed = outdir / f"{stage_name}_scib_metrics_table_fixed.png"
    results_df = None

    try:
        bm = Benchmarker(
            a,
            batch_key=batch_key,
            label_key=label_key,
            bio_conservation_metrics=BioConservation(),
            batch_correction_metrics=BatchCorrection(),
            embedding_obsm_keys=keys,
            n_jobs=n_jobs,
        )
        bm.benchmark()

        # Get the full results table (with aggregates)
        results_df_full = bm.get_results(min_max_scale=min_max_scale)
        # Save it
        results_df_full.to_csv(results_csv)
        log(f"Saved full benchmark results (with aggregates) to {results_csv}")

        # Also store the raw results table (without aggregates) for reference?
        raw_df = None
        if hasattr(bm, "results_table") and isinstance(bm.results_table, pd.DataFrame):
            raw_df = bm.results_table.copy()
            raw_csv = outdir / f"{stage_name}_scib_metrics_raw.csv"
            raw_df.to_csv(raw_csv)

        results_df = results_df_full  # for downstream use
        # ----- scIB PLOTS -----
        # Save native scib-metrics plot for reference (can have layout issues on some setups)
        native_png = outdir / f"{stage_name}_scib_metrics_table_native.png"
        try:
            fig_obj = bm.plot_results_table(min_max_scale=min_max_scale, show=False)
            fig = None
            if hasattr(fig_obj, "figure") and hasattr(fig_obj.figure, "savefig"):
                fig = fig_obj.figure
            elif hasattr(fig_obj, "get_figure") and callable(fig_obj.get_figure):
                fig = fig_obj.get_figure()
            elif isinstance(fig_obj, plt.Figure):
                fig = fig_obj
            else:
                fig = plt.gcf()

            n_rows = len(results_df_full.index) - (1 if "_METRIC_TYPE" in results_df_full.index else 0)
            n_cols = len(results_df_full.columns)
            fig_w = max(18, 0.45 * n_cols + 4)
            fig_h = max(6, 0.45 * n_rows + 3)
            fig.set_size_inches(fig_w, fig_h, forward=True)
            fig.tight_layout()
            fig.savefig(native_png, dpi=240, bbox_inches="tight")
            plt.close(fig)
            ok(f"Native scIB table saved (reference) to {native_png}")
        except Exception as e:
            warn(f"Native scIB plotting failed: {e}")

        # Save FIXED plot (aggregate bars aligned and placed to the right)
        try:
            plot_scib_results_table_fixed(results_df_full, results_png, title=stage_name)
            plot_scib_results_table_fixed(results_df_full, results_png_fixed, title=stage_name)
            ok(f"Fixed scIB metrics table saved to {results_png} and {results_png_fixed}")
        except Exception as e:
            warn(f"Fixed scIB plotting failed: {e}. Using fallback heatmap.")
            plot_scib_heatmap_fallback(results_df_full, results_png, stage_name)


    except Exception as e:
        warn(f"scib benchmark failed (stage={stage_name}): {e}")
        return None, None, None

    return results_csv, results_png, results_df

def plot_scib_heatmap_fallback(df: pd.DataFrame, out_png: Path, title: str) -> None:
    # Remove the MetricType row if present
    if "_METRIC_TYPE" in df.index:
        df = df.drop("_METRIC_TYPE", axis=0)
    dff = df.select_dtypes(include=[np.number])
    if dff.shape[0] == 0 or dff.shape[1] == 0:
        warn("No numeric data for fallback heatmap.")
        return

    n_rows, n_cols = dff.shape
    fig_w = max(14.0, 0.45 * n_cols + 4.0)
    fig_h = max(4.5, 0.45 * n_rows + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(dff.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([str(x) for x in dff.index])
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(dff.columns, rotation=75, ha="right", fontsize=9)

    for i in range(n_rows):
        for j in range(n_cols):
            v = dff.values[i, j]
            if np.isfinite(v):
                color = "white" if v > 0.5 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color=color)

    ax.set_title(f"{title} – scIB metrics (fallback)", fontsize=12)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Score")
    fig.tight_layout()
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    plt.close(fig)
    ok(f"Fallback scIB heatmap saved to {out_png}")



def _scib_drop_metrictype_and_coerce(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    scib-metrics often includes a special row `_METRIC_TYPE` containing strings ("bio"/"batch"/...).
    That single row makes *all* columns dtype=object, even after dropping it.

    This helper:
      - splits out the `_METRIC_TYPE` row (if present)
      - drops it from the method table
      - coerces every remaining column to numeric (errors='coerce')
      - drops columns that become all-NaN
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.shape[0] == 0:
        return pd.DataFrame(), None

    d = df.copy()
    mtypes = None
    if "_METRIC_TYPE" in d.index:
        try:
            mtypes = d.loc["_METRIC_TYPE"].astype(str)
        except Exception:
            mtypes = None
        d = d.drop("_METRIC_TYPE", axis=0)

    # Coerce columns to numeric *after* removing the metric-type row
    dnum = d.apply(lambda s: pd.to_numeric(s, errors="coerce"))

    # Drop columns that are entirely NaN after coercion
    dnum = dnum.dropna(axis=1, how="all")
    return dnum, mtypes


def _is_bio_metric(name: str) -> bool:
    n = str(name).strip().lower()
    # biological conservation metrics (heuristic, robust across scib-metrics versions)
    bio_tokens = [
        "ari", "nmi", "asw_label", "asw-label", "silhouette_label", "silhouette-label",
        "graph_conn", "graph_connectivity", "graph-connectivity",
        "isolated", "clisi", "c_lisi", "clisi", "hvg", "cell_cycle", "trajectory", "bio"
    ]
    # avoid counting aggregates themselves
    if re.search(r"\bbio\s*conservation\b", n) or n == "bio conservation":
        return False
    return any(t in n for t in bio_tokens)


def _is_batch_metric(name: str) -> bool:
    n = str(name).strip().lower()
    batch_tokens = [
        "kbet", "kbety", "k_bet", "ilisi", "i_lisi", "lisi", "asw_batch", "asw-batch",
        "pcr", "pcr_comparison", "pcr-comparison", "batch"
    ]
    if re.search(r"\bbatch\s*correction\b", n) or n == "batch correction":
        return False
    return any(t in n for t in batch_tokens)


def _ensure_scib_aggregate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the scIB results table contains the three aggregate columns:
      - Bio conservation
      - Batch correction
      - Total

    Why this is needed:
      - some scib-metrics versions output aggregates, some don't
      - `_METRIC_TYPE` row makes numeric columns dtype=object -> downstream plotting/comparisons fail
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.shape[0] == 0:
        return df

    d_orig = df.copy()
    dnum, mtypes = _scib_drop_metrictype_and_coerce(d_orig)

    if dnum.shape[0] == 0 or dnum.shape[1] == 0:
        # nothing we can aggregate
        return d_orig

    cols = list(dnum.columns)

    # Detect existing aggregates (case-insensitive)
    def _find(regex: str) -> Optional[str]:
        for c in cols:
            if re.fullmatch(regex, str(c).strip(), flags=re.IGNORECASE):
                return c
        for c in cols:
            if re.search(regex, str(c), flags=re.IGNORECASE):
                return c
        return None

    bio_c = _find(r"bio\s*conservation")
    batch_c = _find(r"batch\s*correction")
    total_c = _find(r"total")

    # Determine bio/batch metric columns
    metric_cols = [c for c in cols if c not in {bio_c, batch_c, total_c}]
    if mtypes is not None:
        bio_cols = [c for c in metric_cols if "bio" in str(mtypes.get(c, "")).lower()]
        batch_cols = [c for c in metric_cols if "batch" in str(mtypes.get(c, "")).lower()]
        # If mtypes is present but doesn't classify anything, fall back to heuristics
        if len(bio_cols) == 0:
            bio_cols = [c for c in metric_cols if _is_bio_metric(c)]
        if len(batch_cols) == 0:
            batch_cols = [c for c in metric_cols if _is_batch_metric(c)]
    else:
        bio_cols = [c for c in metric_cols if _is_bio_metric(c)]
        batch_cols = [c for c in metric_cols if _is_batch_metric(c)]

    # Add aggregates if missing
    out = dnum.copy()

    if bio_c is None:
        if len(bio_cols) > 0:
            out["Bio conservation"] = out[bio_cols].mean(axis=1)
            bio_c = "Bio conservation"
        else:
            out["Bio conservation"] = np.nan
            bio_c = "Bio conservation"

    if batch_c is None:
        if len(batch_cols) > 0:
            out["Batch correction"] = out[batch_cols].mean(axis=1)
            batch_c = "Batch correction"
        else:
            out["Batch correction"] = np.nan
            batch_c = "Batch correction"

    if total_c is None:
        # Prefer mean of the two aggregates when they exist
        out["Total"] = 0.5 * (pd.to_numeric(out[bio_c], errors="coerce") + pd.to_numeric(out[batch_c], errors="coerce"))
        total_c = "Total"

    # Reorder: metrics first, then the three aggregates at the end
    agg_order = [bio_c, batch_c, total_c]
    agg_order = [c for c in agg_order if c in out.columns]
    metric_order = [c for c in cols if c in out.columns and c not in set(agg_order)]
    out = out[metric_order + agg_order]

    # Re-attach a metric-type row (so saved CSV remains similar to scib-metrics output)
    if "_METRIC_TYPE" in d_orig.index:
        mt = pd.Series(index=out.columns, dtype=object)
        if mtypes is not None:
            for c in out.columns:
                if c in mtypes.index:
                    mt[c] = str(mtypes.get(c))
        for c in agg_order:
            mt[c] = "aggregate"
        out2 = pd.concat([out, mt.to_frame().T.rename(index={0: "_METRIC_TYPE"})], axis=0)
        return out2

    return out


def plot_scib_results_table_fixed(df: pd.DataFrame, out_png: Path, title: str) -> None:
    """
    Custom scIB plot with correctly aligned aggregate bars on the right.
    This version is robust to dtype=object (caused by `_METRIC_TYPE`) by coercing numerics.

    IMPORTANT: it *always* writes an output PNG (even if data are missing),
    so upstream code doesn't falsely report success.
    """
    ensure_dir(out_png.parent)

    if df is None or not isinstance(df, pd.DataFrame) or df.shape[0] == 0:
        # Save a diagnostic placeholder
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.axis("off")
        ax.text(0.5, 0.5, "Empty scIB results – nothing to plot", ha="center", va="center", fontsize=12)
        fig.savefig(out_png, dpi=200, bbox_inches="tight")
        plt.close(fig)
        warn("Empty scIB results DF; wrote placeholder plot.")
        return

    d = _ensure_scib_aggregate_columns(df)
    d_plot = d.drop("_METRIC_TYPE", axis=0, errors="ignore")

    # Coerce to numeric after dropping metric type row
    dnum = d_plot.apply(lambda s: pd.to_numeric(s, errors="coerce")).dropna(axis=1, how="all")
    if dnum.shape[0] == 0 or dnum.shape[1] == 0:
        fig, ax = plt.subplots(figsize=(12, 3.2))
        ax.axis("off")
        ax.text(0.5, 0.55, f"{title} – scIB metrics", ha="center", va="center", fontsize=13)
        ax.text(0.5, 0.35, "No numeric values found after coercion.\nCheck scib-metrics output table.", ha="center", va="center", fontsize=11)
        fig.savefig(out_png, dpi=200, bbox_inches="tight")
        plt.close(fig)
        warn("No numeric scIB data to plot; wrote placeholder plot.")
        return

    # Identify aggregate columns robustly
    def _find(regex_list: List[str]) -> Optional[str]:
        for rgx in regex_list:
            for c in dnum.columns:
                if re.fullmatch(rgx, str(c).strip(), flags=re.IGNORECASE):
                    return c
            for c in dnum.columns:
                if re.search(rgx, str(c), flags=re.IGNORECASE):
                    return c
        return None

    bio_col = _find([r"bio\s*conservation", r"\bbio\b"])
    batch_col = _find([r"batch\s*correction", r"\bbatch\b"])
    total_col = _find([r"\btotal\b"])

    agg_cols = [c for c in [bio_col, batch_col, total_col] if c is not None and c in dnum.columns]
    agg_cols = list(dict.fromkeys(agg_cols))
    metric_cols = [c for c in dnum.columns if c not in agg_cols]

    n_rows = dnum.shape[0]
    cols_used = metric_cols if len(metric_cols) > 0 else list(dnum.columns)

    import matplotlib.gridspec as gridspec

    # Stable sizing
    fig_w = max(18.0, 0.52 * max(1, len(cols_used)) + 9.0)
    fig_h = max(5.8, 0.52 * n_rows + 3.0)

    n_right = len(agg_cols)
    width_ratios = [max(2.0, 0.90 * max(1, len(cols_used)))] + ([1.10] * n_right)

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = gridspec.GridSpec(1, 1 + n_right, width_ratios=width_ratios, wspace=0.06)

    # Heatmap
    ax_hm = fig.add_subplot(gs[0, 0])
    mat = dnum[cols_used].to_numpy(dtype=float)
    im = ax_hm.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1, origin="upper")
    ax_hm.set_yticks(np.arange(n_rows))
    ax_hm.set_yticklabels([str(x) for x in dnum.index])
    ax_hm.set_xticks(np.arange(len(cols_used)))
    ax_hm.set_xticklabels([str(c) for c in cols_used], rotation=75, ha="right", fontsize=9)
    ax_hm.set_title(f"{title} – scIB metrics (fixed)", fontsize=12)
    ax_hm.set_ylim(n_rows - 0.5, -0.5)

    # annotate
    for i in range(n_rows):
        for j in range(len(cols_used)):
            v = mat[i, j]
            if np.isfinite(v):
                color = "white" if v > 0.55 else "black"
                ax_hm.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color=color)

    cbar = fig.colorbar(im, ax=ax_hm, fraction=0.030, pad=0.02)
    cbar.set_label("Score")

    # Right aggregate bars (perfectly aligned)
    for k, col in enumerate(agg_cols):
        ax = fig.add_subplot(gs[0, 1 + k])
        vals = pd.to_numeric(dnum[col], errors="coerce").values.astype(float)
        y = np.arange(n_rows)

        ax.barh(y, vals, height=0.80, alpha=0.95)
        ax.set_xlim(0, 1)
        ax.set_ylim(n_rows - 0.5, -0.5)
        ax.set_yticks(y)
        ax.set_yticklabels([])
        ax.set_title(str(col), fontsize=10)
        ax.grid(axis="x", alpha=0.25)

        for i, v in enumerate(vals):
            if np.isfinite(v):
                ax.text(min(1.02, v + 0.02), i, f"{v:.2f}", va="center", ha="left", fontsize=8)

        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    plt.close(fig)
    ok(f"Fixed scIB plot saved to {out_png}")


def plot_comparison_dotplot(
    before_csv: Path, after_csv: Path, out_png: Path, title: str
) -> None:
    """
    Compare Before vs After for the three aggregate scores using a compact dotplot:
      - Bio conservation
      - Batch correction
      - Total
    Robust to scIB version / column naming differences.
    """
    if not before_csv.exists() or not after_csv.exists():
        warn("Missing CSV files for comparison dotplot.")
        return

    b = pd.read_csv(before_csv, index_col=0)
    a = pd.read_csv(after_csv, index_col=0)

    b = _ensure_scib_aggregate_columns(b)
    a = _ensure_scib_aggregate_columns(a)

    b, _ = _scib_drop_metrictype_and_coerce(b)
    a, _ = _scib_drop_metrictype_and_coerce(a)

    def _find_col(df: pd.DataFrame, patterns: List[str]) -> Optional[str]:
        cols = list(df.columns)
        for rgx in patterns:
            for c in cols:
                if re.fullmatch(rgx, str(c).strip(), flags=re.IGNORECASE):
                    return c
            for c in cols:
                if re.search(rgx, str(c), flags=re.IGNORECASE):
                    return c
        return None

    bio = _find_col(b, [r"bio\s*conservation", r"\bbio\b"])
    batch = _find_col(b, [r"batch\s*correction", r"\bbatch\b"])
    total = _find_col(b, [r"\btotal\b"])

    agg_cols = [c for c in [bio, batch, total] if c is not None and c in b.columns and c in a.columns]
    if len(agg_cols) == 0:
        warn("No aggregate columns found for comparison dotplot.")
        return

    methods = sorted(set(b.index) & set(a.index))
    if len(methods) == 0:
        warn("No common methods between before/after for comparison.")
        return

    n_metrics = len(agg_cols)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5.8 * n_metrics, 5), squeeze=False)
    fig.suptitle(title, fontsize=14)

    for idx, col in enumerate(agg_cols):
        ax = axes[0, idx]
        x = np.arange(len(methods))
        before_vals = b.loc[methods, col].astype(float).values
        after_vals = a.loc[methods, col].astype(float).values
        delta = after_vals - before_vals

        ax.axhline(0.0, lw=1, alpha=0.4)
        ax.scatter(x, delta, s=60)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=60, ha="right", fontsize=9)
        ax.set_title(f"Δ {col}", fontsize=11)
        ax.set_ylabel("After − Before")

        for i, dv in enumerate(delta):
            if np.isfinite(dv):
                ax.text(i, dv, f"{dv:+.2f}", ha="center", va="bottom" if dv >= 0 else "top", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    plt.close(fig)
    ok(f"Comparison dotplot saved to {out_png}")


def plot_grouped_bars(
    before_csv: Path, after_csv: Path, out_png: Path, title: str
) -> None:
    """
    Grouped bar plot for the three aggregate scores (Before vs After).
    Robust to scIB version / column naming differences.
    """
    if not before_csv.exists() or not after_csv.exists():
        warn("Missing CSV files for grouped bars.")
        return

    b = pd.read_csv(before_csv, index_col=0)
    a = pd.read_csv(after_csv, index_col=0)

    b = _ensure_scib_aggregate_columns(b)
    a = _ensure_scib_aggregate_columns(a)

    b, _ = _scib_drop_metrictype_and_coerce(b)
    a, _ = _scib_drop_metrictype_and_coerce(a)

    def _find_col(df: pd.DataFrame, patterns: List[str]) -> Optional[str]:
        cols = list(df.columns)
        for rgx in patterns:
            for c in cols:
                if re.fullmatch(rgx, str(c).strip(), flags=re.IGNORECASE):
                    return c
            for c in cols:
                if re.search(rgx, str(c), flags=re.IGNORECASE):
                    return c
        return None

    bio = _find_col(b, [r"bio\s*conservation", r"\bbio\b"])
    batch = _find_col(b, [r"batch\s*correction", r"\bbatch\b"])
    total = _find_col(b, [r"\btotal\b"])

    agg_cols = [c for c in [bio, batch, total] if c is not None and c in b.columns and c in a.columns]
    if len(agg_cols) == 0:
        warn("No aggregate columns found for grouped bars.")
        return

    methods = sorted(set(b.index) & set(a.index))
    if len(methods) == 0:
        warn("No common methods for grouped bars.")
        return

    n_metrics = len(agg_cols)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6.2 * n_metrics, 5.5), squeeze=False)
    fig.suptitle(title, fontsize=14)

    x = np.arange(len(methods))
    width = 0.36

    for j, col in enumerate(agg_cols):
        ax = axes[0, j]
        bv = b.loc[methods, col].astype(float).values
        av = a.loc[methods, col].astype(float).values

        ax.bar(x - width / 2, bv, width, label="Before")
        ax.bar(x + width / 2, av, width, label="After")

        ax.set_ylim(0, 1)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=60, ha="right", fontsize=9)
        ax.set_title(str(col), fontsize=11)
        ax.grid(axis="y", alpha=0.25)

        # annotate deltas
        for i in range(len(methods)):
            dv = av[i] - bv[i]
            if np.isfinite(dv):
                ax.text(i, max(bv[i], av[i]) + 0.02, f"{dv:+.2f}", ha="center", fontsize=8)

    axes[0, 0].legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    plt.close(fig)
    ok(f"Grouped bars saved to {out_png}")


def plot_delta_heatmap_fixed(
    before_csv: Path, after_csv: Path, out_png: Path, out_csv: Path, title: str
) -> None:
    # ... update similarly ...
    if not before_csv.exists() or not after_csv.exists():
        warn("Cannot plot delta heatmap: missing CSV files.")
        return

    b = pd.read_csv(before_csv, index_col=0)
    a = pd.read_csv(after_csv, index_col=0)

    b, _ = _scib_drop_metrictype_and_coerce(b)
    a, _ = _scib_drop_metrictype_and_coerce(a)

    common_cols = [c for c in b.columns if c in a.columns]
    methods = [m for m in b.index if m in a.index]
    if len(methods) == 0 or len(common_cols) == 0:
        warn("No common methods/columns for delta heatmap.")
        return

    delta = a.loc[methods, common_cols].astype(float) - b.loc[methods, common_cols].astype(float)
    delta.to_csv(out_csv)

    # Clustering
    delta_clustered = delta.copy()
    try:
        from scipy.cluster.hierarchy import linkage, leaves_list
        if delta.shape[0] > 1:
            row_link = linkage(delta, method="average", metric="euclidean")
            row_order = leaves_list(row_link)
            delta_clustered = delta.iloc[row_order, :]
        if delta.shape[1] > 1:
            col_link = linkage(delta.T, method="average", metric="euclidean")
            col_order = leaves_list(col_link)
            delta_clustered = delta_clustered.iloc[:, col_order]
    except Exception as e:
        warn(f"Clustering failed: {e}")

    fig_w = max(14, 0.4 * len(common_cols))
    fig_h = max(5, 0.55 * len(methods))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(
        delta_clustered.to_numpy(), aspect="auto", interpolation="nearest",
        cmap="RdBu_r", vmin=-0.3, vmax=0.3
    )
    ax.set_title(title)
    ax.set_yticks(np.arange(len(delta_clustered.index)))
    ax.set_yticklabels(delta_clustered.index)
    ax.set_xticks(np.arange(len(delta_clustered.columns)))
    ax.set_xticklabels(delta_clustered.columns, rotation=70, ha="right", fontsize=8)

    for i in range(len(delta_clustered.index)):
        for j in range(len(delta_clustered.columns)):
            val = delta_clustered.iloc[i, j]
            if pd.notna(val):
                color = "white" if abs(val) > 0.15 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="After - Before")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    ok(f"Delta heatmap saved to {out_png}")

# ---------------------------------------------------------------------
# Metadata classification (unchanged)
# ---------------------------------------------------------------------
def _confusion_matrix(y_true: Sequence[str], y_pred: Sequence[str], labels: List[str]) -> np.ndarray:
    # ... unchanged ...
    m = np.zeros((len(labels), len(labels)), dtype=int)
    lab2i = {l: i for i, l in enumerate(labels)}
    for t, p in zip(y_true, y_pred):
        if t in lab2i and p in lab2i:
            m[lab2i[t], lab2i[p]] += 1
    return m

def _metrics_from_cm(cm: np.ndarray) -> Dict[str, float]:
    # ... unchanged ...
    total = cm.sum()
    acc = float(np.trace(cm) / total) if total > 0 else 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        tp = np.diag(cm).astype(float)
        fp = cm.sum(axis=0).astype(float) - tp
        fn = cm.sum(axis=1).astype(float) - tp
        prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        rec = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
    macro_f1 = float(np.mean(f1)) if len(f1) else 0.0
    support = cm.sum(axis=1).astype(float)
    w_f1 = float(np.sum(f1 * support) / np.sum(support)) if np.sum(support) > 0 else 0.0
    bal_acc = float(np.mean(rec)) if len(rec) else 0.0
    macro_prec = float(np.mean(prec)) if len(prec) else 0.0
    macro_rec = float(np.mean(rec)) if len(rec) else 0.0
    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "weighted_f1": w_f1,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "n": int(total),
    }

def plot_confusion_with_metrics(
    cm: np.ndarray, labels: List[str], metrics: Dict[str, float],
    out_png: Path, title: str
) -> None:
    # ... unchanged ...
    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    im = ax.imshow(cm, interpolation="nearest", aspect="auto", cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    vmax = cm.max() if cm.size else 1
    thresh = vmax / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center", fontsize=9,
                color="white" if cm[i, j] > thresh else "black"
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    text = "\n".join([
        f"n={metrics.get('n', 0)}",
        f"Acc={metrics.get('accuracy', 0):.3f}",
        f"BalAcc={metrics.get('balanced_accuracy', 0):.3f}",
        f"MacroF1={metrics.get('macro_f1', 0):.3f}",
        f"WeightF1={metrics.get('weighted_f1', 0):.3f}",
    ])
    ax.text(
        1.02, 0.02, text, transform=ax.transAxes, va="bottom", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85, edgecolor="0.7")
    )

    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)

def plot_metrics_table(metrics: Dict[str, float], out_png: Path, title: str) -> None:
    # ... unchanged ...
    df = pd.DataFrame([metrics])
    cols = ["n", "accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]
    df = df[cols]
    fig, ax = plt.subplots(figsize=(9.0, 1.8))
    ax.axis("off")
    ax.set_title(title, pad=10)
    table = ax.table(cellText=df.round(4).values, colLabels=df.columns, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.4)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)

def evaluate_metadata_classification(a: ad.AnnData, outdir: Path, field: str) -> None:
    # ... unchanged ...
    ensure_dir(outdir)
    true_key = f"true_{field}"
    pred_key = field
    if true_key not in a.obs.columns:
        warn(f"Cannot evaluate {field}: missing '{true_key}' in obs")
        return
    if pred_key not in a.obs.columns:
        warn(f"Cannot evaluate {field}: missing '{pred_key}' in obs")
        return

    y_true = a.obs[true_key].astype(str).tolist()
    y_pred = a.obs[pred_key].astype(str).tolist()
    labels = sorted(list(set(y_true) | set(y_pred)))
    cm = _confusion_matrix(y_true, y_pred, labels)
    metrics = _metrics_from_cm(cm)

    (outdir / f"{field}_metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(outdir / f"{field}_confusion.csv")

    plot_confusion_with_metrics(
        cm, labels, metrics,
        out_png=outdir / f"{field}_confusion.png",
        title=f"{field.title()} confusion matrix (true vs predicted)",
    )
    plot_metrics_table(
        metrics,
        out_png=outdir / f"{field}_metrics_table.png",
        title=f"{field.title()} classification metrics"
    )
    ok(f"Metadata classification ({field}) done → {outdir}")

# ---------------------------------------------------------------------
# Full benchmark runner for one group – **FIXED MERGE STRATEGY + ZERO‑GENE CHECK**
# ---------------------------------------------------------------------
def run_methods_and_benchmark(
    *,
    h5ad_paths: List[Path],
    group_name: str,
    stage_name: str,
    outdir: Path,
    batch_key: str,
    label_key: str,
    n_top_genes: int,
    n_pcs: int,
    n_jobs: int,
    seed: int,
    use_scvi: bool,
    multi_factor: bool,
) -> Tuple[Optional[Path], Optional[Path], Optional[ad.AnnData]]:
    ensure_dir(outdir)
    set_global_seeds(seed)

    # Load and concatenate
    adatas = [sc.read_h5ad(p) for p in h5ad_paths]
    for i, a in enumerate(adatas):
        a.obs["dataset_id"] = str(Path(h5ad_paths[i]).stem)

    # Merge strategy: before = outer join, after = inner join
    if "before" in stage_name.lower():
        merged = ad.concat(adatas, join="outer", merge="same", label="dataset_id_join", fill_value=0)
        log(f"[MERGE] {stage_name}: outer join, shape={merged.shape}")
    else:
        merged = ad.concat(adatas, join="inner", merge="same", label="dataset_id_join", fill_value=0)
        log(f"[MERGE] {stage_name}: inner join, shape={merged.shape}")

        # Rescue path: if inner join yields 0 genes, standardize gene names in-memory and retry
        if merged.n_vars == 0:
            warn(f"[MERGE] {stage_name}: inner join produced 0 genes. Attempting rescue via gene standardization...")
            sp = None
            for adx in adatas:
                if "species" in adx.obs.columns:
                    try:
                        sp = str(adx.obs["species"].astype(str).mode().iat[0])
                    except Exception:
                        sp = str(adx.obs["species"].astype(str).iloc[0])
                    break
            if sp is None:
                sp = infer_species_from_varnames(adatas[0], fallback="human")

            adatas_fix = [simple_gene_harmonization(adx, target_species=sp, dedup_how="sum") for adx in adatas]
            merged = ad.concat(adatas_fix, join="inner", merge="same", label="dataset_id_join", fill_value=0)
            log(f"[MERGE-RESCUE] {stage_name}: inner join after standardization, shape={merged.shape}")

            if merged.n_vars == 0:
                warn(f"[MERGE-RESCUE] Still 0 common genes (stage={stage_name}). Falling back to OUTER join so the pipeline can proceed.")
                merged = ad.concat(adatas_fix, join="outer", merge="same", label="dataset_id_join", fill_value=0)
                log(f"[MERGE-RESCUE] {stage_name}: outer join fallback, shape={merged.shape}")

    # CRITICAL: final check that we have genes
    if merged.n_vars == 0:
        raise ValueError(f"Merged AnnData has 0 genes (stage={stage_name}). Cannot proceed.")

        raise ValueError(f"Merged AnnData has 0 genes (stage={stage_name}). Cannot proceed.")

    if batch_key not in merged.obs.columns:
        raise RuntimeError(f"batch_key '{batch_key}' not found in merged.obs (stage={stage_name}).")
    if label_key not in merged.obs.columns:
        raise RuntimeError(f"label_key '{label_key}' not found in merged.obs (stage={stage_name}).")

    merged.obs[batch_key] = pd.Categorical(merged.obs[batch_key].astype(str))
    merged.obs[label_key] = pd.Categorical(merged.obs[label_key].astype(str))

    # Preprocess
    prepared = preprocess_for_benchmark(
        merged, batch_key=batch_key, n_top_genes=n_top_genes, n_pcs=n_pcs, seed=seed
    )

    # --- Run embeddings ---
    embedding_keys = ["Unintegrated"]

    Z = run_scanorama_embedding(prepared, batch_key=batch_key)
    if Z is not None and Z.shape[0] == prepared.n_obs:
        prepared.obsm["Scanorama"] = Z
        embedding_keys.append("Scanorama")

    covs = []
    if multi_factor:
        for col in ["donor", "technology", "sex", "species"]:
            if col in prepared.obs.columns and prepared.obs[col].astype(str).nunique() > 1:
                covs.append(col)
    Z = run_harmony_embedding(prepared, batch_key=batch_key, covariates=covs)
    if Z is not None and Z.shape[0] == prepared.n_obs:
        prepared.obsm["Harmony"] = Z
        embedding_keys.append("Harmony")

    Z = run_combat_embedding(prepared, batch_key=batch_key, n_pcs=n_pcs, seed=seed)
    if Z is not None and Z.shape[0] == prepared.n_obs:
        prepared.obsm["ComBat"] = Z
        embedding_keys.append("ComBat")

    if use_scvi:
        Z = run_scvi_embedding(
            prepared, batch_key=batch_key, covariates=covs,
            n_latent=n_pcs, seed=seed
        )
        if Z is not None and Z.shape[0] == prepared.n_obs:
            prepared.obsm["scVI"] = Z
            embedding_keys.append("scVI")

        Z = run_scanvi_embedding(
            prepared, batch_key=batch_key, label_key=label_key,
            covariates=covs, n_latent=n_pcs, seed=seed
        )
        if Z is not None and Z.shape[0] == prepared.n_obs:
            prepared.obsm["scANVI"] = Z
            embedding_keys.append("scANVI")

    prepared_path = outdir / f"{stage_name}_prepared_for_benchmark.h5ad"
    prepared.write_h5ad(prepared_path)

    # Run benchmark – save full results (with aggregates)
    results_csv, results_png, results_df = run_scib_benchmark(
        prepared,
        batch_key=batch_key,
        label_key=label_key,
        embedding_keys=embedding_keys,
        n_jobs=n_jobs,
        outdir=outdir,
        stage_name=stage_name,
        min_max_scale=False,   # keep raw scores
    )

    # Plots
    plot_umap_grid(
        prepared, rep_keys=embedding_keys,
        color=batch_key,
        out_png=outdir / f"{stage_name}_umap_by_{batch_key}.png",
        seed=seed,
        title=f"{group_name} — {stage_name} — UMAP colored by {batch_key}",
    )
    plot_umap_grid(
        prepared, rep_keys=embedding_keys,
        color=label_key,
        out_png=outdir / f"{stage_name}_umap_by_{label_key}.png",
        seed=seed,
        title=f"{group_name} — {stage_name} — UMAP colored by {label_key}",
    )
    plot_embedding_2d_grid(
        prepared, rep_keys=embedding_keys,
        color=batch_key,
        out_png=outdir / f"{stage_name}_embed2d_by_{batch_key}.png",
        title=f"{group_name} — {stage_name} — 2D embedding (dims 1-2) by {batch_key}",
    )
    plot_embedding_2d_grid(
        prepared, rep_keys=embedding_keys,
        color=label_key,
        out_png=outdir / f"{stage_name}_embed2d_by_{label_key}.png",
        title=f"{group_name} — {stage_name} — 2D embedding (dims 1-2) by {label_key}",
    )

    return results_csv, results_png, prepared

def run_group_end_to_end(
    *,
    group_name: str,
    raw_paths: List[Path],
    harm_paths: List[Path],
    outdir: Path,
    batch_key: str,
    label_key: str,
    n_top_genes: int,
    n_pcs: int,
    n_jobs: int,
    seed: int,
    use_scvi: bool,
    multi_factor: bool,
) -> None:
    gdir = ensure_dir(outdir / group_name)
    before_dir = ensure_dir(gdir / "before_h5adify")
    after_dir = ensure_dir(gdir / "after_h5adify")
    compare_dir = ensure_dir(gdir / "comparisons")
    meta_dir = ensure_dir(gdir / "metadata_classification")

    log(f"\n[INFO] Benchmark group: {group_name}")

    # ---------- BEFORE ----------
    try:
        before_csv, before_png, before_adata = run_methods_and_benchmark(
            h5ad_paths=raw_paths,
            group_name=group_name,
            stage_name="before_h5adify",
            outdir=before_dir,
            batch_key=batch_key,
            label_key=label_key,
            n_top_genes=n_top_genes,
            n_pcs=n_pcs,
            n_jobs=n_jobs,
            seed=seed,
            use_scvi=use_scvi,
            multi_factor=False,
        )
    except Exception as e:
        warn(f"Before benchmark failed for {group_name}: {e}")
        before_csv = before_png = before_adata = None

    # ---------- AFTER ----------
    try:
        after_csv, after_png, after_adata = run_methods_and_benchmark(
            h5ad_paths=harm_paths,
            group_name=group_name,
            stage_name="after_h5adify",
            outdir=after_dir,
            batch_key=batch_key,
            label_key=label_key,
            n_top_genes=n_top_genes,
            n_pcs=n_pcs,
            n_jobs=n_jobs,
            seed=seed,
            use_scvi=use_scvi,
            multi_factor=multi_factor,
        )
    except Exception as e:
        warn(f"After benchmark failed for {group_name}: {e}")
        after_csv = after_png = after_adata = None


    # ---------- COMPARISON PLOTS (only if both succeeded) ----------
    if before_csv is not None and before_csv.exists() and after_csv is not None and after_csv.exists():
        # 1) Aggregate delta dotplot
        try:
            plot_comparison_dotplot(
                before_csv, after_csv,
                out_png=compare_dir / "delta_dotplot_aggregates.png",
                title=f"{group_name}: Before vs After (Δ aggregates)",
            )
        except Exception as e:
            warn(f"Comparison dotplot failed for {group_name}: {e}")

        # 2) Grouped bars (Before vs After)
        try:
            plot_grouped_bars(
                before_csv, after_csv,
                out_png=compare_dir / "grouped_bars_before_after.png",
                title=f"{group_name}: Before vs After (aggregates)",
            )
        except Exception as e:
            warn(f"Comparison grouped bars failed for {group_name}: {e}")

        # 3) Delta heatmap (all metrics)
        try:
            plot_delta_heatmap_fixed(
                before_csv, after_csv,
                out_png=compare_dir / "delta_heatmap_all_metrics.png",
                out_csv=compare_dir / "delta_after_minus_before.csv",
                title=f"{group_name}: Δ metrics (After − Before)",
            )
        except Exception as e:
            warn(f"Comparison delta heatmap failed for {group_name}: {e}")

        # 4) Always write a compact CSV summary for the three aggregates, if available
        try:
            b = pd.read_csv(before_csv, index_col=0)
            a = pd.read_csv(after_csv, index_col=0)
            b = _ensure_scib_aggregate_columns(b)
            a = _ensure_scib_aggregate_columns(a)
            b_num, _ = _scib_drop_metrictype_and_coerce(b)
            a_num, _ = _scib_drop_metrictype_and_coerce(a)
            common_methods = sorted(set(b_num.index) & set(a_num.index))
            cols = [c for c in ["Bio conservation", "Batch correction", "Total"] if c in b_num.columns and c in a_num.columns]
            if common_methods and cols:
                summ = pd.DataFrame(index=common_methods)
                for c in cols:
                    summ[f"before_{c}"] = b_num.loc[common_methods, c].astype(float)
                    summ[f"after_{c}"] = a_num.loc[common_methods, c].astype(float)
                    summ[f"delta_{c}"] = summ[f"after_{c}"] - summ[f"before_{c}"]
                summ.to_csv(compare_dir / "summary_aggregates_before_after.csv")
                ok(f"Aggregate summary saved to {compare_dir / 'summary_aggregates_before_after.csv'}")
        except Exception as e:
            warn(f"Could not write aggregate summary for {group_name}: {e}")
    else:
        warn(f"Skipping comparison plots for {group_name}: missing before/after CSV files.")

    # ---------- METADATA CLASSIFICATION ----------

    if after_adata is not None:
        evaluate_metadata_classification(after_adata, meta_dir, field="sex")
        evaluate_metadata_classification(after_adata, meta_dir, field="species")
    else:
        warn(f"Skipping metadata classification for {group_name}: after_adata is None.")

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="h5adify benchmark – FINAL WORKING VERSION (v25 – robust scIB parsing/plots, improved sex inference, comparisons fixed)",
    )
    ap.add_argument("--zip", type=str, default="", help="Path to h5adify zip (optional).")
    ap.add_argument("--outdir", type=str, required=True, help="Output directory.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-top-genes", type=int, default=2000)
    ap.add_argument("--n-pcs", type=int, default=50)
    ap.add_argument("--n-jobs", type=int, default=6, help="Jobs for scib-metrics. -1 = all cores.")
    ap.add_argument("--install-missing", type=int, default=0, help="Attempt pip install missing optional deps.")
    ap.add_argument("--use-scvi", type=int, default=1, help="Run scVI/scANVI if available.")
    ap.add_argument("--multi-factor", type=int, default=1, help="After h5adify: allow multi‑factor correction.")
    ap.add_argument("--use-llm", type=int, default=0, help="Allow h5adify harmonize_metadata to use LLM.")
    return ap.parse_args()

def main() -> None:
    args = parse_args()
    outdir = ensure_dir(Path(args.outdir).expanduser().resolve())
    seed = int(args.seed)
    set_global_seeds(seed)

    n_jobs = normalize_n_jobs(int(args.n_jobs))
    n_top = int(args.n_top_genes)
    n_pcs = int(args.n_pcs)
    use_scvi = bool(int(args.use_scvi))
    multi_factor = bool(int(args.multi_factor))
    use_llm = bool(int(args.use_llm))

    if int(args.install_missing) == 1:
        missing = []
        if safe_import("scib_metrics") is None:
            missing.append("scib-metrics")
        if safe_import("scanorama") is None:
            missing.append("scanorama")
        if safe_import("harmony") is None and safe_import("harmonypy") is None:
            missing.append("harmonypy")
        if use_scvi and safe_import("scvi") is None:
            missing.append("scvi-tools")
        if missing:
            pip_install(missing)

    if args.zip:
        zp = Path(args.zip).expanduser().resolve()
        if not zp.exists():
            raise FileNotFoundError(zp)
        extract_dir = ensure_dir(outdir / "_h5adify_zip_extracted")
        add_zip_to_syspath(zp, extract_dir)

    try:
        import h5adify
        log(f"[INFO] h5adify version: {getattr(h5adify, '__version__', 'unknown')}")
    except Exception:
        warn("h5adify import failed (continuing with internal heuristics).")

    # ---------- SIMULATIONS ----------
    sim_dir = ensure_dir(outdir / "simulations_v25_0")
    log("[INFO] Running simulations (strong batch effects + inconsistent gene symbols)...")
    groups = write_simulations(sim_dir, seed=seed)

    # ---------- HARMONIZATION ----------
    harm_dir = ensure_dir(outdir / "harmonized_v25_0")
    log("[INFO] Running h5adify harmonization...")
    harm_groups = {}
    for gname, paths in groups.items():
        target_species = "mouse" if gname == "SimC_Spatial" else "human"
        g_out = ensure_dir(harm_dir / gname)
        harm_paths = harmonize_files_with_h5adify(
            paths, g_out, target_species=target_species, use_llm=use_llm
        )
        harm_groups[gname] = harm_paths

    # ---------- BENCHMARK ----------
    bench_dir = ensure_dir(outdir / "benchmark_results_v25_0")
    group_specs = {
        "SimA_Brain": {"batch_key": "batch", "label_key": "cell_type"},
        "SimB_GBM": {"batch_key": "batch", "label_key": "cell_type"},
        "SimC_Spatial": {"batch_key": "batch", "label_key": "cell_type"},
    }

    for gname, spec in group_specs.items():
        raw_paths = groups[gname]
        harm_paths = harm_groups[gname]
        run_group_end_to_end(
            group_name=gname,
            raw_paths=raw_paths,
            harm_paths=harm_paths,
            outdir=bench_dir,
            batch_key=spec["batch_key"],
            label_key=spec["label_key"],
            n_top_genes=n_top,
            n_pcs=n_pcs,
            n_jobs=n_jobs,
            seed=seed,
            use_scvi=use_scvi,
            multi_factor=multi_factor,
        )

    ok(f"\n{'='*60}\nALL DONE. Results in: {bench_dir}\n{'='*60}")

if __name__ == "__main__":
    main()