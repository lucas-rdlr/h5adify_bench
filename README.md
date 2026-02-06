# Benchmarking LLM-Assisted Metadata Harmonization in Omics Data

## Abstract

```diff
- Just an idea of quick abstract and possible article name. Probably include the reference to the main repository with the actual *h5adify* module.
```

This repository contains the **analysis pipelines, benchmarking frameworks, and reproducibility scripts** for the study:

> **“Evaluation of Large Language Models for Metadata and Feature Harmonization in Single-Cell and Spatial Transcriptomics.”**

The rapid expansion of public omics repositories is increasingly hindered by:

- **Inconsistent metadata annotations**  
  (e.g., non-standardized labels for *sex*, *species*, and *technology*)
- **Divergent gene naming conventions**

This study systematically benchmarks the performance of the **`h5adify` Python library** in resolving these inconsistencies across real and synthetic datasets.

## Harmonization Strategies Evaluated

### 1. Deterministic Harmonization
Baseline **rule-based mapping** using predefined synonym dictionaries and normalization rules.

### 2. LLM-Assisted Annotation
Utilizes **Large Language Models (LLMs)** to infer structured metadata from unstructured `.obs` fields.

### 3. Paper-Aware LLM Annotation
Enhances LLM inference by **retrieving and parsing the full-text associated publication** (PDF or HTML) via DOI.

## Benchmark Scope

- **Real-world datasets (DOI20)**  
  Metadata harmonization benchmark against a manually curated gold standard.

- **Synthetic stress-test datasets**  
  Designed to quantify **hallucination rates** and robustness.

- **Biological application (Glioblastoma, GBM)**  
  Harmonization and integration of GBM datasets to identify **sex-specific markers**.

## Reproducibility Statement

```diff
- Include information/instructions for reproducibility
```

All experiments, data retrieval steps, and scoring metrics detailed in the associated manuscript can be **fully reproduced** using the scripts provided in this repository.

### Benchmark Parts

- **Part 1 (DOI20)** – Metadata harmonization benchmark  
- **Part 2 (Simulation)** – Hallucination and robustness tests  
- **Part 3 (Application)** – GBM integration and biological analysis

## Installation & Setup

```diff
- Either include full installation details or link the other GitHub with the module and the installation instructions.
```

### 1. Core Module Installation

```bash
bash scripts/install.sh /absolute/path/to/h5adify_v0.0.7_final.zip
```

### 2. Analysis Dependencies

```bash
python -m pip install -U pyyaml requests tqdm beautifulsoup4 lxml pymupdf
```

### 3. Configuration

Configuration files are located in `/configs`:

- `normalization.yaml`: makes the scoring robust to synonyms / formatting (e.g., “Homo sapiens” vs “human”, “10x Visium” vs “Visium”).
- `field_weights.yaml`: lets you compute a **single composite score (weighted average)**.
- `field_map.yaml`: because prediction JSONs often differ by method, we map each method’s schema to the canonical fields.

## Part 1 - DOI20 metadata harmonization benchmark

This part evaluates how accurately metadata fields can be identified and standardized from single-cell datasets, with and without LLM assistance.

### Part 1a - Dataset-only harmonization (deterministic) 🧪  
*download → gold → run models → score*

#### 1.1. Dataset retrieval and preparation

This step maps a predefined list of DOIs (the DOI20 benchmark set) to publicly available single-cell datasets hosted in the CellxGene Census, downloads the corresponding data, and prepares standardized `.h5ad` files for downstream benchmarking.

<details><summary>Show more</summary>

All download settings, including the output directory, are defined in the configuration file (e.g. `configs/doi20.yaml`).

For each DOI, the script:
1. Searches the CellxGene Census dataset registry for matching datasets
   (exact match on `collection_doi`, with a citation-based fallback).
2. If multiple datasets match a DOI, selects the largest dataset by cell count.
3. Downloads the original source `.h5ad` file as deposited in the Census.
4. Optionally creates a reduced “small” version by subsampling cells and genes
   (for faster benchmarking and LLM evaluation).
5. Records all results in a manifest file, including missing DOIs and errors.

**Output structure**

Datasets are downloaded to the directory specified by `download.out_dir`
in `configs/doi20.yaml`, which by default has the following structure:

```
data/
├── doi20/
│   ├── dataset1/ 
│   │    ├── dataset1_id.source.h5ad
│   │    └── dataset1_id.small.h5ad # optional
│   └── ...
└── manifest.json
```

**Note:** This step does not involve any LLMs. It ensures that all downstream
comparisons are grounded in the same curated, reproducible datasets.

</details>

**Download the datasets**

```bash
python scripts/part1_download_doi20.py --doi-config configs/doi20.yaml
```

#### 1.2. Gold standard construction

This step constructs the gold-standard annotations used to evaluate metadata harmonization performance in Part 1. Rather than creating new annotations, the gold standard defines, for each dataset, which existing metadata fields and labels should be considered the correct targets for harmonization.

<details><summary>Show more</summary>

For each dataset listed in the DOI20 manifest, the script:

1. Loads the dataset metadata (`.obs`) from the corresponding `.h5ad` file
   (optionally using the reduced “small” version).
2. Identifies candidate `.obs` columns for a predefined set of semantic fields
   (e.g. batch, sample, donor, sex) using curated synonym lists and fuzzy matching.
3. Selects a single “best” column per field, which serves as the gold reference
   during evaluation.
4. Derives canonical dataset-level labels for species and technology based on
   Census metadata and controlled vocabularies.
5. Records all results in a structured gold JSON file.

**Output**

The gold standard is written to: `gold/doi20_gold.json`

Each entry specifies:
- The expected `.obs` column for each semantic field (or `null` if absent)
- Canonical species and technology labels
- Dataset-level metadata for traceability

This gold file defines the reference against which both LLM-based and deterministic
metadata harmonization methods are evaluated in subsequent steps.

</details>

**Run the gold standard**

```bash
python scripts/part1_make_gold.py --manifest data/doi20/manifest.json --out gold/doi20_gold.json --use-small
```

#### 1.3. Metadata harmonization benchmark run

This step executes metadata harmonization on the DOI20 datasets using either LLM-assisted or deterministic methods and records structured predictions for subsequent evaluation. It isolates harmonization behavior from evaluation, allowing prompt, model, and method variations to be explored independently.

<details><summary>Show more</summary>

For each dataset and each configured model, the script:
1. Loads the input `.h5ad` file (optionally using a reduced “small” version).
2. Runs the metadata harmonization pipeline on a predefined set of semantic fields
   (e.g. batch, sample, donor, sex, species, technology).
3. Optionally uses an LLM to assist with column selection and label normalization,
   based on a specified prompt template.
4. Writes a structured prediction record (`pred.json`) capturing the harmonized
   outputs, execution metadata, and runtime diagnostics.
5. Optionally saves a harmonized `.h5ad` file with standardized metadata columns
   (prefixed with `h5adify_`) for qualitative inspection.

Both LLM-based and non-LLM (deterministic) runs use the same datasets and gold
definitions, enabling direct comparison.

**Output structure**

Results are written to a model-organized directory structure:

```
results/part1/
├── <model_name>/
│   ├── <doi_slug>/<dataset_id>/
│   │   ├── pred.json
│   │   └── harmonized.h5ad # optional
│   └── runlog.json
└── ...
```

The outputs of this step are evaluated against the gold standard in the scoring stage of Part 1.

</details>

**Run llm benchmark**

```bash
python scripts/part1_run_benchmark.py --use-llm --prompt-name metadata_harmonize_v1_default --save-h5ad --prefer-small
```

**Run deterministic (no benchmark)**

```bash
python scripts/part1_run_benchmark.py --prompt-name metadata_harmonize_v1_default --prefer-small
```

#### 1.4. Scoring and evaluation

This step evaluates metadata harmonization predictions against the gold standard defined in Part 1 and computes quantitative performance metrics.

<details><summary>Show more</summary>

For each model, prompt, and dataset, the script compares predicted metadata mappings to the gold reference and reports:

- **Field mapping accuracy**: whether the correct `.obs` column was selected for
  each semantic field (batch, sample, donor, domain, sex, species, technology).
- **Hallucination rate**: whether a predicted metadata key does not exist in the
  dataset’s `.obs` table.
- **Completeness**: the fraction of fields for which a prediction was produced.
- **Canonical value accuracy**: correctness of normalized species and technology
  labels.
- **Runtime**: elapsed time per dataset.

**Outputs**

Two evaluation tables are written to the output directory:

```
results/part1_scores/
├── scores_long.csv # one row per dataset
└── scores_summary.csv # aggregated means per model and prompt
```

The long-format table enables detailed error analysis, while the summary table supports direct comparison across models, prompts, and harmonization methods.

</details>

**Run scores of part 1**

```bash
python scripts/part1_score.py --gold gold/doi20_gold.json --results results/part1 --outdir results/part1_scores
```

### Part 1b - Paper-aware harmonization (optional) 📄

In addition to dataset-only harmonization, Part 1 includes an optional paper-aware setting in which LLMs are provided with unstructured text from the associated publication (PDF or HTML) as additional context. This extension evaluates whether access to manuscript text improves metadata harmonization accuracy, while keeping all other aspects of the benchmark unchanged (datasets, gold standard, and scoring metrics).

<details><summary>Show more</summary>

Associated manuscripts are fetched using the DOI list and converted into plain text suitable for LLM input:

```bash
python scripts/part1_fetch_manuscripts.py --doi-config configs/doi20.yaml --outdir papers
python scripts/part1_extract_manuscript_text.py --papers-dir papers
```

**Run example with qwen2.5:3b**

```bash
python scripts/part1_run_annotation_paperaware.py \
  --doi-config configs/doi20.yaml \
  --papers-dir papers \
  --outdir results_part1_paperaware \
  --ollama-model qwen2.5:3b \
  --verify
```

The `scripts/part1_eval_compare_methods.py` produces:

   - per-dataset/per-field metrics,
   - macro/micro aggregates,
   - composite weighted score,
   - LLM-style: EM, token-F1, ROUGE-L-F1 for titles, Slot Error Rate (SER) for set fields,
   - coverage/hallucination proxy metrics.

```bash
python scripts/part1_eval_compare_methods.py \
  --gold configs/doi20_gold_verbose.json \
  --field-map configs/field_map.yaml \
  --normalize configs/normalization.yaml \
  --weights configs/field_weights.yaml \
  --pred deterministic=results_part1/deterministic \
  --pred llm=results_part1/llm_only \
  --pred hybrid=results_part1/hybrid \
  --outdir eval_part1
```

The `part1_eval_evidence_support.py` (faithfulness / “supported by paper”), we implement a simple but effective proxy: does the predicted string appear in the paper text (papers/<doi_slug>/paper_fulltext.txt)? it’s useful to quantify “hallucination vs extractive grounding” in structured metadata.

```bash
python scripts/part1_eval_evidence_support.py \
  --pred-dir results_part1/hybrid \
  --papers-dir papers \
  --out eval_part1/evidence_support_hybrid.json
```
</details>

### Part 1c - Avatar vs TextGrad prompt optimization (metadata mapping prompt)

```bash
python scripts/part2_optimize_prompts.py --method avatar --opt-model qwen2.5:3b
python scripts/part2_optimize_prompts.py --method textgrad --opt-model qwen2.5:3b
```

Build JSONL training set (paper-aware)

```bash
python scripts/part2_build_training_jsonl_paperaware.py \
  --doi-config configs/doi20.yaml \
  --papers-dir papers \
  --gold-json gold/doi20_gold.json \
  --out-jsonl train_paperaware.jsonl
```

#### The re-run Part 1 with each optimized prompt

```bash
python scripts/part1_run_benchmark.py --use-llm --prompt-name metadata_harmonize_v1_avatar_opt --save-h5ad --prefer-small
python scripts/part1_run_benchmark.py --use-llm --prompt-name metadata_harmonize_v1_textgrad_opt --save-h5ad --prefer-small
python scripts/part1_score.py --outdir results/part1_scores
```

Run Avatar-style optimization (paper-aware)

```bash
python scripts/part2_optimize_avatar_paperaware.py \
  --train-jsonl train_paperaware.jsonl \
  --ollama-model qwen2.5:3b
```

Run TextGrad-style optimization (paper-aware)

```bash
python scripts/part2_optimize_textgrad_paperaware.py \
  --train-jsonl train_paperaware.jsonl \
  --ollama-model qwen2.5:3b
```

Compare prompt variants (baseline vs Avatar vs TextGrad) + model grid 

```bash
python scripts/part2_eval_prompt_variants.py \
  --gold configs/doi20_gold_verbose.json \
  --field-map configs/field_map.yaml \
  --normalize configs/normalization.yaml \
  --weights configs/field_weights.yaml \
  --root results_part2 \
  --models qwen2.5_3b llama3_latest mistral_nemo_latest deepseek_r1_8b \
  --variants baseline avatar textgrad \
  --outdir eval_part2
```

### Part 2 - Simulations (5–10 synthetic h5ad) + evaluation

```diff
- TODO
```

```bash
python scripts/part3_eval_simulations.py --use-llm --prompt-name metadata_harmonize_v1_default
```

### Part 3 - GBM analysis: GBM-Space + extra GBM datasets → merge → integration → sex markers

```diff
- TODO
```

```bash
python scripts/part4_gbm_pipeline.py --config configs/gbm.yaml --models configs/models.yaml
```

Probably here there could be used other additional ways to deal with the batch effects as analysis using [Harmony](https://github.com/lilab-bcb/harmony-pytorch) or [scVI](https://docs.scvi-tools.org/en/stable/tutorials/notebooks/quick_start/api_overview.html)

## Project maturity checklist

<details><summary>Show more</summary>

### Core infrastructure
- ✅ Repository structure and scripts stabilized
- ✅ Configuration-driven workflows (`configs/*.yaml`)
- ✅ Deterministic (non-LLM) harmonization baseline
- ✅ LLM-assisted harmonization pipeline
- ✅ Unified prediction format (`pred.json`)
- 🚧 Containerized / Docker-based execution

### Part 1 — DOI20 metadata harmonization benchmark
- ✅ DOI → CellxGene Census dataset resolution
- ✅ Dataset download and manifest generation
- ✅ Small `.h5ad` subsampling for scalability
- ⚠️ Gold-standard construction from `.obs` metadata
- 🔍 Dataset-only harmonization benchmark
- 🔍 Quantitative scoring framework (accuracy, completeness, hallucination)
- 🔍 Cross-model benchmarking (multiple LLMs)
- 🔍 Statistical significance testing across models/prompts

### Part 1b — Paper-aware harmonization
- 🔍 DOI-based manuscript retrieval (PDF / HTML)
- 🔍 Manuscript text extraction and preprocessing
- 🔍 Paper-aware LLM annotation pipeline
- 🔍 Paper-aware vs dataset-only comparison
- 🔍 Evidence-support (faithfulness) proxy metric
- 🔍 Section-level citation grounding
- 🔍 Sensitivity analysis to manuscript noise/length

### Part 1c — Prompt optimization (Avatar / TextGrad)
- 🔍 Avatar-style prompt optimization
- 🔍 TextGrad-style prompt optimization
- 🔍 Paper-aware training JSONL generation
- 🔍 Re-running Part 1 with optimized prompts
- 🔍 Prompt-variant comparison across models
- 🔍 Robustness analysis across random seeds

### Part 2 — Synthetic simulations
- 🔍 Synthetic `.h5ad` simulation pipeline
- 🔍 Hallucination and robustness evaluation
- 🔍 Expanded simulation scenarios (missing/contradictory metadata)
- 🔍 Stress tests under extreme class imbalance

### Part 3 — Biological application (GBM)
- ❌ Multi-dataset GBM harmonization pipeline
- ❌ Dataset merging and integration
- ❌ Sex-specific marker analysis
- ❌ Harmony-based batch correction baseline
- ❌ scVI-based batch correction baseline
- ❌ External biological validation

### Evaluation, documentation, and release
- ❌ Final manuscript-aligned abstract
- ❌ Public documentation polish
- ❌ Contribution guidelines

### Legend

- ✅ stable
- ❌ not implemented
- 🔍 needs review
- 🧪 experimental
- 🚧 in progress
- ⚠️ known limitations
- 📝 documentation needed
- 💡 open question
</details>
