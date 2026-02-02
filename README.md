# Initial installation

```
bash scripts/install.sh /absolute/path/to/h5adify_v0.0.7_final.zip
```

For the analysis of the pdf and html documents, this should be installed.

```
python -m pip install -U pyyaml requests tqdm beautifulsoup4 lxml pymupdf
```

The `normalization.yaml` included in `/configs` makes the scoring robust to synonyms / formatting (e.g., “Homo sapiens” vs “human”, “10x Visium” vs “Visium”).

The `configs/field_weights.yaml` lets you compute a **single composite score (weighted average)**.

# Part 1 - DOI20 metadata harmonization benchmark

This part evaluates how accurately metadata fields can be identified and standardized from single-cell datasets, with and without LLM assistance.

## Part 1a - Dataset-only harmonization 🧪  
*download → gold → run models → score*

### 1.1. Dataset retrieval and preparation

This step maps a predefined list of DOIs (the DOI20 benchmark set) to publicly available single-cell datasets hosted in the CellxGene Census, downloads the corresponding data, and prepares standardized `.h5ad` files for downstream benchmarking.

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


**Download the datasets**

```
source .venv/bin/activate
python scripts/part1_download_doi20.py --doi-config configs/doi20.yaml
```

### 1.2. Gold standard construction

This step constructs the gold-standard annotations used to evaluate metadata
harmonization performance in Part 1.

Rather than creating new annotations, the gold standard defines, for each dataset,
which existing metadata fields and labels should be considered the correct targets
for harmonization.

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

**Run the gold standard**

```
python scripts/part1_make_gold.py --manifest data/doi20/manifest.json --out gold/doi20_gold.json --use-small
```

### 1.3. Metadata harmonization benchmark run

This step executes metadata harmonization on the DOI20 datasets using either LLM-assisted or deterministic methods and records structured predictions for subsequent evaluation. It isolates harmonization behavior from evaluation, allowing prompt, model, and method variations to be explored independently.

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

**Run llm benchmark**

```
python scripts/part1_run_benchmark.py --use-llm --prompt-name metadata_harmonize_v1_default --save-h5ad --prefer-small
```

**Run deterministic (no benchmark)**

```
python scripts/part1_run_benchmark.py --prompt-name metadata_harmonize_v1_default --prefer-small
```

### 1.4. Scoring and evaluation

This step evaluates metadata harmonization predictions against the gold standard
defined in Part 1 and computes quantitative performance metrics.

For each model, prompt, and dataset, the script compares predicted metadata
mappings to the gold reference and reports:

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

**Run scores of part 1**

```
python scripts/part1_score.py --gold gold/doi20_gold.json --results results/part1 --outdir results/part1_scores
```

## Part 1b — Paper-aware harmonization (optional) 📄

In addition to dataset-only harmonization, Part 1 includes an optional paper-aware
setting in which LLMs are provided with unstructured text from the associated
publication (PDF or HTML) as additional context.

This extension evaluates whether access to manuscript text improves metadata
harmonization accuracy, while keeping all other aspects of the benchmark unchanged
(datasets, gold standard, and scoring metrics).

### Manuscript retrieval and preprocessing

Associated manuscripts are fetched using the DOI list and converted into
plain text suitable for LLM input:

```
bash
python scripts/part1_fetch_manuscripts.py --doi-config configs/doi20.yaml --outdir papers
python scripts/part1_extract_manuscript_text.py --papers-dir papers
```

**Run example with qwen2.5:3b**

```
python scripts/part1_run_annotation_paperaware.py \
  --doi-config configs/doi20.yaml \
  --papers-dir papers \
  --outdir results_part1_paperaware \
  --ollama-model qwen2.5:3b \
  --verify
```

# Part 2 - Avatar vs TextGrad prompt optimization (metadata mapping prompt)

```
python scripts/part2_optimize_prompts.py --method avatar --opt-model qwen2.5:3b
python scripts/part2_optimize_prompts.py --method textgrad --opt-model qwen2.5:3b
```

Build JSONL training set (paper-aware)

```
python scripts/part2_build_training_jsonl_paperaware.py \
  --doi-config configs/doi20.yaml \
  --papers-dir papers \
  --gold-json gold/doi20_gold.json \
  --out-jsonl train_paperaware.jsonl
```

### The re-run Part 1 with each optimized prompt

```
python scripts/part1_run_benchmark.py --use-llm --prompt-name metadata_harmonize_v1_avatar_opt --save-h5ad --prefer-small
python scripts/part1_run_benchmark.py --use-llm --prompt-name metadata_harmonize_v1_textgrad_opt --save-h5ad --prefer-small
python scripts/part1_score.py --outdir results/part1_scores
```

Run Avatar-style optimization (paper-aware)

```
python scripts/part2_optimize_avatar_paperaware.py \
  --train-jsonl train_paperaware.jsonl \
  --ollama-model qwen2.5:3b
```

Run TextGrad-style optimization (paper-aware)

```
python scripts/part2_optimize_textgrad_paperaware.py \
  --train-jsonl train_paperaware.jsonl \
  --ollama-model qwen2.5:3b
```


## Part 3 — Simulations (5–10 synthetic h5ad) + evaluation

```
python scripts/part3_eval_simulations.py --use-llm --prompt-name metadata_harmonize_v1_default
```

## Part 4 — GBM analysis: GBM-Space + extra GBM datasets → merge → integration → sex markers

```
python scripts/part4_gbm_pipeline.py --config configs/gbm.yaml --models configs/models.yaml
```

Probably here there could be used other additional ways to deal with the batch effects as analysis using [Harmony](https://github.com/lilab-bcb/harmony-pytorch) or [scVI](https://docs.scvi-tools.org/en/stable/tutorials/notebooks/quick_start/api_overview.html)


