#!/usr/bin/env python3
# scripts/part1_eval_compare_methods.py

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from eval_normalize import load_synonyms, canonize_set, canonize_value, to_str_list
from eval_utils import (
    exact_match, token_f1, rouge_l_f1, set_prf, jaccard, slot_error_rate, coverage
)
from eval_adapters import gold_to_canonical, pred_to_canonical

CANON_FIELDS_SET = ["species", "assay_modalities", "technology"]
CANON_FIELDS_TEXT = ["title", "tissue_scope", "disease_or_context"]

def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))

def load_field_map(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))

def normalize_repositories(repos_any: Any, inv: Dict[str, Dict[str, str]]) -> Dict[str, set]:
    """
    Canonicalize repositories to:
      { repo_type_canon : set(accessions) }
    Accepts list[dict{type,accession}] or dict-like or raw.
    """
    out: Dict[str, set] = {}
    if repos_any is None:
        return out
    if isinstance(repos_any, dict):
        # already dict -> treat keys as types
        for k, v in repos_any.items():
            rt = canonize_value("repositories", str(k), inv)
            out.setdefault(rt, set()).update(canonize_set("repositories", to_str_list(v), inv))
        return out
    if isinstance(repos_any, list):
        for r in repos_any:
            if isinstance(r, dict):
                t = r.get("type") or r.get("db") or r.get("repository")
                a = r.get("accession") or r.get("id")
                if t and a:
                    rt = canonize_value("repositories", str(t), inv)
                    out.setdefault(rt, set()).add(str(a).strip())
            else:
                # unknown list items
                out.setdefault("unknown", set()).add(str(r).strip())
        return out
    # fallback
    out.setdefault("unknown", set()).update(to_str_list(repos_any))
    return out

def score_one(pred: Dict[str, Any], gold: Dict[str, Any], inv: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
    """
    Returns per-field metrics dict.
    """
    res: Dict[str, Any] = {"fields": {}}

    # Set fields
    for f in CANON_FIELDS_SET:
        pset = canonize_set(f if f != "assay_modalities" else "modalities", pred.get(f), inv)
        gset = canonize_set(f if f != "assay_modalities" else "modalities", gold.get(f), inv)
        p, r, f1 = set_prf(pset, gset)
        res["fields"][f] = {
            "pred": sorted(list(pset)),
            "gold": sorted(list(gset)),
            "precision": p,
            "recall": r,
            "f1": f1,
            "jaccard": jaccard(pset, gset),
            "ser": slot_error_rate(pset, gset),
            "coverage": coverage(bool(pset))
        }

    # Repositories: treat as (type,accession) pairs. We score:
    # 1) repo-type F1
    # 2) accession F1 pooled across all types
    preg = normalize_repositories(pred.get("repositories"), inv)
    greg = normalize_repositories(gold.get("repositories"), inv)

    ptypes = set(preg.keys())
    gtypes = set(greg.keys())
    p, r, f1 = set_prf(ptypes, gtypes)
    res["fields"]["repositories.repo_types"] = {
        "pred": sorted(list(ptypes)),
        "gold": sorted(list(gtypes)),
        "precision": p, "recall": r, "f1": f1,
        "jaccard": jaccard(ptypes, gtypes),
        "ser": slot_error_rate(ptypes, gtypes),
        "coverage": coverage(bool(ptypes))
    }

    pacc = set()
    gacc = set()
    for t, ids in preg.items():
        for x in ids:
            pacc.add(f"{t}:{x}")
    for t, ids in greg.items():
        for x in ids:
            gacc.add(f"{t}:{x}")
    p, r, f1 = set_prf(pacc, gacc)
    res["fields"]["repositories.accessions"] = {
        "pred": sorted(list(pacc)),
        "gold": sorted(list(gacc)),
        "precision": p, "recall": r, "f1": f1,
        "jaccard": jaccard(pacc, gacc),
        "ser": slot_error_rate(pacc, gacc),
        "coverage": coverage(bool(pacc))
    }

    # Text-ish fields: title (use EM + tokenF1 + ROUGE-L)
    for f in CANON_FIELDS_TEXT:
        pv = pred.get(f) or ""
        gv = gold.get(f) or ""
        pv = str(pv)
        gv = str(gv)
        res["fields"][f] = {
            "pred": pv,
            "gold": gv,
            "em": exact_match(pv.strip().lower(), gv.strip().lower()),
            "token_f1": token_f1(pv, gv),
            "rougeL_f1": rouge_l_f1(pv, gv),
            "coverage": coverage(bool(pv.strip()))
        }

    # Hallucination proxy: ratio of predicted items not in gold for the set fields
    # (use pooled acc + modalities + tech + species)
    pooled_pred = set(res["fields"]["species"]["pred"]) | set(res["fields"]["assay_modalities"]["pred"]) | set(res["fields"]["technology"]["pred"]) | set(res["fields"]["repositories.accessions"]["pred"])
    pooled_gold = set(res["fields"]["species"]["gold"]) | set(res["fields"]["assay_modalities"]["gold"]) | set(res["fields"]["technology"]["gold"]) | set(res["fields"]["repositories.accessions"]["gold"])
    fp = len(pooled_pred - pooled_gold)
    tp = len(pooled_pred & pooled_gold)
    res["hallucination_fp_rate"] = 0.0 if (tp + fp) == 0 else fp / (tp + fp)

    return res

def aggregate(reports: Dict[str, Dict[str, Any]], weights: Dict[str, float]) -> Dict[str, Any]:
    """
    reports: doi -> per-field metrics
    """
    # Macro averages over DOIs
    fields = {}
    for doi, rep in reports.items():
        for k, v in rep["fields"].items():
            fields.setdefault(k, []).append(v)

    macro = {}
    for k, lst in fields.items():
        # pick the best available metric for a canonical summary
        if "f1" in lst[0]:
            macro[k] = {
                "macro_f1": sum(x["f1"] for x in lst) / len(lst),
                "macro_precision": sum(x["precision"] for x in lst) / len(lst),
                "macro_recall": sum(x["recall"] for x in lst) / len(lst),
                "macro_jaccard": sum(x["jaccard"] for x in lst) / len(lst),
                "macro_ser": sum(x["ser"] for x in lst) / len(lst),
                "macro_coverage": sum(x["coverage"] for x in lst) / len(lst),
            }
        else:
            macro[k] = {
                "macro_em": sum(x["em"] for x in lst) / len(lst),
                "macro_token_f1": sum(x["token_f1"] for x in lst) / len(lst),
                "macro_rougeL_f1": sum(x["rougeL_f1"] for x in lst) / len(lst),
                "macro_coverage": sum(x["coverage"] for x in lst) / len(lst),
            }

    # Composite weighted score: use chosen metrics per field
    # Set-fields use F1; text-fields use token_f1; repositories.accessions uses F1
    def field_score(field_key: str, agg: Dict[str, Any]) -> float:
        if "macro_f1" in agg:
            return float(agg["macro_f1"])
        return float(agg.get("macro_token_f1", 0.0))

    total_w = 0.0
    total = 0.0
    for f, w in weights.items():
        # map weight field names to macro keys
        if f == "assay_modalities":
            key = "assay_modalities"
        elif f == "repositories":
            key = "repositories.accessions"
        else:
            key = f
        if key in macro:
            total += float(w) * field_score(key, macro[key])
            total_w += float(w)
    composite = total / total_w if total_w else 0.0

    hall = sum(rep.get("hallucination_fp_rate", 0.0) for rep in reports.values()) / max(1, len(reports))

    return {"macro": macro, "composite_weighted_score": composite, "macro_hallucination_fp_rate": hall}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True, help="configs/doi20_gold_verbose.json")
    ap.add_argument("--field-map", required=True, help="configs/field_map.yaml")
    ap.add_argument("--normalize", required=True, help="configs/normalization.yaml")
    ap.add_argument("--weights", required=True, help="configs/field_weights.yaml")
    ap.add_argument("--pred", action="append", required=True,
                    help="METHOD=DIR, e.g. deterministic=results_part1/deterministic")
    ap.add_argument("--outdir", default="eval_part1")
    args = ap.parse_args()

    gold_json = read_json(Path(args.gold))
    gold_items = gold_json.get("items") or gold_json.get("datasets") or []
    gold_by_doi = {it["doi"]: gold_to_canonical(it) for it in gold_items if it.get("doi")}

    field_map = load_field_map(Path(args.field_map))
    inv = load_synonyms(yaml.safe_load(Path(args.normalize).read_text(encoding="utf-8")))
    weights = yaml.safe_load(Path(args.weights).read_text(encoding="utf-8")).get("weights", {})

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_method_summaries = {}

    for spec in args.pred:
        method, pdir = spec.split("=", 1)
        pdir = Path(pdir)
        per_doi = {}

        for fp in sorted(pdir.glob("*.json")):
            obj = read_json(fp)
            # robust DOI discovery
            doi = obj.get("doi") or (obj.get("schema", {}) or {}).get("doi") or obj.get("deterministic_facts", {}).get("doi")
            if not doi:
                continue
            if doi not in gold_by_doi:
                continue

            pred_can = pred_to_canonical(obj, field_map, method)
            gold_can = gold_by_doi[doi]
            per_doi[doi] = score_one(pred_can, gold_can, inv)

        summary = aggregate(per_doi, weights)
        all_method_summaries[method] = summary

        # Save method report
        out = {
            "method": method,
            "n_compared": len(per_doi),
            "per_doi": per_doi,
            "summary": summary
        }
        (outdir / f"report_{method}.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    (outdir / "summary_all_methods.json").write_text(json.dumps(all_method_summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✅ Wrote evaluation reports to {outdir}")

if __name__ == "__main__":
    main()
