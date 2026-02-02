#!/usr/bin/env python3
# scripts/part1_eval_evidence_support.py

from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Set

def read_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))

def slugify_doi(doi: str) -> str:
    doi = doi.lower().strip()
    doi = re.sub(r"^https?://doi\.org/", "", doi)
    doi = re.sub(r"[^a-z0-9._/-]+", "", doi)
    return doi.replace("/", "_")

def find_support(text: str, value: str) -> bool:
    v = (value or "").strip()
    if not v or len(v) < 4:
        return False
    return v.lower() in text.lower()

def flatten_values(obj: Any) -> List[str]:
    if obj is None:
        return []
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        out = []
        for k, v in obj.items():
            out.extend(flatten_values(v))
        return out
    if isinstance(obj, (list, tuple, set)):
        out = []
        for x in obj:
            out.extend(flatten_values(x))
        return out
    return [str(obj)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True, help="results_part1/hybrid (or llm_only)")
    ap.add_argument("--papers-dir", required=True, help="papers/<doi_slug>/paper_fulltext.txt")
    ap.add_argument("--out", default="eval_part1/evidence_support.json")
    args = ap.parse_args()

    pred_dir = Path(args.pred_dir)
    papers_dir = Path(args.papers_dir)

    report = {"pred_dir": str(pred_dir), "items": [], "summary": {}}

    total = 0
    supported = 0

    for fp in sorted(pred_dir.glob("*.json")):
        obj = read_json(fp)
        doi = obj.get("doi") or (obj.get("schema", {}) or {}).get("doi") or obj.get("deterministic_facts", {}).get("doi")
        if not doi:
            continue
        slug = slugify_doi(doi)
        txt = papers_dir / slug / "paper_fulltext.txt"
        if not txt.exists():
            continue
        paper_text = txt.read_text(encoding="utf-8", errors="ignore")

        schema = obj.get("schema") or obj.get("deterministic_facts") or {}
        vals = flatten_values(schema)

        # Only check “meaningful” candidates
        candidates = [v for v in vals if isinstance(v, str) and 4 <= len(v) <= 120]

        item_total = 0
        item_supported = 0
        for v in candidates:
            item_total += 1
            if find_support(paper_text, v):
                item_supported += 1

        total += item_total
        supported += item_supported

        report["items"].append({
            "doi": doi,
            "n_candidates": item_total,
            "n_supported": item_supported,
            "support_rate": (item_supported / item_total) if item_total else None
        })

    report["summary"] = {
        "n_items": len(report["items"]),
        "total_candidates": total,
        "supported_candidates": supported,
        "overall_support_rate": (supported / total) if total else None
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✅ Evidence support written to {args.out}")

if __name__ == "__main__":
    main()
