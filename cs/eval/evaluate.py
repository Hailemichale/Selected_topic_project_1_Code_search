"""Evaluate search methods.

Defaults to the DEV split so you can tune freely.
Run with --on test once, at the end, for the numbers you report.
"""
import argparse
import csv
import json
from pathlib import Path

from src.data import extract_functions, split
from src.metrics import recall_at_k, reciprocal_rank
from src.search import build

DEPTH = 10  # MRR is MRR@10


def evaluate(search, queries):
    rows = []
    for r in queries:
        ranked = search.search(r["query"], k=DEPTH)
        rows.append({"query": r["query"], "gold": r["id"], "top1": ranked[0] if ranked else "",
                     "r1": recall_at_k(ranked, r["id"], 1), "r5": recall_at_k(ranked, r["id"], 5),
                     "rr": reciprocal_rank(ranked, r["id"])})
    n = max(len(rows), 1)
    summary = {"n": len(rows), "recall@1": sum(x["r1"] for x in rows) / n,
               "recall@5": sum(x["r5"] for x in rows) / n, "mrr@10": sum(x["rr"] for x in rows) / n}
    return summary, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--methods", nargs="+", default=["bm25", "embed"])
    ap.add_argument("--on", choices=["dev", "test"], default="dev")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-queries", type=int, default=None)
    a = ap.parse_args()

    records = extract_functions(a.repo)
    dev, test = split(records, seed=a.seed)
    queries = (dev if a.on == "dev" else test)[: a.max_queries]
    out = Path("results")
    out.mkdir(exist_ok=True)
    summary = {}
    for m in a.methods:
        s, rows = evaluate(build(m, records), queries)
        summary[m] = s
        with open(out / f"failures_{a.on}_{m}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["query", "gold", "top1", "r1", "r5", "rr"])
            w.writeheader()
            w.writerows(x for x in rows if x["r1"] == 0)
        print(f"{m:6} n={s['n']}  R@1={s['recall@1']:.3f}  R@5={s['recall@5']:.3f}  MRR@10={s['mrr@10']:.3f}")
    (out / f"results_{a.on}.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
