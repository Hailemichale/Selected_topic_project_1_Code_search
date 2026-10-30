"""Command line search:  python -m src.search "parse a date" --repo path/to/code"""
import argparse

from src.baseline import BM25Search
from src.data import extract_functions


def build(method, records):
    if method == "bm25":
        return BM25Search(records)
    if method == "embed":
        from src.embed import EmbeddingSearch
        return EmbeddingSearch(records)
    raise ValueError(f"unknown method: {method}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--method", default="embed", choices=["bm25", "embed"])
    ap.add_argument("--k", type=int, default=5)
    a = ap.parse_args()
    records = extract_functions(a.repo)
    by_id = {r["id"]: r for r in records}
    for i, rid in enumerate(build(a.method, records).search(a.query, a.k), 1):
        print(f"{i}. {rid}")
        print("   " + by_id[rid]["code"].splitlines()[0])
