"""Baseline: BM25 keyword search. No Transformer."""
import re

from rank_bm25 import BM25Okapi


def tokenize(text):
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)  # parseDate -> parse Date
    return [t for t in re.split(r"[^A-Za-z0-9]+", text.lower()) if t]


class BM25Search:
    def __init__(self, records):
        self.records = records
        self.bm25 = BM25Okapi([tokenize(r["name"] + " " + r["code"]) for r in records])

    def search(self, query, k=10):
        scores = self.bm25.get_scores(tokenize(query))
        top = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [self.records[i]["id"] for i in top]
