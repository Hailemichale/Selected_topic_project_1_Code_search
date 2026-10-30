"""Transformer search: a pretrained sentence encoder plus cosine similarity."""
import numpy as np


class EmbeddingSearch:
    def __init__(self, records, model_name="all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # imported here so BM25 runs without it

        self.records = records
        self.model = SentenceTransformer(model_name)
        texts = [r["name"] + " " + r["code"] for r in records]
        self.vecs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    def search(self, query, k=10):
        q = self.model.encode([query], normalize_embeddings=True)[0]
        top = np.argsort(-(self.vecs @ q))[:k]
        return [self.records[i]["id"] for i in top]
