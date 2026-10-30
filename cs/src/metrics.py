def recall_at_k(ranked, gold, k):
    return 1.0 if gold in ranked[:k] else 0.0


def reciprocal_rank(ranked, gold):
    return 1.0 / (ranked.index(gold) + 1) if gold in ranked else 0.0
