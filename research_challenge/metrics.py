import math

import numpy as np


def ndcg_at_k(ranking, qrels, k=100):
    def dcg(values):
        return sum((2**rel - 1) / math.log2(rank + 2) for rank, rel in enumerate(values))

    observed = [qrels.get(doc_id, 0) for doc_id in ranking[:k]]
    ideal = sorted(qrels.values(), reverse=True)[:k]
    denominator = dcg(ideal)
    return dcg(observed) / denominator if denominator else 0.0


def evaluate(run, qrels, k=100):
    values = [
        ndcg_at_k(run.get(qid, []), judgments, k)
        for qid, judgments in qrels.items()
    ]
    return float(np.mean(values)) if values else 0.0


def recall_at_k(run, qrels, k):
    values = []
    for qid, judgments in qrels.items():
        relevant = {doc_id for doc_id, rel in judgments.items() if rel > 0}
        if relevant:
            values.append(len(relevant & set(run.get(qid, [])[:k])) / len(relevant))
    return float(np.mean(values)) if values else 0.0
