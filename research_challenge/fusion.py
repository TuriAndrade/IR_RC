from collections import defaultdict

from .metrics import evaluate


def rrf(sparse_run, dense_run, dense_weight, k=60, limit=None):
    qids = set(sparse_run) | set(dense_run)
    output = {}
    for qid in qids:
        scores = defaultdict(float)
        for rank, doc_id in enumerate(sparse_run.get(qid, [])):
            scores[doc_id] += 1.0 / (k + rank + 1)
        for rank, doc_id in enumerate(dense_run.get(qid, [])):
            scores[doc_id] += dense_weight / (k + rank + 1)
        ranking = sorted(scores, key=scores.get, reverse=True)
        output[qid] = ranking[:limit] if limit else ranking
    return output


def tune_dense_weight(sparse_run, dense_run, qrels, config):
    best = (-1.0, None, None)
    for weight in config["retrieval"]["dense_weight_grid"]:
        run = rrf(
            sparse_run,
            dense_run,
            weight,
            config["retrieval"]["rrf_k"],
            config["retrieval"]["candidate_k"],
        )
        score = evaluate(run, qrels, config["retrieval"]["final_k"])
        print(f"dense_weight={weight}: nDCG@100={score:.4f}")
        if score > best[0]:
            best = (score, weight, run)
    return best
