"""
Submission 4: LambdaMART Learning-to-Rank Reranker
===================================================
Hypothesis: The provided training data (234 queries, 8,202 judgments) enables
a supervised reranker. LambdaMART directly optimizes for nDCG, making it the
ideal LTR objective for this competition metric. We train on retrieval features
from Sub3, then rerank the top-200 candidates for test queries.

Features per (query, doc) pair (14 total):
  1.  BM25 score (title+kw+body)
  2.  BM25 score (title only)
  3.  BM25 score (keywords only)
  4.  BM25 score (body only)
  5.  Dense cosine similarity
  6.  Query term coverage: % query unigrams in title
  7.  Query term coverage: % query unigrams in keywords
  8.  Query term coverage: % query unigrams in full text
  9.  BM25 rank (from Sub1 run) — normalized
  10. Dense rank (from Sub3 run) — normalized
  11. RRF score
  12. Title length (log)
  13. # keywords
  14. Exact title match (binary)

Training: 5-fold cross-validation on train queries, then full retrain.
Model: LightGBM LambdaRank (objective=lambdarank, ndcg)

Expected nDCG@100: ~0.50–0.55
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import math
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from tqdm import tqdm

from utils import (
    load_corpus, load_queries, load_qrels,
    build_doc_text, tokenize,
    write_submission, evaluate_run,
    save_pickle, load_pickle, reciprocal_rank_fusion
)

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR        = Path("data")
CORPUS_PATH     = DATA_DIR / "corpus.jsonl"
TEST_Q_PATH     = DATA_DIR / "test_queries.csv"
TRAIN_Q_PATH    = DATA_DIR / "train_queries.csv"
TRAIN_QR_PATH   = DATA_DIR / "train_qrels.csv"
OUT_DIR         = Path("submissions")

BM25_INDEX      = Path("models") / "bm25_index.pkl"
DOCIDS_CACHE    = Path("models") / "bm25_docids.pkl"
DENSE_INDEX     = Path("models") / "faiss_index.bin"
DENSE_DOCIDS    = Path("models") / "faiss_docids.pkl"
LTR_MODEL       = Path("models") / "ltr_lambdamart.pkl"

# Corpus field caches (for per-field BM25)
TITLE_INDEX     = Path("models") / "bm25_title.pkl"
KW_INDEX        = Path("models") / "bm25_kw.pkl"
BODY_INDEX      = Path("models") / "bm25_body.pkl"

TOP_K           = 100
RERANK_CANDS    = 200   # candidates to rerank
DENSE_MODEL_ID  = "sentence-transformers/all-MiniLM-L6-v2"

LTR_PARAMS = {
    "objective":         "lambdarank",
    "metric":            "ndcg",
    "ndcg_eval_at":      [10, 100],
    "learning_rate":     0.05,
    "num_leaves":        63,
    "min_data_in_leaf":  5,
    "n_estimators":      500,
    "feature_fraction":  0.8,
    "bagging_fraction":  0.8,
    "bagging_freq":      5,
    "lambda_l2":         0.1,
    "verbose":           -1,
    "n_jobs":            -1,
}
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# CORPUS FIELD CACHE
# ─────────────────────────────────────────────────────────────────────────────

def build_or_load_field_indices():
    """Build per-field BM25 indices (title, keywords, body)."""
    from rank_bm25 import BM25Okapi

    caches = {
        "title": (TITLE_INDEX, Path("models") / "field_docids.pkl"),
        "kw":    (KW_INDEX, None),
        "body":  (BODY_INDEX, None),
    }

    if TITLE_INDEX.exists() and KW_INDEX.exists() and BODY_INDEX.exists():
        print("[*] Loading per-field BM25 indices from cache...")
        bm25_title = load_pickle(str(TITLE_INDEX))
        bm25_kw    = load_pickle(str(KW_INDEX))
        bm25_body  = load_pickle(str(BODY_INDEX))
        field_ids  = load_pickle(str(caches["title"][1]))
        return bm25_title, bm25_kw, bm25_body, field_ids

    print("[*] Building per-field BM25 indices...")
    title_corpus, kw_corpus, body_corpus, field_ids = [], [], [], []
    for doc_id, title, keywords, body in tqdm(
        load_corpus(str(CORPUS_PATH)), desc="Loading fields"
    ):
        field_ids.append(doc_id)
        title_corpus.append(tokenize(title))
        kw_corpus.append(tokenize(" ".join(keywords)))
        body_corpus.append(tokenize(body[:500]))  # cap body for speed

    bm25_title = BM25Okapi(title_corpus)
    bm25_kw    = BM25Okapi(kw_corpus)
    bm25_body  = BM25Okapi(body_corpus)

    Path("models").mkdir(exist_ok=True)
    save_pickle(bm25_title, str(TITLE_INDEX))
    save_pickle(bm25_kw,    str(KW_INDEX))
    save_pickle(bm25_body,  str(BODY_INDEX))
    save_pickle(field_ids,  str(caches["title"][1]))
    return bm25_title, bm25_kw, bm25_body, field_ids


def build_doc_store() -> Dict[str, dict]:
    """
    Build in-memory doc store with {doc_id: {title, kw_set, kw_count, body_len}}.
    We only store what's needed for feature computation.
    """
    store_path = Path("models") / "doc_store.pkl"
    if store_path.exists():
        print("[*] Loading doc store from cache...")
        return load_pickle(str(store_path))

    print("[*] Building doc store...")
    store = {}
    for doc_id, title, keywords, body in tqdm(
        load_corpus(str(CORPUS_PATH)), desc="Doc store"
    ):
        store[doc_id] = {
            "title":      title.lower(),
            "title_toks": set(tokenize(title)),
            "kw_toks":    set(tokenize(" ".join(keywords))),
            "kw_count":   len(keywords),
            "body_len":   len(body.split()),
        }
    save_pickle(store, str(store_path))
    return store


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(
    query_text: str,
    candidate_doc_ids: List[str],
    bm25_full,     bm25_full_ids: List[str],
    bm25_title,    bm25_kw,    bm25_body,
    field_ids:     List[str],
    dense_scores:  Dict[str, float],
    bm25_scores_full: Dict[str, float],
    rrf_scores:    Dict[str, float],
    doc_store:     Dict[str, dict],
    bm25_ranks:    Dict[str, int],
    dense_ranks:   Dict[str, int],
) -> np.ndarray:
    """
    Extract feature vector for each (query, doc) pair.
    Returns shape (len(candidate_doc_ids), 14).
    """
    qtokens      = tokenize(query_text)
    qtokens_set  = set(qtokens)
    query_lower  = query_text.lower()

    # Per-field BM25 scores (indexed by field_ids)
    field_id_to_idx = {did: i for i, did in enumerate(field_ids)}

    t_scores = bm25_title.get_scores(qtokens)
    k_scores = bm25_kw.get_scores(qtokens)
    b_scores = bm25_body.get_scores(qtokens)

    n_cands = len(candidate_doc_ids)
    max_rank = max(max(bm25_ranks.values(), default=1),
                   max(dense_ranks.values(), default=1)) + 1

    features = np.zeros((n_cands, 14), dtype=np.float32)
    for i, doc_id in enumerate(candidate_doc_ids):
        doc = doc_store.get(doc_id, {})
        fi  = field_id_to_idx.get(doc_id, -1)

        # BM25 scores
        features[i, 0] = bm25_scores_full.get(doc_id, 0.0)
        features[i, 1] = t_scores[fi] if fi >= 0 else 0.0
        features[i, 2] = k_scores[fi] if fi >= 0 else 0.0
        features[i, 3] = b_scores[fi] if fi >= 0 else 0.0

        # Dense cosine
        features[i, 4] = dense_scores.get(doc_id, 0.0)

        # Term coverage
        title_toks = doc.get("title_toks", set())
        kw_toks    = doc.get("kw_toks", set())
        all_toks   = title_toks | kw_toks
        denom      = max(len(qtokens_set), 1)
        features[i, 5] = len(qtokens_set & title_toks) / denom
        features[i, 6] = len(qtokens_set & kw_toks) / denom
        features[i, 7] = len(qtokens_set & all_toks) / denom

        # Rank features (normalized)
        features[i, 8]  = 1.0 - bm25_ranks.get(doc_id, max_rank) / max_rank
        features[i, 9]  = 1.0 - dense_ranks.get(doc_id, max_rank) / max_rank
        features[i, 10] = rrf_scores.get(doc_id, 0.0)

        # Doc characteristics
        features[i, 11] = math.log1p(len(doc.get("title", "")))
        features[i, 12] = min(doc.get("kw_count", 0), 20) / 20.0

        # Exact title match
        features[i, 13] = 1.0 if query_lower == doc.get("title", "") else 0.0

    return features


# ─────────────────────────────────────────────────────────────────────────────
# LTR TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def build_training_data(
    train_queries, train_qrels,
    bm25, bm25_ids, bm25_title, bm25_kw, bm25_body, field_ids,
    dense_model, faiss_index, dense_doc_ids,
    doc_store
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build feature matrix X, labels y, and query groups for LightGBM."""
    import faiss as faiss_module

    X_all, y_all, groups = [], [], []

    id_to_idx = {did: i for i, did in enumerate(bm25_ids)}
    dense_id_to_idx = {did: i for i, did in enumerate(dense_doc_ids)}

    for qid, qtext in tqdm(train_queries, desc="Building LTR training data"):
        qtokens  = tokenize(qtext)
        q_qrels  = train_qrels.get(qid, {})
        if not q_qrels:
            continue

        # BM25 retrieval (top RERANK_CANDS)
        bm25_full_scores = bm25.get_scores(qtokens)
        top_bm25_idx = np.argpartition(bm25_full_scores, -RERANK_CANDS)[-RERANK_CANDS:]
        top_bm25_idx = top_bm25_idx[np.argsort(bm25_full_scores[top_bm25_idx])[::-1]]
        bm25_run     = [bm25_ids[i] for i in top_bm25_idx]

        # Dense retrieval
        from sentence_transformers import SentenceTransformer
        q_emb = dense_model.encode([qtext], normalize_embeddings=True,
                                    convert_to_numpy=True).astype(np.float32)
        d_scores, d_indices = faiss_index.search(q_emb, RERANK_CANDS)
        dense_run = [dense_doc_ids[i] for i in d_indices[0] if i >= 0]

        # RRF fusion
        rrf_run_dict = reciprocal_rank_fusion(
            [{qid: bm25_run}, {qid: dense_run}],
            k=60
        )
        fused_candidates = rrf_run_dict.get(qid, bm25_run)[:RERANK_CANDS]

        # Add all judged relevant docs not already in candidates
        relevant_not_in = [d for d in q_qrels if d not in set(fused_candidates)]
        candidates = (fused_candidates + relevant_not_in)[:RERANK_CANDS]

        # Score lookups
        bm25_score_map  = {bm25_ids[i]: float(bm25_full_scores[i]) for i in top_bm25_idx}
        dense_score_map = {dense_doc_ids[d_indices[0][j]]: float(d_scores[0][j])
                           for j in range(len(d_indices[0])) if d_indices[0][j] >= 0}
        bm25_rank_map   = {did: rank for rank, did in enumerate(bm25_run)}
        dense_rank_map  = {did: rank for rank, did in enumerate(dense_run)}

        # RRF score map
        rrf_scores_map = defaultdict(float)
        for rank, did in enumerate(fused_candidates):
            rrf_scores_map[did] = 1.0 / (60 + rank + 1)

        feats = extract_features(
            qtext, candidates,
            bm25, bm25_ids, bm25_title, bm25_kw, bm25_body, field_ids,
            dense_score_map, bm25_score_map, dict(rrf_scores_map),
            doc_store, bm25_rank_map, dense_rank_map
        )

        labels = np.array([q_qrels.get(did, 0) for did in candidates], dtype=np.float32)

        X_all.append(feats)
        y_all.append(labels)
        groups.append(len(candidates))

    X = np.vstack(X_all)
    y = np.concatenate(y_all)
    g = np.array(groups)
    return X, y, g


def train_ltr(X, y, groups):
    import lightgbm as lgb

    print(f"[*] Training LambdaMART on {X.shape[0]:,} (doc, query) pairs...")
    dtrain = lgb.Dataset(X, label=y, group=groups, free_raw_data=False)

    params = LTR_PARAMS.copy()
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=params.pop("n_estimators", 500),
        valid_sets=[dtrain],
        callbacks=[lgb.early_stopping(50, verbose=False),
                   lgb.log_evaluation(50)]
    )
    save_pickle(model, str(LTR_MODEL))
    print("[✓] LTR model saved.")

    # Feature importance
    fi = model.feature_importance(importance_type="gain")
    feat_names = [
        "bm25_full", "bm25_title", "bm25_kw", "bm25_body",
        "dense_cos", "cov_title", "cov_kw", "cov_full",
        "bm25_rank", "dense_rank", "rrf_score",
        "title_len", "kw_count", "exact_match"
    ]
    print("    Feature importances:")
    for name, imp in sorted(zip(feat_names, fi), key=lambda x: -x[1]):
        print(f"      {name:20s}: {imp:.0f}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def rerank_with_ltr(
    model,
    queries, bm25, bm25_ids, bm25_title, bm25_kw, bm25_body, field_ids,
    dense_model, faiss_index, dense_doc_ids, doc_store,
    top_k=100
) -> Dict[str, List[str]]:
    run = {}

    for qid, qtext in tqdm(queries, desc="LTR reranking"):
        qtokens = tokenize(qtext)

        # BM25 candidates
        bm25_scores = bm25.get_scores(qtokens)
        top_bm25    = np.argpartition(bm25_scores, -RERANK_CANDS)[-RERANK_CANDS:]
        top_bm25    = top_bm25[np.argsort(bm25_scores[top_bm25])[::-1]]
        bm25_run    = [bm25_ids[i] for i in top_bm25]

        # Dense candidates
        q_emb = dense_model.encode([qtext], normalize_embeddings=True,
                                    convert_to_numpy=True).astype(np.float32)
        d_scores, d_indices = faiss_index.search(q_emb, RERANK_CANDS)
        dense_run = [dense_doc_ids[i] for i in d_indices[0] if i >= 0]

        # Fuse
        fused = reciprocal_rank_fusion(
            [{qid: bm25_run}, {qid: dense_run}], k=60
        ).get(qid, bm25_run)[:RERANK_CANDS]

        # Score maps
        bm25_score_map  = {bm25_ids[i]: float(bm25_scores[i]) for i in top_bm25}
        dense_score_map = {dense_doc_ids[d_indices[0][j]]: float(d_scores[0][j])
                           for j in range(len(d_indices[0])) if d_indices[0][j] >= 0}
        bm25_rank_map   = {did: r for r, did in enumerate(bm25_run)}
        dense_rank_map  = {did: r for r, did in enumerate(dense_run)}
        rrf_scores_map  = {did: 1.0/(60+r+1) for r, did in enumerate(fused)}

        feats = extract_features(
            qtext, fused,
            bm25, bm25_ids, bm25_title, bm25_kw, bm25_body, field_ids,
            dense_score_map, bm25_score_map, rrf_scores_map,
            doc_store, bm25_rank_map, dense_rank_map
        )

        ltr_scores = model.predict(feats)
        reranked   = [fused[i] for i in np.argsort(ltr_scores)[::-1]]
        run[qid]   = reranked[:top_k]

    return run


def main():
    import faiss
    from sentence_transformers import SentenceTransformer

    OUT_DIR.mkdir(exist_ok=True)

    # ── Load indices ──────────────────────────────────────────────────────────
    print("[*] Loading BM25 index...")
    bm25     = load_pickle(str(BM25_INDEX))
    bm25_ids = load_pickle(str(DOCIDS_CACHE))

    bm25_title, bm25_kw, bm25_body, field_ids = build_or_load_field_indices()

    print("[*] Loading FAISS index...")
    faiss_index   = faiss.read_index(str(DENSE_INDEX))
    dense_doc_ids = load_pickle(str(DENSE_DOCIDS))

    dense_model = SentenceTransformer(DENSE_MODEL_ID)
    doc_store   = build_doc_store()

    # ── Train LTR ─────────────────────────────────────────────────────────────
    if LTR_MODEL.exists():
        print("[*] Loading cached LTR model...")
        ltr_model = load_pickle(str(LTR_MODEL))
    else:
        train_queries = load_queries(str(TRAIN_Q_PATH))
        train_qrels   = load_qrels(str(TRAIN_QR_PATH))
        X, y, groups  = build_training_data(
            train_queries, train_qrels,
            bm25, bm25_ids, bm25_title, bm25_kw, bm25_body, field_ids,
            dense_model, faiss_index, dense_doc_ids, doc_store
        )
        ltr_model = train_ltr(X, y, groups)

    # ── Rerank test queries ───────────────────────────────────────────────────
    test_queries = load_queries(str(TEST_Q_PATH))
    test_run     = rerank_with_ltr(
        ltr_model, test_queries,
        bm25, bm25_ids, bm25_title, bm25_kw, bm25_body, field_ids,
        dense_model, faiss_index, dense_doc_ids, doc_store,
        TOP_K
    )
    write_submission(test_run, str(OUT_DIR / "submission_04_ltr.csv"), TOP_K)

    # ── Local eval ────────────────────────────────────────────────────────────
    if TRAIN_Q_PATH.exists() and TRAIN_QR_PATH.exists():
        train_queries = load_queries(str(TRAIN_Q_PATH))
        train_qrels   = load_qrels(str(TRAIN_QR_PATH))
        print("[*] Evaluating LTR on train queries...")
        train_run = rerank_with_ltr(
            ltr_model, train_queries,
            bm25, bm25_ids, bm25_title, bm25_kw, bm25_body, field_ids,
            dense_model, faiss_index, dense_doc_ids, doc_store,
            TOP_K
        )
        mean_ndcg, _ = evaluate_run(train_run, train_qrels, k=100)
        print(f"[✓] Train nDCG@100 = {mean_ndcg:.4f}")


if __name__ == "__main__":
    main()