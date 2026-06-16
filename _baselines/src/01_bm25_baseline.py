"""
Submission 1: Field-Weighted BM25 Baseline
==========================================
Hypothesis: A well-tuned BM25 with field weighting (title > keywords > body)
and no stemming is a strong baseline for entity search — entities are often
identified by their title/name, so boosting title matches strongly improves nDCG.

Indexing:  rank-bm25 (BM25Okapi), corpus text = title×3 + keywords×2 + body×1
Tokenizer: simple whitespace + lowercase, no stemming, no stopword removal
           (for entity names, stemming can hurt: "Brazil" → "brazil" fine,
            but aggressive stemming may conflate entity names)
Ranking:   BM25 score directly, top 100 per query

Expected nDCG@100: ~0.33–0.38
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import time
import pickle
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

from rank_bm25 import BM25Okapi
from utils import (
    load_corpus, load_queries, load_qrels,
    build_doc_text, tokenize,
    write_submission, evaluate_run, save_pickle, load_pickle
)

# ── Configuration ────────────────────────────────────────────────────────────
DATA_DIR       = Path("data")
CORPUS_PATH    = DATA_DIR / "corpus.jsonl"
TEST_Q_PATH    = DATA_DIR / "test_queries.csv"
TRAIN_Q_PATH   = DATA_DIR / "train_queries.csv"
TRAIN_QR_PATH  = DATA_DIR / "train_qrels.csv"
OUT_DIR        = Path("submissions")
INDEX_CACHE    = Path("models") / "bm25_index.pkl"
DOCIDS_CACHE   = Path("models") / "bm25_docids.pkl"

TOP_K          = 100
TITLE_WEIGHT   = 3   # repeat title N times in concatenated text
KW_WEIGHT      = 2   # repeat keywords N times
STEM           = False
REMOVE_STOPS   = False
# ─────────────────────────────────────────────────────────────────────────────


def build_or_load_index():
    """Build BM25 index from corpus, or load from cache."""
    if INDEX_CACHE.exists() and DOCIDS_CACHE.exists():
        print("[*] Loading cached BM25 index...")
        bm25 = load_pickle(str(INDEX_CACHE))
        doc_ids = load_pickle(str(DOCIDS_CACHE))
        print(f"    Loaded index for {len(doc_ids):,} documents.")
        return bm25, doc_ids

    print("[*] Building BM25 index from corpus (this will take a while)...")
    t0 = time.time()

    doc_ids = []
    corpus_tokens = []

    for i, (doc_id, title, keywords, body) in enumerate(
        tqdm(load_corpus(str(CORPUS_PATH)), desc="Indexing", unit=" docs")
    ):
        doc_ids.append(doc_id)
        combined = build_doc_text(title, keywords, body, TITLE_WEIGHT, KW_WEIGHT)
        tokens = tokenize(combined, stem=STEM, remove_stops=REMOVE_STOPS)
        corpus_tokens.append(tokens)

        if i % 500_000 == 0 and i > 0:
            print(f"    {i:,} docs indexed in {time.time()-t0:.0f}s")

    print(f"[*] Building BM25 model for {len(doc_ids):,} docs...")
    bm25 = BM25Okapi(corpus_tokens)

    print("[*] Saving index to cache...")
    Path("models").mkdir(exist_ok=True)
    save_pickle(bm25, str(INDEX_CACHE))
    save_pickle(doc_ids, str(DOCIDS_CACHE))
    print(f"[✓] Index built in {time.time()-t0:.0f}s")
    return bm25, doc_ids


def retrieve(bm25: BM25Okapi, doc_ids: List[str],
             queries: List[tuple], top_k: int = 100) -> Dict[str, List[str]]:
    """Retrieve top-k docs per query using BM25."""
    import numpy as np
    run = {}
    for qid, qtext in tqdm(queries, desc="Retrieving"):
        qtokens = tokenize(qtext, stem=STEM, remove_stops=REMOVE_STOPS)
        scores = bm25.get_scores(qtokens)
        # argsort descending — get top_k indices
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        run[qid] = [doc_ids[i] for i in top_indices]
    return run


def main():
    OUT_DIR.mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)

    # 1. Build / load index
    bm25, doc_ids = build_or_load_index()

    # 2. Load queries
    test_queries = load_queries(str(TEST_Q_PATH))
    print(f"[*] {len(test_queries)} test queries loaded.")

    # 3. Retrieve
    print("[*] Running retrieval on test queries...")
    test_run = retrieve(bm25, doc_ids, test_queries, TOP_K)

    # 4. Write submission
    write_submission(test_run, str(OUT_DIR / "submission_01_bm25_baseline.csv"), TOP_K)

    # 5. Local evaluation on training queries (sanity check)
    if TRAIN_Q_PATH.exists() and TRAIN_QR_PATH.exists():
        print("[*] Evaluating on train queries (local nDCG@100)...")
        train_queries = load_queries(str(TRAIN_Q_PATH))
        train_qrels   = load_qrels(str(TRAIN_QR_PATH))
        train_run     = retrieve(bm25, doc_ids, train_queries, TOP_K)
        mean_ndcg, per_query = evaluate_run(train_run, train_qrels, k=100)
        print(f"[✓] Train nDCG@100 = {mean_ndcg:.4f}")
        # Show worst 5 queries for diagnosis
        worst = sorted(per_query.items(), key=lambda x: x[1])[:5]
        print("    Worst 5 queries:")
        tq_map = dict(train_queries)
        for qid, score in worst:
            print(f"      qid={qid}  nDCG={score:.4f}  query='{tq_map.get(qid, '?')}'")


if __name__ == "__main__":
    main()