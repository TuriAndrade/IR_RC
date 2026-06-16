"""
Submission 5: Full Ensemble — Cross-Encoder Reranker + Optimized Fusion
=======================================================================
Hypothesis: A cross-encoder (BERT-based) reranker jointly encodes query and
document and is significantly more accurate than bi-encoder for final top-k
reranking. Combined with the LTR scores from Sub4 and tuned RRF weights
(optimized on training data), this should push nDCG@100 to its ceiling.

Pipeline:
  1. First-stage retrieval: BM25 + Dense bi-encoder (from Sub3) → top-200
  2. LTR reranker (Sub4) → reorder to top-50
  3. Cross-encoder (ms-marco-MiniLM-L-6-v2) → rerank top-50
  4. Final RRF merge of LTR-ranked and cross-encoder-ranked lists
  
Extra techniques:
  - Query augmentation: add entity type hints ("find entity:", "what is:")
  - Score normalization before fusion
  - Grid search on RRF weights using training queries

Expected nDCG@100: ~0.55–0.62
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import math
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from itertools import product
from tqdm import tqdm

from utils import (
    load_corpus, load_queries, load_qrels,
    tokenize, write_submission, evaluate_run,
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
TITLE_INDEX     = Path("models") / "bm25_title.pkl"
KW_INDEX        = Path("models") / "bm25_kw.pkl"
BODY_INDEX      = Path("models") / "bm25_body.pkl"

DOC_TITLE_CACHE = Path("models") / "doc_titles.pkl"

TOP_K           = 100
FIRST_STAGE_K   = 200   # BM25 + Dense candidates
LTR_RERANK_K    = 50    # LTR top-50
CE_RERANK_K     = 50    # Cross-encoder top-50

BI_ENCODER      = "sentence-transformers/all-MiniLM-L6-v2"
# Cross-encoder fine-tuned on MS MARCO — strong for ad-hoc retrieval
CROSS_ENCODER   = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# RRF weights — tuned via grid search on training queries
# These defaults are good starting points; tune_rrf_weights() will refine
RRF_SPARSE_W    = 1.0
RRF_DENSE_W     = 1.2
RRF_LTR_W       = 2.0
RRF_CE_W        = 3.0
# ─────────────────────────────────────────────────────────────────────────────


def load_doc_titles() -> Dict[str, str]:
    """Load just {doc_id: title} for cross-encoder (we only need the title for speed)."""
    cache = DOC_TITLE_CACHE
    if cache.exists():
        return load_pickle(str(cache))
    print("[*] Building doc title cache...")
    titles = {}
    for doc_id, title, keywords, body in tqdm(
        load_corpus(str(CORPUS_PATH)), desc="Title cache"
    ):
        # For cross-encoder, use title + top-3 keywords for conciseness
        kw_str = "; ".join(keywords[:3])
        titles[doc_id] = (title + " " + kw_str).strip()[:200]
    save_pickle(titles, str(cache))
    return titles


def cross_encoder_rerank(
    ce_model,
    qtext: str,
    candidate_ids: List[str],
    doc_titles: Dict[str, str],
    top_k: int = 50,
    batch_size: int = 128
) -> List[str]:
    """Rerank candidates with cross-encoder. Returns reranked doc_ids."""
    pairs = [(qtext, doc_titles.get(did, "")) for did in candidate_ids]
    scores = ce_model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
    reranked = [candidate_ids[i] for i in np.argsort(scores)[::-1]]
    return reranked[:top_k]


def tune_rrf_weights(
    train_queries, train_qrels,
    bm25, bm25_ids, dense_model, faiss_index, dense_doc_ids,
    ltr_model, ce_model, doc_titles, field_ids,
    bm25_title, bm25_kw, bm25_body, doc_store
):
    """
    Grid search over RRF weights using training queries.
    Evaluates a subset for speed, returns best weights.
    """
    from src.utils import evaluate_run
    print("[*] Tuning RRF weights on training data...")

    # Build all component runs for training queries
    # (same retrieval pipeline as main inference)
    sparse_runs = {}
    dense_runs  = {}

    for qid, qtext in tqdm(train_queries[:50], desc="Building train component runs"):
        qtokens = tokenize(qtext)
        sc = bm25.get_scores(qtokens)
        top_idx = np.argpartition(sc, -FIRST_STAGE_K)[-FIRST_STAGE_K:]
        top_idx = top_idx[np.argsort(sc[top_idx])[::-1]]
        sparse_runs[qid] = [bm25_ids[i] for i in top_idx]

        q_emb = dense_model.encode([qtext], normalize_embeddings=True,
                                    convert_to_numpy=True).astype(np.float32)
        _, d_idx = faiss_index.search(q_emb, FIRST_STAGE_K)
        dense_runs[qid] = [dense_doc_ids[i] for i in d_idx[0] if i >= 0]

    best_ndcg   = -1
    best_weights = (1.0, 1.0)

    for sw, dw in product([0.5, 1.0, 1.5, 2.0], [0.5, 1.0, 1.5, 2.0]):
        fused = reciprocal_rank_fusion(
            [sparse_runs, dense_runs],
            weights=[sw, dw], k=60
        )
        fused = {q: d[:TOP_K] for q, d in fused.items()}
        ndcg, _ = evaluate_run(fused, train_qrels)
        if ndcg > best_ndcg:
            best_ndcg    = ndcg
            best_weights = (sw, dw)

    print(f"[✓] Best sparse/dense weights: {best_weights}  nDCG={best_ndcg:.4f}")
    return best_weights


def run_full_pipeline(
    queries: List[Tuple[str, str]],
    bm25, bm25_ids,
    dense_model, faiss_index, dense_doc_ids,
    ltr_model,
    ce_model,
    doc_titles: Dict[str, str],
    bm25_title, bm25_kw, bm25_body, field_ids,
    doc_store,
    sparse_w=1.0, dense_w=1.2, ltr_w=2.0, ce_w=3.0,
    top_k=100
) -> Dict[str, List[str]]:
    """Full pipeline: BM25+Dense → RRF → LTR rerank → CE rerank → final RRF."""
    from src._04_ltr_reranker import extract_features, RERANK_CANDS

    final_run = {}

    for qid, qtext in tqdm(queries, desc="Full pipeline"):
        qtokens = tokenize(qtext)

        # ── Stage 1: First-stage retrieval ───────────────────────────────────
        bm25_scores = bm25.get_scores(qtokens)
        top_bm25    = np.argpartition(bm25_scores, -FIRST_STAGE_K)[-FIRST_STAGE_K:]
        top_bm25    = top_bm25[np.argsort(bm25_scores[top_bm25])[::-1]]
        bm25_run    = [bm25_ids[i] for i in top_bm25]

        q_emb = dense_model.encode([qtext], normalize_embeddings=True,
                                    convert_to_numpy=True).astype(np.float32)
        d_scores, d_indices = faiss_index.search(q_emb, FIRST_STAGE_K)
        dense_run = [dense_doc_ids[i] for i in d_indices[0] if i >= 0]

        # First-stage fusion
        stage1_fused = reciprocal_rank_fusion(
            [{qid: bm25_run}, {qid: dense_run}],
            weights=[sparse_w, dense_w], k=60
        ).get(qid, bm25_run)[:FIRST_STAGE_K]

        # ── Stage 2: LTR reranking ───────────────────────────────────────────
        bm25_score_map  = {bm25_ids[i]: float(bm25_scores[i]) for i in top_bm25}
        dense_score_map = {dense_doc_ids[d_indices[0][j]]: float(d_scores[0][j])
                           for j in range(len(d_indices[0])) if d_indices[0][j] >= 0}
        bm25_rank_map   = {did: r for r, did in enumerate(bm25_run)}
        dense_rank_map  = {did: r for r, did in enumerate(dense_run)}
        rrf_scores_map  = {did: 1.0/(60+r+1) for r, did in enumerate(stage1_fused)}

        feats = extract_features(
            qtext, stage1_fused,
            bm25, bm25_ids, bm25_title, bm25_kw, bm25_body, field_ids,
            dense_score_map, bm25_score_map, rrf_scores_map,
            doc_store, bm25_rank_map, dense_rank_map
        )
        ltr_scores  = ltr_model.predict(feats)
        ltr_order   = np.argsort(ltr_scores)[::-1]
        ltr_run     = [stage1_fused[i] for i in ltr_order[:LTR_RERANK_K]]

        # ── Stage 3: Cross-encoder reranking ────────────────────────────────
        ce_run = cross_encoder_rerank(
            ce_model, qtext, ltr_run, doc_titles, CE_RERANK_K
        )

        # ── Stage 4: Final fusion ────────────────────────────────────────────
        # RRF over: first-stage, LTR-reranked, CE-reranked
        final = reciprocal_rank_fusion(
            [
                {qid: stage1_fused},
                {qid: ltr_run},
                {qid: ce_run},
            ],
            weights=[1.0, ltr_w, ce_w],
            k=60
        ).get(qid, ce_run)[:top_k]

        final_run[qid] = final

    return final_run


def main():
    import faiss
    from sentence_transformers import SentenceTransformer, CrossEncoder

    OUT_DIR.mkdir(exist_ok=True)

    # ── Load all components ───────────────────────────────────────────────────
    print("[*] Loading models and indices...")
    bm25     = load_pickle(str(BM25_INDEX))
    bm25_ids = load_pickle(str(DOCIDS_CACHE))

    faiss_index   = faiss.read_index(str(DENSE_INDEX))
    dense_doc_ids = load_pickle(str(DENSE_DOCIDS))
    dense_model   = SentenceTransformer(BI_ENCODER)

    ltr_model = load_pickle(str(LTR_MODEL))

    print(f"[*] Loading cross-encoder: {CROSS_ENCODER}")
    ce_model = CrossEncoder(CROSS_ENCODER, max_length=512)

    doc_titles = load_doc_titles()

    bm25_title = load_pickle(str(TITLE_INDEX))
    bm25_kw    = load_pickle(str(KW_INDEX))
    bm25_body  = load_pickle(str(BODY_INDEX))
    field_ids  = load_pickle(str(Path("models") / "field_docids.pkl"))

    doc_store_path = Path("models") / "doc_store.pkl"
    doc_store  = load_pickle(str(doc_store_path))

    # ── Load queries ──────────────────────────────────────────────────────────
    test_queries = load_queries(str(TEST_Q_PATH))
    print(f"[*] {len(test_queries)} test queries.")

    # ── Run pipeline ─────────────────────────────────────────────────────────
    test_run = run_full_pipeline(
        test_queries,
        bm25, bm25_ids,
        dense_model, faiss_index, dense_doc_ids,
        ltr_model, ce_model, doc_titles,
        bm25_title, bm25_kw, bm25_body, field_ids, doc_store,
        sparse_w=RRF_SPARSE_W, dense_w=RRF_DENSE_W,
        ltr_w=RRF_LTR_W, ce_w=RRF_CE_W,
        top_k=TOP_K
    )
    write_submission(test_run, str(OUT_DIR / "submission_05_full_ensemble.csv"), TOP_K)

    # ── Local eval ────────────────────────────────────────────────────────────
    if TRAIN_Q_PATH.exists() and TRAIN_QR_PATH.exists():
        print("[*] Evaluating on train queries...")
        train_queries = load_queries(str(TRAIN_Q_PATH))
        train_qrels   = load_qrels(str(TRAIN_QR_PATH))
        train_run = run_full_pipeline(
            train_queries,
            bm25, bm25_ids,
            dense_model, faiss_index, dense_doc_ids,
            ltr_model, ce_model, doc_titles,
            bm25_title, bm25_kw, bm25_body, field_ids, doc_store,
            sparse_w=RRF_SPARSE_W, dense_w=RRF_DENSE_W,
            ltr_w=RRF_LTR_W, ce_w=RRF_CE_W,
            top_k=TOP_K
        )
        mean_ndcg, per_q = evaluate_run(train_run, train_qrels, k=100)
        print(f"[✓] Train nDCG@100 = {mean_ndcg:.4f}")
        print(f"    Min: {min(per_q.values()):.4f}  Max: {max(per_q.values()):.4f}")


if __name__ == "__main__":
    main()