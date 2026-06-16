"""
Submission 3: Hybrid Retrieval — BM25 + Dense (FAISS) + RRF Fusion
===================================================================
Hypothesis: Sparse BM25 misses semantically relevant entities when surface
forms differ from query terms. Dense retrieval with a pretrained bi-encoder
captures semantic similarity. Fusing both with Reciprocal Rank Fusion (RRF)
is consistently better than either alone (proven in BEIR benchmark).

Dense model: sentence-transformers/all-MiniLM-L6-v2
  - Fast (384-dim embeddings), good for entity titles
  - If GPU available, use "intfloat/e5-base-v2" for better accuracy

Indexing:
  - BM25: same as Sub1 (cached)
  - Dense: FAISS flat L2 index over entity title+keyword embeddings
    (we skip body for dense to keep memory <16GB: title+kw ≈ 50 chars avg)

Fusion: RRF with k=60, equal weights for sparse and dense

Expected nDCG@100: ~0.43–0.48
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
import math

from utils import (
    load_corpus, load_queries, load_qrels,
    build_doc_text, tokenize,
    write_submission, evaluate_run,
    save_pickle, load_pickle, reciprocal_rank_fusion
)

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR      = Path("data")
CORPUS_PATH   = DATA_DIR / "corpus.jsonl"
TEST_Q_PATH   = DATA_DIR / "test_queries.csv"
TRAIN_Q_PATH  = DATA_DIR / "train_queries.csv"
TRAIN_QR_PATH = DATA_DIR / "train_qrels.csv"
OUT_DIR       = Path("submissions")

# BM25 cache (from Sub1)
BM25_INDEX    = Path("models") / "bm25_index.pkl"
DOCIDS_CACHE  = Path("models") / "bm25_docids.pkl"

# Dense cache
DENSE_INDEX   = Path("models") / "faiss_index.bin"
DENSE_DOCIDS  = Path("models") / "faiss_docids.pkl"

# Retrieval
TOP_K         = 100
DENSE_CANDS   = 200   # retrieve more from each system, fuse, take top 100
BM25_CANDS    = 200

# Dense model — swap to "intfloat/e5-base-v2" for +2-3 nDCG if GPU available
DENSE_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE    = 512
USE_E5_PREFIX = False   # set True if using E5 model (adds "query: " prefix)

# RRF
RRF_K         = 60
BM25_WEIGHT   = 1.0
DENSE_WEIGHT  = 1.0
# ─────────────────────────────────────────────────────────────────────────────


def build_or_load_dense_index(doc_ids_ref: List[str]):
    """Build FAISS dense index over entity title+keywords, or load cache."""
    try:
        import faiss
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("[!] Install faiss-cpu and sentence-transformers")
        raise

    if DENSE_INDEX.exists() and DENSE_DOCIDS.exists():
        print("[*] Loading cached FAISS index...")
        index = faiss.read_index(str(DENSE_INDEX))
        dense_doc_ids = load_pickle(str(DENSE_DOCIDS))
        print(f"    Loaded FAISS index for {len(dense_doc_ids):,} docs.")
        return index, dense_doc_ids

    print(f"[*] Building dense index with {DENSE_MODEL}...")
    model = SentenceTransformer(DENSE_MODEL)

    dense_doc_ids = []
    texts_buffer = []
    all_embeddings = []
    t0 = time.time()

    def flush_buffer():
        if not texts_buffer:
            return
        embs = model.encode(
            texts_buffer,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True,   # cosine via inner product
            convert_to_numpy=True
        )
        all_embeddings.append(embs)
        texts_buffer.clear()

    for i, (doc_id, title, keywords, body) in enumerate(
        tqdm(load_corpus(str(CORPUS_PATH)), desc="Encoding", unit=" docs")
    ):
        dense_doc_ids.append(doc_id)
        # For dense: only title + keywords (concise, no noise from body)
        kw_str = " ".join(keywords[:10])  # cap keywords
        text = (title + " " + kw_str).strip()[:256]
        texts_buffer.append(text)

        if len(texts_buffer) >= 10_000:
            flush_buffer()
            if i % 500_000 == 0 and i > 0:
                print(f"    {i:,} docs in {time.time()-t0:.0f}s")

    flush_buffer()

    print("[*] Stacking embeddings...")
    embeddings = np.vstack(all_embeddings).astype(np.float32)
    print(f"    Embedding matrix: {embeddings.shape}")

    print("[*] Building FAISS index...")
    import faiss
    dim = embeddings.shape[1]
    # Use IVF for large corpora (>1M), Flat for smaller
    n = len(dense_doc_ids)
    if n > 1_000_000:
        # IVFFlat: fast approximate search
        nlist = min(4096, int(math.sqrt(n)))
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        print(f"    Training IVF index (nlist={nlist})...")
        train_sample = embeddings[::max(1, n // 500_000)]  # ~500k sample
        index.train(train_sample)
        index.nprobe = 64   # search 64 clusters per query
    else:
        index = faiss.IndexFlatIP(dim)

    index.add(embeddings)
    print(f"    FAISS index has {index.ntotal:,} vectors.")

    # Save
    Path("models").mkdir(exist_ok=True)
    faiss.write_index(index, str(DENSE_INDEX))
    save_pickle(dense_doc_ids, str(DENSE_DOCIDS))
    print(f"[✓] Dense index built in {time.time()-t0:.0f}s")
    return index, dense_doc_ids


def dense_retrieve(
    model,
    faiss_index,
    dense_doc_ids: List[str],
    queries: List[Tuple[str, str]],
    top_k: int = 200
) -> Dict[str, List[str]]:
    """Dense retrieval for all queries."""
    run = {}
    qids = [q[0] for q in queries]
    qtexts = [q[1] for q in queries]

    if USE_E5_PREFIX:
        qtexts = ["query: " + t for t in qtexts]

    print("[*] Encoding queries (dense)...")
    from sentence_transformers import SentenceTransformer
    q_embs = model.encode(
        qtexts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype(np.float32)

    print("[*] FAISS search...")
    scores, indices = faiss_index.search(q_embs, top_k)

    for i, qid in enumerate(qids):
        run[qid] = [dense_doc_ids[idx] for idx in indices[i] if idx >= 0]
    return run


def bm25_retrieve(bm25, doc_ids, queries, top_k=200):
    """BM25 retrieval (reusing Sub1 logic)."""
    run = {}
    for qid, qtext in tqdm(queries, desc="BM25 retrieval"):
        tokens = tokenize(qtext)
        scores = bm25.get_scores(tokens)
        top_idx = np.argpartition(scores, -top_k)[-top_k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        run[qid] = [doc_ids[i] for i in top_idx]
    return run


def main():
    import math
    OUT_DIR.mkdir(exist_ok=True)

    # ── Load BM25 index ──
    print("[*] Loading BM25 index...")
    bm25 = load_pickle(str(BM25_INDEX))
    bm25_docids = load_pickle(str(DOCIDS_CACHE))

    # ── Load/build dense index ──
    faiss_index, dense_doc_ids = build_or_load_dense_index(bm25_docids)

    from sentence_transformers import SentenceTransformer
    dense_model = SentenceTransformer(DENSE_MODEL)

    # ── Load queries ──
    test_queries  = load_queries(str(TEST_Q_PATH))
    print(f"[*] {len(test_queries)} test queries.")

    # ── Retrieve (both systems) ──
    bm25_run  = bm25_retrieve(bm25, bm25_docids, test_queries, BM25_CANDS)
    dense_run = dense_retrieve(dense_model, faiss_index, dense_doc_ids,
                                test_queries, DENSE_CANDS)

    # ── Fuse ──
    print("[*] Fusing with RRF...")
    fused_run = reciprocal_rank_fusion(
        [bm25_run, dense_run],
        weights=[BM25_WEIGHT, DENSE_WEIGHT],
        k=RRF_K
    )
    # Trim to top 100
    fused_run = {qid: docs[:TOP_K] for qid, docs in fused_run.items()}

    write_submission(fused_run, str(OUT_DIR / "submission_03_hybrid_rrf.csv"), TOP_K)

    # ── Local eval ──
    if TRAIN_Q_PATH.exists() and TRAIN_QR_PATH.exists():
        print("[*] Evaluating on train queries...")
        train_queries = load_queries(str(TRAIN_Q_PATH))
        train_qrels   = load_qrels(str(TRAIN_QR_PATH))

        tr_bm25  = bm25_retrieve(bm25, bm25_docids, train_queries, BM25_CANDS)
        tr_dense = dense_retrieve(dense_model, faiss_index, dense_doc_ids,
                                   train_queries, DENSE_CANDS)
        tr_fused = reciprocal_rank_fusion(
            [tr_bm25, tr_dense],
            weights=[BM25_WEIGHT, DENSE_WEIGHT], k=RRF_K)
        tr_fused = {qid: docs[:TOP_K] for qid, docs in tr_fused.items()}

        mean_ndcg, _ = evaluate_run(tr_fused, train_qrels, k=100)
        print(f"[✓] Train nDCG@100 = {mean_ndcg:.4f}")

        # Compare components
        bm25_ndcg, _ = evaluate_run(
            {qid: docs[:TOP_K] for qid, docs in tr_bm25.items()}, train_qrels)
        dense_ndcg, _ = evaluate_run(
            {qid: docs[:TOP_K] for qid, docs in tr_dense.items()}, train_qrels)
        print(f"    BM25-only:  {bm25_ndcg:.4f}")
        print(f"    Dense-only: {dense_ndcg:.4f}")
        print(f"    Fused:      {mean_ndcg:.4f}")


if __name__ == "__main__":
    main()