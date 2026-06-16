"""
Submission 2: BM25 + Query Expansion (PRF + WordNet)
=====================================================
Hypothesis: Entity queries like "electricity source in france" are short and
ambiguous. Expanding them with (a) Pseudo-Relevance Feedback terms from top-k
BM25 results and (b) WordNet synonyms for content words improves recall for
entities not directly matching the original tokens.

Indexing:  Same BM25 as Sub1 (reuses cache)
Expansion: 
  - RM3-style PRF: take top-10 docs from initial BM25, extract top-15
    discriminative terms by P(t|R) * log(P(t|R)/P(t|C)), interpolate
    with original query (α=0.7 original, 0.3 expansion)
  - WordNet: add 1 synonym per content word (if unambiguous)
Ranking:   BM25 on expanded query, top 100

Expected nDCG@100: ~0.38–0.43
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import math
import time
import re
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter, defaultdict
from tqdm import tqdm

import numpy as np
import nltk
from nltk.corpus import wordnet

from rank_bm25 import BM25Okapi
from utils import (
    load_corpus, load_queries, load_qrels,
    build_doc_text, tokenize,
    write_submission, evaluate_run, save_pickle, load_pickle
)

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR      = Path("data")
CORPUS_PATH   = DATA_DIR / "corpus.jsonl"
TEST_Q_PATH   = DATA_DIR / "test_queries.csv"
TRAIN_Q_PATH  = DATA_DIR / "train_queries.csv"
TRAIN_QR_PATH = DATA_DIR / "train_qrels.csv"
OUT_DIR       = Path("submissions")
INDEX_CACHE   = Path("models") / "bm25_index.pkl"
DOCIDS_CACHE  = Path("models") / "bm25_docids.pkl"

TOP_K         = 100
PRF_DOCS      = 10    # pseudo-relevant docs for feedback
PRF_TERMS     = 15    # expansion terms to add
RM3_ALPHA     = 0.70  # weight for original query (1-alpha for expansion)
USE_WORDNET   = True
STEM          = False
REMOVE_STOPS  = False
# ─────────────────────────────────────────────────────────────────────────────


def load_index():
    """Load cached BM25 index (built by submission 1)."""
    print("[*] Loading BM25 index from cache...")
    bm25 = load_pickle(str(INDEX_CACHE))
    doc_ids = load_pickle(str(DOCIDS_CACHE))
    print(f"    Loaded index for {len(doc_ids):,} documents.")
    return bm25, doc_ids


def estimate_collection_probs(bm25: BM25Okapi) -> Dict[str, float]:
    """
    Estimate P(t|Collection) for each term using BM25's internal doc_freqs.
    """
    total = sum(bm25.doc_len)
    col_prob = {}
    for term, freq in bm25.idf.items():
        # approximate with doc frequency / total docs
        df = bm25.doc_freq.get(term, 0) if hasattr(bm25, 'doc_freq') else 1
        col_prob[term] = df / len(bm25.doc_len)
    return col_prob


def rm3_expand(
    query_tokens: List[str],
    bm25: BM25Okapi,
    doc_ids: List[str],
    col_prob: Dict[str, float],
    prf_docs: int = 10,
    prf_terms: int = 15,
    alpha: float = 0.7
) -> List[str]:
    """
    RM3-style query expansion.
    1. Retrieve top prf_docs docs with original query
    2. Score terms in those docs by P(t|R) * log(P(t|R) / P(t|C))
    3. Take top prf_terms expansion terms
    4. Merge with original query: alpha * original + (1-alpha) * expansion
    Returns a new token list (weighted by repetition for BM25 compatibility).
    """
    # Initial retrieval
    scores = bm25.get_scores(query_tokens)
    top_idx = np.argpartition(scores, -prf_docs)[-prf_docs:]
    top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

    # Count terms in top docs
    term_counts: Counter = Counter()
    total_terms = 0
    for idx in top_idx:
        doc_tokens = bm25.corpus[idx] if hasattr(bm25, 'corpus') else []
        # rank-bm25 stores tokenized corpus
        for tok in doc_tokens:
            term_counts[tok] += 1
            total_terms += 1

    if total_terms == 0:
        return query_tokens

    # Score terms
    term_scores = {}
    for term, cnt in term_counts.items():
        p_t_R = cnt / total_terms
        p_t_C = col_prob.get(term, 1e-10)
        if p_t_C > 0:
            term_scores[term] = p_t_R * math.log(p_t_R / p_t_C + 1e-10)

    # Exclude original query terms from pure expansion set
    orig_set = set(query_tokens)
    expansion_terms = [
        t for t, _ in sorted(term_scores.items(), key=lambda x: -x[1])
        if t not in orig_set
    ][:prf_terms]

    # Weighted merge via token repetition (crude but BM25-compatible)
    orig_reps  = max(1, round(alpha * 10))
    expan_reps = max(1, round((1 - alpha) * 10))
    expanded = (query_tokens * orig_reps) + (expansion_terms * expan_reps)
    return expanded


def wordnet_expand(query_tokens: List[str]) -> List[str]:
    """
    Add one synonym per content word using WordNet.
    Only adds when there's a single dominant synset (low ambiguity).
    """
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    extra = []
    for tok in query_tokens:
        synsets = wordnet.synsets(tok, pos=wordnet.NOUN)
        if not synsets:
            synsets = wordnet.synsets(tok, pos=wordnet.VERB)
        if len(synsets) == 1:  # unambiguous
            lemmas = synsets[0].lemma_names()
            for lem in lemmas:
                clean = lem.replace("_", " ").lower()
                if clean != tok and " " not in clean:
                    extra.append(clean)
                    break
    return extra


def retrieve_with_expansion(
    bm25: BM25Okapi,
    doc_ids: List[str],
    queries: List[Tuple[str, str]],
    col_prob: Dict[str, float],
    top_k: int = 100
) -> Dict[str, List[str]]:
    run = {}
    for qid, qtext in tqdm(queries, desc="Retrieving+Expanding"):
        qtokens = tokenize(qtext, stem=STEM, remove_stops=REMOVE_STOPS)

        # WordNet expansion
        if USE_WORDNET:
            wn_extra = wordnet_expand(qtokens)
            qtokens = qtokens + wn_extra

        # RM3 PRF expansion
        expanded_tokens = rm3_expand(
            qtokens, bm25, doc_ids, col_prob,
            prf_docs=PRF_DOCS, prf_terms=PRF_TERMS, alpha=RM3_ALPHA
        )

        scores = bm25.get_scores(expanded_tokens)
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        run[qid] = [doc_ids[i] for i in top_indices]
    return run


def main():
    OUT_DIR.mkdir(exist_ok=True)

    bm25, doc_ids = load_index()

    # Pre-compute collection language model
    print("[*] Estimating collection probabilities...")
    col_prob = estimate_collection_probs(bm25)

    # Test queries
    test_queries = load_queries(str(TEST_Q_PATH))
    print(f"[*] {len(test_queries)} test queries.")

    print("[*] Retrieving with query expansion...")
    test_run = retrieve_with_expansion(bm25, doc_ids, test_queries, col_prob, TOP_K)
    write_submission(test_run, str(OUT_DIR / "submission_02_bm25_qe.csv"), TOP_K)

    # Local eval
    if TRAIN_Q_PATH.exists() and TRAIN_QR_PATH.exists():
        print("[*] Evaluating on train queries...")
        train_queries = load_queries(str(TRAIN_Q_PATH))
        train_qrels   = load_qrels(str(TRAIN_QR_PATH))
        train_run     = retrieve_with_expansion(
            bm25, doc_ids, train_queries, col_prob, TOP_K)
        mean_ndcg, _ = evaluate_run(train_run, train_qrels, k=100)
        print(f"[✓] Train nDCG@100 = {mean_ndcg:.4f}")


if __name__ == "__main__":
    main()