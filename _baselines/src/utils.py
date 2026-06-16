"""
Shared utilities: corpus loading, query processing, nDCG evaluation.
Used by all 5 submissions.
"""
import json
import csv
import math
import re
import os
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────

def load_corpus(path: str, limit: Optional[int] = None):
    """
    Yields (doc_id_str, title, keywords_list, body_text) tuples.
    Handles both 'id' and '_id' field names defensively.
    """
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            doc = json.loads(line)
            doc_id = str(doc.get("id") or doc.get("_id", ""))
            title = doc.get("title", "") or ""
            # keywords may be a list or a comma-separated string
            kw_raw = doc.get("keywords", [])
            if isinstance(kw_raw, str):
                keywords = [k.strip() for k in kw_raw.split(",") if k.strip()]
            else:
                keywords = [str(k) for k in (kw_raw or [])]
            text = doc.get("text", "") or doc.get("description", "") or ""
            yield doc_id, title, keywords, text


def load_queries(path: str) -> List[Tuple[str, str]]:
    """Returns list of (query_id, query_text) pairs."""
    queries = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = str(row.get("QueryId") or row.get("query_id") or row.get("id", ""))
            qtext = str(row.get("Query") or row.get("query") or row.get("text", ""))
            queries.append((qid, qtext))
    return queries


def load_qrels(path: str) -> Dict[str, Dict[str, int]]:
    """Returns {query_id: {doc_id: relevance}} dict."""
    qrels = defaultdict(dict)
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = str(row.get("QueryId") or row.get("query_id", ""))
            did = str(row.get("EntityId") or row.get("doc_id", ""))
            rel = int(row.get("Relevance") or row.get("relevance", 0))
            qrels[qid][did] = rel
    return dict(qrels)


# ─────────────────────────────────────────────
# 2. TEXT PREPROCESSING
# ─────────────────────────────────────────────

import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

_STEMMER = None
_STOPWORDS = None

def _ensure_nltk():
    global _STEMMER, _STOPWORDS
    if _STEMMER is None:
        nltk.download("stopwords", quiet=True)
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        _STEMMER = PorterStemmer()
        _STOPWORDS = set(stopwords.words("english"))


def tokenize(text: str, stem: bool = False, remove_stops: bool = False) -> List[str]:
    _ensure_nltk()
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    if remove_stops:
        tokens = [t for t in tokens if t not in _STOPWORDS]
    if stem:
        tokens = [_STEMMER.stem(t) for t in tokens]
    return tokens


def build_doc_text(title: str, keywords: List[str], body: str,
                   title_weight: int = 3, kw_weight: int = 2) -> str:
    """
    Create a weighted concatenated document string.
    Title is repeated title_weight times, keywords kw_weight times.
    This boosts field importance in BM25 without field-specific indexing.
    """
    kw_str = " ".join(keywords)
    parts = (
        " ".join([title] * title_weight)
        + " " + " ".join([kw_str] * kw_weight)
        + " " + body
    )
    return parts.strip()


# ─────────────────────────────────────────────
# 3. nDCG EVALUATION (local)
# ─────────────────────────────────────────────

def dcg(relevances: List[int], k: int) -> float:
    """Compute DCG@k given a list of relevance scores in rank order."""
    score = 0.0
    for i, rel in enumerate(relevances[:k]):
        score += (2 ** rel - 1) / math.log2(i + 2)
    return score


def ndcg_at_k(ranking: List[str], qrels: Dict[str, int], k: int = 100) -> float:
    """Compute nDCG@k for a single query."""
    rels = [qrels.get(doc_id, 0) for doc_id in ranking[:k]]
    ideal = sorted(qrels.values(), reverse=True)
    idcg = dcg(ideal, k)
    if idcg == 0:
        return 0.0
    return dcg(rels, k) / idcg


def evaluate_run(
    run: Dict[str, List[str]],
    qrels: Dict[str, Dict[str, int]],
    k: int = 100
) -> Tuple[float, Dict[str, float]]:
    """
    Evaluate a complete run.
    run: {query_id: [ranked doc_ids]}
    Returns (mean_ndcg, per_query_ndcg)
    """
    per_query = {}
    for qid, ranking in run.items():
        q_qrels = qrels.get(qid, {})
        per_query[qid] = ndcg_at_k(ranking, q_qrels, k)
    mean = float(np.mean(list(per_query.values()))) if per_query else 0.0
    return mean, per_query


# ─────────────────────────────────────────────
# 4. SUBMISSION I/O
# ─────────────────────────────────────────────

def write_submission(run: Dict[str, List[str]], path: str, k: int = 100):
    """Write a run to Kaggle submission CSV format."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    rows = []
    for qid, doc_ids in sorted(run.items()):
        for did in doc_ids[:k]:
            rows.append({"QueryId": qid, "EntityId": did})
    df = pd.DataFrame(rows, columns=["QueryId", "EntityId"])
    df.to_csv(path, index=False)
    print(f"[✓] Submission written to {path}  ({len(df)} lines)")


def load_submission(path: str) -> Dict[str, List[str]]:
    """Load a submission CSV back into a run dict."""
    df = pd.read_csv(path)
    run = defaultdict(list)
    for _, row in df.iterrows():
        run[str(row["QueryId"])].append(str(row["EntityId"]))
    return dict(run)


# ─────────────────────────────────────────────
# 5. RECIPROCAL RANK FUSION
# ─────────────────────────────────────────────

def reciprocal_rank_fusion(
    runs: List[Dict[str, List[str]]],
    weights: Optional[List[float]] = None,
    k: int = 60
) -> Dict[str, List[str]]:
    """
    Fuse multiple ranked lists using Reciprocal Rank Fusion.
    k=60 is the standard constant that reduces sensitivity to top ranks.
    weights: per-run multipliers (default: equal)
    """
    if weights is None:
        weights = [1.0] * len(runs)

    # Collect all query ids
    all_qids = set()
    for run in runs:
        all_qids.update(run.keys())

    fused = {}
    for qid in all_qids:
        scores: Dict[str, float] = defaultdict(float)
        for run, w in zip(runs, weights):
            if qid not in run:
                continue
            for rank, doc_id in enumerate(run[qid]):
                scores[doc_id] += w / (k + rank + 1)
        fused[qid] = sorted(scores.keys(), key=lambda d: -scores[d])
    return fused


# ─────────────────────────────────────────────
# 6. MISC HELPERS
# ─────────────────────────────────────────────

def save_pickle(obj, path: str):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=4)
    print(f"[✓] Saved {path}")


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def pad_run(run: Dict[str, List[str]], all_doc_ids: List[str], k: int = 100):
    """Pad any query ranking shorter than k with random docs (last resort)."""
    for qid in run:
        if len(run[qid]) < k:
            existing = set(run[qid])
            extras = [d for d in all_doc_ids if d not in existing]
            run[qid] = (run[qid] + extras)[:k]
    return run