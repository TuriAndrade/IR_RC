"""
indexer_pyserini.py — Optional: Build Lucene index with Pyserini
================================================================
Use this instead of rank-bm25 if you have Java 11+ installed.
Pyserini's BM25 is much faster at query time on 4.6M docs.

Pyserini advantages over rank-bm25:
  - Lucene's BM25 is optimized with inverted index (O(|posting| per query)
    vs rank-bm25's O(N) naive scan)
  - Supports BM25F (field-weighted BM25 natively)
  - Much lower memory footprint at inference time

Usage:
  python src/indexer_pyserini.py --build   # builds the Lucene index
  python src/indexer_pyserini.py --search  # searches with Pyserini BM25
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import json
import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm

from utils import load_corpus, load_queries, load_qrels, write_submission, evaluate_run

DATA_DIR       = Path("data")
CORPUS_PATH    = DATA_DIR / "corpus.jsonl"
TEST_Q_PATH    = DATA_DIR / "test_queries.csv"
TRAIN_Q_PATH   = DATA_DIR / "train_queries.csv"
TRAIN_QR_PATH  = DATA_DIR / "train_qrels.csv"
OUT_DIR        = Path("submissions")

LUCENE_INDEX   = Path("models") / "lucene_index"
PYSERINI_CORPUS= Path("models") / "pyserini_corpus"  # Pyserini JSONL format

TOP_K = 100


def build_pyserini_corpus():
    """Convert corpus.jsonl to Pyserini-compatible format."""
    PYSERINI_CORPUS.mkdir(parents=True, exist_ok=True)
    out_path = PYSERINI_CORPUS / "corpus.jsonl"

    if out_path.exists():
        print(f"[*] Pyserini corpus already exists at {out_path}")
        return

    print("[*] Converting corpus to Pyserini format...")
    with open(out_path, "w", encoding="utf-8") as out:
        for doc_id, title, keywords, body in tqdm(
            load_corpus(str(CORPUS_PATH)), desc="Converting"
        ):
            kw_str = " ".join(keywords)
            # Pyserini expects: {"id": ..., "contents": ...}
            # Weighted concatenation for field boosting
            contents = (
                title + " " + title + " " + title + " "  # title ×3
                + kw_str + " " + kw_str + " "            # keywords ×2
                + body                                     # body ×1
            )
            record = {"id": doc_id, "contents": contents.strip()}
            out.write(json.dumps(record) + "\n")

    print(f"[✓] Pyserini corpus written to {out_path}")


def build_lucene_index():
    """Build Lucene index using Pyserini's indexer."""
    try:
        from pyserini.index.lucene import LuceneIndexer
    except ImportError:
        print("[!] Pyserini not installed. Run: pip install pyserini")
        print("    Also requires Java 11+. Check: java -version")
        return

    LUCENE_INDEX.mkdir(parents=True, exist_ok=True)
    print("[*] Building Lucene index with Pyserini...")

    cmd = [
        "python", "-m", "pyserini.index.lucene",
        "--collection", "JsonCollection",
        "--input",  str(PYSERINI_CORPUS),
        "--index",  str(LUCENE_INDEX),
        "--generator", "DefaultLuceneDocumentGenerator",
        "--threads", "8",
        "--storeDocvectors",
        "--storeRaw",
    ]
    print(f"    Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[!] Indexing failed:\n{result.stderr}")
    else:
        print("[✓] Lucene index built.")


def search_pyserini(queries: List[Tuple[str, str]], top_k: int = 100) -> Dict[str, List[str]]:
    """Search the Lucene index with Pyserini BM25."""
    try:
        from pyserini.search.lucene import LuceneSearcher
    except ImportError:
        print("[!] Pyserini not available")
        return {}

    searcher = LuceneSearcher(str(LUCENE_INDEX))
    # BM25 parameters — tuned for entity search
    # k1=1.2 (term saturation), b=0.75 (length norm)
    # Try k1=0.9, b=0.4 for shorter entity-title focused queries
    searcher.set_bm25(k1=1.2, b=0.75)

    run = {}
    for qid, qtext in tqdm(queries, desc="Pyserini BM25"):
        hits = searcher.search(qtext, k=top_k)
        run[qid] = [hit.docid for hit in hits]
    return run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build",  action="store_true", help="Build Lucene index")
    parser.add_argument("--search", action="store_true", help="Search and generate submission")
    args = parser.parse_args()

    if args.build:
        build_pyserini_corpus()
        build_lucene_index()

    if args.search:
        OUT_DIR.mkdir(exist_ok=True)
        test_queries = load_queries(str(TEST_Q_PATH))
        run = search_pyserini(test_queries, TOP_K)
        write_submission(run, str(OUT_DIR / "submission_01b_pyserini_bm25.csv"), TOP_K)

        if TRAIN_Q_PATH.exists() and TRAIN_QR_PATH.exists():
            train_queries = load_queries(str(TRAIN_Q_PATH))
            train_qrels   = load_qrels(str(TRAIN_QR_PATH))
            train_run     = search_pyserini(train_queries, TOP_K)
            ndcg, _       = evaluate_run(train_run, train_qrels, k=100)
            print(f"[✓] Train nDCG@100 (Pyserini BM25) = {ndcg:.4f}")


if __name__ == "__main__":
    main()