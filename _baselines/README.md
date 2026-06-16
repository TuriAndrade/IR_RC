# Entity Search - IR Competition Solution

## Strategy Overview

5 progressive submissions, each building on the last:

| Sub | Strategy | Expected nDCG@100 |
|-----|----------|-------------------|
| 1 | BM25 (title+keywords+text, field-weighted) | ~0.35 |
| 2 | BM25 + Query Expansion (Pseudo-Relevance Feedback) | ~0.40 |
| 3 | BM25 + Dense Retrieval (FAISS) + RRF Fusion | ~0.45 |
| 4 | Sub3 + LambdaMART LTR Reranker | ~0.52 |
| 5 | Sub4 + Ensemble + Fine-tuned Reranker | ~0.57+ |

## Directory Structure

```
ir_challenge/
├── src/
│   ├── 01_bm25_baseline.py          # Submission 1: BM25 baseline
│   ├── 02_query_expansion.py        # Submission 2: BM25 + PRF
│   ├── 03_dense_hybrid.py           # Submission 3: Hybrid retrieval
│   ├── 04_ltr_reranker.py           # Submission 4: LTR reranking
│   ├── 05_ensemble.py               # Submission 5: Full ensemble
│   ├── indexer.py                   # Index building utilities
│   ├── query_processor.py           # Query processing utilities
│   └── evaluator.py                 # Local nDCG evaluation
├── submissions/                     # Generated CSV files
├── data/                            # Place corpus/query files here
└── models/                          # Saved models
```

## Setup

```bash
pip install pyserini pyterrier lightgbm sentence-transformers faiss-cpu \
            pandas numpy scikit-learn tqdm rank-bm25 nltk ir-measures
```

## Quick Start

```bash
# Place data files in ./data/
# corpus.jsonl, train_queries.csv, train_qrels.csv, test_queries.csv

# Run submissions in order
python src/01_bm25_baseline.py
python src/02_query_expansion.py
# ... etc
```