"""
evaluator.py — Analyze submission quality against training judgments.

Usage:
  python src/evaluator.py submissions/submission_01_bm25_baseline.csv
  python src/evaluator.py submissions/sub1.csv submissions/sub2.csv  # compare
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

from utils import load_queries, load_qrels, load_submission, evaluate_run, ndcg_at_k

DATA_DIR      = Path("data")
TRAIN_Q_PATH  = DATA_DIR / "train_queries.csv"
TRAIN_QR_PATH = DATA_DIR / "train_qrels.csv"


def recall_at_k(ranking, qrels, k):
    relevant = set(d for d, r in qrels.items() if r > 0)
    if not relevant:
        return 0.0
    retrieved = set(ranking[:k])
    return len(relevant & retrieved) / len(relevant)


def precision_at_k(ranking, qrels, k):
    relevant = set(d for d, r in qrels.items() if r > 0)
    return sum(1 for d in ranking[:k] if d in relevant) / k


def mrr(ranking, qrels):
    relevant = set(d for d, r in qrels.items() if r > 0)
    for i, d in enumerate(ranking):
        if d in relevant:
            return 1.0 / (i + 1)
    return 0.0


def analyze_submission(run_path: str, qrels_path: str, queries_path: str):
    run     = load_submission(run_path)
    qrels   = load_qrels(qrels_path)
    queries = dict(load_queries(queries_path))

    print(f"\n{'='*60}")
    print(f"Submission: {run_path}")
    print(f"{'='*60}")
    print(f"Queries in run:    {len(run)}")
    print(f"Queries with qrels:{len(qrels)}")

    rows = []
    for qid in sorted(qrels.keys()):
        if qid not in run:
            continue
        ranking = run[qid]
        q_qrels = qrels[qid]
        rows.append({
            "qid":      qid,
            "query":    queries.get(qid, "?")[:50],
            "ndcg100":  ndcg_at_k(ranking, q_qrels, 100),
            "recall100":recall_at_k(ranking, q_qrels, 100),
            "p10":      precision_at_k(ranking, q_qrels, 10),
            "mrr":      mrr(ranking, q_qrels),
            "n_rel":    sum(1 for r in q_qrels.values() if r > 0),
            "n_retrieved": len(ranking),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        print("No overlap between run and qrels!")
        return

    print(f"\n{'─'*60}")
    print("Aggregate Metrics:")
    print(f"  nDCG@100:   {df['ndcg100'].mean():.4f}  (±{df['ndcg100'].std():.4f})")
    print(f"  Recall@100: {df['recall100'].mean():.4f}  (±{df['recall100'].std():.4f})")
    print(f"  P@10:       {df['p10'].mean():.4f}  (±{df['p10'].std():.4f})")
    print(f"  MRR:        {df['mrr'].mean():.4f}  (±{df['mrr'].std():.4f})")

    print(f"\n{'─'*60}")
    print("Worst 10 queries (by nDCG@100):")
    worst = df.nsmallest(10, "ndcg100")[["qid","query","ndcg100","recall100","n_rel"]]
    print(worst.to_string(index=False))

    print(f"\n{'─'*60}")
    print("Best 10 queries:")
    best = df.nlargest(10, "ndcg100")[["qid","query","ndcg100","recall100","n_rel"]]
    print(best.to_string(index=False))

    # Queries with zero nDCG — nothing retrieved that was relevant
    zero = df[df["ndcg100"] == 0.0]
    print(f"\n{'─'*60}")
    print(f"Queries with nDCG=0: {len(zero)}/{len(df)}")
    if len(zero) > 0:
        print(zero[["qid","query","n_rel"]].to_string(index=False))

    return df


def compare_submissions(paths, qrels_path, queries_path):
    qrels   = load_qrels(qrels_path)
    queries = dict(load_queries(queries_path))

    dfs = {}
    for path in paths:
        run = load_submission(path)
        rows = []
        for qid in sorted(qrels.keys()):
            if qid not in run:
                continue
            rows.append({
                "qid":     qid,
                "ndcg100": ndcg_at_k(run[qid], qrels[qid], 100),
            })
        dfs[Path(path).name] = pd.DataFrame(rows).set_index("qid")["ndcg100"]

    comparison = pd.DataFrame(dfs)
    print("\n=== Submission Comparison (nDCG@100) ===")
    print(comparison.describe().round(4))
    print("\nMean nDCG@100 per submission:")
    for col in comparison.columns:
        print(f"  {col}: {comparison[col].mean():.4f}")

    # Per-query winner
    comparison["best"] = comparison.idxmax(axis=1)
    print("\nQuery-level wins:")
    print(comparison["best"].value_counts())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("submissions", nargs="+", help="CSV submission file(s)")
    parser.add_argument("--qrels",   default=str(TRAIN_QR_PATH))
    parser.add_argument("--queries", default=str(TRAIN_Q_PATH))
    args = parser.parse_args()

    if len(args.submissions) == 1:
        analyze_submission(args.submissions[0], args.qrels, args.queries)
    else:
        for path in args.submissions:
            analyze_submission(path, args.qrels, args.queries)
        compare_submissions(args.submissions, args.qrels, args.queries)


if __name__ == "__main__":
    main()