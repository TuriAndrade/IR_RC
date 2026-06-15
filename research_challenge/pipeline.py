import json

from .data import DocumentStore, load_qrels, load_queries
from .dense import DenseSearcher
from .fusion import rrf, tune_dense_weight
from .io import write_submission
from .metrics import recall_at_k
from .rerank import BGEReranker, combine_scores, tune_retrieval_score
from .sparse import SparseSearcher


def _retrieve_split(name, queries, config, sparse_searcher, dense_searcher):
    del name
    sparse_run, sparse_scores = sparse_searcher.search(
        queries, config["retrieval"]["bm25_k"]
    )
    dense_run, dense_scores = dense_searcher.search(
        queries, config["retrieval"]["dense_k"]
    )
    return sparse_run, sparse_scores, dense_run, dense_scores


def run_pipeline(config):
    required = (
        config["paths"]["work_dir"] / "documents.sqlite",
        config["paths"]["work_dir"] / "bge_m3.faiss",
        config["paths"]["work_dir"] / "bge_m3_docids.json",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing prepared artifacts. Run `python3 run.py all` or the "
            f"prepare/encode/index stages first: {missing}"
        )

    train_queries = load_queries(config["paths"]["train_queries"])
    test_queries = load_queries(config["paths"]["test_queries"])
    qrels = load_qrels(config["paths"]["train_qrels"])

    sparse_searcher = SparseSearcher(config)
    dense_searcher = DenseSearcher(config)

    train_sparse, _, train_dense, _ = _retrieve_split(
        "train", train_queries, config, sparse_searcher, dense_searcher
    )
    print("Candidate recall:")
    for name, run in (("BM25", train_sparse), ("BGE-M3", train_dense)):
        for k in (100, 200, 500, 1000):
            print(f"  {name} Recall@{k}: {recall_at_k(run, qrels, k):.4f}")

    retrieval_ndcg, dense_weight, train_candidates = tune_dense_weight(
        train_sparse, train_dense, qrels, config
    )
    print(
        f"Selected dense weight {dense_weight} "
        f"(retrieval nDCG@100={retrieval_ndcg:.4f})"
    )
    print(
        f"  Hybrid Recall@{config['retrieval']['candidate_k']}: "
        f"{recall_at_k(train_candidates, qrels, config['retrieval']['candidate_k']):.4f}"
    )

    document_store = DocumentStore(
        config["paths"]["work_dir"] / "documents.sqlite"
    )
    reranker = BGEReranker(config)
    try:
        train_reranker_scores = reranker.score(
            train_queries, train_candidates, document_store
        )
        rerank_ndcg, retrieval_weight, _ = tune_retrieval_score(
            train_candidates, train_reranker_scores, qrels, config
        )
        print(
            f"Selected retrieval score weight {retrieval_weight} "
            f"(reranked train nDCG@100={rerank_ndcg:.4f})"
        )

        test_sparse, _, test_dense, _ = _retrieve_split(
            "test", test_queries, config, sparse_searcher, dense_searcher
        )
        test_candidates = rrf(
            test_sparse,
            test_dense,
            dense_weight,
            config["retrieval"]["rrf_k"],
            config["retrieval"]["candidate_k"],
        )
        test_reranker_scores = reranker.score(
            test_queries, test_candidates, document_store
        )
        test_run = combine_scores(
            test_candidates,
            test_reranker_scores,
            retrieval_weight,
            config["retrieval"]["final_k"],
        )
    finally:
        document_store.close()

    submission_path = (
        config["paths"]["submissions_dir"] / "submission_bge_m3_reranker.csv"
    )
    write_submission(test_run, submission_path, config["retrieval"]["final_k"])
    parameters = {
        "dense_weight": dense_weight,
        "retrieval_score_weight": retrieval_weight,
        "retrieval_train_ndcg100": retrieval_ndcg,
        "reranked_train_ndcg100": rerank_ndcg,
    }
    (config["paths"]["work_dir"] / "selected_parameters.json").write_text(
        json.dumps(parameters, indent=2), encoding="utf-8"
    )
    return submission_path
