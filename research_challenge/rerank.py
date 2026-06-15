import numpy as np
from tqdm import tqdm

from .data import reranker_text
from .metrics import evaluate


class BGEReranker:
    """Direct Hugging Face inference for BAAI/bge-reranker-v2-m3."""

    def __init__(self, config):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        requested = config["reranker"].get("device", "cuda:0")
        self.device = requested if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(
            config["reranker"]["model"],
            use_fast=True,
        )
        dtype = (
            torch.float16
            if self.device.startswith("cuda") and config["dense"]["use_fp16"]
            else torch.float32
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config["reranker"]["model"],
            torch_dtype=dtype,
        ).to(self.device)
        self.model.eval()
        self.config = config
        print(f"Reranker device: {self.device}")

    def _score_pairs(self, pairs):
        import torch

        output = []
        batch_size = self.config["reranker"]["batch_size"]
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            queries = [query for query, _ in batch]
            documents = [document for _, document in batch]
            encoded = self.tokenizer(
                queries,
                documents,
                padding=True,
                truncation=True,
                max_length=self.config["reranker"]["max_length"],
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.inference_mode():
                logits = self.model(**encoded).logits.view(-1).float()
                scores = torch.sigmoid(logits)
            output.extend(scores.cpu().numpy().tolist())
        return np.asarray(output, dtype=np.float32)

    def score(self, queries, candidate_run, document_store):
        scores = {}
        query_map = dict(queries)
        for qid in tqdm(candidate_run, desc="BGE reranking"):
            candidates = candidate_run[qid][: self.config["reranker"]["candidate_k"]]
            documents = document_store.fetch_many(candidates)
            valid = [doc_id for doc_id in candidates if doc_id in documents]
            pairs = [
                (query_map[qid], reranker_text(documents[doc_id], self.config))
                for doc_id in valid
            ]
            values = self._score_pairs(pairs)
            scores[qid] = {
                doc_id: float(score) for doc_id, score in zip(valid, values)
            }
        return scores


def combine_scores(candidate_run, reranker_scores, retrieval_weight, final_k):
    output = {}
    for qid, candidates in candidate_run.items():
        denominator = max(len(candidates) - 1, 1)
        values = {}
        for rank, doc_id in enumerate(candidates):
            retrieval_score = 1.0 - rank / denominator
            values[doc_id] = (
                reranker_scores.get(qid, {}).get(doc_id, 0.0)
                + retrieval_weight * retrieval_score
            )
        output[qid] = sorted(values, key=values.get, reverse=True)[:final_k]
    return output


def tune_retrieval_score(candidate_run, reranker_scores, qrels, config):
    best = (-1.0, None, None)
    for weight in config["reranker"]["retrieval_score_weight_grid"]:
        run = combine_scores(
            candidate_run,
            reranker_scores,
            weight,
            config["retrieval"]["final_k"],
        )
        score = evaluate(run, qrels, config["retrieval"]["final_k"])
        print(f"retrieval_score_weight={weight}: nDCG@100={score:.4f}")
        if score > best[0]:
            best = (score, weight, run)
    return best
