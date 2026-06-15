# Research Challenge

A modular entity-search pipeline for the `ir-20261-rc` competition:

1. Lucene BM25 retrieves lexical candidates.
2. `BAAI/bge-m3` retrieves semantic candidates using title, keywords, and body.
3. Reciprocal-rank fusion is tuned on training qrels.
4. `BAAI/bge-reranker-v2-m3` reranks the top candidates using fuller entity text.
5. A Kaggle submission is generated.

## Requirements

- Python 3.10-3.12 recommended
- A full JDK with both `java` and `javac`
- CUDA-enabled PyTorch for GPU acceleration
- Roughly 25-35 GB of free disk for embeddings, FAISS, and the document store

The project is fully independent. Data is read from `./data/`, and every
generated artifact is written under `./artifacts/`.

## Installation

```bash
cd /sonic_home/turirezende/research_challenge
conda create -n research-ir python=3.11 -y
conda activate research-ir
bash setup.sh
```

Verify GPU visibility:

```bash
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

## Run

Expose both GPUs when available:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash run_pipeline.sh
```

Expected input layout:

```text
data/
├── corpus.jsonl
├── train_queries.csv
├── train_qrels.csv
└── test_queries.csv
```

`encode` uses every visible GPU through SentenceTransformers multiprocessing.
The reranker defaults to `cuda:0` and uses direct Hugging Face batched
sequence-classification inference. Lucene and standard `faiss-cpu` indexing
remain CPU workloads.

Or run resumable stages:

```bash
python3 run.py prepare
CUDA_VISIBLE_DEVICES=0,1 python3 run.py encode
python3 run.py index
CUDA_VISIBLE_DEVICES=0,1 python3 run.py run
```

The submission is written to:

```text
submissions/submission_bge_m3_reranker.csv
```

## Resuming

Corpus embeddings are written in shards under `artifacts/dense_shards/`.
Completed shards are reused. Retrieval runs are also cached under `artifacts/`.

To rebuild an artifact, remove only that artifact and rerun its stage.

Lucene, BGE-M3 embeddings, FAISS, and the SQLite document store are built by
this project. Provenance metadata prevents artifacts from another corpus or
configuration from being silently reused.

## Important Evaluation Note

Training metrics are useful for diagnostics but optimistic because fusion and
reranker blending are tuned on the same 234 queries. For reliable model
selection, the next extension should use query-level cross-validation.
