import json
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .data import dense_text, iter_corpus
from .provenance import expected_metadata, metadata_matches, write_metadata


def visible_devices():
    value = os.environ.get("CUDA_VISIBLE_DEVICES")
    if value:
        return [f"cuda:{index}" for index, _ in enumerate(value.split(","))]
    try:
        import torch

        return [f"cuda:{index}" for index in range(torch.cuda.device_count())]
    except ImportError:
        return []


def load_encoder(model_name, use_fp16=True):
    from sentence_transformers import SentenceTransformer

    device = "cuda:0" if visible_devices() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    if use_fp16 and device.startswith("cuda"):
        model.half()
    print(f"Dense encoder device: {device}")
    return model


def _encode(model, texts, config, pool=None):
    if pool is not None:
        return model.encode_multi_process(
            texts,
            pool,
            batch_size=config["dense"]["batch_size_per_gpu"],
            normalize_embeddings=config["dense"]["normalize"],
        )
    return model.encode(
        texts,
        batch_size=config["dense"]["batch_size_per_gpu"],
        normalize_embeddings=config["dense"]["normalize"],
        convert_to_numpy=True,
        show_progress_bar=False,
    )


def encode_corpus(config):
    work_dir = config["paths"]["work_dir"]
    shard_dir = work_dir / "dense_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = work_dir / "dense_manifest.json"
    metadata_path = work_dir / "dense_metadata.json"
    metadata = expected_metadata(config, "dense")
    if manifest_path.exists() and not metadata_matches(metadata_path, metadata):
        raise RuntimeError(
            f"Dense artifact provenance mismatch. Remove {shard_dir}, "
            f"{manifest_path}, and {metadata_path}, then rerun encode."
        )

    model = load_encoder(
        config["dense"]["model"], config["dense"]["use_fp16"]
    )
    devices = visible_devices()
    pool = None
    if len(devices) > 1:
        pool = model.start_multi_process_pool(target_devices=devices)
        print(f"Using {len(devices)} GPUs: {devices}")

    shard_size = config["dense"]["shard_size"]
    texts, doc_ids = [], []
    shard_number = 0
    manifest = []
    existing_ids = {}
    for path in shard_dir.glob("docids_*.json"):
        existing_ids[path.stem.split("_")[-1]] = json.loads(
            path.read_text(encoding="utf-8")
        )

    def flush():
        nonlocal shard_number
        if not texts:
            return
        embedding_path = shard_dir / f"embeddings_{shard_number:05d}.npy"
        ids_path = shard_dir / f"docids_{shard_number:05d}.json"
        if embedding_path.exists() and ids_path.exists():
            saved_ids = existing_ids.get(f"{shard_number:05d}")
            if saved_ids != doc_ids:
                raise RuntimeError(f"Existing shard does not match corpus: {ids_path}")
            embeddings = np.load(embedding_path, mmap_mode="r")
        else:
            embeddings = _encode(model, texts, config, pool).astype(np.float16)
            np.save(embedding_path, embeddings)
            ids_path.write_text(json.dumps(doc_ids), encoding="utf-8")
        manifest.append(
            {
                "embeddings": str(embedding_path),
                "docids": str(ids_path),
                "count": len(doc_ids),
                "dimension": int(embeddings.shape[1]),
            }
        )
        shard_number += 1
        texts.clear()
        doc_ids.clear()

    try:
        for doc in tqdm(
            iter_corpus(config["paths"]["corpus"]),
            desc="BGE-M3 corpus encoding",
            unit=" docs",
        ):
            doc_ids.append(doc["id"])
            texts.append(dense_text(doc, config))
            if len(texts) >= shard_size:
                flush()
        flush()
    finally:
        if pool is not None:
            model.stop_multi_process_pool(pool)

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_metadata(metadata_path, metadata)
    print(f"Wrote dense manifest: {manifest_path}")


def build_faiss_index(config):
    import faiss

    work_dir = config["paths"]["work_dir"]
    manifest = json.loads((work_dir / "dense_manifest.json").read_text())
    metadata_path = work_dir / "dense_metadata.json"
    metadata = expected_metadata(config, "dense")
    if not metadata_matches(metadata_path, metadata):
        raise RuntimeError("Dense embeddings do not match the configured corpus/model.")
    index_path = work_dir / "bge_m3.faiss"
    docids_path = work_dir / "bge_m3_docids.json"
    index_metadata_path = work_dir / "faiss_metadata.json"
    if index_path.exists() and docids_path.exists():
        if metadata_matches(index_metadata_path, metadata):
            print(f"FAISS index already exists: {index_path}")
            return
        raise RuntimeError(
            f"FAISS provenance mismatch. Remove {index_path}, {docids_path}, "
            f"and {index_metadata_path}, then rerun index."
        )

    dimension = manifest[0]["dimension"]
    factory = config["dense"]["faiss_factory"]
    index = faiss.index_factory(dimension, factory, faiss.METRIC_INNER_PRODUCT)

    if not index.is_trained:
        sample_target = config["dense"]["train_vectors"]
        per_shard = max(1, sample_target // len(manifest))
        samples = []
        for shard in manifest:
            values = np.load(shard["embeddings"], mmap_mode="r")
            step = max(1, len(values) // per_shard)
            samples.append(np.asarray(values[::step][:per_shard], dtype=np.float32))
        training = np.concatenate(samples)
        print(f"Training FAISS {factory} with {len(training):,} vectors")
        index.train(training)

    all_doc_ids = []
    for shard in tqdm(manifest, desc="Adding FAISS shards"):
        embeddings = np.load(shard["embeddings"], mmap_mode="r")
        index.add(np.asarray(embeddings, dtype=np.float32))
        all_doc_ids.extend(json.loads(Path(shard["docids"]).read_text()))

    if hasattr(index, "nprobe"):
        index.nprobe = config["dense"]["faiss_nprobe"]
    faiss.write_index(index, str(index_path))
    docids_path.write_text(json.dumps(all_doc_ids), encoding="utf-8")
    write_metadata(index_metadata_path, metadata)
    print(f"Wrote FAISS index with {index.ntotal:,} vectors")


class DenseSearcher:
    def __init__(self, config):
        import faiss

        work_dir = config["paths"]["work_dir"]
        self.config = config
        self.model = load_encoder(
            config["dense"]["model"], config["dense"]["use_fp16"]
        )
        self.index = faiss.read_index(str(work_dir / "bge_m3.faiss"))
        if hasattr(self.index, "nprobe"):
            self.index.nprobe = config["dense"]["faiss_nprobe"]
        self.doc_ids = json.loads((work_dir / "bge_m3_docids.json").read_text())

    def search(self, queries, k):
        texts = [text for _, text in queries]
        embeddings = self.model.encode(
            texts,
            batch_size=self.config["dense"]["query_batch_size"],
            normalize_embeddings=self.config["dense"]["normalize"],
            convert_to_numpy=True,
            show_progress_bar=True,
        ).astype(np.float32)
        scores, indices = self.index.search(embeddings, k)
        run, score_run = {}, {}
        for row, (qid, _) in enumerate(queries):
            valid = [(int(index), float(score)) for index, score in zip(indices[row], scores[row]) if index >= 0]
            run[qid] = [self.doc_ids[index] for index, _ in valid]
            score_run[qid] = {self.doc_ids[index]: score for index, score in valid}
        return run, score_run
