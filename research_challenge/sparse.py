import json
import shutil
import subprocess
import sys
from pathlib import Path

from tqdm import tqdm

from .data import iter_corpus
from .provenance import expected_metadata, metadata_matches, write_metadata


def build_lucene(config):
    index_path = config["paths"]["lucene_index"]
    metadata_path = index_path.parent / "lucene_metadata.json"
    metadata = expected_metadata(config, "lucene")
    if any(index_path.glob("segments_*")):
        if metadata_matches(metadata_path, metadata):
            print(f"Using project Lucene index: {index_path}")
            return
        raise RuntimeError(
            f"Lucene index provenance mismatch. Remove {index_path} and "
            f"{metadata_path}, then rerun prepare."
        )

    if index_path.exists() and any(index_path.iterdir()):
        print(f"Removing incomplete Lucene index: {index_path}")
        shutil.rmtree(index_path)

    corpus_dir = config["paths"]["work_dir"] / "lucene_corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    converted = corpus_dir / "corpus.jsonl"
    # Pyserini treats every file under --input as a collection document.
    converted_metadata = config["paths"]["work_dir"] / "lucene_corpus_metadata.json"

    if not converted.exists():
        temporary = converted.with_suffix(".jsonl.tmp")
        with temporary.open("w", encoding="utf-8") as output:
            for doc in tqdm(iter_corpus(config["paths"]["corpus"]), desc="Lucene corpus"):
                keywords = " ".join(doc["keywords"])
                contents = f"{doc['title']} {doc['title']} {doc['title']} {keywords} {keywords} {doc['text']}"
                output.write(json.dumps({"id": doc["id"], "contents": contents}) + "\n")
        temporary.replace(converted)
        write_metadata(converted_metadata, metadata)
    elif not metadata_matches(converted_metadata, metadata):
        raise RuntimeError(
            f"Lucene source corpus provenance mismatch. Remove {corpus_dir} "
            f"and {converted_metadata}, then rerun prepare."
        )

    index_path.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pyserini.index.lucene",
            "--collection",
            "JsonCollection",
            "--input",
            str(corpus_dir),
            "--index",
            str(index_path),
            "--generator",
            "DefaultLuceneDocumentGenerator",
            "--threads",
            str(config["runtime"]["num_workers"]),
            "--storeRaw",
        ],
        check=True,
    )
    if not any(index_path.glob("segments_*")):
        raise RuntimeError(f"Lucene indexing did not create a valid index: {index_path}")
    write_metadata(metadata_path, metadata)


class SparseSearcher:
    def __init__(self, config):
        from pyserini.search.lucene import LuceneSearcher

        build_lucene(config)
        self.searcher = LuceneSearcher(str(config["paths"]["lucene_index"]))
        self.searcher.set_bm25(k1=1.2, b=0.75)

    def search(self, queries, k):
        run, score_run = {}, {}
        for qid, text in tqdm(queries, desc="Lucene BM25"):
            hits = self.searcher.search(text, k=k)
            run[qid] = [hit.docid for hit in hits]
            score_run[qid] = {hit.docid: float(hit.score) for hit in hits}
        return run, score_run
