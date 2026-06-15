import csv
import json
import sqlite3
from pathlib import Path

from tqdm import tqdm

from .provenance import expected_metadata, metadata_matches, write_metadata


def iter_corpus(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            doc = json.loads(line)
            keywords = doc.get("keywords") or []
            if isinstance(keywords, str):
                keywords = [part.strip() for part in keywords.split(",") if part.strip()]
            yield {
                "id": str(doc.get("id") or doc.get("_id") or ""),
                "title": doc.get("title") or "",
                "keywords": [str(value) for value in keywords],
                "text": doc.get("text") or doc.get("description") or "",
            }


def load_queries(path):
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            qid = str(row.get("QueryId") or row.get("query_id") or row.get("id"))
            text = str(row.get("Query") or row.get("query") or row.get("text"))
            rows.append((qid, text))
    return rows


def load_qrels(path):
    qrels = {}
    with Path(path).open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            qid = str(row.get("QueryId") or row.get("query_id"))
            doc_id = str(row.get("EntityId") or row.get("doc_id"))
            relevance = int(row.get("Relevance") or row.get("relevance") or 0)
            qrels.setdefault(qid, {})[doc_id] = relevance
    return qrels


def dense_text(doc, config):
    keywords = " ; ".join(doc["keywords"][: config["text"]["max_keywords"]])
    body = doc["text"][: config["text"]["dense_body_chars"]]
    return f"Title: {doc['title']}\nKeywords: {keywords}\nDescription: {body}"


def reranker_text(doc, config):
    keywords = " ; ".join(doc["keywords"][: config["text"]["max_keywords"]])
    body = doc["text"][: config["text"]["reranker_body_chars"]]
    return f"Title: {doc['title']}\nKeywords: {keywords}\nDescription: {body}"


class DocumentStore:
    def __init__(self, path):
        self.path = Path(path)
        self.connection = sqlite3.connect(self.path)

    def close(self):
        self.connection.close()

    def fetch_many(self, doc_ids):
        unique_ids = list(dict.fromkeys(doc_ids))
        result = {}
        for start in range(0, len(unique_ids), 900):
            batch = unique_ids[start : start + 900]
            placeholders = ",".join("?" for _ in batch)
            rows = self.connection.execute(
                f"SELECT id, title, keywords, body FROM documents "
                f"WHERE id IN ({placeholders})",
                batch,
            )
            for doc_id, title, keywords, body in rows:
                result[doc_id] = {
                    "id": doc_id,
                    "title": title,
                    "keywords": json.loads(keywords),
                    "text": body,
                }
        return result


def build_document_store(config):
    corpus_path = config["paths"]["corpus"]
    output_path = config["paths"]["work_dir"] / "documents.sqlite"
    output_path = Path(output_path)
    metadata_path = output_path.with_suffix(".metadata.json")
    metadata = expected_metadata(config, "document_store")
    if output_path.exists():
        if metadata_matches(metadata_path, metadata):
            print(f"Document store already exists: {output_path}")
            return
        raise RuntimeError(
            f"Document store provenance mismatch. Remove {output_path} and "
            f"{metadata_path}, then rerun prepare."
        )

    connection = sqlite3.connect(output_path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute(
        "CREATE TABLE documents ("
        "id TEXT PRIMARY KEY, title TEXT NOT NULL, keywords TEXT NOT NULL, body TEXT NOT NULL)"
    )
    batch = []
    for doc in tqdm(iter_corpus(corpus_path), desc="Document store", unit=" docs"):
        batch.append(
            (doc["id"], doc["title"], json.dumps(doc["keywords"]), doc["text"])
        )
        if len(batch) == 10000:
            connection.executemany("INSERT INTO documents VALUES (?, ?, ?, ?)", batch)
            connection.commit()
            batch.clear()
    if batch:
        connection.executemany("INSERT INTO documents VALUES (?, ?, ?, ?)", batch)
        connection.commit()
    connection.close()
    write_metadata(metadata_path, metadata)
