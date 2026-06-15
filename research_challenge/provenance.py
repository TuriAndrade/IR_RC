import hashlib
import json
from pathlib import Path


def file_identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read(1024 * 1024))
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "head_sha256": digest.hexdigest(),
    }


def expected_metadata(config, artifact_type):
    metadata = {
        "artifact_type": artifact_type,
        "corpus": file_identity(config["paths"]["corpus"]),
    }
    if artifact_type == "dense":
        metadata["model"] = config["dense"]["model"]
        metadata["text"] = config["text"]
        metadata["normalize"] = config["dense"]["normalize"]
    elif artifact_type == "lucene":
        metadata["field_weights"] = {"title": 3, "keywords": 2, "body": 1}
    elif artifact_type == "document_store":
        metadata["schema_version"] = 1
    return metadata


def metadata_matches(path, expected):
    path = Path(path)
    if not path.exists():
        return False
    return json.loads(path.read_text(encoding="utf-8")) == expected


def write_metadata(path, metadata):
    Path(path).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
