import csv
import json
from pathlib import Path


def save_run(run, path, scores=None):
    payload = {"run": run}
    if scores is not None:
        payload["scores"] = scores
    Path(path).write_text(json.dumps(payload), encoding="utf-8")


def load_run(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_submission(run, path, k=100):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["QueryId", "EntityId"])
        writer.writeheader()
        for qid in sorted(run):
            for doc_id in run[qid][:k]:
                writer.writerow({"QueryId": qid, "EntityId": doc_id})
    print(f"Wrote submission: {path}")
