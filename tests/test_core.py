import csv
import json
import tempfile
import unittest
from pathlib import Path

from research_challenge.data import (
    DocumentStore,
    build_document_store,
    iter_corpus,
    load_qrels,
    load_queries,
)
from research_challenge.fusion import rrf
from research_challenge.io import write_submission
from research_challenge.metrics import evaluate, recall_at_k


class CorePipelineTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.corpus = self.root / "corpus.jsonl"
        documents = [
            {"id": "d1", "title": "Solar plant", "text": "Power in Europe", "keywords": ["energy"]},
            {"id": "d2", "title": "Wind farm", "text": "Renewable power", "keywords": ["wind"]},
            {"id": "d3", "title": "Restaurant", "text": "Food", "keywords": ["cuisine"]},
        ]
        self.corpus.write_text(
            "".join(json.dumps(doc) + "\n" for doc in documents),
            encoding="utf-8",
        )
        (self.root / "queries.csv").write_text(
            "QueryId,Query\nq1,europe solar power\n", encoding="utf-8"
        )
        (self.root / "qrels.csv").write_text(
            "QueryId,EntityId,Relevance\nq1,d1,2\nq1,d2,1\n",
            encoding="utf-8",
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_data_store_metrics_fusion_and_submission(self):
        self.assertEqual(len(list(iter_corpus(self.corpus))), 3)
        queries = load_queries(self.root / "queries.csv")
        qrels = load_qrels(self.root / "qrels.csv")
        self.assertEqual(queries, [("q1", "europe solar power")])

        database = self.root / "documents.sqlite"
        config = {
            "paths": {"corpus": self.corpus, "work_dir": self.root},
        }
        build_document_store(config)
        store = DocumentStore(database)
        try:
            self.assertEqual(store.fetch_many(["d1"])["d1"]["title"], "Solar plant")
        finally:
            store.close()

        run = rrf({"q1": ["d1", "d3"]}, {"q1": ["d2", "d1"]}, 0.5, limit=3)
        self.assertEqual(run["q1"][0], "d1")
        self.assertGreater(evaluate(run, qrels), 0.0)
        self.assertEqual(recall_at_k(run, qrels, 3), 1.0)

        output = self.root / "submission.csv"
        write_submission(run, output, k=2)
        with output.open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 2)
        self.assertEqual(set(rows[0]), {"QueryId", "EntityId"})


if __name__ == "__main__":
    unittest.main()
