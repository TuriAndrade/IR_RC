import argparse

from .config import load_config
from .data import build_document_store
from .dense import build_faiss_index, encode_corpus
from .pipeline import run_pipeline
from .sparse import build_lucene


def parser():
    result = argparse.ArgumentParser(description="BGE-M3 entity search pipeline")
    result.add_argument(
        "--config", default="config.yaml", help="Path to the YAML configuration"
    )
    subparsers = result.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare", help="Build document store and Lucene index")
    subparsers.add_parser("encode", help="Encode corpus with BGE-M3")
    subparsers.add_parser("index", help="Build the FAISS index")
    subparsers.add_parser("run", help="Retrieve, tune, rerank, and write submission")
    subparsers.add_parser("all", help="Run every stage in order")
    return result


def prepare(config):
    build_document_store(config)
    build_lucene(config)


def main():
    args = parser().parse_args()
    config = load_config(args.config)
    if args.command in ("prepare", "all"):
        prepare(config)
    if args.command in ("encode", "all"):
        encode_corpus(config)
    if args.command in ("index", "all"):
        build_faiss_index(config)
    if args.command in ("run", "all"):
        run_pipeline(config)


if __name__ == "__main__":
    main()
