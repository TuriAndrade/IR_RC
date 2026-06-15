from pathlib import Path

import yaml


def load_config(path):
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    root = path.parent
    for key in (
        "corpus",
        "train_queries",
        "train_qrels",
        "test_queries",
        "work_dir",
        "submissions_dir",
        "lucene_index",
    ):
        value = Path(config["paths"][key])
        config["paths"][key] = value if value.is_absolute() else (root / value).resolve()

    config["paths"]["work_dir"].mkdir(parents=True, exist_ok=True)
    config["paths"]["submissions_dir"].mkdir(parents=True, exist_ok=True)

    required_files = ("corpus", "train_queries", "train_qrels", "test_queries")
    missing = [
        str(config["paths"][key])
        for key in required_files
        if not config["paths"][key].is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing required data files: {missing}")
    return config
