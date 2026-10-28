"""Fit the isolation forest on train-server traffic and write app/ml_forest.json.

This is the only place that calls fit. Startup loads the saved trees.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.ml_filter import (  # noqa: E402
    FOREST_PATH,
    FastFilter,
    _benign_training_set,
    fit_compiled_forest,
)
from app.rag_engine import load_schema_document  # noqa: E402

spec = importlib.util.spec_from_file_location("run_holdout", ROOT / "scripts" / "run_holdout.py")
holdout = importlib.util.module_from_spec(spec)
spec.loader.exec_module(holdout)

TRAIN_DIRS = (
    ROOT / "examples" / "holdout" / "train" / "1-EchoGrid",
    ROOT / "examples" / "holdout" / "train" / "2-RiftNote",
    ROOT / "examples" / "holdout" / "train" / "3-StratumNotes",
)
SYNTHETIC_ROWS = 400


def step_vector(sample: dict) -> tuple[tuple[float, ...], int, int]:
    body = b"" if not sample.get("body") else json.dumps(sample["body"]).encode()
    features = FastFilter.extract_features(
        sample["method"],
        sample["path"],
        sample.get("query") or "",
        body,
        window_rate=3,
        headers=sample.get("headers") or None,
    )
    return FastFilter.to_vector(features), features.traversal_hits, features.injection_hits


def real_rows() -> tuple[np.ndarray, dict[str, int]]:
    vectors: list[tuple[float, ...]] = []
    counts = {"benign": 0, "suspicious_kept": 0, "suspicious_skipped": 0}
    for directory in TRAIN_DIRS:
        document = load_schema_document((directory / "openapi.yaml").read_bytes())
        prefix = holdout.server_prefix(document)
        samples = holdout.load_sessions(sorted((directory / "sessions").glob("*.json")), prefix)
        for sample in samples:
            label = sample.get("label")
            if label not in ("benign", "suspicious_benign"):
                continue
            vector, traversal_hits, injection_hits = step_vector(sample)
            if label == "suspicious_benign" and (traversal_hits or injection_hits):
                counts["suspicious_skipped"] += 1
                continue
            vectors.append(vector)
            counts["benign" if label == "benign" else "suspicious_kept"] += 1
    return np.array(vectors, dtype=float), counts


def main() -> None:
    real, counts = real_rows()
    synthetic = _benign_training_set(n=SYNTHETIC_ROWS, seed=1337)
    rows = np.vstack([real, synthetic])
    compiled = fit_compiled_forest(rows, n_estimators=64, max_samples=2048, seed=1337)
    compiled.dump(FOREST_PATH)
    print("real", counts, "synthetic", len(synthetic), "total", len(rows))
    print("wrote", FOREST_PATH, "bytes", FOREST_PATH.stat().st_size)


if __name__ == "__main__":
    main()
