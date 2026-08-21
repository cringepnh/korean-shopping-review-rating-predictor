"""Shared data integrity and metric helpers for real Naver Shopping ratings."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "naver_shopping_reviews.tsv"
SOURCE_MANIFEST_PATH = DATA_DIR / "source_manifest.json"
SPLIT_MANIFEST_PATH = DATA_DIR / "split_manifest.json"
TRAIN_PATH = DATA_DIR / "train.csv"
VALIDATION_PATH = DATA_DIR / "validation.csv"
TEST_PATH = DATA_DIR / "test.csv"

SOURCE_URL = (
    "https://raw.githubusercontent.com/bab2min/corpus/master/"
    "sentiment/naver_shopping.txt"
)
SOURCE_PAGE = "https://github.com/bab2min/corpus/tree/master/sentiment"
LICENSE_PAGE = "https://github.com/bab2min/corpus"
SOURCE_SHA256 = "dc4d1aca0a148671cbe80bcb81962eee297370acab42be93c1617ce9336479c0"
EXPECTED_SOURCE_ROWS = 200_000
VALID_RATINGS = (1, 2, 4, 5)
ALL_SCALE_VALUES = (1, 2, 3, 4, 5)
SEED = 42


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_real_source() -> dict:
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"{RAW_DATA_PATH} is missing. Run: python download_data.py"
        )
    actual_hash = sha256_file(RAW_DATA_PATH)
    if actual_hash != SOURCE_SHA256:
        raise RuntimeError(
            "Downloaded ratings file failed its SHA-256 check. Refusing to train on "
            "an unknown or modified target source."
        )
    if not SOURCE_MANIFEST_PATH.exists():
        raise RuntimeError("Source manifest is missing; run python download_data.py")
    manifest = json.loads(SOURCE_MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("sha256") != SOURCE_SHA256 or manifest.get("target_origin") != "source_file":
        raise RuntimeError("Source manifest does not prove that ratings came from the real source file")
    return manifest


def print_target_distribution(name: str, frame: pd.DataFrame) -> None:
    print(f"\n{name} target distribution ({len(frame):,} rows):")
    counts = frame["rating"].value_counts().sort_index()
    for rating in ALL_SCALE_VALUES:
        count = int(counts.get(rating, 0))
        print(f"  {rating} star: {count:>7,} ({count / len(frame):6.2%})")
        if count == 0:
            print(f"  WARNING: rating {rating} has ZERO examples in {name}!")


def load_real_reviews() -> pd.DataFrame:
    verify_real_source()
    frame = pd.read_csv(
        RAW_DATA_PATH,
        sep="\t",
        names=["rating", "review"],
        header=None,
        dtype={"rating": "int64", "review": "string"},
    )
    if len(frame) != EXPECTED_SOURCE_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_SOURCE_ROWS:,} source rows, got {len(frame):,}")
    if frame["review"].isna().any():
        raise RuntimeError("The verified source unexpectedly contains missing review text")
    observed = set(frame["rating"].unique().tolist())
    if observed != set(VALID_RATINGS):
        raise RuntimeError(f"Expected real ratings {VALID_RATINGS}, got {sorted(observed)}")

    # This assertion is intentionally explicit: targets must be parsed directly
    # from column 1 of the verified source, never generated from review text.
    frame["target_origin"] = "source_file"
    if not (frame["target_origin"] == "source_file").all():
        raise AssertionError("Every rating target must come from the verified source file")
    frame["review"] = frame["review"].str.strip()
    frame = frame[frame["review"].str.len() > 0]
    rating_counts_per_text = frame.groupby("review")["rating"].transform("nunique")
    ambiguous_rows = int((rating_counts_per_text > 1).sum())
    if ambiguous_rows:
        print(
            f"Removed {ambiguous_rows:,} rows whose identical review text has "
            "conflicting source ratings."
        )
    frame = frame[rating_counts_per_text == 1]
    frame = frame.drop_duplicates(subset=["review"], keep="first").reset_index(drop=True)
    print_target_distribution("CLEANED SOURCE", frame)
    return frame[["review", "rating", "target_origin"]]


def load_prepared_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    verify_real_source()
    for path in (TRAIN_PATH, VALIDATION_PATH, TEST_PATH, SPLIT_MANIFEST_PATH):
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing. Run: python prepare_data.py")
    manifest = json.loads(SPLIT_MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("source_sha256") != SOURCE_SHA256:
        raise RuntimeError("Prepared splits do not match the verified real-rating source")

    frames = [pd.read_csv(path) for path in (TRAIN_PATH, VALIDATION_PATH, TEST_PATH)]
    for name, frame in zip(("TRAIN", "VALIDATION", "TEST"), frames):
        if set(frame.columns) != {"review", "rating", "target_origin"}:
            raise RuntimeError(f"{name} split has an unexpected schema")
        if not (frame["target_origin"] == "source_file").all():
            raise RuntimeError(f"{name} contains targets that are not source ratings")
        if not set(frame["rating"].unique()).issubset(set(VALID_RATINGS)):
            raise RuntimeError(f"{name} contains invalid ratings")
        print_target_distribution(name, frame)
    return frames[0], frames[1], frames[2], manifest


def rating_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    errors = predictions - labels
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "exact_accuracy": float(np.mean(np.isclose(predictions, labels))),
        "within_one_accuracy": float(np.mean(np.abs(errors) <= 1.0)),
    }
