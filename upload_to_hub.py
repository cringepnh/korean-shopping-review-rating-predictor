"""Upload the verified shopping-rating model and model card to Hugging Face Hub."""

from __future__ import annotations

import json
from pathlib import Path

from huggingface_hub import HfApi, create_repo, get_token
from transformers import AutoModelForSequenceClassification, AutoTokenizer


ROOT = Path(__file__).resolve().parent
HF_USERNAME = "cringepnh"
REPO_NAME = "koelectra-korean-shopping-rating"
FULL_REPO_ID = f"{HF_USERNAME}/{REPO_NAME}"
MODEL_DIR = ROOT / "models" / "rating-classifier"
METRICS_PATH = ROOT / "evaluation_results.json"
MODEL_CARD_PATH = ROOT / "hf_model_card.md"
EXPECTED_BASE = "monologg/koelectra-base-v3-discriminator"
EXPECTED_SOURCE_SHA256 = (
    "dc4d1aca0a148671cbe80bcb81962eee297370acab42be93c1617ce9336479c0"
)


def upload() -> None:
    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Saved model not found at {MODEL_DIR}; run python train_model.py first"
        )
    if not METRICS_PATH.exists():
        raise FileNotFoundError("evaluation_results.json is missing")
    if not MODEL_CARD_PATH.exists():
        raise FileNotFoundError("hf_model_card.md is missing")

    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    if metrics.get("base_checkpoint") != EXPECTED_BASE:
        raise RuntimeError("Refusing to upload a model from an unexpected base checkpoint")
    if metrics.get("source_sha256") != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("Refusing to upload metrics from an unverified rating source")
    if metrics.get("ratings") != [1, 2, 4, 5]:
        raise RuntimeError("Refusing to upload a model with unexpected rating classes")

    if not get_token():
        raise RuntimeError("Not logged in to Hugging Face. Run: hf auth login")
    create_repo(repo_id=FULL_REPO_ID, repo_type="model", exist_ok=True)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model.push_to_hub(FULL_REPO_ID)
    tokenizer.push_to_hub(FULL_REPO_ID)
    HfApi().upload_file(
        path_or_fileobj=str(MODEL_CARD_PATH),
        path_in_repo="README.md",
        repo_id=FULL_REPO_ID,
        repo_type="model",
    )
    print(f"Published: https://huggingface.co/{FULL_REPO_ID}")


if __name__ == "__main__":
    upload()
