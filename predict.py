"""Predict one of the observed real Naver Shopping ratings: 1, 2, 4, or 5."""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from train_model import MODEL_DIR


def predict_rating(text: str) -> dict:
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Saved model not found at {MODEL_DIR}; run python train_model.py")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.inference_mode():
        probabilities = torch.softmax(model(**inputs).logits, dim=-1).squeeze(0)
    index = int(probabilities.argmax())
    rating = int(model.config.id2label[index])
    return {
        "rating": rating,
        "confidence": float(probabilities[index]),
        "probabilities": {
            int(model.config.id2label[i]): float(value)
            for i, value in enumerate(probabilities)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("text", nargs="?")
    args = parser.parse_args()
    text = args.text or input("Korean shopping review: ").strip()
    if not text:
        raise ValueError("Review text must not be empty")
    result = predict_rating(text)
    print(f"Predicted rating: {result['rating']}/5")
    print(f"Confidence: {result['confidence']:.1%}")
    print("Observed-class probabilities:")
    for rating, probability in result["probabilities"].items():
        print(f"  {rating}: {probability:.1%}")


if __name__ == "__main__":
    main()
