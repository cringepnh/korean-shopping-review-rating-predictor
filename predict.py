"""Predict one of the observed real Naver Shopping ratings: 1, 2, 4, or 5.

Supports both decoders compared in evaluate_model.py: argmax (the classifier's
native decision rule) and expected value (minimizes MAE but is not one of the
four observed ratings unless snapped to the nearest one). See decoding.py for
why these differ and coral.py/README for the CORAL alternative.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from decoding import decode_expected_value_rounded, decode_median
from train_model import MODEL_DIR


def predict_rating(text: str, decoder: str = "argmax") -> dict:
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Saved model not found at {MODEL_DIR}; run python train_model.py")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.inference_mode():
        probabilities = torch.softmax(model(**inputs).logits, dim=-1).squeeze(0)

    ratings = np.array(
        [int(model.config.id2label[i]) for i in range(len(probabilities))], dtype=np.int64
    )
    probability_array = probabilities.numpy()[None, :]

    if decoder == "argmax":
        index = int(probabilities.argmax())
        rating = int(model.config.id2label[index])
        confidence = float(probabilities[index])
    elif decoder == "expected":
        rating = int(decode_expected_value_rounded(probability_array, ratings)[0])
        confidence = float(probability_array[0, list(ratings).index(rating)])
    elif decoder == "median":
        rating = int(decode_median(probability_array, ratings)[0])
        confidence = float(probability_array[0, list(ratings).index(rating)])
    else:
        raise ValueError(f"Unknown decoder: {decoder!r}. Use 'argmax', 'expected', or 'median'.")

    return {
        "rating": rating,
        "decoder": decoder,
        "confidence": confidence,
        "probabilities": {
            int(model.config.id2label[i]): float(value)
            for i, value in enumerate(probabilities)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("text", nargs="?")
    parser.add_argument(
        "--decoder",
        choices=["argmax", "expected", "median"],
        default="argmax",
        help="argmax = classifier's native decision rule (best exact accuracy); "
        "expected = mean of the distribution, snapped to the nearest observed "
        "rating (minimizes RMSE, not MAE -- see decoding.py); median = the "
        "MAE-minimizing decoder.",
    )
    args = parser.parse_args()
    text = args.text or input("Korean shopping review: ").strip()
    if not text:
        raise ValueError("Review text must not be empty")
    result = predict_rating(text, decoder=args.decoder)
    print(f"Predicted rating ({result['decoder']} decoder): {result['rating']}/5")
    print(f"Confidence: {result['confidence']:.1%}")
    print("Observed-class probabilities:")
    for rating, probability in result["probabilities"].items():
        print(f"  {rating}: {probability:.1%}")


if __name__ == "__main__":
    main()
