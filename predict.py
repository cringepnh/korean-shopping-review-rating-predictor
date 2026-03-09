"""
predict.py
==========
STEP 6: Predict the review score for any Korean text input.

This script provides:
  1. predict_score(text, approach)
       → returns predicted score (1-10) with explanation
  2. Interactive mode: type a review, get a prediction in real time

Usage:
  # Interactive mode
  python predict.py

  # Single prediction (from another script)
  from predict import predict_score
  result = predict_score("정말 재미있는 영화였습니다!")
  print(result)

Both 'regression' and 'ordinal' models are supported.
The script loads both if they exist and shows combined output.
"""

import torch
import numpy as np
from pathlib import Path
from transformers import ElectraTokenizer
from train_model import (
    KoELECTRARegressor,
    KoELECTRAOrdinalClassifier,
    MODEL_NAME,
    MAX_LENGTH,
    NUM_CLASSES,
)

MODELS_DIR = Path("models")
SCORES     = list(range(1, 11))  # [1, 2, ..., 10]

# Human-readable score labels
SCORE_LABELS = {
    1:  "완전 최악 😡 (Terrible)",
    2:  "매우 나쁨 😤 (Very Bad)",
    3:  "나쁨 😞 (Bad)",
    4:  "별로임 😕 (Below Average)",
    5:  "그냥 그럼 😐 (Mediocre)",
    6:  "나쁘지 않음 🙂 (Decent)",
    7:  "좋음 😊 (Good)",
    8:  "꽤 좋음 😄 (Pretty Good)",
    9:  "매우 좋음 🤩 (Excellent)",
    10: "완전 최고 🎉 (Masterpiece)",
}


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ScorePredictor:
    """
    Loads trained models and predicts review scores.

    Attributes:
        tokenizer     : KoELECTRA tokenizer
        models        : dict of loaded model objects
        device        : CPU or GPU
    """

    def __init__(self):
        """Load tokenizer and any available trained models."""
        self.device    = get_device()
        print(f"Loading tokenizer ({MODEL_NAME})...")
        self.tokenizer = ElectraTokenizer.from_pretrained(MODEL_NAME)
        self.models    = {}

        # Try to load regression model
        reg_ckpt = MODELS_DIR / "regression" / "best_model.pt"
        if reg_ckpt.exists():
            print("Loading regression model...")
            m = KoELECTRARegressor(MODEL_NAME)
            m.load_state_dict(torch.load(reg_ckpt, map_location=self.device, weights_only=True))
            m = m.to(self.device).eval()
            self.models["regression"] = m
            print("  ✓ Regression model loaded")
        else:
            print(f"  ⚠ No regression model at {reg_ckpt}")

        # Try to load ordinal model
        ord_ckpt = MODELS_DIR / "ordinal" / "best_model.pt"
        if ord_ckpt.exists():
            print("Loading ordinal model...")
            m = KoELECTRAOrdinalClassifier(MODEL_NAME, NUM_CLASSES)
            m.load_state_dict(torch.load(ord_ckpt, map_location=self.device, weights_only=True))
            m = m.to(self.device).eval()
            self.models["ordinal"] = m
            print("  ✓ Ordinal model loaded")
        else:
            print(f"  ⚠ No ordinal model at {ord_ckpt}")

        if not self.models:
            raise RuntimeError(
                "No trained models found. Please run train_model.py first."
            )

    def tokenize(self, text: str) -> dict:
        """Tokenize a single review string."""
        return self.tokenizer(
            text,
            max_length     = MAX_LENGTH,
            padding        = "max_length",
            truncation     = True,
            return_tensors = "pt",
        )

    def predict_regression(self, text: str) -> dict:
        """Use regression model to predict score."""
        enc        = self.tokenize(text)
        input_ids  = enc["input_ids"].to(self.device)
        attn_mask  = enc["attention_mask"].to(self.device)

        with torch.no_grad():
            raw_score = self.models["regression"](input_ids, attn_mask).item()

        score       = round(np.clip(raw_score, 1, 10))
        confidence  = 1.0 - abs(raw_score - score)  # How close to an integer

        return {
            "approach"    : "regression",
            "raw_score"   : round(raw_score, 2),
            "score"       : score,
            "confidence"  : round(max(0, min(1, confidence)), 2),
            "label"       : SCORE_LABELS[score],
        }

    def predict_ordinal(self, text: str) -> dict:
        """Use ordinal classification model to predict score."""
        enc        = self.tokenize(text)
        input_ids  = enc["input_ids"].to(self.device)
        attn_mask  = enc["attention_mask"].to(self.device)

        with torch.no_grad():
            logits = self.models["ordinal"](input_ids, attn_mask)
            probs  = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        score      = int(np.argmax(probs)) + 1  # argmax gives 0-9, add 1 → 1-10
        confidence = float(probs[score - 1])

        return {
            "approach"        : "ordinal",
            "score"           : score,
            "confidence"      : round(confidence, 3),
            "label"           : SCORE_LABELS[score],
            "probabilities"   : {s: round(float(p), 4) for s, p in zip(SCORES, probs)},
        }

    def predict(self, text: str) -> list[dict]:
        """
        Predict score using all available models.
        Returns a list of result dicts (one per model).
        """
        results = []
        if "regression" in self.models:
            results.append(self.predict_regression(text))
        if "ordinal" in self.models:
            results.append(self.predict_ordinal(text))
        return results


def predict_score(text: str, predictor=None) -> list[dict]:
    """
    Convenience function — predict score for a single Korean review.

    Args:
        text      : Korean movie review text
        predictor : Optional pre-loaded ScorePredictor (reuse for speed)

    Returns:
        List of prediction dicts from each model

    Example:
        result = predict_score("영화가 정말 재미있었어요!")
        print(result[0]["score"])  # e.g., 9
    """
    if predictor is None:
        predictor = ScorePredictor()
    return predictor.predict(text)


def print_prediction(text: str, results: list[dict]) -> None:
    """Pretty-print prediction results to the terminal."""
    print("\n" + "─"*60)
    print(f"📝 Review: {text[:100]}{'...' if len(text) > 100 else ''}")
    print("─"*60)

    for r in results:
        approach = r["approach"].upper()
        score    = r["score"]
        label    = r["label"]
        conf     = r.get("confidence", 0) * 100

        print(f"\n[{approach}]")
        print(f"  Predicted Score : {score}/10  {label}")
        print(f"  Confidence      : {conf:.1f}%")

        if approach == "REGRESSION" and "raw_score" in r:
            print(f"  Raw output      : {r['raw_score']:.2f}")

        if approach == "ORDINAL" and "probabilities" in r:
            probs = r["probabilities"]
            print(f"  Score probabilities:")
            for s in SCORES:
                bar  = "█" * int(probs[s] * 30)
                mark = " ◄" if s == score else ""
                print(f"    Score {s:2d}: {bar:<30} {probs[s]*100:5.1f}%{mark}")

    print("─"*60)


def interactive_mode(predictor: ScorePredictor) -> None:
    """Loop that accepts review input and prints predictions."""
    print("\n" + "="*60)
    print("Korean Movie Review Score Predictor — Interactive Mode")
    print("="*60)
    print("Type a Korean movie review and press Enter.")
    print("Type 'quit' or 'q' to exit.\n")

    while True:
        try:
            text = input("🎬 Enter review: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if text.lower() in ("quit", "q", "exit"):
            print("Goodbye!")
            break

        if not text:
            print("  (Please enter some text)")
            continue

        results = predictor.predict(text)
        print_prediction(text, results)


def demo_predictions(predictor: ScorePredictor) -> None:
    """Run a few demo predictions to show the model working."""
    demo_reviews = [
        "정말 재미있는 영화였습니다! 배우들의 연기가 훌륭하고 스토리가 탄탄했어요. 올해 최고의 영화입니다!",
        "시간 낭비였습니다. 스토리가 너무 지루하고 배우들 연기도 어색했어요. 돈이 아까웠습니다.",
        "보통 수준의 영화였습니다. 나쁘지는 않았지만 특별히 기억에 남는 것도 없었어요.",
        "눈물이 날 정도로 감동적이었어요. 강력 추천합니다! 꼭 보세요!",
        "기대 이하였습니다. 예고편이 훨씬 재미있었어요. 실망스럽습니다.",
    ]

    print("\n" + "="*60)
    print("DEMO: Sample Predictions")
    print("="*60)

    for review in demo_reviews:
        results = predictor.predict(review)
        print_prediction(review, results)


def main():
    print("Loading models (please wait)...")
    try:
        predictor = ScorePredictor()
    except RuntimeError as e:
        print(f"\n❌ {e}")
        return

    # Show demo predictions
    demo_predictions(predictor)

    # Enter interactive mode
    interactive_mode(predictor)


if __name__ == "__main__":
    main()
