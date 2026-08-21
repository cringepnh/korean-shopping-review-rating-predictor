"""Re-evaluate the saved shopping-rating model with all required baselines."""

from __future__ import annotations

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from data_utils import load_prepared_splits, rating_metrics
from train_model import (
    CLASS_TO_RATING,
    MODEL_DIR,
    constant_baselines,
    make_dataset,
    predict_rating_classes,
    print_comparison,
)


def main() -> None:
    train, _, test, _ = load_prepared_splits()
    results = constant_baselines(train, test)
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Saved model not found at {MODEL_DIR}; run python train_model.py")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    test_ds = make_dataset(test, tokenizer)
    predictions = CLASS_TO_RATING[
        predict_rating_classes(model, test_ds, tokenizer, batch_size=128)
    ]
    results["Fine-tuned 4-class KoELECTRA"] = rating_metrics(
        test["rating"].to_numpy(), predictions
    )
    print_comparison(results)


if __name__ == "__main__":
    main()
