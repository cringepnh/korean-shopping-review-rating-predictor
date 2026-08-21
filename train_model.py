"""Fine-tune raw KoELECTRA for four-class real shopping-rating prediction."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from data_utils import ROOT, SEED, VALID_RATINGS, load_prepared_splits, rating_metrics


BASE_MODEL = "monologg/koelectra-base-v3-discriminator"
MODEL_DIR = ROOT / "models" / "rating-classifier"
RESULTS_DIR = ROOT / "results" / "rating-classifier"
METRICS_PATH = ROOT / "evaluation_results.json"
MAX_LENGTH = 128
RATING_TO_CLASS = {rating: index for index, rating in enumerate(VALID_RATINGS)}
CLASS_TO_RATING = np.array(VALID_RATINGS, dtype=np.int64)


def seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    set_seed(SEED)


def constant_baselines(train, test) -> dict[str, dict[str, float]]:
    labels = test["rating"].to_numpy()
    median = float(train["rating"].median())
    mean = float(train["rating"].mean())
    majority = float(train["rating"].mode().iloc[0])
    systems = {
        "Constant: training median": np.full(len(test), median),
        "Constant: training mean": np.full(len(test), mean),
        "Majority rating class": np.full(len(test), majority),
    }
    results = {}
    for name, predictions in systems.items():
        metrics = rating_metrics(labels, predictions)
        metrics["constant_prediction"] = float(predictions[0])
        results[name] = metrics
    return results


def make_dataset(frame, tokenizer) -> Dataset:
    converted = frame[["review", "rating"]].copy()
    converted["labels"] = converted["rating"].map(RATING_TO_CLASS)
    if converted["labels"].isna().any():
        raise RuntimeError("Found a rating with no classifier class")
    dataset = Dataset.from_pandas(
        converted[["review", "labels"]].rename(columns={"review": "text"}),
        preserve_index=False,
    )

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

    return dataset.map(tokenize, batched=True, remove_columns=["text"])


def predict_rating_classes(model, dataset: Dataset, tokenizer, batch_size: int) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collator
    )
    predictions: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            batch.pop("labels", None)
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast(
                device_type=device.type,
                enabled=device.type == "cuda",
                dtype=torch.float16,
            ):
                logits = model(**batch).logits
            predictions.append(logits.argmax(dim=-1).cpu().numpy())
    return np.concatenate(predictions)


def predict_rating_probabilities(model, dataset: Dataset, tokenizer, batch_size: int) -> np.ndarray:
    """Softmax class probabilities, one row per example, columns ordered by
    VALID_RATINGS (i.e. column 0 = P(rating=1), ..., column 3 = P(rating=5)).

    Kept separate from predict_rating_classes (which only returns the argmax)
    so that alternative decoders (see decoding.py) can be evaluated from the
    same forward pass without retraining or re-running inference twice.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collator
    )
    all_probabilities: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            batch.pop("labels", None)
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast(
                device_type=device.type,
                enabled=device.type == "cuda",
                dtype=torch.float16,
            ):
                logits = model(**batch).logits
            probabilities = torch.softmax(logits.float(), dim=-1)
            all_probabilities.append(probabilities.cpu().numpy())
    return np.concatenate(all_probabilities)


def print_comparison(results: dict[str, dict[str, float]]) -> None:
    print("\n" + "=" * 91)
    print("REAL-RATING BASELINE COMPARISON — SAME HELD-OUT TEST SET")
    print("=" * 91)
    print(
        f"{'System':<38} {'MAE':>9} {'RMSE':>9} {'Exact':>10} {'Within ±1':>12}"
    )
    print("-" * 91)
    for name, metrics in results.items():
        print(
            f"{name:<38} {metrics['mae']:>9.4f} {metrics['rmse']:>9.4f} "
            f"{metrics['exact_accuracy']:>9.2%} {metrics['within_one_accuracy']:>11.2%}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument(
        "--baselines-only",
        action="store_true",
        help="Verify the real data and print same-test constant baselines without loading a model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything()
    train, validation, test, split_manifest = load_prepared_splits()
    baseline_results = constant_baselines(train, test)
    print_comparison(baseline_results)
    if args.baselines_only:
        return

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    train_ds = make_dataset(train, tokenizer)
    validation_ds = make_dataset(validation, tokenizer)
    test_ds = make_dataset(test, tokenizer)

    id2label = {index: str(rating) for index, rating in enumerate(VALID_RATINGS)}
    label2id = {label: index for index, label in id2label.items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(VALID_RATINGS),
        id2label=id2label,
        label2id=label2id,
    )

    def compute_metrics(eval_prediction):
        logits, class_labels = eval_prediction
        predicted_ratings = CLASS_TO_RATING[np.argmax(logits, axis=-1)]
        true_ratings = CLASS_TO_RATING[class_labels.astype(int)]
        return rating_metrics(true_ratings, predicted_ratings)

    use_cuda = torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir=str(RESULTS_DIR),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=max(
            1,
            int(np.ceil(len(train) / args.train_batch_size) * args.epochs * 0.1),
        ),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_mae",
        greater_is_better=False,
        save_total_limit=2,
        fp16=use_cuda,
        report_to="none",
        seed=SEED,
        data_seed=SEED,
        dataloader_num_workers=0,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=validation_ds,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )
    trainer.train()

    # Reload explicitly so legacy LayerNorm beta/gamma names in a saved
    # checkpoint are converted correctly by Transformers 5.x.
    best_checkpoint = trainer.state.best_model_checkpoint
    if not best_checkpoint:
        raise RuntimeError("Training completed without a selected best checkpoint")
    best_model = AutoModelForSequenceClassification.from_pretrained(best_checkpoint)
    predicted_ratings = CLASS_TO_RATING[
        predict_rating_classes(best_model, test_ds, tokenizer, args.eval_batch_size)
    ]
    true_ratings = test["rating"].to_numpy()
    results = dict(baseline_results)
    results["Fine-tuned 4-class KoELECTRA"] = rating_metrics(
        true_ratings, predicted_ratings
    )
    print_comparison(results)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    best_model.save_pretrained(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))

    # Per-epoch validation history, sourced directly from the Trainer's own
    # log, not typed in by hand. A previous version of this file had a
    # validation_history block hand-edited into evaluation_results.json with
    # broken JSON indentation; if the pipeline were rerun it would have been
    # silently dropped because nothing here ever wrote it. Capturing it from
    # trainer.state.log_history makes the metrics file fully regenerable by
    # this script alone.
    validation_history = [
        {k: v for k, v in h.items() if k.startswith("eval_") or k == "epoch"}
        for h in trainer.state.log_history
        if "eval_mae" in h
    ]

    payload = {
        "task": "Naver Shopping review rating classification",
        "method": "4-class classification with cross-entropy",
        "ratings": list(VALID_RATINGS),
        "base_checkpoint": BASE_MODEL,
        "source_sha256": split_manifest["source_sha256"],
        "split_manifest": split_manifest,
        "validation_history": validation_history,
        "training": {
            "best_checkpoint": best_checkpoint,
            "checkpoint_reload": "explicit AutoModelForSequenceClassification.from_pretrained",
            "epochs_requested": args.epochs,
            "global_steps": trainer.state.global_step,
            "best_validation_mae": trainer.state.best_metric,
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
        },
        "test_results": results,
    }
    METRICS_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nSaved model: {MODEL_DIR}")
    print(f"Saved auditable metrics: {METRICS_PATH}")


if __name__ == "__main__":
    main()
