"""Train the CORAL ordinal-regression head and compare it against the 4-class
cross-entropy classifier on the same held-out test split.

This is the one part of the project that requires new training (the CE model
is already trained and saved). Run with --smoke-test first on a tiny slice to
catch integration bugs before committing to the full ~30-45 min run.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from coral import (
    NUM_THRESHOLDS,
    CoralForOrdinalRegression,
    check_bias_monotonicity,
    cumulative_logits_to_ranks,
    ranks_to_ratings,
    ratings_to_ranks,
)
from data_utils import ROOT, SEED, VALID_RATINGS, load_prepared_splits, rating_metrics
from train_model import BASE_MODEL, MAX_LENGTH, seed_everything

MODEL_DIR = ROOT / "models" / "rating-coral"
RESULTS_DIR = ROOT / "results" / "rating-coral"
METRICS_PATH = ROOT / "coral_results.json"


def make_dataset(frame, tokenizer) -> Dataset:
    converted = frame[["review", "rating"]].copy()
    converted["labels"] = ratings_to_ranks(converted["rating"].to_numpy())
    dataset = Dataset.from_pandas(
        converted[["review", "labels"]].rename(columns={"review": "text"}),
        preserve_index=False,
    )

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

    return dataset.map(tokenize, batched=True, remove_columns=["text"])


def predict_ranks(model, dataset: Dataset, tokenizer, batch_size: int) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collator
    )
    all_logits = []
    with torch.inference_mode():
        for batch in loader:
            batch.pop("labels", None)
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**batch)
            all_logits.append(output["logits"].float().cpu().numpy())
    logits = np.concatenate(all_logits)
    return cumulative_logits_to_ranks(logits)


def compute_metrics_fn(eval_prediction):
    logits, labels = eval_prediction
    predicted_ranks = cumulative_logits_to_ranks(logits)
    predicted_ratings = ranks_to_ratings(predicted_ranks)
    true_ratings = ranks_to_ratings(labels.astype(int))
    return rating_metrics(true_ratings, predicted_ratings)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Train on a 2000-row slice for a few steps to verify the pipeline works.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything()
    train, validation, test, split_manifest = load_prepared_splits()

    if args.smoke_test:
        train = train.sample(n=min(2000, len(train)), random_state=SEED).reset_index(drop=True)
        validation = validation.sample(n=min(500, len(validation)), random_state=SEED).reset_index(drop=True)
        test = test.sample(n=min(500, len(test)), random_state=SEED).reset_index(drop=True)
        args.epochs = 1.0
        print(f"SMOKE TEST: train={len(train)} validation={len(validation)} test={len(test)}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    train_ds = make_dataset(train, tokenizer)
    validation_ds = make_dataset(validation, tokenizer)
    test_ds = make_dataset(test, tokenizer)

    config = AutoConfig.from_pretrained(BASE_MODEL)
    model = CoralForOrdinalRegression(config, base_model_name=BASE_MODEL)

    use_cuda = torch.cuda.is_available()
    output_dir = RESULTS_DIR if not args.smoke_test else RESULTS_DIR / "smoke"
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=max(
            1, int(np.ceil(len(train) / args.train_batch_size) * args.epochs * 0.1)
        ),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=20 if args.smoke_test else 100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_mae",
        greater_is_better=False,
        save_total_limit=2,
        fp16=use_cuda,
        report_to="none",
        seed=SEED,
        data_seed=SEED,
        dataloader_num_workers=0,
        label_names=["labels"],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=validation_ds,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        processing_class=tokenizer,
        compute_metrics=compute_metrics_fn,
        callbacks=[] if args.smoke_test else [EarlyStoppingCallback(early_stopping_patience=1)],
    )

    start = time.time()
    trainer.train()
    train_seconds = time.time() - start

    best_checkpoint = trainer.state.best_model_checkpoint
    if best_checkpoint:
        # CoralForOrdinalRegression has no config_class set (it is not
        # registered with AutoConfig for any single architecture, since the
        # encoder is chosen at runtime), so from_pretrained can't resolve its
        # own config type automatically. Passing the already-saved config
        # explicitly skips that resolution step.
        reload_config = AutoConfig.from_pretrained(best_checkpoint)
        best_model = CoralForOrdinalRegression.from_pretrained(
            best_checkpoint, config=reload_config, base_model_name=None
        )
    else:
        best_model = model  # smoke test: no checkpointing pressure

    predicted_ranks = predict_ranks(best_model, test_ds, tokenizer, args.eval_batch_size)
    predicted_ratings = ranks_to_ratings(predicted_ranks)
    true_ratings = test["rating"].to_numpy()
    test_metrics = rating_metrics(true_ratings, predicted_ratings)

    monotonicity = check_bias_monotonicity(
        best_model.head.biases.detach().cpu().numpy()
    )

    print("\nCORAL test metrics:", json.dumps(test_metrics, indent=2))
    print("Bias monotonicity check:", json.dumps(monotonicity, indent=2))

    if args.smoke_test:
        print("\nSMOKE TEST PASSED: pipeline runs end to end.")
        return

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    best_model.save_pretrained(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))

    validation_history = [
        {k: v for k, v in h.items() if k.startswith("eval_") or k == "epoch"}
        for h in trainer.state.log_history
        if "eval_mae" in h
    ]

    payload = {
        "method": "CORAL ordinal regression (Cao, Mirjalili & Raschka 2020)",
        "base_checkpoint": BASE_MODEL,
        "ratings": list(VALID_RATINGS),
        "source_sha256": split_manifest["source_sha256"],
        "train_seconds": train_seconds,
        "training": {
            "best_checkpoint": best_checkpoint,
            "epochs_requested": args.epochs,
            "global_steps": trainer.state.global_step,
            "best_validation_mae": trainer.state.best_metric,
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
        },
        "validation_history": validation_history,
        "bias_monotonicity": monotonicity,
        "test_results": {"CORAL ordinal regression": test_metrics},
    }
    METRICS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved CORAL model: {MODEL_DIR}")
    print(f"Saved CORAL metrics: {METRICS_PATH}")


if __name__ == "__main__":
    main()
