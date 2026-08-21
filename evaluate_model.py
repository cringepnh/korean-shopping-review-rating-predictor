"""Regenerate evaluation_results.json from scratch.

Every number in evaluation_results.json is produced here by loading saved
models and running inference on the same held-out test split — none of it is
typed in by hand. A previous version of this file had a validation_history
block manually pasted into the JSON with broken indentation; this script is
the fix, and it is meant to be the *only* way that file is ever written after
training (see also train_model.py, which now captures the same history from
trainer.state.log_history during the original run).

Run after train_model.py. CORAL comparison rows are included automatically
once models/rating-coral exists (produced by train_coral.py); until then this
script still runs and reports the CE-only comparison.
"""

from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import cohen_kappa_score
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from data_utils import ROOT, VALID_RATINGS, load_prepared_splits, rating_metrics
from decoding import (
    decode_argmax,
    decode_expected_value,
    decode_expected_value_rounded,
    decode_median,
)
from reporting import (
    bootstrap_ci,
    confusion_markdown,
    mcnemar,
    paired_bootstrap_delta,
    per_class_markdown,
)
from train_model import (
    MODEL_DIR,
    RESULTS_DIR,
    constant_baselines,
    make_dataset,
    predict_rating_probabilities,
    print_comparison,
)

CORAL_MODEL_DIR = ROOT / "models" / "rating-coral"
METRICS_PATH = ROOT / "evaluation_results.json"
RATING_ARRAY = np.array(VALID_RATINGS, dtype=np.int64)
RATING_TO_RANK = {rating: rank for rank, rating in enumerate(VALID_RATINGS)}


def to_ranks(ratings: np.ndarray) -> np.ndarray:
    return np.array([RATING_TO_RANK[int(r)] for r in ratings], dtype=np.int64)


def nearest_valid_rating(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    distances = np.abs(values[:, None] - RATING_ARRAY.astype(np.float64)[None, :])
    return RATING_ARRAY[np.argmin(distances, axis=1)]


def qwk(true_ratings: np.ndarray, predicted_ratings: np.ndarray) -> float:
    """Quadratic weighted kappa, computed on rank indices {0,1,2,3}, not on
    raw rating values {1,2,4,5}.

    sklearn's cohen_kappa_score builds its weight matrix from the *positions*
    of sorted unique labels, not their numeric spacing — so confusing rating 2
    with rating 4 (a real gap of 2, because 3-star reviews are absent from the
    source corpus) would otherwise be weighted identically to confusing rating
    1 with rating 2 (a gap of 1). Mapping to ranks first makes explicit that
    this project treats the four observed ratings as four *equally spaced*
    ordinal categories, matching how CORAL's rank space is defined elsewhere
    in this project, rather than silently inheriting sklearn's default.
    """
    predicted_ratings = nearest_valid_rating(predicted_ratings)
    return float(
        cohen_kappa_score(
            to_ranks(true_ratings), to_ranks(predicted_ratings), weights="quadratic"
        )
    )


def add_ci(metrics: dict, true_ratings: np.ndarray, predicted_ratings: np.ndarray) -> dict:
    rounded = nearest_valid_rating(predicted_ratings)
    metrics["mae_ci95"] = bootstrap_ci(
        true_ratings, predicted_ratings, lambda y, p: np.mean(np.abs(p - y))
    )
    metrics["exact_accuracy_ci95"] = bootstrap_ci(
        true_ratings, predicted_ratings, lambda y, p: np.mean(np.isclose(p, y))
    )
    metrics["qwk"] = qwk(true_ratings, predicted_ratings)
    metrics["qwk_ci95"] = bootstrap_ci(true_ratings, rounded, lambda y, p: qwk(y, p))
    return metrics


def load_ce_training_metadata() -> dict:
    """Read training/validation history from the CE model's own saved
    checkpoints, rather than re-typing it into this script.

    The largest step-count checkpoint under results/rating-classifier holds
    the trainer's cumulative log_history for the whole run (Trainer appends
    to the same in-memory state across epochs and each checkpoint save
    snapshots it as of that point), so the final checkpoint has the complete
    per-epoch validation record and the recorded best_model_checkpoint.
    """
    checkpoint_dirs = sorted(
        RESULTS_DIR.glob("checkpoint-*"),
        key=lambda p: int(p.name.rsplit("-", 1)[1]),
    )
    if not checkpoint_dirs:
        return {"validation_history": [], "best_checkpoint": None, "best_validation_mae": None}
    state = json.loads((checkpoint_dirs[-1] / "trainer_state.json").read_text(encoding="utf-8"))
    validation_history = [
        {k: v for k, v in h.items() if k.startswith("eval_") or k == "epoch"}
        for h in state.get("log_history", [])
        if "eval_mae" in h
    ]
    return {
        "validation_history": validation_history,
        "best_checkpoint": state.get("best_model_checkpoint"),
        "best_validation_mae": state.get("best_metric"),
    }


def load_coral_predictions(test):
    """Load the trained CORAL model and predict on the test split, if it
    exists. Returns None (not a placeholder result) when CORAL hasn't been
    trained yet, so the rest of the pipeline degrades gracefully rather than
    fabricating a row.
    """
    if not CORAL_MODEL_DIR.exists():
        return None
    from coral import CoralForOrdinalRegression, cumulative_logits_to_ranks, ranks_to_ratings
    from train_coral import make_dataset as make_coral_dataset

    config = AutoConfig.from_pretrained(CORAL_MODEL_DIR)
    model = CoralForOrdinalRegression.from_pretrained(
        CORAL_MODEL_DIR, config=config, base_model_name=None
    )
    # CORAL_MODEL_DIR holds a flat, standard save_pretrained layout (config.json
    # + model.safetensors + tokenizer files all at the top level) -- see
    # coral.py's note on why the encoder is NOT saved to a separate
    # subdirectory. Loading the tokenizer from an "encoder/" subpath was a
    # leftover from that earlier (broken) two-directory format.
    tokenizer = AutoTokenizer.from_pretrained(CORAL_MODEL_DIR)
    test_ds = make_coral_dataset(test, tokenizer)

    import torch
    from transformers import DataCollatorWithPadding

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    loader = torch.utils.data.DataLoader(test_ds, batch_size=128, shuffle=False, collate_fn=collator)
    all_logits = []
    with torch.inference_mode():
        for batch in loader:
            batch.pop("labels", None)
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**batch)
            all_logits.append(output["logits"].float().cpu().numpy())
    logits = np.concatenate(all_logits)
    ranks = cumulative_logits_to_ranks(logits)
    return ranks_to_ratings(ranks)


def main() -> None:
    train, _, test, split_manifest = load_prepared_splits()
    true_ratings = test["rating"].to_numpy()

    results: dict[str, dict] = {}
    predictions_by_system: dict[str, np.ndarray] = {}

    baseline_results = constant_baselines(train, test)
    for name, metrics in baseline_results.items():
        predictions = np.full(len(test), metrics["constant_prediction"])
        predictions_by_system[name] = predictions
        results[name] = add_ci(metrics, true_ratings, predictions)

    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Saved model not found at {MODEL_DIR}; run python train_model.py")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    test_ds = make_dataset(test, tokenizer)
    probabilities = predict_rating_probabilities(model, test_ds, tokenizer, batch_size=128)

    ce_decoders = {
        "Fine-tuned 4-class KoELECTRA (argmax)": decode_argmax(probabilities, RATING_ARRAY),
        "Fine-tuned 4-class KoELECTRA (expected value)": decode_expected_value(
            probabilities, RATING_ARRAY
        ),
        "Fine-tuned 4-class KoELECTRA (expected value, rounded)": decode_expected_value_rounded(
            probabilities, RATING_ARRAY
        ),
        "Fine-tuned 4-class KoELECTRA (median)": decode_median(probabilities, RATING_ARRAY),
    }
    for name, predictions in ce_decoders.items():
        predictions_by_system[name] = predictions
        metrics = rating_metrics(true_ratings, predictions)
        results[name] = add_ci(metrics, true_ratings, predictions)

    coral_predictions = load_coral_predictions(test)
    if coral_predictions is not None:
        predictions_by_system["CORAL ordinal regression"] = coral_predictions
        metrics = rating_metrics(true_ratings, coral_predictions)
        results["CORAL ordinal regression"] = add_ci(metrics, true_ratings, coral_predictions)

    print_comparison(results)

    payload = {
        "task": "Naver Shopping review rating prediction",
        "base_checkpoint": "monologg/koelectra-base-v3-discriminator",
        # upload_to_hub.py checks this list before publishing (refuses to
        # push a model with unexpected rating classes). An earlier rewrite of
        # this payload dropped the field entirely, which made that safety
        # check fail closed (good) rather than silently pass (bad) -- but it
        # needs to actually be here for a legitimate upload to succeed.
        "ratings": list(VALID_RATINGS),
        "source_sha256": split_manifest["source_sha256"],
        "split_manifest": split_manifest,
        "training": load_ce_training_metadata(),
        "notes": {
            "polarity_accuracy": (
                "'within_one_accuracy' is numerically identical to polarity "
                "accuracy on this rating scale {1,2,4,5}: since there is no "
                "3-star class, |pred-true|<=1 holds exactly for the pairs "
                "{1,2}x{1,2} and {4,5}x{4,5}, i.e. exactly when the predicted "
                "and true ratings fall on the same side of the 2/4 gap. It is "
                "not an ordinal-distance metric on this scale; see qwk for that."
            ),
            "qwk_scale": (
                "qwk is computed on rank indices {0,1,2,3} (equally spaced), "
                "not on raw rating values {1,2,4,5}, matching sklearn's "
                "cohen_kappa_score convention and CORAL's own rank space."
            ),
            "expected_value_decoder": (
                "all decoders below are evaluated on the SAME trained "
                "classifier's output probabilities as argmax -- no "
                "retraining. An earlier version of this project assumed "
                "expected value (the distribution's mean) minimizes MAE; the "
                "results below disprove that: expected value improves RMSE "
                "over argmax but makes MAE worse, because the mean minimizes "
                "squared error while the median minimizes absolute error. "
                "The median decoder is the one that actually targets MAE. "
                "The continuous expected-value variant has 0% exact_accuracy "
                "by the same logic as the 'constant: training mean' baseline "
                "(it is never one of the four observed rating values)."
            ),
        },
        "test_results": results,
    }

    if coral_predictions is not None:
        argmax_predictions = predictions_by_system["Fine-tuned 4-class KoELECTRA (argmax)"]
        mae_fn = lambda y, p: np.mean(np.abs(nearest_valid_rating(p).astype(np.float64) - y))
        payload["ce_vs_coral_mae_delta"] = paired_bootstrap_delta(
            true_ratings, argmax_predictions, coral_predictions, mae_fn
        )
        payload["ce_vs_coral_mcnemar_exact_match"] = mcnemar(
            true_ratings, argmax_predictions, coral_predictions
        )

    best_system = "Fine-tuned 4-class KoELECTRA (argmax)"
    payload["per_class_report"] = {
        best_system: per_class_markdown(true_ratings, predictions_by_system[best_system], VALID_RATINGS)
    }
    payload["confusion_matrix"] = {
        best_system: confusion_markdown(true_ratings, predictions_by_system[best_system], VALID_RATINGS)
    }
    if coral_predictions is not None:
        payload["per_class_report"]["CORAL ordinal regression"] = per_class_markdown(
            true_ratings, coral_predictions, VALID_RATINGS
        )
        payload["confusion_matrix"]["CORAL ordinal regression"] = confusion_markdown(
            true_ratings, coral_predictions, VALID_RATINGS
        )

    METRICS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved regenerated, auditable metrics: {METRICS_PATH}")
    if coral_predictions is None:
        print(
            "\nNOTE: models/rating-coral not found yet -- CORAL rows were "
            "skipped. Re-run this script after train_coral.py finishes."
        )


if __name__ == "__main__":
    main()
