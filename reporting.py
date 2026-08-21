"""Shared, model-free reporting helpers: bootstrap CIs, McNemar, per-class
tables, confusion matrices.

Deliberately imports only numpy/scipy — no torch, no transformers. That keeps
`pytest` fast in CI (no GPU wheel needed) and keeps this module reusable for
any prediction array regardless of which model produced it.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import binomtest


def bootstrap_ci(
    labels: np.ndarray,
    predictions: np.ndarray,
    metric_fn,
    n_resamples: int = 1000,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Percentile bootstrap CI for a metric computed on paired (labels, predictions).

    metric_fn(labels, predictions) -> float. Resamples row indices with
    replacement, recomputes the metric each time. Returns the point estimate
    plus the CI bounds.
    """
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    n = len(labels)
    rng = np.random.default_rng(seed)
    point = float(metric_fn(labels, predictions))
    resampled = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        resampled[i] = metric_fn(labels[idx], predictions[idx])
    alpha = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(resampled, [alpha, 1.0 - alpha])
    return {"point": point, "ci_low": float(lo), "ci_high": float(hi), "n_resamples": n_resamples}


def paired_bootstrap_delta(
    labels: np.ndarray,
    predictions_a: np.ndarray,
    predictions_b: np.ndarray,
    metric_fn,
    n_resamples: int = 1000,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict[str, float]:
    """CI for metric(a) - metric(b) on the same resampled rows each iteration.

    Use this rather than comparing two independent bootstrap_ci intervals:
    independent CIs can each be individually plausible while their difference
    is not, because they ignore that both systems are scored on the *same*
    test rows (positively correlated errors).
    """
    labels = np.asarray(labels)
    predictions_a = np.asarray(predictions_a)
    predictions_b = np.asarray(predictions_b)
    n = len(labels)
    rng = np.random.default_rng(seed)
    point = float(metric_fn(labels, predictions_a) - metric_fn(labels, predictions_b))
    deltas = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        deltas[i] = metric_fn(labels[idx], predictions_a[idx]) - metric_fn(
            labels[idx], predictions_b[idx]
        )
    alpha = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(deltas, [alpha, 1.0 - alpha])
    return {
        "point_delta": point,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "excludes_zero": bool(lo > 0 or hi < 0),
        "n_resamples": n_resamples,
    }


def mcnemar(labels: np.ndarray, predictions_a: np.ndarray, predictions_b: np.ndarray) -> dict:
    """Exact McNemar test for two classifiers' correctness on the same test set.

    Compares only the discordant pairs: b = a correct & b wrong, c = a wrong &
    b correct. Uses an exact two-sided binomial test on b vs (b+c), which is
    the standard exact form of McNemar's test and avoids the chi-square
    approximation's small-sample instability.
    """
    labels = np.asarray(labels)
    correct_a = np.asarray(predictions_a) == labels
    correct_b = np.asarray(predictions_b) == labels
    b = int(np.sum(correct_a & ~correct_b))  # a right, b wrong
    c = int(np.sum(~correct_a & correct_b))  # a wrong, b right
    if b + c == 0:
        p_value = 1.0
    else:
        p_value = binomtest(min(b, c), b + c, 0.5).pvalue
    return {
        "a_right_b_wrong": b,
        "a_wrong_b_right": c,
        "p_value": float(p_value),
        "significant_at_0.05": bool(p_value < 0.05),
    }


def per_class_table(labels: np.ndarray, predictions: np.ndarray, class_values) -> list[dict]:
    """Precision/recall/F1/support per class, without sklearn's aggregate averaging."""
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    rows = []
    for value in class_values:
        true_positive = int(np.sum((labels == value) & (predictions == value)))
        false_positive = int(np.sum((labels != value) & (predictions == value)))
        false_negative = int(np.sum((labels == value) & (predictions != value)))
        support = int(np.sum(labels == value))
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows.append(
            {
                "class": value,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )
    return rows


def per_class_markdown(labels: np.ndarray, predictions: np.ndarray, class_values) -> str:
    rows = per_class_table(labels, predictions, class_values)
    lines = ["| Class | Precision | Recall | F1 | Support |", "|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(
            f"| {row['class']} | {row['precision']:.2%} | {row['recall']:.2%} | "
            f"{row['f1']:.2%} | {row['support']:,} |"
        )
    return "\n".join(lines)


def confusion_matrix(labels: np.ndarray, predictions: np.ndarray, class_values) -> np.ndarray:
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    index = {value: i for i, value in enumerate(class_values)}
    matrix = np.zeros((len(class_values), len(class_values)), dtype=np.int64)
    for true_value, predicted_value in zip(labels, predictions):
        if true_value in index and predicted_value in index:
            matrix[index[true_value], index[predicted_value]] += 1
    return matrix


def confusion_markdown(labels: np.ndarray, predictions: np.ndarray, class_values) -> str:
    matrix = confusion_matrix(labels, predictions, class_values)
    header = "| True \\ Pred | " + " | ".join(str(v) for v in class_values) + " |"
    separator = "|---|" + "---:|" * len(class_values)
    lines = [header, separator]
    for i, true_value in enumerate(class_values):
        row = " | ".join(f"{matrix[i, j]:,}" for j in range(len(class_values)))
        lines.append(f"| {true_value} | {row} |")
    return "\n".join(lines)
