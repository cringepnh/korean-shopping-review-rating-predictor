import numpy as np
import pytest

from reporting import (
    bootstrap_ci,
    confusion_markdown,
    mcnemar,
    paired_bootstrap_delta,
    per_class_table,
)


def accuracy(labels, predictions):
    return float(np.mean(labels == predictions))


def test_bootstrap_ci_is_deterministic_for_a_fixed_seed():
    labels = np.array([0, 1] * 100)
    predictions = np.array([0, 1] * 95 + [1, 0] * 5)
    first = bootstrap_ci(labels, predictions, accuracy, n_resamples=200, seed=42)
    second = bootstrap_ci(labels, predictions, accuracy, n_resamples=200, seed=42)
    assert first == second


def test_bootstrap_ci_different_seeds_can_differ_but_bracket_the_point_estimate():
    labels = np.random.default_rng(1).integers(0, 2, 300)
    predictions = np.random.default_rng(2).integers(0, 2, 300)
    result = bootstrap_ci(labels, predictions, accuracy, n_resamples=500, seed=7)
    assert result["ci_low"] <= result["point"] <= result["ci_high"]


def test_bootstrap_ci_zero_variance_when_metric_is_constant():
    labels = np.zeros(50, dtype=int)
    predictions = np.zeros(50, dtype=int)
    result = bootstrap_ci(labels, predictions, accuracy, n_resamples=200, seed=0)
    assert result["point"] == 1.0
    assert result["ci_low"] == 1.0
    assert result["ci_high"] == 1.0


def test_paired_bootstrap_delta_detects_a_clear_winner():
    labels = np.random.default_rng(0).integers(0, 2, 500)
    predictions_a = labels.copy()  # perfect
    predictions_b = labels.copy()
    predictions_b[:250] = 1 - predictions_b[:250]  # 50% accuracy
    result = paired_bootstrap_delta(labels, predictions_a, predictions_b, accuracy, n_resamples=500)
    assert result["point_delta"] == pytest.approx(0.5)
    assert result["excludes_zero"] is True


def test_paired_bootstrap_delta_is_zero_for_identical_predictions():
    labels = np.random.default_rng(3).integers(0, 2, 200)
    result = paired_bootstrap_delta(labels, labels, labels, accuracy, n_resamples=200)
    assert result["point_delta"] == 0.0
    assert result["excludes_zero"] is False


def test_mcnemar_significant_when_one_system_dominates_discordant_pairs():
    labels = np.ones(100, dtype=int)
    predictions_a = np.ones(100, dtype=int)  # always correct
    predictions_b = np.ones(100, dtype=int)
    predictions_b[:40] = 0  # wrong on 40/100
    result = mcnemar(labels, predictions_a, predictions_b)
    assert result["a_right_b_wrong"] == 40
    assert result["a_wrong_b_right"] == 0
    assert result["significant_at_0.05"] is True


def test_mcnemar_not_significant_with_no_discordant_pairs():
    labels = np.array([0, 1, 0, 1])
    predictions = labels.copy()
    result = mcnemar(labels, predictions, predictions)
    assert result["p_value"] == 1.0
    assert result["significant_at_0.05"] is False


def test_per_class_table_reports_support_and_zero_division_safely():
    labels = np.array([1, 1, 2, 2, 2])
    predictions = np.array([1, 2, 2, 2, 2])
    rows = per_class_table(labels, predictions, [1, 2, 3])
    by_class = {row["class"]: row for row in rows}
    assert by_class[1]["support"] == 2
    assert by_class[1]["recall"] == 0.5  # one of two 1s predicted correctly
    assert by_class[3]["support"] == 0
    assert by_class[3]["precision"] == 0.0  # no zero-division crash
    assert by_class[3]["recall"] == 0.0


def test_confusion_markdown_contains_all_class_labels():
    labels = np.array([1, 2, 1, 2])
    predictions = np.array([1, 1, 2, 2])
    table = confusion_markdown(labels, predictions, [1, 2])
    assert "| True \\ Pred | 1 | 2 |" in table
    assert table.count("|") > 0
