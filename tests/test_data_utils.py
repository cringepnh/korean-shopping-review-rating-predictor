import numpy as np
import pytest

from data_utils import rating_metrics, sets_are_disjoint


def test_rating_metrics_perfect_predictions():
    labels = np.array([1, 2, 4, 5])
    metrics = rating_metrics(labels, labels)
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["exact_accuracy"] == 1.0
    assert metrics["within_one_accuracy"] == 1.0


def test_rating_metrics_known_errors():
    labels = np.array([1, 2, 4, 5])
    predictions = np.array([2, 2, 5, 4])  # errors: 1, 0, 1, 1
    metrics = rating_metrics(labels, predictions)
    assert metrics["mae"] == pytest.approx(0.75)
    assert metrics["exact_accuracy"] == pytest.approx(0.25)
    # every error here has magnitude 1 or 0, so within_one_accuracy is 100%
    assert metrics["within_one_accuracy"] == pytest.approx(1.0)


def test_within_one_accuracy_is_polarity_accuracy_on_this_rating_scale():
    """Regression test for the README caveat: on the scale {1,2,4,5} (no
    3-star class), |pred-true|<=1 is true exactly for the pairs
    {1,2}x{1,2} and {4,5}x{4,5} -- i.e. it is numerically identical to
    "predicted and true rating are on the same side of the 2/4 gap"
    (polarity accuracy), not a genuine ordinal-distance metric on this scale.
    This locks that claim into the test suite rather than leaving it only in
    prose, per the audit finding that the README's "within +/-1" framing
    overstated what the metric measures.
    """
    rng = np.random.default_rng(0)
    valid_ratings = np.array([1, 2, 4, 5])
    labels = rng.choice(valid_ratings, size=2000)
    predictions = rng.choice(valid_ratings, size=2000)

    def polarity(values):
        return values >= 4  # True for {4,5}, False for {1,2}

    within_one = np.abs(predictions.astype(float) - labels.astype(float)) <= 1.0
    same_polarity = polarity(predictions) == polarity(labels)
    assert np.array_equal(within_one, same_polarity)


def test_sets_are_disjoint_true_for_no_overlap():
    assert sets_are_disjoint({"a", "b"}, {"c", "d"}, {"e"})


def test_sets_are_disjoint_false_for_any_pairwise_overlap():
    assert not sets_are_disjoint({"a", "b"}, {"b", "c"}, {"d"})
    assert not sets_are_disjoint({"a"}, {"b"}, {"a", "c"})


def test_sets_are_disjoint_handles_single_and_empty_sets():
    assert sets_are_disjoint()
    assert sets_are_disjoint(set())
    assert sets_are_disjoint(set(), set())
