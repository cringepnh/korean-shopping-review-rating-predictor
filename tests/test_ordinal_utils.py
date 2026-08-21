import numpy as np

from ordinal_utils import (
    RATING_ARRAY,
    check_bias_monotonicity,
    cumulative_logits_to_ranks,
    ranks_to_ratings,
    ratings_to_ranks,
)


def test_ratings_to_ranks_and_back_is_a_round_trip():
    ratings = np.array([1, 2, 4, 5, 1, 5, 2, 4])
    ranks = ratings_to_ranks(ratings)
    assert np.array_equal(ranks, np.array([0, 1, 2, 3, 0, 3, 1, 2]))
    assert np.array_equal(ranks_to_ratings(ranks), ratings)


def test_cumulative_logits_to_ranks_all_positive_gives_top_rank():
    logits = np.full((1, 3), 5.0)  # every threshold exceeded
    assert cumulative_logits_to_ranks(logits)[0] == 3


def test_cumulative_logits_to_ranks_all_negative_gives_bottom_rank():
    logits = np.full((1, 3), -5.0)  # no threshold exceeded
    assert cumulative_logits_to_ranks(logits)[0] == 0


def test_cumulative_logits_to_ranks_monotone_case_matches_threshold_count():
    # rank 2 means: exceeds threshold 0 and 1, not threshold 2
    logits = np.array([[2.0, 1.0, -1.0]])
    assert cumulative_logits_to_ranks(logits)[0] == 2


def test_check_bias_monotonicity_flags_sorted_and_unsorted_biases():
    assert check_bias_monotonicity(np.array([1.0, 0.5, 0.1]))["monotone_decreasing"] is True
    assert check_bias_monotonicity(np.array([0.1, 0.5, 1.0]))["monotone_decreasing"] is False
    # ties are allowed (non-increasing, not strictly decreasing)
    assert check_bias_monotonicity(np.array([0.5, 0.5, 0.5]))["monotone_decreasing"] is True


def test_rating_array_matches_valid_ratings_order():
    assert list(RATING_ARRAY) == [1, 2, 4, 5]
