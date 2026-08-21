import numpy as np

from decoding import (
    decode_argmax,
    decode_expected_value,
    decode_expected_value_rounded,
    decode_median,
)

RATINGS = np.array([1, 2, 4, 5])


def test_decode_argmax_picks_highest_probability_class():
    probabilities = np.array(
        [
            [0.7, 0.1, 0.1, 0.1],  # rating 1
            [0.1, 0.1, 0.1, 0.7],  # rating 5
        ]
    )
    assert np.array_equal(decode_argmax(probabilities, RATINGS), np.array([1, 5]))


def test_decode_expected_value_is_probability_weighted_average():
    probabilities = np.array([[0.5, 0.5, 0.0, 0.0]])  # midpoint of 1 and 2
    expected = decode_expected_value(probabilities, RATINGS)
    assert expected[0] == 1.5


def test_decode_expected_value_matches_certain_prediction():
    probabilities = np.array([[0.0, 0.0, 1.0, 0.0]])  # certain rating 4
    assert decode_expected_value(probabilities, RATINGS)[0] == 4.0


def test_decode_expected_value_rounded_snaps_to_nearest_valid_rating():
    # E[rating] = 1*0.5 + 2*0.5 = 1.5, equidistant from 1 and 2 -> argmin picks
    # the first (lowest index) on ties, i.e. rating 1.
    probabilities = np.array([[0.5, 0.5, 0.0, 0.0]])
    rounded = decode_expected_value_rounded(probabilities, RATINGS)
    assert rounded[0] in (1, 2)  # tie-break implementation detail, but must be valid
    assert rounded[0] in RATINGS


def test_decode_expected_value_rounded_handles_the_missing_three_star_gap():
    # E[rating] = 2*0.5 + 4*0.5 = 3.0, exactly between 2 and 4 (the gap where
    # 3-star reviews would be). Must still snap to one of the *valid* ratings.
    probabilities = np.array([[0.0, 0.5, 0.5, 0.0]])
    rounded = decode_expected_value_rounded(probabilities, RATINGS)
    assert rounded[0] in RATINGS


def test_decode_expected_value_rounded_always_returns_valid_ratings():
    rng = np.random.default_rng(0)
    raw = rng.random((50, 4))
    probabilities = raw / raw.sum(axis=1, keepdims=True)
    rounded = decode_expected_value_rounded(probabilities, RATINGS)
    assert set(np.unique(rounded)).issubset(set(RATINGS.tolist()))


def test_decode_median_matches_certain_prediction():
    probabilities = np.array([[0.0, 1.0, 0.0, 0.0]])  # certain rating 2
    assert decode_median(probabilities, RATINGS)[0] == 2


def test_decode_median_picks_the_class_where_cumulative_probability_crosses_half():
    # cumulative: 1 -> 0.3, 2 -> 0.7 (crosses 0.5 here), 4 -> 0.9, 5 -> 1.0
    probabilities = np.array([[0.3, 0.4, 0.2, 0.1]])
    assert decode_median(probabilities, RATINGS)[0] == 2


def test_decode_median_always_returns_valid_ratings():
    rng = np.random.default_rng(1)
    raw = rng.random((50, 4))
    probabilities = raw / raw.sum(axis=1, keepdims=True)
    medians = decode_median(probabilities, RATINGS)
    assert set(np.unique(medians)).issubset(set(RATINGS.tolist()))


def test_decode_median_differs_from_argmax_when_distribution_is_skewed():
    # argmax picks rating 5 (highest single probability), but the cumulative
    # mass crosses 0.5 at rating 2: median and argmax should disagree here.
    probabilities = np.array([[0.2, 0.35, 0.05, 0.4]])
    assert decode_argmax(probabilities, RATINGS)[0] == 5
    assert decode_median(probabilities, RATINGS)[0] == 2
