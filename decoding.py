"""Decoders that turn a 4-class probability distribution over ratings
{1,2,4,5} into a single predicted rating.

The trained classifier optimizes cross-entropy, whose Bayes-optimal decision
rule under 0-1 loss is argmax. The headline metric in this project is MAE, and
argmax is not the MAE-optimal decoder for a given predicted distribution.

An earlier version of this module claimed the *expected value* (mean) of the
distribution minimizes MAE. That is wrong: the mean minimizes expected
*squared* error (RMSE); the statistic that minimizes expected *absolute*
error is the *median*. Running all four decoders here on the same trained
model's predictions confirms the theory rather than the earlier claim:
expected value improves RMSE over argmax (0.7370 vs 0.8364) but makes MAE
*worse* (0.4862 vs 0.3856), while the median decoder (added after finding
this) is the one that actually targets MAE. See evaluate_model.py's saved
comparison table and its "notes" field for the exact numbers this was
checked against.
"""

from __future__ import annotations

import numpy as np


def decode_argmax(probabilities: np.ndarray, ratings: np.ndarray) -> np.ndarray:
    """Standard decoder: the class with the highest probability. Bayes-optimal
    under 0-1 loss, not under MAE or RMSE."""
    return ratings[np.argmax(probabilities, axis=1)]


def decode_expected_value(probabilities: np.ndarray, ratings: np.ndarray) -> np.ndarray:
    """Continuous expected rating E[rating] = sum_i p_i * rating_i.

    Minimizes expected *squared* error (RMSE) under the model's own predicted
    distribution -- not MAE (see module docstring). It is never one of the
    four observed rating values, so exact_accuracy is necessarily 0% for this
    decoder, the same reason the "constant: training mean" baseline has 0%
    exact accuracy. Read this decoder's row as RMSE-only.
    """
    return probabilities @ ratings.astype(np.float64)


def decode_expected_value_rounded(probabilities: np.ndarray, ratings: np.ndarray) -> np.ndarray:
    """Expected value snapped to the nearest valid rating in {1,2,4,5}.

    Restores a usable exact_accuracy relative to the continuous expected
    value, by rounding to the closest element of the (unevenly spaced) valid
    rating set rather than the closest integer. Still not the MAE-optimal
    decoder -- see decode_median for that.
    """
    continuous = decode_expected_value(probabilities, ratings)
    ratings = ratings.astype(np.float64)
    distances = np.abs(continuous[:, None] - ratings[None, :])
    nearest_index = np.argmin(distances, axis=1)
    return ratings[nearest_index].astype(np.int64)


def decode_median(probabilities: np.ndarray, ratings: np.ndarray) -> np.ndarray:
    """Discrete median of the predicted distribution: the smallest rating at
    which the cumulative probability (in rating order) reaches 0.5.

    This, not the mean, is the decoder that actually minimizes expected
    absolute error (MAE) for a fixed predicted distribution -- the standard
    fact that the median minimizes E[|X - c|] over c, while the mean
    minimizes E[(X - c)^2]. Always returns one of the observed rating values,
    so unlike decode_expected_value this has a meaningful exact_accuracy.
    """
    order = np.argsort(ratings)
    sorted_ratings = ratings[order]
    sorted_probabilities = probabilities[:, order]
    cumulative = np.cumsum(sorted_probabilities, axis=1)
    median_index = (cumulative >= 0.5).argmax(axis=1)  # first index reaching 0.5
    return sorted_ratings[median_index]
