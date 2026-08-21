"""Pure numpy helpers for the CORAL ordinal rank <-> rating mapping.

Split out of coral.py specifically so these can be unit-tested (and used by
evaluate_model.py) without importing torch/transformers. coral.py imports
from here and re-exports the same names for backward compatibility.
"""

from __future__ import annotations

import numpy as np

from data_utils import VALID_RATINGS

NUM_RANKS = len(VALID_RATINGS)  # 4
NUM_THRESHOLDS = NUM_RANKS - 1  # 3
RATING_ARRAY = np.array(VALID_RATINGS, dtype=np.int64)


def cumulative_logits_to_ranks(cumulative_logits: np.ndarray) -> np.ndarray:
    """Decode CORAL cumulative logits into integer ranks 0..K-1.

    rank = number of thresholds exceeded = sum_k 1[sigmoid(logit_k) > 0.5],
    equivalently sum_k 1[logit_k > 0]. This is the standard CORAL decode; it
    does not require the thresholds to be sorted to produce a valid rank in
    [0, K-1], but rank-monotone *probabilities* still require sorted biases,
    which is checked separately by check_bias_monotonicity.
    """
    exceeded = cumulative_logits > 0.0
    return exceeded.sum(axis=1).astype(np.int64)


def ranks_to_ratings(ranks: np.ndarray) -> np.ndarray:
    return RATING_ARRAY[np.clip(ranks, 0, NUM_RANKS - 1)]


def ratings_to_ranks(ratings: np.ndarray) -> np.ndarray:
    rating_to_rank = {rating: rank for rank, rating in enumerate(VALID_RATINGS)}
    return np.array([rating_to_rank[int(r)] for r in ratings], dtype=np.int64)


def check_bias_monotonicity(biases: np.ndarray) -> dict:
    """CORAL's rank-monotone guarantee requires biases[0] >= biases[1] >= ...

    This is NOT architecturally enforced in CoralHead (no reparameterization
    with e.g. softplus increments was used), so it is verified empirically
    post-training and reported as whichever is true, rather than asserted.
    """
    sorted_desc = bool(np.all(np.diff(np.asarray(biases)) <= 1e-6))
    return {
        "biases": np.asarray(biases).tolist(),
        "monotone_decreasing": sorted_desc,
        "note": (
            "Biases are monotonically non-increasing, confirming rank-monotone "
            "cumulative probabilities at convergence."
            if sorted_desc
            else "Biases are NOT monotonic: rank-monotonicity is violated for "
            "this trained model despite CORAL's shared-weight design."
        ),
    }
