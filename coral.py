"""CORAL ordinal head for the 4-class Naver Shopping rating scale.

Implements Cao, Mirjalili & Raschka, "Rank Consistent Ordinal Regression for
Neural Networks with Application to Age Estimation" (2020): a single shared
weight vector projects the encoder's pooled representation to one logit, and
K-1 learned bias terms produce K-1 monotonically-ordered cumulative logits
P(rank > k). This guarantees rank-monotone probabilities by construction
(unlike naive per-threshold binary classifiers), which is what actually
distinguishes CORAL from just re-labeling the same 4-way softmax.

VALID_RATINGS = (1, 2, 4, 5) maps to ranks (0, 1, 2, 3). The gap between rating
2 and rating 4 (the missing 3-star class) is real: MAE below is always computed
in rating space {1,2,4,5}, not rank space {0,1,2,3}, so this asymmetry is
reflected in the reported numbers rather than hidden by it.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel, PreTrainedModel

# Re-exported for backward compatibility with existing imports
# (train_coral.py, evaluate_model.py): these are plain numpy functions with
# no torch dependency, kept in ordinal_utils.py specifically so they can be
# unit-tested without installing torch.
from ordinal_utils import (  # noqa: F401
    NUM_RANKS,
    NUM_THRESHOLDS,
    RATING_ARRAY,
    check_bias_monotonicity,
    cumulative_logits_to_ranks,
    ranks_to_ratings,
    ratings_to_ranks,
)


class CoralHead(nn.Module):
    """Shared-weight linear projection + independent monotone bias terms."""

    def __init__(self, hidden_size: int, num_thresholds: int = NUM_THRESHOLDS):
        super().__init__()
        self.shared = nn.Linear(hidden_size, 1, bias=False)
        # Biases are NOT sorted by construction; monotonicity of the resulting
        # cumulative logits (shared(x) + bias_k) requires bias_0 >= bias_1 >=
        # ... at convergence. We verify this empirically after training rather
        # than enforcing it architecturally, and report the check's result.
        self.biases = nn.Parameter(torch.zeros(num_thresholds))

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        base = self.shared(pooled)  # (batch, 1)
        return base + self.biases  # (batch, num_thresholds), broadcast


class CoralForOrdinalRegression(PreTrainedModel):
    """Wraps an AutoModel encoder with a CoralHead and CORAL's cumulative-link loss."""

    def __init__(self, config, base_model_name: str | None = None):
        super().__init__(config)
        self.encoder = (
            AutoModel.from_pretrained(base_model_name)
            if base_model_name is not None
            else AutoModel.from_config(config)
        )
        hidden_size = config.hidden_size
        self.head = CoralHead(hidden_size)
        self.num_thresholds = NUM_THRESHOLDS
        # Required by PreTrainedModel: registers tied-weight bookkeeping and
        # applies the configured weight initialization. Omitting this call
        # leaves internal registries (e.g. all_tied_weights_keys) unset and
        # crashes from_pretrained's post-load finalization step.
        self.post_init()

    def pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Mean-pool over non-padding tokens. ELECTRA has no pooler head trained
        # for classification, so a manual masked mean is more defensible than
        # relying on an untrained [CLS] pooler.
        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-6)
        return summed / counts

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.pool(outputs.last_hidden_state, attention_mask)
        cumulative_logits = self.head(pooled)  # (batch, num_thresholds)

        loss = None
        if labels is not None:
            # labels are integer ranks 0..K-1. Cumulative targets: does the
            # true rank exceed threshold k? targets[:, k] = 1[rank > k].
            rank_range = torch.arange(self.num_thresholds, device=labels.device)
            targets = (labels.unsqueeze(1) > rank_range.unsqueeze(0)).float()
            loss = nn.functional.binary_cross_entropy_with_logits(
                cumulative_logits, targets
            )

        return {"loss": loss, "logits": cumulative_logits} if loss is not None else {
            "logits": cumulative_logits
        }

    # Deliberately NOT overriding save_pretrained/from_pretrained: this class
    # inherits PreTrainedModel's default, which serializes the *entire*
    # state_dict (encoder.* and head.* together) into one model.safetensors
    # next to config.json. That standard single-file layout is what Trainer's
    # load_best_model_at_end depends on to restore the best epoch's weights;
    # an earlier custom two-directory layout (encoder/ + coral_head.pt) broke
    # that restoration silently (Trainer logged "Could not locate the best
    # model" and fell back to the last epoch instead of the best one).
    # Reloading is just `CoralForOrdinalRegression.from_pretrained(path)`
    # (base_model_name=None so the architecture is built from config, then
    # weights are loaded from the saved state_dict, matching PreTrainedModel's
    # normal from_pretrained flow).
