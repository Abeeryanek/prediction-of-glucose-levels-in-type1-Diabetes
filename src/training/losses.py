"""
Shared loss functions for the unified training pipeline.

clinically_weighted_mse() ports Abeer's clinical sample-weighting scheme —
originally duplicated verbatim across all 7 model files in
src/models/clean_treaining/ plus src/training/rf_and_gb_pipeline.py and
src/training/lstm_and_cnnlstm_pipeline.py (9 copies total) — into a single
reusable loss. It preserves her exact thresholds and weight multipliers.

Reference implementation (unchanged across all 9 original copies), e.g.:
    src/models/clean_treaining/lstm.py:216-222

    def calculate_clinical_weights(y_true):
        weights = np.ones(len(y_true), dtype=np.float32)
        weights[y_true < 54] = 3.0
        weights[(y_true >= 54) & (y_true < 70)] = 2.5
        weights[(y_true > 180) & (y_true <= 250)] = 1.5
        weights[y_true > 250] = 2.0
        return weights

Weighting scheme (glucose in mg/dL):
    < 54            severe hypoglycemia    -> weight 3.0
    54 <= x < 70    moderate hypoglycemia  -> weight 2.5
    70 <= x <= 180  normal range           -> weight 1.0 (default)
    180 < x <= 250  moderate hyperglycemia -> weight 1.5
    > 250           severe hyperglycemia   -> weight 2.0

Not yet wired into any model's train_model() — see FINAL_EXPERIMENTS_PLAN_DETAILED.md
for the porting plan. This module only provides the standalone building block.
"""

from __future__ import annotations

import numpy as np
import torch


def calculate_clinical_weights(y_true: np.ndarray) -> np.ndarray:
    """
    NumPy port of Abeer's calculate_clinical_weights — thresholds and weight
    multipliers unchanged. Intended for tree-model `sample_weight=` (RF/GB)
    and as the reference this module's torch loss is verified against.
    """
    y_true = np.asarray(y_true)
    weights = np.ones(len(y_true), dtype=np.float32)
    weights[y_true < 54] = 3.0
    weights[(y_true >= 54) & (y_true < 70)] = 2.5
    weights[(y_true > 180) & (y_true <= 250)] = 1.5
    weights[y_true > 250] = 2.0
    return weights


def clinically_weighted_mse(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    Clinically-weighted MSE, callable with the same (y_pred, y_true) signature
    as nn.MSELoss()(y_pred, y_true) — a drop-in replacement for any model's
    `criterion` so train_model() can accept it via a loss_fn argument.

    Reproduces (nn.MSELoss(reduction="none")(preds, y) * weights).mean() from
    Abeer's original training loops (e.g. clean_treaining/cnn_lstm_bigideas.py:285),
    but computes weights on the fly from y_true instead of precomputing and
    storing them as a third Dataset tensor. Since weight(y_true[i]) depends
    only on that single label, this is mathematically identical regardless
    of batching — just architecturally simpler.
    """
    weights = torch.ones_like(y_true, dtype=torch.float32)
    weights[y_true < 54] = 3.0
    weights[(y_true >= 54) & (y_true < 70)] = 2.5
    weights[(y_true > 180) & (y_true <= 250)] = 1.5
    weights[y_true > 250] = 2.0

    squared_error = (y_pred - y_true) ** 2
    return (squared_error * weights).mean()
