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


def calculate_clinical_weights(glucose_mgdl: np.ndarray) -> np.ndarray:
    """Clinical sample weights from glucose in mg/dL.
    Weights dangerous ranges more heavily (hypo hardest):
      <54 -> 3.0, 54-70 -> 2.5, in-range -> 1.0, 180-250 -> 1.5, >250 -> 2.0
    Used by tree models (via sample_weight) and the weighted MSE loss,
    so all models share one weighting definition.
    Input MUST be mg/dL, not scaled.

    NumPy port of Abeer's calculate_clinical_weights (gb_bigideas.py:177-183)
    — thresholds and weight multipliers unchanged.
    """
    y_true = np.asarray(glucose_mgdl)
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


def clinically_weighted_mse_scaled(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    y_mean: float,
    y_std: float,
) -> torch.Tensor:
    """
    Option-A variant of clinically_weighted_mse() for our unified pipeline,
    which z-scores glucose (X *and* y) before training — unlike Abeer's
    original, which only ever scales the features (see
    clean_treaining/lstm.py build_seq_dataset(): StandardScaler is fit on
    `features` only; `y_train`/`y_test` are the raw target column, mg/dL).
    Decision made with Abeer: keep z-scoring the model's target, but
    un-scale back to mg/dL *inside the loss* so her thresholds
    (54/70/180/250) apply to real glucose.

    y_mean, y_std: the mean/std the glucose target was scaled with (i.e.
    the fitted target scaler's parameters), so
        y_true_mgdl = y_true * y_std + y_mean
    recovers real glucose from the z-scored label.

    Design decision — squared error computed in mg/dL space, not scaled
    space:
    Abeer's loss is ((y_pred_mgdl - y_true_mgdl) ** 2 * weights).mean() —
    both the weights AND the error term operate on raw mg/dL, because her
    model's output is never scaled to begin with. Under Option A our
    model's raw output is a z-score, so matching her numbers requires
    un-scaling y_pred as well as y_true before differencing — not just
    un-scaling y_true for the weight lookup. If the squared-error term were
    left in scaled space (only weights computed from un-scaled y_true), the
    result would carry correct weights but the wrong units: scaled MSE is
    related to mg/dL MSE by (mg/dL_error)^2 = (y_std^2) * (scaled_error)^2
    only when comparing the same points, and diverges further once you're
    averaging different weighted subsets — it would NOT reduce to the same
    scalar as Abeer's loss, only be proportional to it in the unweighted
    case. So both y_true and y_pred are un-scaled to mg/dL first; weights
    and squared error are then both computed on those mg/dL values, exactly
    mirroring Abeer's function. This does not change training dynamics:
    y_pred_mgdl is an affine function of y_pred_scaled with constant slope
    y_std, so gradients w.r.t. model parameters point the same direction,
    just rescaled by y_std^2 (equivalent to a learning-rate rescale).
    """
    y_true_mgdl = y_true * y_std + y_mean
    y_pred_mgdl = y_pred * y_std + y_mean

    # calculate_clinical_weights sizes its output off len(input) (first-dim
    # only), so a 2D (batch, horizon) tensor must be flattened before the
    # call and reshaped back — weights are computed elementwise off each
    # value's own mg/dL level regardless of shape, so this is a no-op for
    # the actual weight assigned to any given element.
    weights_np = calculate_clinical_weights(
        y_true_mgdl.detach().cpu().numpy().ravel()
    )
    weights = torch.as_tensor(
        weights_np, dtype=torch.float32, device=y_true_mgdl.device
    ).reshape(y_true_mgdl.shape)

    squared_error = (y_pred_mgdl - y_true_mgdl) ** 2
    return (squared_error * weights).mean()
