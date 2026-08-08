"""Gradient Boosting baseline for multi-step blood glucose forecasting.

Ported from Abeer's src/models/clean_treaining/gb_bigideas.py into the
shared interface src/models/random_forest.py exposes — train()/evaluate(),
not the neural train_model(...X_val, y_val...) shape, since GB (like RF)
has no early stopping to do. Only the calling interface and data shape
were adapted; the model itself (GradientBoostingRegressor, its fixed
subsample, and its grid-searched hyperparameter ranges) is unchanged from
Abeer's original.

Clinical sample weighting (Option 2, decided 2026-08-08): GB keeps its
weighting exactly as Abeer built it, but computed via the single shared
src.training.losses.calculate_clinical_weights function — never
re-implemented here — so tree models (sample_weight) and the neural
weighted-MSE loss all share one weighting definition.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

from src.evaluation.metrics import rmse, mae
from src.training.losses import calculate_clinical_weights

# n_estimators/max_depth/min_samples_leaf mirror random_forest.py's
# _DEFAULT_PARAMS (100/15/2) for baseline consistency between the two tree
# models — Abeer's gb_bigideas.py never had a single "default" for these
# (they're grid-searched per window size via GB_PARAM_GRID, same grid RF
# uses). subsample=0.8 and random_state=42 ARE Abeer's fixed, non-grid-
# searched GB hyperparameters (gb_bigideas.py:490):
#     gb_params = {**best_params, "random_state": RANDOM_SEED, "subsample": 0.8}
_DEFAULT_PARAMS = {
    "n_estimators": 100,
    "max_depth": 15,
    "min_samples_leaf": 2,
    "subsample": 0.8,
    "random_state": 42,
}


def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: dict | None = None,
    y_train_raw: np.ndarray | None = None,
) -> MultiOutputRegressor:
    """
    Fit a Gradient Boosting ensemble for multi-step glucose forecasting.

    Parameters
    ----------
    X_train : (n_samples, window_size * n_features)
        Flat feature matrix — identical shape contract to
        random_forest.train() (pipeline.create_windows(flat=True) /
        make_splits(..., flat=True)).
    y_train : (n_samples, horizon)
        Target glucose values (may be normalised — GB is indifferent,
        same as RF).
    params : dict, optional
        GradientBoostingRegressor kwargs. Defaults to _DEFAULT_PARAMS.
    y_train_raw : (n_samples, horizon), optional
        Unscaled glucose targets in mg/dL — e.g. splits["y_train_raw"]
        from pipeline.make_splits(). Required to enable clinical sample
        weighting (Option 2); if omitted, GB trains unweighted (same as
        an unweighted RF call). This param is new relative to
        random_forest.train()'s 3-argument signature — it's purely
        additive with a None default, so existing RF-style call sites
        (`train(X_train, y_train, params=...)`) work unchanged for GB.
        It has to exist because sample_weight has no scaled-space
        equivalent: clinical thresholds (54/70/180/250) are mg/dL-only.

    Returns
    -------
    Fitted MultiOutputRegressor wrapping a GradientBoostingRegressor.
    """
    cfg = {**_DEFAULT_PARAMS, **(params or {})}

    sample_weight = None
    if y_train_raw is not None:
        # calculate_clinical_weights returns one weight per element of its
        # input, but sklearn's sample_weight wants exactly one scalar per
        # row (MultiOutputRegressor applies the same weight to every
        # horizon step of a given sample) — y_train_raw is 2D
        # (n_samples, horizon) under multi-step targets, so it must be
        # collapsed to one mg/dL reference value per sample first.
        # Convention used here: the nearest-horizon (first) target column.
        # This is a new design decision, not present in Abeer's original —
        # her script trains one single-output model per horizon and so
        # never had a multi-step array to collapse. Flagged for review.
        ref = y_train_raw[:, 0] if np.ndim(y_train_raw) > 1 else y_train_raw
        sample_weight = calculate_clinical_weights(ref)

    # GradientBoostingRegressor has no internal n_jobs (unlike
    # RandomForestRegressor), so the nested-process-pool problem RF's
    # comment describes doesn't apply here. n_jobs=1 on the outer wrapper
    # is kept anyway, purely for interface symmetry with random_forest.train().
    model = MultiOutputRegressor(GradientBoostingRegressor(**cfg), n_jobs=1)
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model


def evaluate(
    model: MultiOutputRegressor,
    X_test: np.ndarray,
    y_test_raw: np.ndarray,
    y_mean: float,
    y_std: float,
) -> tuple[float, float, np.ndarray]:
    """
    Evaluate a fitted Gradient Boosting model on test data.

    Identical contract to random_forest.evaluate(): predictions are in
    normalised space (GB is trained on scaled y from make_splits, same as
    RF), inverse-transformed to mg/dL before computing metrics.

    Parameters
    ----------
    X_test      : (n_samples, window_size * n_features)
    y_test_raw  : (n_samples, horizon)  original scale [mg/dL]
    y_mean, y_std : glucose normalisation stats from the pipeline scaler
                    (scaler.mean_[glucose_idx], scaler.scale_[glucose_idx])

    Returns
    -------
    (rmse_val, mae_val, predictions_mgdl)
        predictions_mgdl has shape (n_samples, horizon) in mg/dL.
    """
    pred_norm = model.predict(X_test)
    predictions_mgdl = pred_norm * y_std + y_mean
    return rmse(y_test_raw, predictions_mgdl), mae(y_test_raw, predictions_mgdl), predictions_mgdl
