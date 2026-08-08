"""Random Forest baseline for multi-step blood glucose forecasting."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

from src.evaluation.metrics import rmse, mae
from src.training.losses import calculate_clinical_weights

_DEFAULT_PARAMS = {
    "n_estimators": 100,
    "max_depth": 15,
    "min_samples_leaf": 2,
    "n_jobs": -1,
    "random_state": 42,
}


def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: dict | None = None,
    y_train_raw: np.ndarray | None = None,
) -> MultiOutputRegressor:
    """
    Fit a Random Forest for multi-step glucose forecasting.

    Parameters
    ----------
    X_train : (n_samples, window_size * n_features)
        Flat feature matrix produced by pipeline.create_windows(flat=True).
    y_train : (n_samples, horizon)
        Target glucose values (may be normalised — the RF is indifferent).
    params : dict, optional
        RandomForestRegressor kwargs. Defaults to _DEFAULT_PARAMS.
    y_train_raw : (n_samples, horizon), optional
        Unscaled glucose targets in mg/dL — e.g. splits["y_train_raw"]
        from pipeline.make_splits(). Required to enable clinical sample
        weighting (Option 2); if omitted, RF trains unweighted (previous
        behaviour, unchanged). NOT the same array as y_train: y_train is
        typically the scaled target the model is fit on, and clinical
        thresholds (54/70/180/250) are mg/dL-only — computing weights
        from scaled y would silently misfire. New relative to the
        previous 3-argument signature — purely additive with a None
        default, so existing call sites (`train(X_train, y_train,
        params=...)`) keep working unweighted, unchanged.

    Returns
    -------
    Fitted MultiOutputRegressor wrapping a RandomForestRegressor.
    """
    cfg = {**_DEFAULT_PARAMS, **(params or {})}

    sample_weight = None
    if y_train_raw is not None:
        # calculate_clinical_weights returns one weight per element of its
        # input, but MultiOutputRegressor's sample_weight wants exactly one
        # scalar per row (the same weight applies to every horizon step of
        # a given sample) — y_train_raw is 2D (n_samples, horizon) under
        # multi-step targets, so it's collapsed to one mg/dL reference
        # value per sample first. Convention: the nearest-horizon (first)
        # target column — same choice made in gradient_boosting.train()
        # for consistency between the two tree models.
        ref = y_train_raw[:, 0] if np.ndim(y_train_raw) > 1 else y_train_raw
        sample_weight = calculate_clinical_weights(ref)

    # RandomForestRegressor already parallelises over its own trees
    # (n_jobs=-1 in _DEFAULT_PARAMS); wrapping it in another n_jobs=-1
    # MultiOutputRegressor causes nested process pools that exhaust OS
    # resources on Windows. Keep the outer wrapper single-process.
    model = MultiOutputRegressor(RandomForestRegressor(**cfg), n_jobs=1)
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
    Evaluate a fitted Random Forest on test data.

    Predictions are in normalised space (the RF is trained on scaled y from
    make_splits). This function inverse-transforms them to mg/dL before
    computing metrics, matching the interface of lstm.evaluate() and
    autoencoder.evaluate().

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
