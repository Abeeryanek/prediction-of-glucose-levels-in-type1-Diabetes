"""Random Forest baseline for multi-step blood glucose forecasting."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

from src.evaluation.metrics import rmse, mae

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

    Returns
    -------
    Fitted MultiOutputRegressor wrapping a RandomForestRegressor.
    """
    cfg = {**_DEFAULT_PARAMS, **(params or {})}
    model = MultiOutputRegressor(RandomForestRegressor(**cfg), n_jobs=-1)
    model.fit(X_train, y_train)
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
