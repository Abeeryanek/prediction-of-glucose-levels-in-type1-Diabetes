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
    y_test: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    """
    Evaluate a fitted Random Forest on test data.

    Parameters
    ----------
    X_test  : (n_samples, window_size * n_features)
    y_test  : (n_samples, horizon)  in the same scale as used during training

    Returns
    -------
    (rmse_val, mae_val, predictions)
        predictions has shape (n_samples, horizon).
    """
    predictions = model.predict(X_test)
    return rmse(y_test, predictions), mae(y_test, predictions), predictions
