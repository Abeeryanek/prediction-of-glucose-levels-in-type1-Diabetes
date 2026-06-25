"""
Hyperparameter search for Random Forest and LSTM glucose forecasting models.

Random Forest grid search uses TimeSeriesSplit cross-validation rather than
standard k-fold, because k-fold allows future data to leak into training folds
for temporally autocorrelated signals — an issue documented by
Cerqueira et al. (2020) and Bergmeir & Benítez (2012).
The specific RF hyperparameter grid (n_estimators, max_depth, min_samples_leaf)
follows the empirical sensitivity analysis of Probst et al. (2019):
"Tunability: Importance of Hyperparameters of Machine Learning Algorithms",
which identifies these three as the most influential for Random Forests.

LSTM grid search sweeps hidden_size and learning rate on a fixed val set.
The learning rate range [0.001, 0.0005] is centred on the default recommended
by Kingma & Ba (2015): "Adam: A Method for Stochastic Optimization".
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import TimeSeriesSplit

from src.models import random_forest as rf_mod
from src.models import lstm as lstm_mod
from src.evaluation.metrics import rmse


# ---------------------------------------------------------------------------
# Random Forest — walk-forward CV grid search
# ---------------------------------------------------------------------------

_RF_PARAM_GRID = {
    "n_estimators": [50, 100, 200],
    "max_depth":    [10, 15, 20],
    "min_samples_leaf": [1, 2, 4],
}


def grid_search_rf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_splits: int = 3,
    param_grid: dict | None = None,
) -> tuple[dict, float]:
    """
    Exhaustive grid search for RandomForestRegressor using TimeSeriesSplit CV.

    Temporal ordering is preserved throughout — TimeSeriesSplit never shuffles,
    and each fold's validation data strictly follows its training data in time.
    This mirrors the walk-forward evaluation strategy recommended for time-series
    models (Cerqueira et al., 2020; Bergmeir & Benítez, 2012).

    The default parameter grid is motivated by Probst et al. (2019), who showed
    that n_estimators, max_depth, and min_samples_leaf account for the bulk of
    Random Forest variance in regression tasks.

    Parameters
    ----------
    X_train     : (n_samples, window_size * n_features)  flat, normalised
    y_train     : (n_samples, horizon)
    n_splits    : number of walk-forward folds (default 3)
    param_grid  : override the default grid if supplied

    Returns
    -------
    best_params : dict  — e.g. {'n_estimators': 100, 'max_depth': 15, ...}
    best_score  : float — mean RMSE across folds for best_params
    """
    grid = param_grid if param_grid is not None else _RF_PARAM_GRID
    tscv = TimeSeriesSplit(n_splits=n_splits)

    from itertools import product
    keys = list(grid.keys())
    combos = list(product(*[grid[k] for k in keys]))

    best_score = float("inf")
    best_params: dict = {}

    for combo in combos:
        params = dict(zip(keys, combo))
        fold_scores: list[float] = []

        for train_idx, val_idx in tscv.split(X_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            model = rf_mod.train(X_tr, y_tr, params=params)
            _, _, preds = rf_mod.evaluate(model, X_val, y_val)
            fold_scores.append(rmse(y_val, preds))

        mean_score = float(np.mean(fold_scores))
        if mean_score < best_score:
            best_score = mean_score
            best_params = params

    return best_params, best_score


# ---------------------------------------------------------------------------
# LSTM — validation-set grid search
# ---------------------------------------------------------------------------

_LSTM_PARAM_GRID = {
    "hidden_size": [32, 64, 128],
    "lr":          [1e-3, 5e-4],
}


def grid_search_lstm(
    X_train_3d: np.ndarray,
    y_train: np.ndarray,
    X_val_3d: np.ndarray,
    y_val: np.ndarray,
    n_features: int,
    horizon: int = 6,
    param_grid: dict | None = None,
    max_epochs: int = 50,
    patience: int = 7,
) -> tuple[dict, float]:
    """
    Grid search over LSTM hidden_size and learning rate on a fixed validation set.

    Uses a single train/val split (the official OhioT1DM or the pipeline's
    val_ratio slice) rather than cross-validation, because re-training deep
    learning models for every fold is prohibitively expensive at this scale.

    The learning rate range [1e-3, 5e-4] brackets the Adam default of α=0.001
    recommended by Kingma & Ba (2015): "Adam: A Method for Stochastic
    Optimization" — a value shown to work well across a broad class of
    gradient-based deep learning problems.

    Parameters
    ----------
    X_train_3d / X_val_3d : (n_samples, seq_len, n_features)  normalised
    y_train / y_val       : (n_samples, horizon)
    n_features            : number of input features per timestep
    horizon               : prediction horizon (default 6)
    param_grid            : override the default grid if supplied
    max_epochs / patience : passed to lstm.train_model

    Returns
    -------
    best_params : dict  — e.g. {'hidden_size': 64, 'lr': 0.001}
    best_score  : float — validation RMSE for best_params
    """
    from itertools import product

    grid = param_grid if param_grid is not None else _LSTM_PARAM_GRID
    keys = list(grid.keys())
    combos = list(product(*[grid[k] for k in keys]))

    best_score = float("inf")
    best_params: dict = {}

    for combo in combos:
        params = dict(zip(keys, combo))
        hidden_size = params["hidden_size"]
        lr = params["lr"]

        model = lstm_mod.GlucoseLSTM(
            n_features=n_features,
            hidden_size=hidden_size,
            horizon=horizon,
        )
        model, _ = lstm_mod.train_model(
            X_train_3d, y_train, X_val_3d, y_val,
            model=model, lr=lr,
            max_epochs=max_epochs, patience=patience,
        )

        # Score in normalised space (consistent ranking — no inverse-transform needed)
        import torch
        import torch.nn as nn
        device = next(model.parameters()).device
        X_t = torch.tensor(X_val_3d, dtype=torch.float32).to(device)
        model.eval()
        with torch.no_grad():
            preds = model(X_t).cpu().numpy()
        score = rmse(y_val, preds)

        if score < best_score:
            best_score = score
            best_params = params

    return best_params, best_score
