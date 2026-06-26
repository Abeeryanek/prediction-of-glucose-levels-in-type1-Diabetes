"""
Training pipeline utilities for blood glucose forecasting.

Temporal ordering is always preserved (shuffle=False). Normalisation is
always derived from the training portion only to prevent data leakage.

References
----------
Cerqueira et al. (2020): "Evaluating time series forecasting models: an
    empirical study on performance estimation methods."
Bergmeir & Benítez (2012): "On the use of cross-validation for time series
    predictor evaluation."
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

# Columns that represent discrete events — absence means the event did not occur
_EVENT_COLS: frozenset[str] = frozenset({"carbs", "bolus"})


def _preprocess(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Prepare a DataFrame for windowing.

    - Event columns (carbs, bolus): NaN → 0 (absence of event, not missing sensor).
    - All other feature columns and 'glucose': rows with NaN are dropped entirely.
      Interpolation is intentionally avoided to preserve real sensor gaps.
    """
    df = df.copy()
    for col in feature_cols:
        if col in _EVENT_COLS:
            df[col] = df[col].fillna(0.0)

    sensor_cols = [c for c in feature_cols if c not in _EVENT_COLS]
    drop_on = list(set(sensor_cols) | ({"glucose"} & set(df.columns)))
    df = df.dropna(subset=drop_on)
    return df.reset_index(drop=True)


def _scale(
    df_train: pd.DataFrame,
    df_other: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Fit a StandardScaler on df_train[feature_cols] and transform both frames.
    df_other statistics never influence the scaler — no leakage.
    """
    scaler = StandardScaler()
    df_train = df_train.copy()
    df_other = df_other.copy()
    df_train[feature_cols] = scaler.fit_transform(df_train[feature_cols])
    df_other[feature_cols] = scaler.transform(df_other[feature_cols])
    return df_train, df_other, scaler


def create_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    window_size: int = 12,
    horizon: int = 6,
    flat: bool = True,
    multi_step: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Slide a fixed-size window over a time-series DataFrame to produce (X, y) arrays.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'glucose' and all feature_cols. Call _preprocess first, or
        ensure sensor NaN rows are already dropped and event NaN already filled.
    feature_cols : list of str
        Ordered list of input feature columns.
    window_size : int
        Past timesteps per sample. Default 12 = 60 min at 5-min CGM resolution.
    horizon : int
        Future glucose timesteps ahead. Default 6 = 30 min ahead.
    flat : bool
        True  → X shape (n_samples, window_size * n_features)  [Random Forest / linear]
        False → X shape (n_samples, window_size, n_features)   [LSTM / TCN]
    multi_step : bool
        False (default) — single-step target: y shape (n_samples,), the glucose
            value at exactly *horizon* steps ahead. Use for RF, LSTM, TCN.
        True  — multi-step target: y shape (n_samples, horizon), glucose values
            at steps 1 … horizon ahead. Use for the Seq2Seq Autoencoder.

    Returns
    -------
    X : np.ndarray  (float32)
    y : np.ndarray  (float32)
        shape (n_samples,)        when multi_step=False
        shape (n_samples, horizon) when multi_step=True
    """
    df = df.copy()
    for col in feature_cols:
        if col in _EVENT_COLS:
            df[col] = df[col].fillna(0.0)

    feat_vals = df[feature_cols].to_numpy(dtype=np.float32)
    glucose_vals = df["glucose"].to_numpy(dtype=np.float32)

    n = len(df)
    n_samples = n - window_size - horizon + 1
    if n_samples <= 0:
        raise ValueError(
            f"Not enough rows ({n}) for window_size={window_size} + horizon={horizon}."
        )

    X_list: list[np.ndarray] = []
    y_list: list = []
    for i in range(n_samples):
        window = feat_vals[i : i + window_size]
        if multi_step:
            target = glucose_vals[i + window_size : i + window_size + horizon]
        else:
            target = glucose_vals[i + window_size + horizon - 1]
        X_list.append(window.flatten() if flat else window)
        y_list.append(target)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


def walk_forward_splits(
    df: pd.DataFrame,
    feature_cols: list[str],
    horizon_steps: int,
    n_splits: int = 3,
    flat: bool = False,
    multi_step: bool = False,
):
    """
    Walk-forward cross-validation using scikit-learn's TimeSeriesSplit.

    Each fold expands the training set chronologically; temporal order is
    preserved throughout (TimeSeriesSplit never shuffles). This strategy is
    preferred over k-fold for time series because it avoids future leakage
    and respects the autocorrelation structure of CGM data
    (Cerqueira et al., 2020; Bergmeir & Benítez, 2012).

    Normalisation is refit independently per fold using only that fold's
    training rows, so no validation statistics ever influence the scaler.

    Parameters
    ----------
    df : pd.DataFrame
        Single-patient DataFrame with a 'glucose' column and feature_cols.
    feature_cols : list of str
    horizon_steps : int
        Prediction horizon in timesteps (e.g. 6 = 30 min).
    n_splits : int
        Number of walk-forward folds (default 3).
    flat : bool
        Passed through to create_windows.
    multi_step : bool
        Passed through to create_windows. False → y shape (n_samples,);
        True → y shape (n_samples, horizon).

    Yields
    ------
    X_train, y_train, X_val, y_val, scaler : tuple
        Arrays are float32. scaler is a fitted StandardScaler for inverse
        transformation of predictions.
    """
    df = _preprocess(df, feature_cols)
    tscv = TimeSeriesSplit(n_splits=n_splits)

    for train_idx, val_idx in tscv.split(np.arange(len(df))):
        df_tr = df.iloc[train_idx].reset_index(drop=True)
        df_val = df.iloc[val_idx].reset_index(drop=True)

        df_tr_sc, df_val_sc, scaler = _scale(df_tr, df_val, feature_cols)

        X_tr, y_tr = create_windows(
            df_tr_sc, feature_cols, horizon=horizon_steps, flat=flat, multi_step=multi_step
        )
        X_val, y_val = create_windows(
            df_val_sc, feature_cols, horizon=horizon_steps, flat=flat, multi_step=multi_step
        )

        yield X_tr, y_tr, X_val, y_val, scaler


def make_splits(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_cols: list[str],
    horizon_steps: int,
    val_ratio: float = 0.20,
    multi_step: bool = False,
    flat: bool = False,
) -> dict[str, np.ndarray | StandardScaler]:
    """
    Prepare train / val / test arrays from the official OhioT1DM split files.

    The validation set is carved from the chronological end of df_train
    (shuffle=False — temporal order is preserved). The scaler is fit only on
    the training rows and then applied to validation and test identically,
    so neither val nor test statistics influence normalisation.

    Raw (unscaled) y arrays in mg/dL are extracted BEFORE scaling and stored
    under 'y_train_raw', 'y_val_raw', 'y_test_raw'. Pass these to evaluate()
    so that RMSE and MAE are reported in mg/dL, not in normalised units.

    Parameters
    ----------
    df_train : pd.DataFrame
        Loaded from the OhioT1DM train XML (e.g. via ohio_loader.load_patient).
    df_test : pd.DataFrame
        Loaded from the OhioT1DM test XML.
    feature_cols : list of str
        Must include 'glucose'. Event columns ('carbs', 'bolus') are allowed
        and will have NaN filled with 0 automatically.
    horizon_steps : int
        Prediction horizon in timesteps.
    val_ratio : float
        Fraction of training rows reserved for validation (default 0.20).
        Applied to the end of df_train to keep temporal ordering intact.
    multi_step : bool
        False (default) → y arrays have shape (n_samples,)          [RF, LSTM, TCN]
        True            → y arrays have shape (n_samples, horizon)  [Seq2Seq Autoencoder]
    flat : bool
        False (default) → X shape (n_samples, window_size, n_features) [LSTM / TCN / Autoencoder]
        True             → X shape (n_samples, window_size * n_features) [Random Forest]

    Returns
    -------
    dict with keys:
        'X_train', 'y_train',
        'X_val',   'y_val',
        'X_test',  'y_test',
        'y_train_raw', 'y_val_raw', 'y_test_raw',  ← unscaled glucose [mg/dL], shape (n, horizon)
        'scaler'   (fitted StandardScaler — use scaler.mean_ / scaler.scale_ for inverse transform)
    """
    df_train = _preprocess(df_train, feature_cols)
    df_test = _preprocess(df_test, feature_cols)

    n_val = max(1, int(len(df_train) * val_ratio))
    df_tr = df_train.iloc[:-n_val].reset_index(drop=True)
    df_val = df_train.iloc[-n_val:].reset_index(drop=True)

    # Extract raw targets BEFORE scaling — always multi-step, always in mg/dL
    _, y_train_raw = create_windows(df_tr,   feature_cols, horizon=horizon_steps, flat=True, multi_step=True)
    _, y_val_raw   = create_windows(df_val,  feature_cols, horizon=horizon_steps, flat=True, multi_step=True)
    _, y_test_raw  = create_windows(df_test, feature_cols, horizon=horizon_steps, flat=True, multi_step=True)

    # Fit scaler on training rows only; apply same scaler to val and test
    df_tr_sc, df_val_sc, scaler = _scale(df_tr, df_val, feature_cols)
    df_test_sc = df_test.copy()
    df_test_sc[feature_cols] = scaler.transform(df_test[feature_cols])

    X_tr,  y_tr  = create_windows(df_tr_sc,   feature_cols, horizon=horizon_steps, flat=flat, multi_step=multi_step)
    X_val, y_val = create_windows(df_val_sc,  feature_cols, horizon=horizon_steps, flat=flat, multi_step=multi_step)
    X_te,  y_te  = create_windows(df_test_sc, feature_cols, horizon=horizon_steps, flat=flat, multi_step=multi_step)

    return {
        "X_train": X_tr,
        "y_train": y_tr,
        "X_val":   X_val,
        "y_val":   y_val,
        "X_test":  X_te,
        "y_test":  y_te,
        "y_train_raw": y_train_raw,
        "y_val_raw":   y_val_raw,
        "y_test_raw":  y_test_raw,
        "scaler":  scaler,
    }
