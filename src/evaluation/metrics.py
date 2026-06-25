"""Regression and clinical accuracy metrics for CGM forecast evaluation."""

from __future__ import annotations

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error [mg/dL]."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error [mg/dL]."""
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """Mean Absolute Percentage Error [%]."""
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100)


def clarke_zone(r: float, p: float) -> str:
    """
    Assign a Clarke Error Grid Analysis zone to one (reference, predicted) pair.

    Parameters
    ----------
    r : float  reference (actual) glucose [mg/dL]
    p : float  predicted glucose [mg/dL]

    Returns
    -------
    One of 'A', 'B', 'C', 'D', 'E' per Clarke et al. (2005).

    Zone definitions
    ----------------
    A — clinically accurate: within 20 % of reference, or both hypoglycaemic
    B — clinically acceptable: erroneous but would not cause incorrect treatment
    C — overcorrection: prediction triggers unnecessary corrective treatment
    D — dangerous failure: clinically significant glucose level not detected
    E — erroneous treatment: prediction leads to treatment in the wrong direction
    """
    # Zone E: treatment direction reversed
    if r <= 70 and p >= 180:
        return "E"
    if r >= 180 and p <= 70:
        return "E"

    # Zone A: clinically accurate
    if r < 70 and p < 70:          # both in hypoglycaemic range
        return "A"
    if abs(p - r) / max(r, 1.0) <= 0.20:
        return "A"

    # Zone D: dangerous failure to detect extreme glucose
    if r >= 240 and 70 < p <= 180:   # severe hyperglycaemia not detected as hyper
        return "D"
    if r <= 70 and 120 <= p <= 180:  # hypoglycaemia not detected (prediction looks normal)
        return "D"

    # Zone C: overcorrection
    # Over-prediction in low-normal range → unnecessary carbohydrate treatment
    if 70 <= r <= 290 and p > r + 110:
        return "C"
    # Under-prediction of borderline-high glucose → excess carbs administered
    if r >= 130 and p <= 70:
        return "C"

    # Zone B: all remaining errors — present but not dangerous
    return "B"


def clarke_error_grid(ref: np.ndarray, pred: np.ndarray) -> dict:
    """
    Compute the Clarke Error Grid zone distribution for paired glucose arrays.

    Parameters
    ----------
    ref  : 1-D array of reference glucose values [mg/dL]
    pred : 1-D array of predicted glucose values  [mg/dL]

    Returns
    -------
    dict
        'counts'      — {zone: int}   absolute count per zone
        'percentages' — {zone: float} percentage of pairs per zone
    """
    ref = np.asarray(ref).ravel()
    pred = np.asarray(pred).ravel()
    if ref.shape != pred.shape:
        raise ValueError(f"ref and pred must have the same shape, got {ref.shape} vs {pred.shape}.")

    zones = [clarke_zone(float(r), float(p)) for r, p in zip(ref, pred)]
    n = len(zones)
    counts = {z: zones.count(z) for z in "ABCDE"}
    percentages = {z: 100.0 * counts[z] / n for z in "ABCDE"}
    return {"counts": counts, "percentages": percentages}


def per_horizon_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: callable = rmse,
) -> list[float]:
    """
    Compute a metric independently for each forecast step.

    Parameters
    ----------
    y_true, y_pred : shape (n_samples, horizon)
    metric_fn : one of rmse, mae, mape

    Returns
    -------
    List of floats, one value per horizon step.
    """
    return [metric_fn(y_true[:, h], y_pred[:, h]) for h in range(y_true.shape[1])]
