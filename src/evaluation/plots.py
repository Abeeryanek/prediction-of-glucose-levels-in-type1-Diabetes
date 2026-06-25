"""Visualisation helpers for glucose forecast evaluation."""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["font.size"] = 11


# ---------------------------------------------------------------------------
# Clarke Error Grid
# ---------------------------------------------------------------------------

def _draw_ceg_boundaries(ax: plt.Axes) -> None:
    """Draw the Clarke Error Grid zone boundary lines on ax."""
    lw = 0.8
    c = "black"

    # Identity line
    ax.plot([0, 400], [0, 400], color=c, linewidth=lw, linestyle="--", zorder=1)

    # Zone A: ±20 % lines (start from origin to avoid artefacts at low values)
    ax.plot([0, 400], [0, 480], color=c, linewidth=lw, zorder=1)   # +20 %  (clipped at 400 by axis limit)
    ax.plot([0, 400], [0, 320], color=c, linewidth=lw, zorder=1)   # -20 %

    # Zone E upper-left boundary box (r ≤ 70, p ≥ 180)
    ax.plot([0, 70],   [180, 180], color=c, linewidth=lw, zorder=1)  # horizontal ceiling
    ax.plot([70, 70],  [180, 400], color=c, linewidth=lw, zorder=1)  # vertical right wall

    # Zone E lower-right boundary box (r ≥ 180, p ≤ 70)
    ax.plot([180, 400], [70, 70],  color=c, linewidth=lw, zorder=1)  # horizontal floor
    ax.plot([180, 180], [0, 70],   color=c, linewidth=lw, zorder=1)  # vertical left wall

    # Zone D right (r ≥ 240, 70 < p ≤ 180)
    ax.plot([240, 240], [70, 180], color=c, linewidth=lw, zorder=1)  # vertical left boundary
    ax.plot([240, 400], [70, 70],  color=c, linewidth=lw, zorder=1)  # horizontal floor extension
    ax.plot([240, 400], [180, 180], color=c, linewidth=lw, zorder=1) # horizontal ceiling extension

    # Zone D left (r ≤ 70, 120 ≤ p ≤ 180)
    ax.plot([0, 70],  [120, 120], color=c, linewidth=lw, zorder=1)   # horizontal floor of left-D

    # Zone C upper boundary: p = r + 110 for r in [70, 290]
    r_c = np.array([70, 290])
    ax.plot(r_c, r_c + 110, color=c, linewidth=lw, zorder=1)


def _label_ceg_zones(ax: plt.Axes) -> None:
    kw = dict(fontsize=13, fontweight="bold", color="dimgrey", zorder=2)
    ax.text(30,  15,  "A", **kw)
    ax.text(370, 340, "A", **kw)
    ax.text(160, 370, "B", **kw)
    ax.text(360, 10,  "B", **kw)
    ax.text(30,  330, "E", **kw)
    ax.text(370, 30,  "E", **kw)
    ax.text(30,  145, "D", **kw)
    ax.text(330, 115, "D", **kw)
    ax.text(160, 370, "C", **kw)


def plot_clarke_error_grid(
    ref: np.ndarray,
    pred: np.ndarray,
    title: str = "Clarke Error Grid Analysis",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """
    Scatter plot of (reference, predicted) glucose pairs with zone boundaries.

    Parameters
    ----------
    ref  : 1-D array of reference glucose [mg/dL]
    pred : 1-D array of predicted glucose [mg/dL]
    title : plot title
    ax    : existing Axes to draw on; a new Figure is created when None

    Returns
    -------
    matplotlib Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    else:
        fig = ax.get_figure()

    _draw_ceg_boundaries(ax)
    _label_ceg_zones(ax)

    ax.scatter(ref, pred, s=6, alpha=0.5, color="steelblue", zorder=3)

    ax.set_xlim(0, 400)
    ax.set_ylim(0, 400)
    ax.set_xlabel("Reference Glucose [mg/dL]")
    ax.set_ylabel("Predicted Glucose [mg/dL]")
    ax.set_title(title)
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Multi-model prediction comparison
# ---------------------------------------------------------------------------

def plot_predictions(
    actual: np.ndarray,
    predictions_dict: dict[str, np.ndarray],
    title: str = "Glucose Forecast Comparison",
    n_samples: int = 288,
    step_minutes: int = 5,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot actual glucose vs. predictions from multiple models.

    Parameters
    ----------
    actual           : 1-D array of actual glucose [mg/dL]
    predictions_dict : {'model_name': predictions_1d_array, ...}
                       predictions are the final horizon step, shape (n_samples,)
    n_samples        : number of consecutive samples to display
    step_minutes     : CGM sampling interval in minutes (default 5)
    save_path        : if given, save the figure to this path

    Returns
    -------
    matplotlib Figure
    """
    n = min(n_samples, len(actual))
    t = np.arange(n) * step_minutes / 60  # hours

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(t, actual[:n], label="Actual", color="black", linewidth=1.8, zorder=5)

    for name, preds in predictions_dict.items():
        ax.plot(t, preds[:n], label=name, linewidth=1.2, linestyle="--", zorder=4)

    # Shade clinically relevant glucose bands
    ax.axhspan(70, 180, alpha=0.06, color="green", label="Target range (70–180)")
    ax.axhline(70,  color="orange", linewidth=0.8, linestyle=":")
    ax.axhline(180, color="orange", linewidth=0.8, linestyle=":")

    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Glucose [mg/dL]")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def plot_feature_importance(
    feature_names: list[str],
    importances: np.ndarray,
    title: str = "Feature Importance",
    top_n: int | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Horizontal bar chart of feature importances (e.g. from Random Forest).

    Parameters
    ----------
    feature_names : list of feature name strings
    importances   : 1-D array of importance scores (same order as feature_names)
    top_n         : if set, display only the top-N features by importance
    save_path     : if given, save the figure to this path

    Returns
    -------
    matplotlib Figure
    """
    names = np.array(feature_names)
    imps = np.array(importances)

    order = np.argsort(imps)
    if top_n is not None:
        order = order[-top_n:]

    fig, ax = plt.subplots(figsize=(8, max(3, len(order) * 0.35)))
    y_pos = np.arange(len(order))
    ax.barh(y_pos, imps[order], align="center", color="steelblue", edgecolor="black")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names[order])
    ax.set_xlabel("Importance")
    ax.set_title(title)
    ax.invert_yaxis()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
