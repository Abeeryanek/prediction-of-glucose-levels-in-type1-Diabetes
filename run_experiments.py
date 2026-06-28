"""Runner script: mirrors experiments/ohio_experiments.ipynb cell by cell."""
import sys
import os

# Project root on path so `src.*` imports resolve
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")   # headless — saves figures without a display

# ── Section 1 — Imports and Configuration ────────────────────────────────────
print("=" * 60)
print("SECTION 1 — Imports and Configuration")
print("=" * 60)

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from src.preprocessing.ohio_loader import (
    load_patient, load_split, COHORT_2018, COHORT_2020
)
from src.training.pipeline import make_splits, walk_forward_splits
from src.training.grid_search import grid_search_rf, grid_search_lstm
from src.models import random_forest as rf
from src.models.lstm import GlucoseLSTM, train_model as lstm_train, evaluate as lstm_eval
from src.models.autoencoder import GlucoseSeq2Seq, train_model as ae_train, evaluate as ae_eval
from src.models.tcn import GlucoseTCN, train_model as tcn_train, evaluate as tcn_eval
from src.evaluation.metrics import rmse, mae, clarke_error_grid
from src.evaluation.plots import plot_clarke_error_grid, plot_predictions

DATA_ROOT   = Path("data/ohio")
RESULTS_DIR = Path("results/ohio")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SIZE = 12
HORIZON     = 6

HORIZONS = {
    0: "5min",
    2: "15min",
    5: "30min",
}

CLINICAL_FEATURES = ["glucose", "bolus", "carbs"]
ALL_PATIENTS      = COHORT_2018 + COHORT_2020
MODEL_NAMES       = ["RF", "LSTM", "Autoencoder", "TCN"]

print("Patients :", ALL_PATIENTS)
print("Features :", CLINICAL_FEATURES)
print("Results  :", RESULTS_DIR)

# ── Section 2 — Load OhioT1DM Data ───────────────────────────────────────────
print()
print("=" * 60)
print("SECTION 2 — Load OhioT1DM Data")
print("=" * 60)

train_data = {}
test_data  = {}

for pid in COHORT_2018:
    train_data[pid] = load_patient(DATA_ROOT / "2018" / "train" / f"{pid}-ws-training.xml")
    test_data[pid]  = load_patient(DATA_ROOT / "2018" / "test"  / f"{pid}-ws-testing.xml")

for pid in COHORT_2020:
    train_data[pid] = load_patient(DATA_ROOT / "2020" / "train" / f"{pid}-ws-training.xml")
    test_data[pid]  = load_patient(DATA_ROOT / "2020" / "test"  / f"{pid}-ws-testing.xml")

rows = []
for pid in ALL_PATIENTS:
    cohort = "2018" if pid in COHORT_2018 else "2020"
    sensor_cols = [c for c in train_data[pid].columns if c not in ("ts", "glucose", "carbs", "bolus")]
    rows.append({
        "Patient": pid,
        "Cohort": cohort,
        "Train rows": len(train_data[pid]),
        "Test rows": len(test_data[pid]),
        "Sensor columns": ", ".join(sensor_cols),
    })

summary = pd.DataFrame(rows)
print(summary.to_string(index=False))

# ── Section 3 — Grid Search ───────────────────────────────────────────────────
print()
print("=" * 60)
print("SECTION 3 — Grid Search")
print("=" * 60)

GS_PATH = RESULTS_DIR / "grid_search_results.json"

if GS_PATH.exists():
    with open(GS_PATH) as f:
        gs_results = json.load(f)
    print("Loaded existing grid search results from", GS_PATH)
else:
    print("Running grid search on patient 559 ...")

    splits_gs_rf = make_splits(
        train_data["559"], test_data["559"],
        CLINICAL_FEATURES, horizon_steps=HORIZON,
        multi_step=True, flat=True,
    )
    best_rf_params, best_rf_score = grid_search_rf(
        splits_gs_rf["X_train"], splits_gs_rf["y_train"]
    )
    print(f"  RF   best params : {best_rf_params}")
    print(f"  RF   RMSE (norm) : {best_rf_score:.4f}")

    splits_gs_dl = make_splits(
        train_data["559"], test_data["559"],
        CLINICAL_FEATURES, horizon_steps=HORIZON,
        multi_step=True, flat=False,
    )
    n_feat_gs = splits_gs_dl["X_train"].shape[2]
    best_lstm_params, best_lstm_score = grid_search_lstm(
        splits_gs_dl["X_train"], splits_gs_dl["y_train"],
        splits_gs_dl["X_val"],   splits_gs_dl["y_val"],
        n_features=n_feat_gs, horizon=HORIZON,
    )
    print(f"  LSTM best params : {best_lstm_params}")
    print(f"  LSTM RMSE (norm) : {best_lstm_score:.4f}")

    gs_results = {"rf": best_rf_params, "lstm": best_lstm_params}
    with open(GS_PATH, "w") as f:
        json.dump(gs_results, f, indent=2)
    print("Grid search results saved to", GS_PATH)

print()
print("Best parameters:")
print("  RF  :", gs_results["rf"])
print("  LSTM:", gs_results["lstm"])

# ── Section 4 — Main Experiment Loop ─────────────────────────────────────────
print()
print("=" * 60)
print("SECTION 4 — Main Experiment Loop")
print("=" * 60)

results_list = []

pooled = {m: {"pred_30": [], "true_30": []} for m in MODEL_NAMES}
patient_preds_559 = {}
patient_true_559  = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print()


def _dl_predict(model, X, y_mean, y_std):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(dev)
    X_t = torch.tensor(X, dtype=torch.float32).to(dev)
    with torch.no_grad():
        return model(X_t).cpu().numpy() * y_std + y_mean


for pid in ALL_PATIENTS:
    try:
        cohort = "2018" if pid in COHORT_2018 else "2020"
        print(f"[{pid}] ({cohort}) ... ", end="", flush=True)

        splits_rf = make_splits(
            train_data[pid], test_data[pid],
            CLINICAL_FEATURES, horizon_steps=HORIZON,
            multi_step=True, flat=True,
        )
        splits_dl = make_splits(
            train_data[pid], test_data[pid],
            CLINICAL_FEATURES, horizon_steps=HORIZON,
            multi_step=True, flat=False,
        )

        glucose_idx = CLINICAL_FEATURES.index("glucose")
        y_mean      = float(splits_dl["scaler"].mean_[glucose_idx])
        y_std       = float(splits_dl["scaler"].scale_[glucose_idx])
        n_features  = splits_dl["X_train"].shape[2]
        y_test_raw  = splits_dl["y_test_raw"]

        rf_model = rf.train(
            splits_rf["X_train"], splits_rf["y_train"],
            params=gs_results["rf"],
        )

        lstm_hs = gs_results["lstm"].get("hidden_size", 64)
        lstm_lr = gs_results["lstm"].get("lr", 1e-3)

        lstm_model = GlucoseLSTM(n_features=n_features, hidden_size=lstm_hs, horizon=HORIZON)
        lstm_model, _ = lstm_train(
            splits_dl["X_train"], splits_dl["y_train"],
            splits_dl["X_val"],   splits_dl["y_val"],
            model=lstm_model, lr=lstm_lr,
        )

        ae_model = GlucoseSeq2Seq(n_features=n_features, horizon=HORIZON)
        ae_model, _ = ae_train(
            splits_dl["X_train"], splits_dl["y_train"],
            splits_dl["X_val"],   splits_dl["y_val"],
            model=ae_model, lr=lstm_lr,
        )

        tcn_model = GlucoseTCN(n_features=n_features, horizon=HORIZON)
        tcn_model, _ = tcn_train(
            splits_dl["X_train"], splits_dl["y_train"],
            splits_dl["X_val"],   splits_dl["y_val"],
            model=tcn_model, lr=lstm_lr,
        )

        rf_pred   = rf_model.predict(splits_rf["X_test"]) * y_std + y_mean
        lstm_pred = _dl_predict(lstm_model, splits_dl["X_test"], y_mean, y_std)
        ae_pred   = _dl_predict(ae_model,   splits_dl["X_test"], y_mean, y_std)
        tcn_pred  = _dl_predict(tcn_model,  splits_dl["X_test"], y_mean, y_std)

        model_preds = {
            "RF": rf_pred, "LSTM": lstm_pred,
            "Autoencoder": ae_pred, "TCN": tcn_pred,
        }

        for step_idx, horizon_name in HORIZONS.items():
            y_true_step = y_test_raw[:, step_idx]
            for model_name, preds in model_preds.items():
                y_pred_step = preds[:, step_idx]
                row = {
                    "patient": pid,
                    "cohort": cohort,
                    "model": model_name,
                    "horizon": horizon_name,
                    "rmse": rmse(y_true_step, y_pred_step),
                    "mae":  mae(y_true_step, y_pred_step),
                }
                if step_idx == 5:
                    ceg = clarke_error_grid(y_true_step, y_pred_step)
                    for zone in "ABCDE":
                        row["zone_" + zone + "_pct"] = ceg["percentages"][zone]
                results_list.append(row)

        for model_name, preds in model_preds.items():
            pooled[model_name]["pred_30"].append(preds[:, 5])
            pooled[model_name]["true_30"].append(y_test_raw[:, 5])

        if pid == "559":
            patient_preds_559 = {m: p[:, 5] for m, p in model_preds.items()}
            patient_true_559  = y_test_raw[:, 5]

        print("done")

    except Exception as exc:
        print(f"ERROR — skipping patient {pid}: {exc}")
        continue

print()
print("Experiment loop complete.")

# ── Section 5 — Results Table ─────────────────────────────────────────────────
print()
print("=" * 60)
print("SECTION 5 — Results Table")
print("=" * 60)

results_df = pd.DataFrame(results_list)

sep = "=" * 72

print(sep)
print("MEAN +/- STD  RMSE [mg/dL] and MAE [mg/dL]  (all patients)")
print(sep)

for horizon in ["5min", "15min", "30min"]:
    print(f"  Horizon: {horizon}")
    sub = results_df[results_df["horizon"] == horizon]
    for model_name in MODEL_NAMES:
        m = sub[sub["model"] == model_name]
        r_mean = m["rmse"].mean()
        r_std  = m["rmse"].std()
        a_mean = m["mae"].mean()
        a_std  = m["mae"].std()
        print(f"    {model_name:14s}  RMSE: {r_mean:.2f} ± {r_std:.2f}   MAE: {a_mean:.2f} ± {a_std:.2f}")
    print()

print(sep)
print("CLARKE ZONE A %  at 30-min horizon  (mean +/- std, all patients)")
print(sep)

sub30 = results_df[results_df["horizon"] == "30min"]
for model_name in MODEL_NAMES:
    m = sub30[sub30["model"] == model_name]
    a_mean = m["zone_A_pct"].mean()
    a_std  = m["zone_A_pct"].std()
    b_mean = m["zone_B_pct"].mean()
    print(f"  {model_name:12s}  Zone A={a_mean:.1f}% +/- {a_std:.1f}%   Zone B={b_mean:.1f}%")

print()

# ── Section 6 — Clarke Error Grid Plots ──────────────────────────────────────
print("=" * 60)
print("SECTION 6 — Clarke Error Grid Plots")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes_flat = axes.ravel()

for ax, model_name in zip(axes_flat, MODEL_NAMES):
    y_true_pooled = np.concatenate(pooled[model_name]["true_30"])
    y_pred_pooled = np.concatenate(pooled[model_name]["pred_30"])

    ceg = clarke_error_grid(y_true_pooled, y_pred_pooled)
    zone_a = ceg["percentages"]["A"]

    plot_clarke_error_grid(
        y_true_pooled, y_pred_pooled,
        title=f"{model_name}  (Zone A={zone_a:.1f}%,  n={len(y_true_pooled)})",
        ax=ax,
    )

fig.suptitle(
    "Clarke Error Grid -- OhioT1DM (all 12 patients, 30-min horizon)",
    fontsize=14, y=1.01,
)
fig.tight_layout()

save_path = RESULTS_DIR / "clarke_30min_all_models.png"
fig.savefig(save_path, dpi=150, bbox_inches="tight")
print("Saved:", save_path)
plt.close(fig)

# ── Section 7 — Prediction Plot (Patient 559) ─────────────────────────────────
print()
print("=" * 60)
print("SECTION 7 — Prediction Plot (Patient 559)")
print("=" * 60)

n_plot = 288
t = np.arange(n_plot) * 5 / 60

fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)

for ax, model_name in zip(axes, MODEL_NAMES):
    true_slice = patient_true_559[:n_plot]
    pred_slice = patient_preds_559[model_name][:n_plot]

    ax.plot(t, true_slice, color="black", linewidth=1.5, label="Actual", zorder=5)
    ax.plot(t, pred_slice, color="steelblue", linewidth=1.2, linestyle="--",
            label=f"{model_name} (+30 min)", zorder=4)
    ax.axhspan(70, 180, alpha=0.07, color="green")
    ax.axhline(70,  color="orange", linewidth=0.8, linestyle=":")
    ax.axhline(180, color="orange", linewidth=0.8, linestyle=":")
    ax.set_ylabel("Glucose [mg/dL]")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(model_name, fontsize=11)

axes[-1].set_xlabel("Time [h]")
fig.suptitle(
    "Glucose Forecast -- Patient 559, 30-min horizon (first 24 h of test set)",
    fontsize=13,
)
fig.tight_layout()

save_path = RESULTS_DIR / "predictions_559_30min.png"
fig.savefig(save_path, dpi=150, bbox_inches="tight")
print("Saved:", save_path)
plt.close(fig)

# ── Section 8 — Save Results ──────────────────────────────────────────────────
print()
print("=" * 60)
print("SECTION 8 — Save Results")
print("=" * 60)

csv_path = RESULTS_DIR / "results_all_models.csv"
results_df.to_csv(csv_path, index=False)

n_rows = len(results_df)
n_cols = len(results_df.columns)

print("Done. Results saved to results/ohio/")
print(f"  results_all_models.csv        {n_rows} rows x {n_cols} columns")
print( "  clarke_30min_all_models.png   Clarke EGA for all 4 models")
print( "  predictions_559_30min.png     24-h forecast comparison, patient 559")
print( "  grid_search_results.json      best hyperparameters")
