"""
Feature ablation study — OhioT1DM 2018 cohort, 30-min horizon.

Tests 8 feature combinations with RF and LSTM.  Results saved to
results/ohio/results_feature_ablation.csv plus two charts.
"""
import sys
import os
import json

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.preprocessing.ohio_loader import load_patient, COHORT_2018
from src.training.pipeline import make_splits
from src.models import random_forest as rf
from src.models.lstm import GlucoseLSTM, train_model as lstm_train
from src.evaluation.metrics import rmse, mae
from src.evaluation.plots import plot_feature_importance

# ── Config ────────────────────────────────────────────────────────────────────
DATA_ROOT   = "data/ohio"
RESULTS_DIR = "results/ohio"
GS_PATH     = "results/ohio/grid_search_results.json"
HORIZON     = 6
STEP_30MIN  = 5      # index 5 = 30-min step in the multi-step output
WINDOW_SIZE = 12     # must match create_windows default

FEAT_SETS = {
    "glucose_only":   ["glucose"],
    "clinical":       ["glucose", "bolus", "carbs"],
    "full":           ["glucose", "bolus", "carbs", "heartrate", "steps"],
    "glucose_hr":     ["glucose", "heartrate"],
    "glucose_steps":  ["glucose", "steps"],
    "clinical_hr":    ["glucose", "bolus", "carbs", "heartrate"],
    "clinical_steps": ["glucose", "bolus", "carbs", "steps"],
    "wearable":       ["glucose", "heartrate", "steps"],
}

# ── Load data ─────────────────────────────────────────────────────────────────
print("=" * 60)
print("Loading OhioT1DM 2018 cohort ...")
print("=" * 60)

train_data, test_data = {}, {}
for pid in COHORT_2018:
    train_data[pid] = load_patient(
        f"{DATA_ROOT}/2018/train/{pid}-ws-training.xml"
    )
    test_data[pid] = load_patient(
        f"{DATA_ROOT}/2018/test/{pid}-ws-testing.xml"
    )
print("Patients:", COHORT_2018)

# ── Best hyperparameters from grid search ─────────────────────────────────────
with open(GS_PATH) as f:
    gs = json.load(f)
rf_params = gs["rf"]
lstm_hs   = gs["lstm"]["hidden_size"]
lstm_lr   = gs["lstm"]["lr"]
print(f"RF params : {rf_params}")
print(f"LSTM      : hidden_size={lstm_hs}, lr={lstm_lr}")

# ── Main loop ─────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("Running ablation ...")
print("=" * 60)

results_list        = []
rf_full_importances = []   # collect per-patient importances for 'full' set

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

for feat_name, feat_cols in FEAT_SETS.items():
    print(f"-- {feat_name}  {feat_cols}")
    for pid in COHORT_2018:
        try:
            # ── Splits ────────────────────────────────────────────────────────
            splits_rf = make_splits(
                train_data[pid], test_data[pid], feat_cols,
                horizon_steps=HORIZON, multi_step=True, flat=True,
            )
            splits_dl = make_splits(
                train_data[pid], test_data[pid], feat_cols,
                horizon_steps=HORIZON, multi_step=True, flat=False,
            )

            glucose_idx = feat_cols.index("glucose")
            y_mean      = float(splits_dl["scaler"].mean_[glucose_idx])
            y_std       = float(splits_dl["scaler"].scale_[glucose_idx])
            y_true_30   = splits_dl["y_test_raw"][:, STEP_30MIN]   # mg/dL
            n_features  = splits_dl["X_train"].shape[2]

            # ── Random Forest ─────────────────────────────────────────────────
            rf_model   = rf.train(splits_rf["X_train"], splits_rf["y_train"],
                                  params=rf_params)
            rf_pred_30 = (rf_model.predict(splits_rf["X_test"]) * y_std + y_mean
                          )[:, STEP_30MIN]

            # ── LSTM ──────────────────────────────────────────────────────────
            lstm_model = GlucoseLSTM(
                n_features=n_features, hidden_size=lstm_hs, horizon=HORIZON
            )
            lstm_model, _ = lstm_train(
                splits_dl["X_train"], splits_dl["y_train"],
                splits_dl["X_val"],   splits_dl["y_val"],
                model=lstm_model, lr=lstm_lr,
            )
            lstm_model.eval().to(device)
            X_t = torch.tensor(
                splits_dl["X_test"], dtype=torch.float32
            ).to(device)
            with torch.no_grad():
                lstm_pred_30 = (
                    lstm_model(X_t).cpu().numpy() * y_std + y_mean
                )[:, STEP_30MIN]

            rf_rmse_val   = rmse(y_true_30, rf_pred_30)
            lstm_rmse_val = rmse(y_true_30, lstm_pred_30)

            results_list.append({
                "feat_set":  feat_name,
                "patient":   pid,
                "rf_rmse":   rf_rmse_val,
                "rf_mae":    mae(y_true_30, rf_pred_30),
                "lstm_rmse": lstm_rmse_val,
                "lstm_mae":  mae(y_true_30, lstm_pred_30),
            })

            # ── RF feature importance (full set only) ─────────────────────────
            if feat_name == "full":
                # MultiOutputRegressor: average importance across 6 estimators
                mean_imp = np.mean(
                    [e.feature_importances_ for e in rf_model.estimators_],
                    axis=0,
                )
                rf_full_importances.append(mean_imp)

            print(f"   [{pid}]  RF={rf_rmse_val:.2f}  LSTM={lstm_rmse_val:.2f}")

        except Exception as exc:
            print(f"   [{pid}]  ERROR — {exc}")

    print()

# ── Summary table ─────────────────────────────────────────────────────────────
results_df = pd.DataFrame(results_list)
ordered    = list(FEAT_SETS.keys())

sep = "=" * 68
print(sep)
print("FEATURE ABLATION — 30-min RMSE (mean ± std, 6 patients, COHORT_2018)")
print(sep)
print(f"  {'Feature Set':22s}  {'RF RMSE':>18s}  {'LSTM RMSE':>18s}")
print("  " + "-" * 62)
for fs in ordered:
    sub = results_df[results_df["feat_set"] == fs]
    r_m, r_s = sub["rf_rmse"].mean(),   sub["rf_rmse"].std()
    l_m, l_s = sub["lstm_rmse"].mean(), sub["lstm_rmse"].std()
    print(f"  {fs:22s}  {r_m:.2f} ± {r_s:.2f}          {l_m:.2f} ± {l_s:.2f}")
print()

# ── Save CSV ──────────────────────────────────────────────────────────────────
csv_path = f"{RESULTS_DIR}/results_feature_ablation.csv"
results_df.to_csv(csv_path, index=False)
print(f"Saved: {csv_path}")

# ── Bar chart — RF vs LSTM per feature set ───────────────────────────────────
rf_means   = [results_df[results_df["feat_set"] == fs]["rf_rmse"].mean()   for fs in ordered]
lstm_means = [results_df[results_df["feat_set"] == fs]["lstm_rmse"].mean() for fs in ordered]
rf_stds    = [results_df[results_df["feat_set"] == fs]["rf_rmse"].std()    for fs in ordered]
lstm_stds  = [results_df[results_df["feat_set"] == fs]["lstm_rmse"].std()  for fs in ordered]

x = np.arange(len(ordered))
w = 0.35

fig, ax = plt.subplots(figsize=(13, 5))
ax.bar(x - w / 2, rf_means,   w, yerr=rf_stds,   label="RF",
       color="#1F3864", capsize=4, error_kw={"elinewidth": 1.2})
ax.bar(x + w / 2, lstm_means, w, yerr=lstm_stds, label="LSTM",
       color="#2E75B6", capsize=4, error_kw={"elinewidth": 1.2})

ax.set_xlabel("Feature Set", fontsize=12)
ax.set_ylabel("RMSE [mg/dL]", fontsize=12)
ax.set_title(
    "Feature Ablation Study — 30-min RMSE\n"
    "OhioT1DM 2018 cohort (6 patients, mean ± std)",
    fontsize=13,
)
ax.set_xticks(x)
ax.set_xticklabels(ordered, rotation=28, ha="right", fontsize=10)
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3, linewidth=0.8)
ax.set_ylim(0, max(rf_means + lstm_means) * 1.25)
fig.tight_layout()

barchart_path = f"{RESULTS_DIR}/feature_ablation_barchart.png"
fig.savefig(barchart_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {barchart_path}")

# ── RF feature importance — 'full' set ───────────────────────────────────────
if rf_full_importances:
    full_cols = FEAT_SETS["full"]          # 5 features
    n_feat    = len(full_cols)

    # Mean across 6 patients → shape (WINDOW_SIZE * n_feat,)
    all_imp = np.mean(rf_full_importances, axis=0)

    # Flatten order is row-major: [f0_t0, f1_t0, ..., fN_t0, f0_t1, ...]
    # Reshape to (WINDOW_SIZE, n_feat) and sum over time steps
    imp_per_feature = all_imp.reshape(WINDOW_SIZE, n_feat).sum(axis=0)

    fig_imp = plot_feature_importance(
        feature_names=full_cols,
        importances=imp_per_feature,
        title=(
            "RF Feature Importance — 'full' feature set\n"
            "30-min horizon, COHORT_2018 (mean across 6 patients × 6 estimators)"
        ),
    )
    imp_path = f"{RESULTS_DIR}/feature_importance_rf.png"
    fig_imp.savefig(imp_path, dpi=150, bbox_inches="tight")
    plt.close(fig_imp)
    print(f"Saved: {imp_path}")

print("\nDone.")
