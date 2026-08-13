"""
Feature ablation study — Glucdict dataset, 30-min horizon.

Glucdict provides CGM + TicWatch heart rate + accelerometer + activity logs
(eat/drink events) for 13 T1D patients. Tests 4 feature combinations with
RF and LSTM, reusing the RF/LSTM hyperparameters tuned via grid search on
OhioT1DM (no Glucdict-specific grid search yet).

Each user's continuous recording is split 80/20 chronologically into
train/test (shuffle=False), matching the OhioT1DM convention since Glucdict
has no official train/test file split.
"""
import sys
import os
import json

# Addresses the duplicate-OpenMP-DLL root cause (torch + numpy/sklearn each
# bundle their own libiomp5md.dll on Windows). Must be set before torch is
# imported. Tested and confirmed NOT sufficient by itself to prevent the
# Windows access-violation crash on interpreter teardown (0xC0000409) seen
# after PyTorch DataLoader-based training completes — kept anyway since it's
# the documented mitigation for the underlying conflict and is harmless; the
# actual fix is the os._exit(0) at the end of this script (after all results
# are saved), which sidesteps teardown entirely.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.preprocessing.glucdict_loader import load_patient, ALL_USERS, GLUCDICT_BASE
from src.training.pipeline import make_splits
from src.models import random_forest as rf
from src.models.lstm import GlucoseLSTM, train_model as lstm_train
from src.models.autoencoder import GlucoseSeq2Seq, train_model as ae_train
from src.models.tcn import GlucoseTCN, train_model as tcn_train
from src.models.transformer import GlucoseTransformer, train_model as tr_train
from src.evaluation.metrics import rmse, mae, clarke_error_grid
from src.evaluation.plots import plot_clarke_error_grid

# ── Config ────────────────────────────────────────────────────────────────────
RESULTS_DIR = "results/glucdict"
GS_PATH     = "results/ohio/grid_search_results.json"   # reused — no Glucdict grid search yet
HORIZON     = 6
STEP_30MIN  = 5      # index 5 = 30-min step in the multi-step output
TRAIN_RATIO = 0.80   # chronological split (shuffle=False)
MIN_ROWS    = 100    # skip users with too little data for window_size(12)+horizon(6)+val split

os.makedirs(RESULTS_DIR, exist_ok=True)

FEAT_SETS = {
    "glucose_only":     ["glucose"],
    "glucose_hr":        ["glucose", "heartrate"],
    "glucose_activity":  ["glucose", "eat_event", "drink_event"],
    "full_wearable":     ["glucose", "heartrate", "acc_magnitude", "eat_event", "drink_event"],
}

# ── Load data ─────────────────────────────────────────────────────────────────
print("=" * 60)
print("Loading Glucdict dataset ...")
print("=" * 60)

train_data, test_data = {}, {}
cohort = []
for user_id in ALL_USERS:
    try:
        df = load_patient(GLUCDICT_BASE / user_id)
    except Exception as exc:
        print(f"  {user_id}: ERROR loading — {exc}")
        continue

    n_split = int(len(df) * TRAIN_RATIO)
    df_train, df_test = df.iloc[:n_split], df.iloc[n_split:]
    if len(df_train) < MIN_ROWS or len(df_test) < MIN_ROWS:
        print(f"  {user_id}: SKIPPED — too few rows (train={len(df_train)}, test={len(df_test)})")
        continue

    train_data[user_id] = df_train.reset_index(drop=True)
    test_data[user_id]  = df_test.reset_index(drop=True)
    cohort.append(user_id)
    print(f"  {user_id}: {len(df)} rows loaded (train={len(df_train)}, test={len(df_test)})")

print("Cohort:", cohort)
N_PATIENTS = len(cohort)

# ── Best hyperparameters from OhioT1DM grid search (reused) ───────────────────
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

results_list = []
pooled_lstm  = {fs: {"true": [], "pred": []} for fs in FEAT_SETS}  # for Clarke EGA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

for feat_name, feat_cols in FEAT_SETS.items():
    print(f"-- {feat_name}  {feat_cols}")
    for pid in cohort:
        try:
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
            lstm_model, _, _ = lstm_train(
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

            pooled_lstm[feat_name]["true"].append(y_true_30)
            pooled_lstm[feat_name]["pred"].append(lstm_pred_30)

            results_list.append({
                "feat_set":  feat_name,
                "patient":   pid,
                "rf_rmse":   rf_rmse_val,
                "rf_mae":    mae(y_true_30, rf_pred_30),
                "lstm_rmse": lstm_rmse_val,
                "lstm_mae":  mae(y_true_30, lstm_pred_30),
            })

            print(f"   [{pid}]  RF={rf_rmse_val:.2f}  LSTM={lstm_rmse_val:.2f}")

        except Exception as exc:
            print(f"   [{pid}]  ERROR — {exc}")

    print()

# ── Summary table ─────────────────────────────────────────────────────────────
results_df = pd.DataFrame(results_list)
ordered    = list(FEAT_SETS.keys())

sep = "=" * 68
print(sep)
print(f"GLUCDICT FEATURE ABLATION — 30-min RMSE (mean ± std, up to {N_PATIENTS} patients)")
print(sep)
print(f"  {'Feature Set':22s}  {'N':>3s}  {'RF RMSE':>18s}  {'LSTM RMSE':>18s}")
print("  " + "-" * 68)
n_per_set = {}
for fs in ordered:
    sub = results_df[results_df["feat_set"] == fs]
    n_per_set[fs] = len(sub)
    r_m, r_s = sub["rf_rmse"].mean(),   sub["rf_rmse"].std()
    l_m, l_s = sub["lstm_rmse"].mean(), sub["lstm_rmse"].std()
    print(f"  {fs:22s}  {len(sub):>3d}  {r_m:.2f} ± {r_s:.2f}          {l_m:.2f} ± {l_s:.2f}")
missing = sorted(set(cohort) - set(results_df[results_df['feat_set'] == 'glucose_hr']['patient']))
if missing:
    print(f"\n  Note: {missing} lack heart-rate coverage in the test split "
          f"(dropped from glucose_hr / full_wearable, not from glucose_only / glucose_activity).")
print()

# ── Save CSV ──────────────────────────────────────────────────────────────────
csv_path = f"{RESULTS_DIR}/results_glucdict_ablation.csv"
results_df.to_csv(csv_path, index=False)
print(f"Saved: {csv_path}")

# ── Bar chart — RF vs LSTM per feature set ───────────────────────────────────
rf_means   = [results_df[results_df["feat_set"] == fs]["rf_rmse"].mean()   for fs in ordered]
lstm_means = [results_df[results_df["feat_set"] == fs]["lstm_rmse"].mean() for fs in ordered]
rf_stds    = [results_df[results_df["feat_set"] == fs]["rf_rmse"].std()    for fs in ordered]
lstm_stds  = [results_df[results_df["feat_set"] == fs]["lstm_rmse"].std()  for fs in ordered]

x = np.arange(len(ordered))
w = 0.35

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - w / 2, rf_means,   w, yerr=rf_stds,   label="RF",
       color="#1F3864", capsize=4, error_kw={"elinewidth": 1.2})
ax.bar(x + w / 2, lstm_means, w, yerr=lstm_stds, label="LSTM",
       color="#2E75B6", capsize=4, error_kw={"elinewidth": 1.2})

ax.set_xlabel("Feature Set", fontsize=12)
ax.set_ylabel("RMSE [mg/dL]", fontsize=12)
ax.set_title(
    "Feature Ablation Study — 30-min RMSE\n"
    f"Glucdict dataset ({N_PATIENTS} patients, mean ± std)",
    fontsize=13,
)
ax.set_xticks(x)
ax.set_xticklabels([f"{fs}\n(n={n_per_set[fs]})" for fs in ordered], fontsize=10)
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3, linewidth=0.8)
ax.set_ylim(0, max(rf_means + lstm_means) * 1.25)
fig.tight_layout()

barchart_path = f"{RESULTS_DIR}/glucdict_ablation_barchart.png"
fig.savefig(barchart_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {barchart_path}")

# ── Clarke Error Grid — every feature combination (LSTM, pooled patients) ─────
print()
print(sep)
print("CLARKE ZONE A/B % — 30-min RMSE, LSTM predictions pooled across patients")
print(sep)

ceg_rows = []
fig_ceg, axes = plt.subplots(2, 2, figsize=(12, 12))
axes_flat = axes.ravel()

for ax, fs in zip(axes_flat, ordered):
    y_true_pooled = np.concatenate(pooled_lstm[fs]["true"])
    y_pred_pooled = np.concatenate(pooled_lstm[fs]["pred"])
    ceg = clarke_error_grid(y_true_pooled, y_pred_pooled)

    ceg_rows.append({"feat_set": fs, "n": len(y_true_pooled), **ceg["percentages"]})
    print(f"  {fs:22s}  A={ceg['percentages']['A']:.1f}%  B={ceg['percentages']['B']:.1f}%"
          f"  C={ceg['percentages']['C']:.1f}%  D={ceg['percentages']['D']:.1f}%"
          f"  E={ceg['percentages']['E']:.1f}%")

    plot_clarke_error_grid(
        y_true_pooled, y_pred_pooled,
        title=f"{fs}  (Zone A={ceg['percentages']['A']:.1f}%, n={len(y_true_pooled)})",
        ax=ax,
    )

fig_ceg.suptitle(
    "Clarke Error Grid — LSTM, 30-min horizon, Glucdict (all feature sets)",
    fontsize=14, y=1.01,
)
fig_ceg.tight_layout()
ceg_grid_path = f"{RESULTS_DIR}/clarke_glucdict_all_featuresets.png"
fig_ceg.savefig(ceg_grid_path, dpi=150, bbox_inches="tight")
plt.close(fig_ceg)
print(f"\nSaved: {ceg_grid_path}")

ceg_csv_path = f"{RESULTS_DIR}/clarke_glucdict_ablation.csv"
pd.DataFrame(ceg_rows).to_csv(ceg_csv_path, index=False)
print(f"Saved: {ceg_csv_path}")

print("\nDone.")

# ── SECTION 2 — 5-Model Comparison (glucose_only) ──────────────────────────────
# The ablation above already showed glucose_only/glucose_activity beat every
# feature set that adds wearable sensors (heartrate, accelerometer) on
# Glucdict — same pattern as OhioT1DM. glucose_only gives the cleanest,
# feature-identical comparison across all 5 architectures, reusing the
# RF/LSTM hyperparameters above plus the OhioT1DM grid search results for
# Autoencoder, TCN, and Transformer (no Glucdict-specific tuning yet).
print()
print("=" * 60)
print("SECTION 2 — Glucdict 5-Model Comparison (glucose_only)")
print("=" * 60)

GLUCOSE_ONLY = ["glucose"]

ae_latent   = gs["autoencoder"]["latent_size"]
ae_lr       = gs["autoencoder"]["lr"]
tcn_filters = gs["tcn"]["num_filters"]
tcn_lr      = gs["tcn"]["lr"]
tr_d_model  = gs["transformer"]["d_model"]
tr_nhead    = gs["transformer"].get("nhead", 4)
tr_lr       = gs["transformer"]["lr"]

print(f"RF         : {rf_params}")
print(f"LSTM       : hidden_size={lstm_hs}, lr={lstm_lr}")
print(f"Autoencoder: latent_size={ae_latent}, lr={ae_lr}")
print(f"TCN        : num_filters={tcn_filters}, lr={tcn_lr}")
print(f"Transformer: d_model={tr_d_model}, nhead={tr_nhead}, lr={tr_lr}")
print()

MODEL_NAMES_5 = ["RF", "LSTM", "Autoencoder", "TCN", "Transformer"]
results_5model = []


def _dl_predict_30(model):
    model.eval().to(device)
    X_t = torch.tensor(splits_dl["X_test"], dtype=torch.float32).to(device)
    with torch.no_grad():
        return (model(X_t).cpu().numpy() * y_std + y_mean)[:, STEP_30MIN]


for pid in cohort:
    print(f"[{pid}] ... ", end="", flush=True)
    try:
        splits_rf = make_splits(
            train_data[pid], test_data[pid], GLUCOSE_ONLY,
            horizon_steps=HORIZON, multi_step=True, flat=True,
        )
        splits_dl = make_splits(
            train_data[pid], test_data[pid], GLUCOSE_ONLY,
            horizon_steps=HORIZON, multi_step=True, flat=False,
        )

        glucose_idx = GLUCOSE_ONLY.index("glucose")
        y_mean      = float(splits_dl["scaler"].mean_[glucose_idx])
        y_std       = float(splits_dl["scaler"].scale_[glucose_idx])
        y_true_30   = splits_dl["y_test_raw"][:, STEP_30MIN]
        n_features  = splits_dl["X_train"].shape[2]

        # ── Random Forest ─────────────────────────────────────────────────
        rf_model = rf.train(splits_rf["X_train"], splits_rf["y_train"], params=rf_params)
        rf_pred_30 = (rf_model.predict(splits_rf["X_test"]) * y_std + y_mean)[:, STEP_30MIN]

        # ── LSTM ──────────────────────────────────────────────────────────
        lstm_model = GlucoseLSTM(n_features=n_features, hidden_size=lstm_hs, horizon=HORIZON)
        lstm_model, _, lstm_epochs = lstm_train(
            splits_dl["X_train"], splits_dl["y_train"],
            splits_dl["X_val"],   splits_dl["y_val"],
            model=lstm_model, lr=lstm_lr,
        )

        # ── Autoencoder ───────────────────────────────────────────────────
        ae_model = GlucoseSeq2Seq(n_features=n_features, latent_size=ae_latent, horizon=HORIZON)
        ae_model, _, ae_epochs = ae_train(
            splits_dl["X_train"], splits_dl["y_train"],
            splits_dl["X_val"],   splits_dl["y_val"],
            model=ae_model, lr=ae_lr,
        )

        # ── TCN ───────────────────────────────────────────────────────────
        tcn_model = GlucoseTCN(n_features=n_features, num_filters=tcn_filters, horizon=HORIZON)
        tcn_model, _, tcn_epochs = tcn_train(
            splits_dl["X_train"], splits_dl["y_train"],
            splits_dl["X_val"],   splits_dl["y_val"],
            model=tcn_model, lr=tcn_lr,
        )

        # ── Transformer ───────────────────────────────────────────────────
        tr_model = GlucoseTransformer(
            n_features=n_features, d_model=tr_d_model, nhead=tr_nhead, horizon=HORIZON
        )
        tr_model, _, tr_epochs = tr_train(
            splits_dl["X_train"], splits_dl["y_train"],
            splits_dl["X_val"],   splits_dl["y_val"],
            model=tr_model, lr=tr_lr,
        )

        preds_30 = {
            "RF":          rf_pred_30,
            "LSTM":        _dl_predict_30(lstm_model),
            "Autoencoder": _dl_predict_30(ae_model),
            "TCN":         _dl_predict_30(tcn_model),
            "Transformer": _dl_predict_30(tr_model),
        }
        epochs_map = {
            "RF": None, "LSTM": lstm_epochs, "Autoencoder": ae_epochs,
            "TCN": tcn_epochs, "Transformer": tr_epochs,
        }

        rmse_map = {}
        for model_name in MODEL_NAMES_5:
            pred_30 = preds_30[model_name]
            rmse_val = rmse(y_true_30, pred_30)
            rmse_map[model_name] = rmse_val
            results_5model.append({
                "user":       pid,
                "model":      model_name,
                "rmse":       rmse_val,
                "mae":        mae(y_true_30, pred_30),
                "zone_a_pct": clarke_error_grid(y_true_30, pred_30)["percentages"]["A"],
                "epochs":     epochs_map[model_name],
            })

        print(
            "RF={:.2f}  LSTM={:.2f}  AE={:.2f}  TCN={:.2f}  Tr={:.2f}".format(
                rmse_map["RF"], rmse_map["LSTM"], rmse_map["Autoencoder"],
                rmse_map["TCN"], rmse_map["Transformer"],
            )
        )

    except Exception as exc:
        print(f"ERROR — {exc}")

print()
print("5-model comparison loop complete.")

# ── Summary table ─────────────────────────────────────────────────────────────
results_df_5 = pd.DataFrame(results_5model)

csv_path_5 = f"{RESULTS_DIR}/results_glucdict_5models.csv"
results_df_5.to_csv(csv_path_5, index=False)
print(f"\nSaved: {csv_path_5}")

sep5 = "=" * 68
print()
print(sep5)
print(f"GLUCDICT 5-MODEL COMPARISON — glucose_only, 30-min horizon "
      f"(mean, up to {N_PATIENTS} users)")
print(sep5)
print(f"  {'Model':14s}  {'Mean RMSE':>10s}  {'Mean MAE':>10s}  {'Mean Zone A':>12s}")
print("  " + "-" * 54)
for model_name in MODEL_NAMES_5:
    sub = results_df_5[results_df_5["model"] == model_name]
    print(
        f"  {model_name:14s}  {sub['rmse'].mean():10.2f}  "
        f"{sub['mae'].mean():10.2f}  {sub['zone_a_pct'].mean():11.1f}%"
    )

print("\nDone.")

# All results (ablation CSV, barchart PNG, Clarke EGA PNG + CSV, 5-model
# CSV) are written above this point. os._exit(0) skips Python's normal
# interpreter teardown (atexit handlers, garbage collection of torch/
# DataLoader objects), which is exactly where the Windows access-violation
# crash (0xC0000409, duplicate OpenMP DLL teardown) occurs after PyTorch
# training — nothing after this line, so nothing is short-circuited.
os._exit(0)
