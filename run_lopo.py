"""
Leave-One-Patient-Out (LOPO) cross-validation — OhioT1DM, 30-min horizon.

For each of the 12 patients, trains RF and LSTM on the pooled data of the
OTHER 11 patients (a population model) and evaluates on the held-out
patient's full record (official train + test XML files combined). This
measures how well a model trained on other patients generalises to an
unseen patient, in contrast to the per-patient personalised models used
in run_experiments.py.

Windowing is done PER PATIENT (never across a patient boundary) before
pooling the resulting (X, y) arrays. Concatenating raw per-patient
DataFrames first and windowing the pooled frame once would let
create_windows slide a window across the boundary between two unrelated
patients' series, splicing one patient's glucose trace onto another's for
a handful of samples at each of the 11 boundaries. Windowing first, then
pooling arrays, avoids this.

Normalisation: a StandardScaler is fit on the pooled RAW training data of
the 11 training patients only, then applied to those same patients and to
the held-out test patient — never fit on test-patient data.

Results saved to results/ohio/results_lopo.csv.
"""
import sys
import os
import json

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from src.preprocessing.ohio_loader import load_patient, COHORT_2018, COHORT_2020
from src.training.pipeline import _preprocess, create_windows
from src.models import random_forest as rf
from src.models.lstm import GlucoseLSTM, train_model as lstm_train
from src.evaluation.metrics import rmse, mae, clarke_error_grid

# ── Config ────────────────────────────────────────────────────────────────────
DATA_ROOT      = "data/ohio"
RESULTS_DIR    = "results/ohio"
GS_PATH        = "results/ohio/grid_search_results.json"
ALL_MODELS_CSV = "results/ohio/results_all_models.csv"

WINDOW_SIZE = 12     # 60 min of history — must match create_windows default
HORIZON     = 6      # 30 min ahead
STEP_30MIN  = 5      # index 5 = 30-min step in the multi-step output
VAL_RATIO   = 0.20   # chronological, carved PER training patient

CLINICAL_FEATURES = ["glucose", "bolus", "carbs"]
ALL_PATIENTS      = COHORT_2018 + COHORT_2020

# ── Load data — combine each patient's official train + test files ────────────
print("=" * 60)
print("Loading OhioT1DM — all 12 patients (train + test combined) ...")
print("=" * 60)

full_data = {}
for pid in COHORT_2018:
    tr = load_patient(f"{DATA_ROOT}/2018/train/{pid}-ws-training.xml")
    te = load_patient(f"{DATA_ROOT}/2018/test/{pid}-ws-testing.xml")
    full_data[pid] = pd.concat([tr, te], ignore_index=True).sort_values("ts").reset_index(drop=True)
for pid in COHORT_2020:
    tr = load_patient(f"{DATA_ROOT}/2020/train/{pid}-ws-training.xml")
    te = load_patient(f"{DATA_ROOT}/2020/test/{pid}-ws-testing.xml")
    full_data[pid] = pd.concat([tr, te], ignore_index=True).sort_values("ts").reset_index(drop=True)

# Preprocess once per patient (drop glucose-NaN rows, fill event NaN with 0)
# — same policy as pipeline.make_splits, applied manually here since LOPO
# defines its own train/test split rather than using the official XML split.
prep_data = {pid: _preprocess(df, CLINICAL_FEATURES) for pid, df in full_data.items()}

for pid in ALL_PATIENTS:
    print(f"  {pid}: {len(prep_data[pid])} rows (train+test combined)")

# ── Best hyperparameters from grid search ─────────────────────────────────────
with open(GS_PATH) as f:
    gs = json.load(f)
rf_params = gs["rf"]
lstm_hs   = gs["lstm"]["hidden_size"]
lstm_lr   = gs["lstm"]["lr"]
print(f"\nRF params : {rf_params}")
print(f"LSTM      : hidden_size={lstm_hs}, lr={lstm_lr}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


def _window_patient(df_scaled: pd.DataFrame, flat: bool):
    """create_windows wrapper — multi-step target, single patient's df only."""
    return create_windows(
        df_scaled, CLINICAL_FEATURES,
        window_size=WINDOW_SIZE, horizon=HORIZON,
        flat=flat, multi_step=True,
    )


# ── LOPO loop ─────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("Running LOPO cross-validation ...")
print("=" * 60)

results_list = []

for test_pid in ALL_PATIENTS:
    cohort = "2018" if test_pid in COHORT_2018 else "2020"
    train_pids = [p for p in ALL_PATIENTS if p != test_pid]
    print(f"[test={test_pid}] ({cohort})  train on {len(train_pids)} patients ... ",
          end="", flush=True)

    try:
        # ── Fit scaler on pooled RAW training data (11 patients only) ───────
        pooled_train_raw = pd.concat(
            [prep_data[p] for p in train_pids], ignore_index=True
        )
        scaler = StandardScaler().fit(pooled_train_raw[CLINICAL_FEATURES])
        glucose_idx = CLINICAL_FEATURES.index("glucose")
        y_mean = float(scaler.mean_[glucose_idx])
        y_std  = float(scaler.scale_[glucose_idx])

        # ── Window each training patient separately, then pool the arrays ───
        # (train/val split carved chronologically PER patient, see docstring)
        X3_tr_list,   X3_val_list   = [], []
        y_tr_list,    y_val_list    = [], []
        Xf_full_list, y_full_list   = [], []   # RF has no early stopping — uses all windows

        for p in train_pids:
            df_p = prep_data[p].copy()
            df_p[CLINICAL_FEATURES] = scaler.transform(df_p[CLINICAL_FEATURES])

            Xf_p, y_p = _window_patient(df_p, flat=True)
            X3_p, _   = _window_patient(df_p, flat=False)

            n_val_p = max(1, int(len(y_p) * VAL_RATIO))

            X3_tr_list.append(X3_p[:-n_val_p]); X3_val_list.append(X3_p[-n_val_p:])
            y_tr_list.append(y_p[:-n_val_p]);    y_val_list.append(y_p[-n_val_p:])

            Xf_full_list.append(Xf_p)
            y_full_list.append(y_p)

        X_train_3d = np.concatenate(X3_tr_list)
        X_val_3d   = np.concatenate(X3_val_list)
        y_train    = np.concatenate(y_tr_list)
        y_val      = np.concatenate(y_val_list)

        X_rf_train = np.concatenate(Xf_full_list)
        y_rf_train = np.concatenate(y_full_list)

        n_features = X_train_3d.shape[2]

        # ── Held-out test patient: raw target (mg/dL) + scaled features ─────
        df_test_raw = prep_data[test_pid]
        _, y_test_raw = _window_patient(df_test_raw, flat=True)   # mg/dL, pre-scaling

        df_test_scaled = df_test_raw.copy()
        df_test_scaled[CLINICAL_FEATURES] = scaler.transform(df_test_raw[CLINICAL_FEATURES])
        X_test_flat, _ = _window_patient(df_test_scaled, flat=True)
        X_test_3d, _   = _window_patient(df_test_scaled, flat=False)

        y_true_30 = y_test_raw[:, STEP_30MIN]

        # ── Random Forest ─────────────────────────────────────────────────
        rf_model = rf.train(X_rf_train, y_rf_train, params=rf_params)
        rf_pred_30 = (rf_model.predict(X_test_flat) * y_std + y_mean)[:, STEP_30MIN]

        # ── LSTM ──────────────────────────────────────────────────────────
        lstm_model = GlucoseLSTM(n_features=n_features, hidden_size=lstm_hs, horizon=HORIZON)
        lstm_model, _, lstm_epochs = lstm_train(
            X_train_3d, y_train, X_val_3d, y_val,
            model=lstm_model, lr=lstm_lr,
        )
        lstm_model.eval().to(device)
        X_t = torch.tensor(X_test_3d, dtype=torch.float32).to(device)
        with torch.no_grad():
            lstm_pred_30 = (lstm_model(X_t).cpu().numpy() * y_std + y_mean)[:, STEP_30MIN]

        rf_rmse_val   = rmse(y_true_30, rf_pred_30)
        rf_mae_val    = mae(y_true_30, rf_pred_30)
        lstm_rmse_val = rmse(y_true_30, lstm_pred_30)
        lstm_mae_val  = mae(y_true_30, lstm_pred_30)
        rf_zoneA      = clarke_error_grid(y_true_30, rf_pred_30)["percentages"]["A"]
        lstm_zoneA    = clarke_error_grid(y_true_30, lstm_pred_30)["percentages"]["A"]

        results_list.append({
            "test_patient": test_pid,
            "cohort":       cohort,
            "rf_rmse":      rf_rmse_val,
            "rf_mae":       rf_mae_val,
            "lstm_rmse":    lstm_rmse_val,
            "lstm_mae":     lstm_mae_val,
            "rf_zoneA":     rf_zoneA,
            "lstm_zoneA":   lstm_zoneA,
        })

        print(f"RF={rf_rmse_val:.2f}  LSTM={lstm_rmse_val:.2f}  (LSTM epochs={lstm_epochs})")

    except Exception as exc:
        print(f"ERROR — {exc}")
        continue

print()
print("LOPO loop complete.")

# ── Summary table ─────────────────────────────────────────────────────────────
results_df = pd.DataFrame(results_list)

sep = "=" * 76
print()
print(sep)
print("LOPO RESULTS — 30-min horizon, per test patient")
print(sep)
print(f"  {'Patient':8s} {'Cohort':7s} {'RF RMSE':>9s} {'RF MAE':>9s} {'RF ZoneA':>10s}  "
      f"{'LSTM RMSE':>10s} {'LSTM MAE':>9s} {'LSTM ZoneA':>11s}")
for _, row in results_df.iterrows():
    print(f"  {row['test_patient']:8s} {row['cohort']:7s} "
          f"{row['rf_rmse']:9.2f} {row['rf_mae']:9.2f} {row['rf_zoneA']:9.1f}%  "
          f"{row['lstm_rmse']:10.2f} {row['lstm_mae']:9.2f} {row['lstm_zoneA']:10.1f}%")

print()
print(sep)
print(f"MEAN +/- STD across {len(results_df)} patients")
print(sep)
print(f"  RF    RMSE: {results_df['rf_rmse'].mean():.2f} +/- {results_df['rf_rmse'].std():.2f}   "
      f"MAE: {results_df['rf_mae'].mean():.2f} +/- {results_df['rf_mae'].std():.2f}   "
      f"Zone A: {results_df['rf_zoneA'].mean():.1f}% +/- {results_df['rf_zoneA'].std():.1f}%")
print(f"  LSTM  RMSE: {results_df['lstm_rmse'].mean():.2f} +/- {results_df['lstm_rmse'].std():.2f}   "
      f"MAE: {results_df['lstm_mae'].mean():.2f} +/- {results_df['lstm_mae'].std():.2f}   "
      f"Zone A: {results_df['lstm_zoneA'].mean():.1f}% +/- {results_df['lstm_zoneA'].std():.1f}%")

# ── Save CSV ──────────────────────────────────────────────────────────────────
csv_path = f"{RESULTS_DIR}/results_lopo.csv"
results_df.to_csv(csv_path, index=False)
print(f"\nSaved: {csv_path}")

# ── Personalised vs. LOPO comparison ───────────────────────────────────────────
print()
print(sep)
print("PERSONALISED vs. LOPO — 30-min horizon (how much does personalisation matter?)")
print(sep)

personalised = pd.read_csv(ALL_MODELS_CSV)
personalised_30 = personalised[personalised["horizon"] == "30min"]

pers_lstm = personalised_30[personalised_30["model"] == "LSTM"]["rmse"].mean()
lopo_lstm = results_df["lstm_rmse"].mean()
print(f"  Personalised LSTM 30-min: {pers_lstm:.2f} mg/dL")
print(f"  LOPO LSTM 30-min:         {lopo_lstm:.2f} mg/dL")
print(f"  Difference:               {lopo_lstm - pers_lstm:.2f} mg/dL")

pers_rf = personalised_30[personalised_30["model"] == "RF"]["rmse"].mean()
lopo_rf = results_df["rf_rmse"].mean()
print()
print(f"  Personalised RF 30-min:   {pers_rf:.2f} mg/dL")
print(f"  LOPO RF 30-min:           {lopo_rf:.2f} mg/dL")
print(f"  Difference:               {lopo_rf - pers_rf:.2f} mg/dL")

print("\nDone.")
