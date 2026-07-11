"""
Cross-dataset transfer — OhioT1DM -> Glucdict, 30-min horizon, zero-shot.

Trains ONE global RF and ONE global LSTM on ALL 12 OhioT1DM patients
pooled together (glucose_only — the only feature common to both datasets,
since Glucdict has no bolus/carbs column), then evaluates those models
DIRECTLY on each Glucdict user's held-out test split, with NO retraining
and NO Glucdict-specific normalisation: the StandardScaler is fit once on
pooled OhioT1DM data and its mean/std (source-domain statistics) are
reused to transform Glucdict's glucose values. This is the standard
zero-shot transfer protocol — the target domain never influences training
or normalisation.

This answers: can a model trained on Type 1 diabetics (OhioT1DM, Medtronic
Enlite CGM) predict glucose for Glucdict participants (Dexcom G6 CGM,
different population) without ever seeing their data during training?

Windowing is done PER PATIENT/PER USER, never across a series boundary —
see run_lopo.py for why concatenating raw DataFrames before windowing
would be incorrect (it would splice unrelated patients' glucose traces
together at each boundary).

The Glucdict test split (last 20% chronologically per user) is identical
to the one used in run_glucdict_experiments.py's Section 2, so the
transfer-vs-personalised RMSE comparison is apples-to-apples: same test
data, only the training regime differs.

Results saved to results/glucdict/results_transfer_ohio_to_glucdict.csv.
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

from src.preprocessing.ohio_loader import load_patient as load_ohio_patient, COHORT_2018, COHORT_2020
from src.preprocessing.glucdict_loader import load_patient as load_glucdict_patient, ALL_USERS, GLUCDICT_BASE
from src.training.pipeline import _preprocess, create_windows
from src.models import random_forest as rf
from src.models.lstm import GlucoseLSTM, train_model as lstm_train
from src.evaluation.metrics import rmse, mae

# ── Config ────────────────────────────────────────────────────────────────────
DATA_ROOT_OHIO       = "data/ohio"
RESULTS_DIR          = "results/glucdict"
GS_PATH              = "results/ohio/grid_search_results.json"
GLUCDICT_5MODEL_CSV  = "results/glucdict/results_glucdict_5models.csv"

WINDOW_SIZE = 12     # 60 min of history — must match create_windows default
HORIZON     = 6      # 30 min ahead
STEP_30MIN  = 5      # index 5 = 30-min step in the multi-step output
VAL_RATIO   = 0.20   # chronological, carved PER OhioT1DM patient (LSTM early stopping)
TRAIN_RATIO = 0.80   # Glucdict chronological split — matches run_glucdict_experiments.py
MIN_ROWS    = 100

GLUCOSE_ONLY      = ["glucose"]   # only feature common to both datasets
ALL_OHIO_PATIENTS = COHORT_2018 + COHORT_2020

os.makedirs(RESULTS_DIR, exist_ok=True)


def _window(df_scaled: pd.DataFrame, flat: bool):
    """create_windows wrapper — multi-step target, glucose_only, single series."""
    return create_windows(
        df_scaled, GLUCOSE_ONLY,
        window_size=WINDOW_SIZE, horizon=HORIZON,
        flat=flat, multi_step=True,
    )


# ── Step 1 — Load & pool ALL OhioT1DM patients (train+test combined) ──────────
print("=" * 60)
print("Loading OhioT1DM — all 12 patients (train + test combined) ...")
print("=" * 60)

ohio_full = {}
for pid in COHORT_2018:
    tr = load_ohio_patient(f"{DATA_ROOT_OHIO}/2018/train/{pid}-ws-training.xml")
    te = load_ohio_patient(f"{DATA_ROOT_OHIO}/2018/test/{pid}-ws-testing.xml")
    ohio_full[pid] = pd.concat([tr, te], ignore_index=True).sort_values("ts").reset_index(drop=True)
for pid in COHORT_2020:
    tr = load_ohio_patient(f"{DATA_ROOT_OHIO}/2020/train/{pid}-ws-training.xml")
    te = load_ohio_patient(f"{DATA_ROOT_OHIO}/2020/test/{pid}-ws-testing.xml")
    ohio_full[pid] = pd.concat([tr, te], ignore_index=True).sort_values("ts").reset_index(drop=True)

ohio_prep = {pid: _preprocess(df, GLUCOSE_ONLY) for pid, df in ohio_full.items()}
for pid in ALL_OHIO_PATIENTS:
    print(f"  {pid}: {len(ohio_prep[pid])} rows")

# ── Fit scaler on ALL pooled OhioT1DM data — source-domain statistics ─────────
pooled_ohio_raw = pd.concat([ohio_prep[p] for p in ALL_OHIO_PATIENTS], ignore_index=True)
scaler = StandardScaler().fit(pooled_ohio_raw[GLUCOSE_ONLY])
glucose_idx = GLUCOSE_ONLY.index("glucose")
y_mean = float(scaler.mean_[glucose_idx])
y_std  = float(scaler.scale_[glucose_idx])
print(f"\nOhioT1DM source-domain glucose stats: mean={y_mean:.2f}  std={y_std:.2f}")

# ── Window each OhioT1DM patient separately, then pool the arrays ─────────────
# (never window across a patient boundary — see module docstring)
X3_tr_list,   X3_val_list   = [], []
y_tr_list,    y_val_list    = [], []
Xf_full_list, y_full_list   = [], []   # RF has no early stopping — trains on all windows

for p in ALL_OHIO_PATIENTS:
    df_p = ohio_prep[p].copy()
    df_p[GLUCOSE_ONLY] = scaler.transform(df_p[GLUCOSE_ONLY])

    Xf_p, y_p = _window(df_p, flat=True)
    X3_p, _   = _window(df_p, flat=False)

    n_val_p = max(1, int(len(y_p) * VAL_RATIO))
    X3_tr_list.append(X3_p[:-n_val_p]); X3_val_list.append(X3_p[-n_val_p:])
    y_tr_list.append(y_p[:-n_val_p]);   y_val_list.append(y_p[-n_val_p:])

    Xf_full_list.append(Xf_p)
    y_full_list.append(y_p)

X_train_3d = np.concatenate(X3_tr_list)
X_val_3d   = np.concatenate(X3_val_list)
y_train    = np.concatenate(y_tr_list)
y_val      = np.concatenate(y_val_list)

X_rf_train = np.concatenate(Xf_full_list)
y_rf_train = np.concatenate(y_full_list)

n_features = X_train_3d.shape[2]
print(f"Pooled OhioT1DM training windows: {len(y_rf_train)}")

# ── Best hyperparameters from grid search ─────────────────────────────────────
with open(GS_PATH) as f:
    gs = json.load(f)
rf_params = gs["rf"]
lstm_hs   = gs["lstm"]["hidden_size"]
lstm_lr   = gs["lstm"]["lr"]
print(f"RF params : {rf_params}")
print(f"LSTM      : hidden_size={lstm_hs}, lr={lstm_lr}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Step 1b — Train ONE global RF + LSTM on ALL of OhioT1DM (glucose_only) ────
print()
print("=" * 60)
print("Training global OhioT1DM models (glucose_only, no held-out patient) ...")
print("=" * 60)

rf_model = rf.train(X_rf_train, y_rf_train, params=rf_params)
print("RF trained.")

lstm_model = GlucoseLSTM(n_features=n_features, hidden_size=lstm_hs, horizon=HORIZON)
lstm_model, _, lstm_epochs = lstm_train(
    X_train_3d, y_train, X_val_3d, y_val,
    model=lstm_model, lr=lstm_lr,
)
lstm_model.eval().to(device)
print(f"LSTM trained ({lstm_epochs} epochs).")

# ── Step 2 — Load Glucdict test split per user (same split as run_glucdict_experiments.py) ─
print()
print("=" * 60)
print("Loading Glucdict — test split per user (80/20 chronological) ...")
print("=" * 60)

glucdict_test = {}
cohort = []
for user_id in ALL_USERS:
    try:
        df = load_glucdict_patient(GLUCDICT_BASE / user_id)
    except Exception as exc:
        print(f"  {user_id}: ERROR loading — {exc}")
        continue

    n_split = int(len(df) * TRAIN_RATIO)
    df_test = df.iloc[n_split:].reset_index(drop=True)
    df_test = _preprocess(df_test, GLUCOSE_ONLY)
    if len(df_test) < MIN_ROWS:
        print(f"  {user_id}: SKIPPED — too few test rows ({len(df_test)})")
        continue

    glucdict_test[user_id] = df_test
    cohort.append(user_id)
    print(f"  {user_id}: {len(df_test)} test rows")

print("Cohort:", cohort)
N_USERS = len(cohort)

# ── Step 3 — Zero-shot transfer: predict on Glucdict, NO retraining ───────────
print()
print("=" * 60)
print("Zero-shot transfer: OhioT1DM -> Glucdict (no retraining) ...")
print("=" * 60)

results_list = []

for uid in cohort:
    try:
        df_raw = glucdict_test[uid]
        _, y_test_raw = _window(df_raw, flat=True)   # mg/dL, pre-scaling

        df_scaled = df_raw.copy()
        # OhioT1DM (source-domain) scaler applied to Glucdict — never refit here
        df_scaled[GLUCOSE_ONLY] = scaler.transform(df_raw[GLUCOSE_ONLY])
        X_test_flat, _ = _window(df_scaled, flat=True)
        X_test_3d, _   = _window(df_scaled, flat=False)

        y_true_30 = y_test_raw[:, STEP_30MIN]

        rf_pred_30 = (rf_model.predict(X_test_flat) * y_std + y_mean)[:, STEP_30MIN]

        X_t = torch.tensor(X_test_3d, dtype=torch.float32).to(device)
        with torch.no_grad():
            lstm_pred_30 = (lstm_model(X_t).cpu().numpy() * y_std + y_mean)[:, STEP_30MIN]

        rf_rmse_val   = rmse(y_true_30, rf_pred_30)
        rf_mae_val    = mae(y_true_30, rf_pred_30)
        lstm_rmse_val = rmse(y_true_30, lstm_pred_30)
        lstm_mae_val  = mae(y_true_30, lstm_pred_30)

        results_list.append({"user": uid, "model": "RF",   "rmse": rf_rmse_val,   "mae": rf_mae_val})
        results_list.append({"user": uid, "model": "LSTM", "rmse": lstm_rmse_val, "mae": lstm_mae_val})

        print(f"  [{uid}]  RF={rf_rmse_val:.2f}  LSTM={lstm_rmse_val:.2f}")

    except Exception as exc:
        print(f"  [{uid}]  ERROR — {exc}")

results_df = pd.DataFrame(results_list)

# ── Per-user summary ────────────────────────────────────────────────────────
sep = "=" * 68
print()
print(sep)
print(f"TRANSFER RESULTS — OhioT1DM -> Glucdict, 30-min horizon, per user ({N_USERS} users)")
print(sep)
print(f"  {'User':8s}  {'RF RMSE':>9s}  {'LSTM RMSE':>10s}")
for uid in cohort:
    sub = results_df[results_df["user"] == uid]
    r = sub[sub["model"] == "RF"]["rmse"]
    l = sub[sub["model"] == "LSTM"]["rmse"]
    if len(r) and len(l):
        print(f"  {uid:8s}  {r.iloc[0]:9.2f}  {l.iloc[0]:10.2f}")

# ── Save CSV ──────────────────────────────────────────────────────────────────
csv_path = f"{RESULTS_DIR}/results_transfer_ohio_to_glucdict.csv"
results_df.to_csv(csv_path, index=False)
print(f"\nSaved: {csv_path}")

# ── Transfer vs. Personalised comparison ───────────────────────────────────────
print()
print(sep)
print("TRANSFER vs. PERSONALISED — glucose_only, 30-min horizon (domain-shift gap)")
print(sep)

personalised = pd.read_csv(GLUCDICT_5MODEL_CSV)

transfer_rf   = results_df[results_df["model"] == "RF"]["rmse"].mean()
transfer_lstm = results_df[results_df["model"] == "LSTM"]["rmse"].mean()
pers_rf       = personalised[personalised["model"] == "RF"]["rmse"].mean()
pers_lstm     = personalised[personalised["model"] == "LSTM"]["rmse"].mean()

print(f"  {'Model':6s}  {'Transfer RMSE':>14s}  {'Personalised RMSE':>18s}  {'Gap':>8s}")
print(f"  {'RF':6s}  {transfer_rf:14.2f}  {pers_rf:18.2f}  {transfer_rf - pers_rf:+8.2f}")
print(f"  {'LSTM':6s}  {transfer_lstm:14.2f}  {pers_lstm:18.2f}  {transfer_lstm - pers_lstm:+8.2f}")

print("\nDone.")
