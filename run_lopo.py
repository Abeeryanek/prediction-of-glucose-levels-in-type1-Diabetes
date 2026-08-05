"""
Leave-One-Patient-Out (LOPO) cross-validation — OhioT1DM, all 5 models,
3 horizons (15/30/45 min), 2 normalisation modes.

For each of the 12 patients, trains all 5 models (RF, LSTM, Autoencoder,
TCN, Transformer) on the pooled data of the OTHER 11 patients (a
population model) and evaluates on the held-out patient's full record
(official train + test XML files combined). This measures how well a
model trained on other patients generalises to an unseen patient, in
contrast to the per-patient personalised models used in run_experiments.py.

Two normalisation modes are compared per fold:
  - "pooled"             : a single StandardScaler fit on the pooled raw
                            training data of the 11 training patients,
                            applied to those patients and to the held-out
                            patient alike (original LOPO behaviour).
  - "per_patient_scaled"  : every patient (including the held-out one) is
                            normalised with its OWN mean/std before
                            pooling, removing inter-patient baseline
                            glucose differences. Predictions for the
                            held-out patient are decoded back to mg/dL
                            using that same patient's own mean/std. Tests
                            whether inter-patient baseline differences are
                            what hurts the population model.

Windowing is done PER PATIENT (never across a patient boundary) before
pooling the resulting (X, y) arrays. Concatenating raw per-patient
DataFrames first and windowing the pooled frame once would let
create_windows slide a window across the boundary between two unrelated
patients' series, splicing one patient's glucose trace onto another's for
a handful of samples at each of the 11 boundaries. Windowing first, then
pooling arrays, avoids this.

Normalisation ("pooled" mode): a StandardScaler is fit on the pooled RAW
training data of the 11 training patients only, then applied to those
same patients and to the held-out test patient — never fit on test-
patient data. ("per_patient_scaled" mode uses each patient's own data by
design — see above.)

Results saved to results/ohio/results_lopo_full.csv.
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
from src.models.autoencoder import GlucoseSeq2Seq, train_model as ae_train, evaluate as ae_eval
from src.models.tcn import GlucoseTCN, train_model as tcn_train, evaluate as tcn_eval
from src.models.transformer import GlucoseTransformer, train_model as tr_train, evaluate as tr_eval
from src.evaluation.metrics import rmse, mae, clarke_error_grid

# ── Config ────────────────────────────────────────────────────────────────────
DATA_ROOT      = "data/ohio"
RESULTS_DIR    = "results/ohio"
GS_PATH        = "results/ohio/grid_search_results.json"
ALL_MODELS_CSV = "results/ohio/results_all_models.csv"

WINDOW_SIZE = 12     # 60 min of history — must match create_windows default
HORIZON     = 9      # 9 steps x 5 min = 45 min max
HORIZONS    = {2: "15min", 5: "30min", 8: "45min"}
VAL_RATIO   = 0.20   # chronological, carved PER training patient
MODES       = ["pooled", "per_patient_scaled"]

CLINICAL_FEATURES = ["glucose", "bolus", "carbs"]
ALL_PATIENTS      = COHORT_2018 + COHORT_2020
MODEL_NAMES       = ["RF", "LSTM", "Autoencoder", "TCN", "Transformer"]

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

glucose_idx = CLINICAL_FEATURES.index("glucose")

# Per-patient scalers for the "per_patient_scaled" mode — fit once per patient
# on that patient's own full record; independent of which fold is running.
patient_scalers = {
    pid: StandardScaler().fit(prep_data[pid][CLINICAL_FEATURES])
    for pid in ALL_PATIENTS
}

# ── Best hyperparameters from grid search ─────────────────────────────────────
with open(GS_PATH) as f:
    gs = json.load(f)

rf_params   = gs["rf"]
lstm_hs     = gs["lstm"]["hidden_size"]
lstm_lr     = gs["lstm"]["lr"]
ae_latent   = gs["autoencoder"].get("latent_size", 32)
ae_lr       = gs["autoencoder"].get("lr", 1e-3)
tcn_filters = gs["tcn"].get("num_filters", 64)
tcn_lr      = gs["tcn"].get("lr", 1e-3)
tr_d_model  = gs["transformer"].get("d_model", 64)
tr_nhead    = gs["transformer"].get("nhead", 4)
tr_lr       = gs["transformer"].get("lr", 1e-3)

print(f"\nRF params   : {rf_params}")
print(f"LSTM        : hidden_size={lstm_hs}, lr={lstm_lr}")
print(f"Autoencoder : latent_size={ae_latent}, lr={ae_lr}")
print(f"TCN         : num_filters={tcn_filters}, lr={tcn_lr}")
print(f"Transformer : d_model={tr_d_model}, nhead={tr_nhead}, lr={tr_lr}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


def _window_patient(df_scaled: pd.DataFrame, flat: bool):
    """create_windows wrapper — multi-step target, single patient's df only."""
    return create_windows(
        df_scaled, CLINICAL_FEATURES,
        window_size=WINDOW_SIZE, horizon=HORIZON,
        flat=flat, multi_step=True,
    )


def _dl_predict(model, X, y_mean, y_std):
    model.eval().to(device)
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        return model(X_t).cpu().numpy() * y_std + y_mean


# ── LOPO loop ─────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("Running LOPO cross-validation — 2 modes x 12 folds x 5 models ...")
print("=" * 60)

results_list = []

for mode in MODES:
    print()
    print("-" * 60)
    print(f"MODE: {mode}")
    print("-" * 60)

    for test_pid in ALL_PATIENTS:
        cohort = "2018" if test_pid in COHORT_2018 else "2020"
        train_pids = [p for p in ALL_PATIENTS if p != test_pid]
        print(f"[{mode} | test={test_pid}] ({cohort})  train on {len(train_pids)} patients ... ",
              end="", flush=True)

        try:
            # ── Determine scaling for this fold, per mode ────────────────────
            if mode == "pooled":
                pooled_train_raw = pd.concat(
                    [prep_data[p] for p in train_pids], ignore_index=True
                )
                fold_scaler = StandardScaler().fit(pooled_train_raw[CLINICAL_FEATURES])
                y_mean = float(fold_scaler.mean_[glucose_idx])
                y_std  = float(fold_scaler.scale_[glucose_idx])

                def _scale(pid):
                    df_p = prep_data[pid].copy()
                    df_p[CLINICAL_FEATURES] = fold_scaler.transform(df_p[CLINICAL_FEATURES])
                    return df_p

                test_y_mean, test_y_std = y_mean, y_std

            else:  # per_patient_scaled — each patient normalised with its own mean/std
                def _scale(pid):
                    df_p = prep_data[pid].copy()
                    df_p[CLINICAL_FEATURES] = patient_scalers[pid].transform(df_p[CLINICAL_FEATURES])
                    return df_p

                test_y_mean = float(patient_scalers[test_pid].mean_[glucose_idx])
                test_y_std  = float(patient_scalers[test_pid].scale_[glucose_idx])

            # ── Window each training patient separately, then pool the arrays ───
            # (train/val split carved chronologically PER patient, see docstring)
            X3_tr_list,   X3_val_list   = [], []
            y_tr_list,    y_val_list    = [], []
            Xf_full_list, y_full_list   = [], []   # RF has no early stopping — uses all windows

            for p in train_pids:
                df_p_scaled = _scale(p)

                Xf_p, y_p = _window_patient(df_p_scaled, flat=True)
                X3_p, _   = _window_patient(df_p_scaled, flat=False)

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

            df_test_scaled = _scale(test_pid)
            X_test_flat, _ = _window_patient(df_test_scaled, flat=True)
            X_test_3d, _   = _window_patient(df_test_scaled, flat=False)

            # ── Train all 5 models ───────────────────────────────────────────
            rf_model = rf.train(X_rf_train, y_rf_train, params=rf_params)
            rf_pred  = rf_model.predict(X_test_flat) * test_y_std + test_y_mean

            lstm_model = GlucoseLSTM(n_features=n_features, hidden_size=lstm_hs, horizon=HORIZON)
            lstm_model, _, lstm_epochs = lstm_train(
                X_train_3d, y_train, X_val_3d, y_val, model=lstm_model, lr=lstm_lr,
            )
            lstm_pred = _dl_predict(lstm_model, X_test_3d, test_y_mean, test_y_std)

            ae_model = GlucoseSeq2Seq(n_features=n_features, latent_size=ae_latent, horizon=HORIZON)
            ae_model, _, ae_epochs = ae_train(
                X_train_3d, y_train, X_val_3d, y_val, model=ae_model, lr=ae_lr,
            )
            ae_pred = _dl_predict(ae_model, X_test_3d, test_y_mean, test_y_std)

            tcn_model = GlucoseTCN(n_features=n_features, num_filters=tcn_filters, horizon=HORIZON)
            tcn_model, _, tcn_epochs = tcn_train(
                X_train_3d, y_train, X_val_3d, y_val, model=tcn_model, lr=tcn_lr,
            )
            tcn_pred = _dl_predict(tcn_model, X_test_3d, test_y_mean, test_y_std)

            tr_model = GlucoseTransformer(
                n_features=n_features, d_model=tr_d_model, nhead=tr_nhead, horizon=HORIZON
            )
            tr_model, _, tr_epochs = tr_train(
                X_train_3d, y_train, X_val_3d, y_val, model=tr_model, lr=tr_lr,
            )
            tr_pred = _dl_predict(tr_model, X_test_3d, test_y_mean, test_y_std)

            model_preds = {
                "RF": rf_pred, "LSTM": lstm_pred,
                "Autoencoder": ae_pred, "TCN": tcn_pred,
                "Transformer": tr_pred,
            }
            model_epochs = {
                "RF": None, "LSTM": lstm_epochs,
                "Autoencoder": ae_epochs, "TCN": tcn_epochs,
                "Transformer": tr_epochs,
            }

            # ── Metrics per horizon, per model ────────────────────────────────
            for step_idx, horizon_name in HORIZONS.items():
                y_true_step = y_test_raw[:, step_idx]
                for model_name, preds in model_preds.items():
                    y_pred_step = preds[:, step_idx]
                    zone_a = clarke_error_grid(y_true_step, y_pred_step)["percentages"]["A"]
                    results_list.append({
                        "test_patient": test_pid,
                        "cohort":       cohort,
                        "mode":         mode,
                        "horizon":      horizon_name,
                        "model":        model_name,
                        "rmse":         rmse(y_true_step, y_pred_step),
                        "mae":          mae(y_true_step, y_pred_step),
                        "zone_a":       zone_a,
                        "epochs":       model_epochs[model_name],
                    })

            print(
                f"RF={rmse(y_test_raw[:, 5], rf_pred[:, 5]):.2f}  "
                f"LSTM={rmse(y_test_raw[:, 5], lstm_pred[:, 5]):.2f}  "
                f"AE={rmse(y_test_raw[:, 5], ae_pred[:, 5]):.2f}  "
                f"TCN={rmse(y_test_raw[:, 5], tcn_pred[:, 5]):.2f}  "
                f"Transformer={rmse(y_test_raw[:, 5], tr_pred[:, 5]):.2f}  "
                f"(epochs LSTM={lstm_epochs} AE={ae_epochs} TCN={tcn_epochs} Tr={tr_epochs})"
            )

        except Exception as exc:
            print(f"ERROR — {exc}")
            continue

print()
print("LOPO loop complete.")

# ── Results table ─────────────────────────────────────────────────────────────
results_df = pd.DataFrame(results_list)

sep = "=" * 88
print()
print(sep)
print("LOPO RESULTS — mean +/- std RMSE [mg/dL], per mode / horizon / model")
print(sep)

for mode in MODES:
    print(f"\nMode: {mode}")
    for horizon_name in HORIZONS.values():
        print(f"  Horizon: {horizon_name}")
        sub = results_df[(results_df["mode"] == mode) & (results_df["horizon"] == horizon_name)]
        for model_name in MODEL_NAMES:
            m = sub[sub["model"] == model_name]
            r_mean  = m["rmse"].mean()
            r_std   = m["rmse"].std()
            a_mean  = m["mae"].mean()
            za_mean = m["zone_a"].mean()
            e_mean  = m["epochs"].mean()   # NaN for RF (no training epochs)
            e_str   = f"{e_mean:.1f}" if pd.notna(e_mean) else "n/a"
            print(f"    {model_name:14s}  RMSE: {r_mean:.2f} +/- {r_std:.2f}   "
                  f"MAE: {a_mean:.2f}   Zone A: {za_mean:.1f}%   epochs: {e_str}")

# ── Save CSV ──────────────────────────────────────────────────────────────────
csv_path = f"{RESULTS_DIR}/results_lopo_full.csv"
results_df.to_csv(csv_path, index=False)
print(f"\nSaved: {csv_path}")

# ── Personalised vs. LOPO comparison ───────────────────────────────────────────
print()
print(sep)
print("PERSONALISED vs. LOPO gap — mean RMSE, per horizon / model / mode")
print(sep)

personalised = pd.read_csv(ALL_MODELS_CSV)

for horizon_name in HORIZONS.values():
    print(f"\nHorizon: {horizon_name}")
    personalised_h = personalised[personalised["horizon"] == horizon_name]
    for model_name in MODEL_NAMES:
        pers_rmse = personalised_h[personalised_h["model"] == model_name]["rmse"].mean()
        print(f"  {model_name:14s}  Personalised: {pers_rmse:.2f} mg/dL")
        for mode in MODES:
            lopo_rmse = results_df[
                (results_df["mode"] == mode) &
                (results_df["horizon"] == horizon_name) &
                (results_df["model"] == model_name)
            ]["rmse"].mean()
            gap = lopo_rmse - pers_rmse
            print(f"    {mode:20s} LOPO: {lopo_rmse:.2f} mg/dL   Gap: {gap:+.2f} mg/dL")

print("\nDone.")
