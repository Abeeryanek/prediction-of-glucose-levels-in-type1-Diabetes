"""
autoencoder.py — Seq2Seq Autoencoder, Window Size x Horizon Walk-Forward Pipeline
=====================================================================================
Self-contained: no external config/utils modules. Everything in one file.
Identical structure to lstm.py / cnn_lstm.py / transformer.py / tcn.py — only the
model architecture differs (LSTM encoder -> bottleneck latent -> decoder head).

NOTE: This is a from-scratch reimplementation of GlucoseSeq2Seq matching plan §2.3
(latent_size grid, tight-bottleneck rationale per Srivastava et al. 2015). It is
NOT copied from your colleague's actual src/models/autoencoder.py, since only the
usage pattern (train_model(X_train, y_train, X_val, y_val, model=..., lr=...)
returning (model, history, epochs)) was visible in the Glucdict ablation script.
Compare against the real implementation once available and reconcile any
architectural differences.

Global DL training standards applied here:
    - max_epochs   = 150
    - patience     = 15    (early stopping, monitor=val_loss, restore_best_weights)
    - batch_size   = 32
    - val_ratio    = 0.20  (last 20% chronological)
    - shuffle      = False (temporal order preserved)
    - seed         = 42    (torch.manual_seed + np.random.seed)

Architecture (per plan §2.3):
    - Encoder LSTM compresses the input window to its final hidden state
    - Bottleneck Linear layer compresses hidden_dim -> latent_size (tight,
      forces the encoder to retain only the dominant glucose trend, reducing
      overfitting on small per-patient datasets — Srivastava et al. 2015)
    - Decoder Linear head maps latent_size -> hidden_dim -> single prediction

STRICT TIME-BASED SHIFTING APPLIED: 
    Rolling windows, lag extraction, target horizon generation, and sequence 
    building are anchored strictly to timestamp math to prevent cross-gap contamination.
"""
import glob
import os
import json
import pickle
import warnings
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import clarke_error_grid as ceg

warnings.filterwarnings("ignore")

# ============================================================================
# CONSTANTS (all inline — no config.py)
# ============================================================================
DATA_PATH = "data/bigideas"
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

MODEL_NAME = "autoencoder"

WINDOW_SIZES = {"1h": 12, "2h": 24, "3h": 36, "6h": 72}   # lookback, in 5-min steps
HORIZONS = {"15min": 3, "30min": 6, "45min": 9}            # prediction horizon, in 5-min steps
N_SPLITS = 5                                                # walk-forward folds per patient

# ---- Global DL training standards ----
MAX_EPOCHS = 150
PATIENCE = 15
BATCH_SIZE = 32
VAL_RATIO = 0.20      # last 20% chronological
SHUFFLE = False       # temporal order preserved

# ---- Autoencoder-specific grid (per plan §2.3) ----
DL_PARAM_GRID = {
    "latent_size": [16, 32, 64],
    "lr": [1e-3, 5e-4],
}
GRID_SEARCH_EPOCHS = 30      # reduced-epoch grid search (documented shortcut, see plan §2.6)
GRID_SEARCH_PATIENCE = 5

print(f"\n{'='*70}\nSEQ2SEQ AUTOENCODER — WINDOW x HORIZON WALK-FORWARD PIPELINE\n{'='*70}")

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1/9] LOADING DATA...")

file_list = glob.glob(os.path.join(DATA_PATH, "clean_patient_*.parquet"))
if not file_list:
    file_list = glob.glob("bent_*.parquet")
if not file_list:
    raise ValueError("No parquet files found!")

df = pd.concat([pd.read_parquet(f) for f in file_list], ignore_index=True)

# ---> NEW: Ensure Timestamp is a true Datetime object for time-based math
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df = df.sort_values(["Patient_ID", "Timestamp"]).reset_index(drop=True)
print(f"Loaded: {len(df):,} rows from {df['Patient_ID'].nunique()} patients")

# ============================================================================
# 2. BASE FEATURE ENGINEERING (independent of window_size — built ONCE)
# ============================================================================
print("\n[2/9] BASE FEATURE ENGINEERING (temporal, sensors, food, activity, physio)...")

df["Hour"] = df["Timestamp"].dt.hour
df["Minute"] = df["Timestamp"].dt.minute
df["DayOfWeek"] = df["Timestamp"].dt.dayofweek
df["MinFromMidnight"] = df["Hour"] * 60 + df["Minute"]
df["Hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
df["Hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)
temporal_features = ["Hour", "Minute", "DayOfWeek", "MinFromMidnight", "Hour_sin", "Hour_cos"]

sensor_cols = [c for c in ["Heart_Rate", "Acc_Vmu", "EDA", "Skin_Temp", "BVP", "IBI"] if c in df.columns]
for col in sensor_cols:
    df[col] = (df.groupby("Patient_ID")[col]
                 .transform(lambda x: x.interpolate(method="linear", limit=6)))
print(f"  Interpolated {len(sensor_cols)} sensors: {sensor_cols}")

food_features_all = [c for c in df.columns if any(
    c.startswith(n) for n in ["calorie_", "total_carb_", "dietary_fiber_", "sugar_", "protein_", "total_fat_"])]
food_features_carbs = [c for c in df.columns if c.startswith("total_carb_")]
print(f"  Food features (all): {len(food_features_all)}  |  (carbs only): {len(food_features_carbs)}")

# ---> NEW: Time-indexed temporary dataframe for Strict Rolling Windows
temp_time_df = df.set_index("Timestamp")

activity_features = []
if "Acc_Vmu" in df.columns:
    for label, periods in {"30min": 6, "1h": 12, "2h": 24, "4h": 48}.items():
        col_mean = f"acc_vmu_mean_{label}"
        col_max = f"acc_vmu_max_{label}"
        # Roll using exactly the time string (e.g. "30min") instead of blind periods
        df[col_mean] = temp_time_df.groupby("Patient_ID")["Acc_Vmu"].rolling(label).mean().values
        df[col_max] = temp_time_df.groupby("Patient_ID")["Acc_Vmu"].rolling(label).max().values
        activity_features += [col_mean, col_max]
print(f"  Activity features: {len(activity_features)}")

physio_features_all, physio_hr, physio_eda, physio_bvp, physio_ibi, physio_skin = ([] for _ in range(6))

def _add_std(col_src, windows, prefix, target_list):
    if col_src in df.columns:
        for label, periods in windows.items():
            col = f"{prefix}_std_{label}"
            # Roll using exactly the time string (e.g. "1h")
            df[col] = temp_time_df.groupby("Patient_ID")[col_src].rolling(label).std().values
            physio_features_all.append(col)
            target_list.append(col)

_add_std("Heart_Rate", {"30min": 6, "1h": 12, "2h": 24}, "hr", physio_hr)
_add_std("EDA", {"30min": 6, "45min": 9, "1h": 12}, "eda", physio_eda)
_add_std("BVP", {"30min": 6, "1h": 12}, "bvp", physio_bvp)
_add_std("IBI", {"30min": 6, "1h": 12}, "ibi", physio_ibi)
_add_std("Skin_Temp", {"30min": 6, "1h": 12}, "skin_temp", physio_skin)
print(f"  Physiological variability features: {len(physio_features_all)}")

# ============================================================================
# 3. WALK-FORWARD FOLD BUILDER (per-patient TimeSeriesSplit)
# ============================================================================
def build_walk_forward_folds(df_in, n_splits):
    fold_train_idx = [[] for _ in range(n_splits)]
    fold_test_idx = [[] for _ in range(n_splits)]

    for pid, group in df_in.groupby("Patient_ID"):
        group = group.sort_values("Timestamp")
        idx = group.index.values
        if len(idx) < n_splits + 1:
            continue
        tscv = TimeSeriesSplit(n_splits=n_splits)
        for fold_i, (tr, te) in enumerate(tscv.split(idx)):
            fold_train_idx[fold_i].extend(idx[tr])
            fold_test_idx[fold_i].extend(idx[te])

    folds = []
    for fold_i in range(n_splits):
        df_tr = df_in.loc[fold_train_idx[fold_i]].sort_values(["Patient_ID", "Timestamp"]).reset_index(drop=True)
        df_te = df_in.loc[fold_test_idx[fold_i]].sort_values(["Patient_ID", "Timestamp"]).reset_index(drop=True)
        folds.append((df_tr, df_te))
    return folds


# ============================================================================
# 4. 3D SEQUENCE BUILDER (Time-Locked)
# ============================================================================
def create_3d_sequences(X_scaled, y_series, patient_ids, timestamps, seq_length):
    patient_ids = np.asarray(patient_ids)
    timestamps = np.asarray(timestamps)
    Xs, ys = [], []
    
    # Expected duration for a perfectly contiguous sequence
    expected_duration = np.timedelta64((seq_length - 1) * 5, 'm')

    for i in range(seq_length - 1, len(X_scaled)):
        if patient_ids[i - seq_length + 1] != patient_ids[i]:
            continue
            
        # ---> NEW: Strict timestamp validation to reject sequences spanning missing data gaps
        if (timestamps[i] - timestamps[i - seq_length + 1]) != expected_duration:
            continue
            
        Xs.append(X_scaled[i - seq_length + 1: i + 1])
        ys.append(y_series.iloc[i])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def build_seq_dataset(df_tr, df_te, features, target_col, seq_length):
    train_clean = df_tr.dropna(subset=[target_col] + features).reset_index(drop=True)
    test_clean = df_te.dropna(subset=[target_col] + features).reset_index(drop=True)

    scaler = StandardScaler()
    X_train_2d = scaler.fit_transform(train_clean[features])
    X_test_2d = scaler.transform(test_clean[features])

    X_train_3d, y_train = create_3d_sequences(
        X_train_2d, train_clean[target_col], train_clean["Patient_ID"], train_clean["Timestamp"], seq_length
    )
    X_test_3d, y_test = create_3d_sequences(
        X_test_2d, test_clean[target_col], test_clean["Patient_ID"], test_clean["Timestamp"], seq_length
    )

    return {
        "X_train": X_train_3d, "y_train": y_train,
        "X_test": X_test_3d, "y_test": y_test,
        "n_train": len(train_clean), "n_test": len(test_clean),
    }


# ============================================================================
# 5. CLINICAL SAMPLE WEIGHTING
# ============================================================================
def calculate_clinical_weights(y_true):
    weights = np.ones(len(y_true), dtype=np.float32)
    weights[y_true < 54] = 3.0
    weights[(y_true >= 54) & (y_true < 70)] = 2.5
    weights[(y_true > 180) & (y_true <= 250)] = 1.5
    weights[y_true > 250] = 2.0
    return weights


# ============================================================================
# 6. MODEL DEFINITION — Seq2Seq Autoencoder
# ============================================================================
class GlucoseSeq2Seq(nn.Module):
    def __init__(self, input_dim, latent_size=16, hidden_dim=64, num_layers=1, dropout=0.3):
        super().__init__()
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                                     batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.to_latent = nn.Linear(hidden_dim, latent_size)
        self.relu = nn.ReLU()
        self.from_latent = nn.Linear(latent_size, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        _, (h_n, _) = self.encoder_lstm(x)
        h_last = h_n[-1]                       # (batch, hidden_dim)
        latent = self.relu(self.to_latent(h_last))   # (batch, latent_size)
        decoded = self.relu(self.from_latent(latent))
        decoded = self.dropout(decoded)
        return self.fc_out(decoded)


# ============================================================================
# 7. TRAINING LOOP
# ============================================================================
def train_weighted_autoencoder(x_train, y_train, sample_weights, input_dim,
                                epochs=MAX_EPOCHS, patience=PATIENCE, val_ratio=VAL_RATIO,
                                **model_kwargs):
    split = int(len(x_train) * (1 - val_ratio))
    xt = torch.tensor(x_train[:split])
    yt = torch.tensor(y_train[:split]).unsqueeze(1)
    wt = torch.tensor(sample_weights[:split]).unsqueeze(1)

    xv = torch.tensor(x_train[split:])
    yv = torch.tensor(y_train[split:]).unsqueeze(1)
    wv = torch.tensor(sample_weights[split:]).unsqueeze(1)

    lr = model_kwargs.pop("lr", 1e-3)
    model = GlucoseSeq2Seq(input_dim=input_dim, **model_kwargs)
    criterion = nn.MSELoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    class _DS(Dataset):
        def __init__(self, x, y, w):
            self.x, self.y, self.w = x, y, w
        def __len__(self):
            return len(self.x)
        def __getitem__(self, idx):
            return self.x[idx], self.y[idx], self.w[idx]

    train_loader = DataLoader(_DS(xt, yt, wt), batch_size=BATCH_SIZE, shuffle=SHUFFLE)

    best_val_loss = float("inf")
    best_weights = None
    patience_counter = 0
    epochs_trained = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb, wb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = (criterion(preds, yb) * wb).mean()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_preds = model(xv)
            val_loss = (criterion(val_preds, yv) * wv).mean().item()

        epochs_trained = epoch + 1

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_weights)
    return model, best_val_loss, epochs_trained


# ============================================================================
# 8. GRID SEARCH
# ============================================================================
def grid_search_dl(x_train, y_train, sample_weights, input_dim, param_grid):
    keys = list(param_grid.keys())
    combos = list(product(*[param_grid[k] for k in keys]))

    best_score, best_params = float("inf"), {}
    for combo in combos:
        params = dict(zip(keys, combo))
        _, val_loss, _ = train_weighted_autoencoder(
            x_train, y_train, sample_weights, input_dim=input_dim,
            epochs=GRID_SEARCH_EPOCHS, patience=GRID_SEARCH_PATIENCE,
            latent_size=params["latent_size"], lr=params["lr"],
        )
        if val_loss < best_score:
            best_score, best_params = val_loss, params
    return best_params, best_score


# ============================================================================
# 9. METRICS
# ============================================================================
def metrics_dict(y_true, y_pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": float(np.mean(np.abs((np.asarray(y_true) - y_pred) / np.asarray(y_true))) * 100),
    }


# ============================================================================
# 10. CLARKE ERROR GRID + ABLATION BAR PLOT
# ============================================================================
def clarke_grid_pooled(pooled_predictions, model_name, out_dir, n_splits):
    for (window_label, subset, horizon), bucket in pooled_predictions.items():
        y_true = np.concatenate(bucket["y_true"])
        y_pred = np.concatenate(bucket["y_pred"])

        print(f"--- {window_label} {horizon.upper()} {model_name} {subset.upper()} "
              f"ZONES (pooled, {n_splits} folds) ---")
        zones = ceg.zone(y_true, y_pred)
        print(zones)

        fig = ceg.plot(y_true, y_pred)
        fig.update_layout(title_text=f"Clarke Error Grid — {model_name} {subset} — "
                                     f"{window_label} {horizon.upper()} (pooled)")
        fig.write_html(str(out_dir / f"clarke_{model_name}_{subset}_{window_label}_{horizon}_pooled.html"))


def create_bar_plot(ablation_results, title, save_path):
    df_results = pd.DataFrame(ablation_results).T
    fig, ax = plt.subplots(figsize=(14, 8), layout="constrained")
    df_results.plot(kind="bar", ax=ax, width=0.7, edgecolor="grey")

    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=3)

    ax.set_ylabel("RMSE (mg/dL)", fontweight="bold")
    ax.set_title(title, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# MAIN PIPELINE
# ============================================================================
grid_cache_path = RESULTS_DIR / f"grid_search_{MODEL_NAME}.json"
grid_cache = json.load(open(grid_cache_path)) if grid_cache_path.exists() else {}

all_results = []
all_ablation_results = []
pooled_predictions = {}

# ============================================================================
# 11. WINDOW SIZE LOOP — STRICT TIME-BASED MATH APPLIED HERE
# ============================================================================
for window_label, window_size in WINDOW_SIZES.items():
    print(f"\n{'#'*70}\nWINDOW SIZE: {window_label} ({window_size} steps)\n{'#'*70}")

    df_w = df.copy()
    SEQ_LENGTH = window_size
    
    # ---> NEW: Exact Time Lookup table
    lookup = df_w.set_index(["Patient_ID", "Timestamp"])["Glucose"]

    # ------------------------------------------------------------------------
    # GLUCOSE LAG — Exact Timestamp Shift (Not row shift)
    # ------------------------------------------------------------------------
    print(f"\n[3/9] GLUCOSE LAG FEATURES for {window_label} (exact time mapping)...")
    candidate_lags = [1, 2, 3, 6, 12, 24, 36, 72]
    lags = sorted(set(l for l in candidate_lags if l <= window_size) | {window_size})

    lag_features = []
    for lag in lags:
        col = f"glucose_lag_{lag}"
        # Shift back exactly `lag` periods of 5 minutes
        past_time = df_w["Timestamp"] - pd.Timedelta(minutes=lag * 5)
        df_w[col] = pd.MultiIndex.from_arrays([df_w["Patient_ID"], past_time]).map(lookup)
        lag_features.append(col)

    roc_lag = max(1, window_size // 2)
    
    # Map exact past glucose for rates of change
    lag_1_val = pd.MultiIndex.from_arrays([df_w["Patient_ID"], df_w["Timestamp"] - pd.Timedelta(minutes=5)]).map(lookup)
    lag_roc_val = pd.MultiIndex.from_arrays([df_w["Patient_ID"], df_w["Timestamp"] - pd.Timedelta(minutes=roc_lag * 5)]).map(lookup)
    
    df_w["glucose_roc_1"] = df_w["Glucose"] - lag_1_val
    df_w[f"glucose_roc_{roc_lag}"] = df_w["Glucose"] - lag_roc_val
    lag_features.extend(["glucose_roc_1", f"glucose_roc_{roc_lag}"])
    print(f"  Created {len(lag_features)} glucose lag/change features for this window")

    # ------------------------------------------------------------------------
    # TARGET EXTRACTION — Exact Timestamp Shift (Not row shift)
    # ------------------------------------------------------------------------
    for label, periods in HORIZONS.items():
        # label is "15min", "30min", etc. Timedelta natively parses these.
        future_times = df_w["Timestamp"] + pd.Timedelta(label.replace("min", "m"))
        df_w[f"Target_{label}"] = pd.MultiIndex.from_arrays([df_w["Patient_ID"], future_times]).map(lookup)

    all_features = (temporal_features + sensor_cols + food_features_all +
                     activity_features + physio_features_all + lag_features)
    all_features = [f for f in all_features if f in df_w.columns]

    all_features_comparable = (temporal_features + sensor_cols + food_features_carbs +
                                 activity_features + physio_hr + lag_features)
    all_features_comparable = [f for f in all_features_comparable if f in df_w.columns]

    # ------------------------------------------------------------------------
    # Ablation groups
    # ------------------------------------------------------------------------
    feature_groups = {
        "physio_only": physio_features_all,
        "EDA": physio_eda,
        "heart_rate": physio_hr,
        "BVP": physio_bvp,
        "IBI": physio_ibi,
        "food_only": food_features_all,
        "carbs_only": food_features_carbs,
        "activity": activity_features,
        "comparable": all_features_comparable, 
        "lag": lag_features, 
        "glucose": ["Glucose"],
        "all": all_features, 
        "physio_food": physio_features_all + food_features_all,
        "food_movement": food_features_all + activity_features,
        "physio_movement": physio_features_all + activity_features,
    }
    feature_groups = {name: [f for f in feats if f in df_w.columns] for name, feats in feature_groups.items()}

    FEATURE_SETS = {"full": all_features, "comparable": all_features_comparable}

    print(f"\n[4/9] BUILDING WALK-FORWARD FOLDS ({window_label})...")
    wf_folds = build_walk_forward_folds(df_w, N_SPLITS)
    df_train_final, df_test_final = wf_folds[-1]
    for i, (tr, te) in enumerate(wf_folds):
        print(f"  Fold {i}: train={len(tr):,}  test={len(te):,}")

    print(f"\n[5/9] GRID SEARCH ({window_label})...")
    gs_key = f"{MODEL_NAME}_{window_label}"
    if gs_key in grid_cache:
        best_params = grid_cache[gs_key]
        print(f"  Loaded cached params: {best_params}")
    else:
        gs_seq = build_seq_dataset(df_train_final, df_test_final, all_features, "Target_30min", SEQ_LENGTH)
        gs_weights = calculate_clinical_weights(gs_seq["y_train"])

        best_params, best_score = grid_search_dl(
            gs_seq["X_train"], gs_seq["y_train"], gs_weights,
            input_dim=len(all_features), param_grid=DL_PARAM_GRID,
        )
        grid_cache[gs_key] = best_params
        with open(grid_cache_path, "w") as f:
            json.dump(grid_cache, f, indent=2)
        print(f"  Best params: {best_params}  (val loss={best_score:.4f})")

    print(f"\n[6/9] MAIN TRAINING LOOP ({window_label})...")
    for fold_i, (df_tr, df_te) in enumerate(wf_folds):
        for subset, features in FEATURE_SETS.items():
            for h_label in HORIZONS:
                target_col = f"Target_{h_label}"
                seq = build_seq_dataset(df_tr, df_te, features, target_col, SEQ_LENGTH)
                weights = calculate_clinical_weights(seq["y_train"])

                model, val_loss, epochs_trained = train_weighted_autoencoder(
                    seq["X_train"], seq["y_train"], weights,
                    input_dim=len(features), latent_size=best_params["latent_size"], lr=best_params["lr"],
                )
                model.eval()
                with torch.no_grad():
                    preds = model(torch.tensor(seq["X_test"])).numpy().flatten()

                m = metrics_dict(seq["y_test"], preds)
                all_results.append({
                    "model": MODEL_NAME, "window": window_label, "horizon": h_label,
                    "subset": subset, "fold": fold_i, "epochs_trained": epochs_trained, **m,
                })

                key = (window_label, subset, h_label)
                pooled_predictions.setdefault(key, {"y_true": [], "y_pred": []})
                pooled_predictions[key]["y_true"].append(seq["y_test"])
                pooled_predictions[key]["y_pred"].append(preds)

                print(f"  [fold {fold_i}] {subset:10s} {h_label:6s} RMSE={m['rmse']:.2f} "
                      f"MAE={m['mae']:.2f} epochs={epochs_trained}")

    print(f"\n[7/9] ABLATION STUDY ({window_label})...")
    for combo_name, combo_features in feature_groups.items():
        if not combo_features:
            continue
        for h_label in HORIZONS:
            target_col = f"Target_{h_label}"
            seq = build_seq_dataset(df_train_final, df_test_final, combo_features, target_col, SEQ_LENGTH)
            weights = calculate_clinical_weights(seq["y_train"])

            model, _, _ = train_weighted_autoencoder(
                seq["X_train"], seq["y_train"], weights,
                input_dim=len(combo_features), latent_size=best_params["latent_size"], lr=best_params["lr"],
            )
            model.eval()
            with torch.no_grad():
                preds = model(torch.tensor(seq["X_test"])).numpy().flatten()
            rmse = float(np.sqrt(mean_squared_error(seq["y_test"], preds)))

            all_ablation_results.append({
                "model": MODEL_NAME, "window": window_label, "combo": combo_name,
                "horizon": h_label, "rmse": rmse,
                "n_train": seq["n_train"], "n_test": seq["n_test"],
            })
        print(f"  [{combo_name:20s}] " + ", ".join(
            f"{h}={r['rmse']:.2f}" for h, r in
            zip(HORIZONS, [x for x in all_ablation_results if x["combo"] == combo_name and x["window"] == window_label])
        ))

# ============================================================================
# 8. SAVE RESULTS
# ============================================================================
print(f"\n[8/9] SAVING RESULTS...")
results_df = pd.DataFrame(all_results)
results_df.to_csv(RESULTS_DIR / f"results_{MODEL_NAME}.csv", index=False)

ablation_df = pd.DataFrame(all_ablation_results)
ablation_df.to_csv(RESULTS_DIR / f"results_{MODEL_NAME}_ablation.csv", index=False)

with open(RESULTS_DIR / f"results_{MODEL_NAME}_pooled_preds.pkl", "wb") as f:
    pickle.dump(pooled_predictions, f)

print(f"  Saved results_{MODEL_NAME}.csv, results_{MODEL_NAME}_ablation.csv, "
      f"results_{MODEL_NAME}_pooled_preds.pkl")

summary = (results_df.groupby(["window", "subset", "horizon"])
           .agg(rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
                mae_mean=("mae", "mean"), mae_std=("mae", "std"),
                r2_mean=("r2", "mean"), mape_mean=("mape", "mean"),
                epochs_mean=("epochs_trained", "mean"),
                n_folds=("rmse", "count"))
           .reset_index())
summary.to_csv(RESULTS_DIR / f"summary_{MODEL_NAME}.csv", index=False)

print(f"\n{'='*70}\nWALK-FORWARD SUMMARY (mean +/- std over {N_SPLITS} folds)\n{'='*70}")
for window_label in WINDOW_SIZES:
    print(f"\n  Window: {window_label}")
    sub = summary[summary["window"] == window_label]
    for _, r in sub.iterrows():
        print(f"    [{r['subset']:10s}] {r['horizon']:6s}  "
              f"RMSE={r['rmse_mean']:.2f}+/-{r['rmse_std']:.2f}  "
              f"MAE={r['mae_mean']:.2f}+/-{r['mae_std']:.2f}  "
              f"R2={r['r2_mean']:.4f}  MAPE={r['mape_mean']:.2f}%  "
              f"epochs~{r['epochs_mean']:.0f}")

# ============================================================================
# 9. CLARKE ERROR GRID (pooled) + ABLATION BAR PLOTS (per window size)
# ============================================================================
print(f"\n[9/9] CLARKE ERROR GRID + ABLATION PLOTS...")
clarke_grid_pooled(pooled_predictions, MODEL_NAME, RESULTS_DIR, N_SPLITS)

for window_label in WINDOW_SIZES:
    sub = ablation_df[ablation_df["window"] == window_label]
    if sub.empty:
        continue
    pivot = sub.pivot(index="combo", columns="horizon", values="rmse")
    create_bar_plot(
        pivot.to_dict("index"),
        f"Ablation Study — {MODEL_NAME.upper()} — {window_label} — RMSE per Feature Group",
        RESULTS_DIR / f"ablation_{MODEL_NAME}_{window_label}.png",
    )

print("\nSeq2Seq Autoencoder pipeline complete.")