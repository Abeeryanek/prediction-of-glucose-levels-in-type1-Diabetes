"""
lstm.py — LSTM, Window Size x Horizon Walk-Forward + LOPO Pipeline
====================================================================
Self-contained: no external config/utils modules. Everything in one file.
Identical structure to cnn_lstm.py — only the model architecture differs
(plain LSTM here, no Conv1d front-end).

Global DL training standards applied here:
    - max_epochs   = 150
    - patience     = 15    (early stopping, monitor=val_loss, restore_best_weights)
    - batch_size   = 32
    - val_ratio    = 0.20  (last 20% chronological)
    - shuffle      = False (temporal order preserved)
    - seed         = 42    (torch.manual_seed + np.random.seed)

Key design points:

1. LAG REMOVED FROM MAIN FEATURE SETS. 'full' and 'comparable' no longer include
   any glucose lag/rate-of-change features. Lag exists ONLY as its own ablation
   study group ('lag'), studied in isolation — never baked into the main models.

2. ALL LOOKBACK-DEPENDENT FEATURES ARE ANCHORED TO THE CURRENT window_size.
   Physiological variability (HR/EDA/BVP/IBI/SkinTemp std), activity (Acc_Vmu
   mean/max), AND the ablation-only glucose lag/roc are all computed using THE
   SAME window_size as their single rolling window, anchored at 5-min
   resolution.

3. ABLATION GROUPS ARE ANCHORED TO A GLUCOSE BASELINE (FIX). Every ablation
   group except 'lag' now includes the current "Glucose" reading alongside
   the signal being tested.

4. LOPO (Leave-One-Patient-Out), nested inside the window_size loop, reusing
   that window's cached grid-search-best hidden_dim/lr. 

5. STRICT TIME-BASED SHIFTING. Rolling windows, lag extraction, target horizon 
   generation, and sequence building are all anchored strictly to timestamp math 
   rather than row-counts to prevent cross-gap contamination.
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

MODEL_NAME = "lstm"

WINDOW_SIZES = {"1h": 12, "2h": 24, "3h": 36, "6h": 72}   # lookback, in 5-min steps
HORIZONS = {"15min": 3, "30min": 6, "45min": 9}            # prediction horizon, in 5-min steps
N_SPLITS = 5                                                # walk-forward folds per patient

# ---- Global DL training standards ----
MAX_EPOCHS = 150
PATIENCE = 15
BATCH_SIZE = 32
VAL_RATIO = 0.20      # last 20% chronological
SHUFFLE = False       # temporal order preserved

DL_PARAM_GRID = {
    "hidden_dim": [32, 64, 128],
    "lr": [1e-3, 5e-4],
}
GRID_SEARCH_EPOCHS = 30      # reduced-epoch grid search (documented shortcut, see plan §2.6)
GRID_SEARCH_PATIENCE = 5

# LOPO minimum row/sequence thresholds (skip patients with too little data)
LOPO_MIN_TRAIN_ROWS = 200
LOPO_MIN_TEST_ROWS = 50

print(f"\n{'='*70}\nLSTM — WINDOW x HORIZON WALK-FORWARD + LOPO PIPELINE\n{'='*70}")

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1/10] LOADING DATA...")

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
# 2. BASE FEATURE ENGINEERING
# ============================================================================
print("\n[2/10] BASE FEATURE ENGINEERING (temporal, sensor interpolation, food)...")

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

# ============================================================================
# 3. WALK-FORWARD FOLD BUILDER
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
            
        # ---> NEW: Strict timestamp validation to reject gap-bridging sequences
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
# 6. MODEL DEFINITION — LSTM
# ============================================================================
class GlucoseDataset(Dataset):
    def __init__(self, x, y, weights):
        self.x = x
        self.y = y
        self.w = weights

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.w[idx]


class GlucoseLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


# ============================================================================
# 7. TRAINING LOOP
# ============================================================================
def train_weighted_lstm(x_train, y_train, sample_weights, input_dim,
                         epochs=MAX_EPOCHS, patience=PATIENCE, val_ratio=VAL_RATIO,
                         **model_kwargs):
    split = int(len(x_train) * (1 - val_ratio))
    xt = torch.tensor(x_train[:split])
    yt = torch.tensor(y_train[:split]).unsqueeze(1)
    wt = torch.tensor(sample_weights[:split]).unsqueeze(1)

    xv = torch.tensor(x_train[split:])
    yv = torch.tensor(y_train[split:]).unsqueeze(1)
    wv = torch.tensor(sample_weights[split:]).unsqueeze(1)

    model = GlucoseLSTM(input_dim=input_dim, **model_kwargs)
    criterion = nn.MSELoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=model_kwargs.get("lr", 1e-3))

    train_loader = DataLoader(GlucoseDataset(xt, yt, wt), batch_size=BATCH_SIZE, shuffle=SHUFFLE)

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
        _, val_loss, _ = train_weighted_lstm(
            x_train, y_train, sample_weights, input_dim=input_dim,
            epochs=GRID_SEARCH_EPOCHS, patience=GRID_SEARCH_PATIENCE,
            hidden_dim=params["hidden_dim"], lr=params["lr"],
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
# 10. LOPO (Leave-One-Patient-Out)
# ============================================================================
def run_lopo_dl(df_w, features, target_col, seq_length, best_params, patient_col="Patient_ID"):
    patients = df_w[patient_col].unique()
    lopo_results = []
    pooled_true, pooled_pred = [], []

    for test_pid in patients:
        train_df = df_w.loc[df_w[patient_col] != test_pid].dropna(subset=[target_col] + features)
        test_df = df_w.loc[df_w[patient_col] == test_pid].dropna(subset=[target_col] + features)

        if len(train_df) < LOPO_MIN_TRAIN_ROWS or len(test_df) < LOPO_MIN_TEST_ROWS:
            continue

        scaler = StandardScaler().fit(train_df[features])
        X_train_2d = scaler.transform(train_df[features])
        X_test_2d = scaler.transform(test_df[features])

        X_train_3d, y_train = create_3d_sequences(
            X_train_2d, train_df[target_col].reset_index(drop=True),
            train_df[patient_col].reset_index(drop=True), train_df["Timestamp"].reset_index(drop=True), seq_length
        )
        X_test_3d, y_test = create_3d_sequences(
            X_test_2d, test_df[target_col].reset_index(drop=True),
            test_df[patient_col].reset_index(drop=True), test_df["Timestamp"].reset_index(drop=True), seq_length
        )

        if len(X_train_3d) < 10 or len(X_test_3d) < 5:
            continue

        weights = calculate_clinical_weights(y_train)
        model, _, epochs_trained = train_weighted_lstm(
            X_train_3d, y_train, weights, input_dim=len(features),
            hidden_dim=best_params["hidden_dim"], lr=best_params["lr"],
        )
        model.eval()
        with torch.no_grad():
            preds = model(torch.tensor(X_test_3d)).numpy().flatten()

        m = metrics_dict(y_test, preds)
        m["patient"] = test_pid
        m["n_train"] = len(X_train_3d)
        m["n_test"] = len(X_test_3d)
        m["epochs_trained"] = epochs_trained
        lopo_results.append(m)

        pooled_true.append(y_test)
        pooled_pred.append(preds)

    return lopo_results, pooled_true, pooled_pred


# ============================================================================
# 11. CLARKE ERROR GRID (pooled across folds)
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


# ============================================================================
# 12. ABLATION BAR PLOT
# ============================================================================
def create_bar_plot(ablation_results, title, save_path):
    df_results = pd.DataFrame(ablation_results).T
    horizons = list(df_results.columns)
    n_groups = len(df_results)

    fig_width = 12
    fig_height = max(6, n_groups * 0.55) * len(horizons)

    fig, axes = plt.subplots(len(horizons), 1, figsize=(fig_width, fig_height), layout="constrained")
    if len(horizons) == 1:
        axes = [axes]

    colors = plt.cm.tab20(np.linspace(0, 1, n_groups))

    for ax, horizon in zip(axes, horizons):
        values = df_results[horizon].dropna().sort_values()
        y_pos = np.arange(len(values))

        bars = ax.barh(y_pos, values.values, height=0.65, edgecolor="black", linewidth=0.6, color=colors[:len(values)])

        max_val = values.values.max() if len(values) else 1.0
        for bar, val in zip(bars, values.values):
            ax.text(bar.get_width() + max_val * 0.015, bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}", ha="left", va="center", fontsize=10, fontweight="bold")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(values.index, fontsize=10)
        ax.set_xlabel("RMSE (mg/dL)", fontweight="bold", fontsize=12)
        ax.set_title(f"Horizon: {horizon}", fontweight="bold", fontsize=14)
        ax.set_xlim(0, max_val * 1.20)
        ax.grid(axis="x", alpha=0.3, linewidth=0.8)
        ax.invert_yaxis()

    fig.suptitle(title, fontsize=16, fontweight="bold")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# MAIN PIPELINE
# ============================================================================
grid_cache_path = RESULTS_DIR / f"grid_search_{MODEL_NAME}.json"
grid_cache = json.load(open(grid_cache_path)) if grid_cache_path.exists() else {}

all_results = []
all_ablation_results = []
all_lopo_results = []
pooled_predictions = {}
pooled_lopo_predictions = {}

# ============================================================================
# 13. WINDOW SIZE LOOP — STRICT TIME-BASED MATH APPLIED HERE
# ============================================================================
for window_label, window_size in WINDOW_SIZES.items():
    print(f"\n{'#'*70}\nWINDOW SIZE: {window_label} ({window_size} steps)\n{'#'*70}")

    df_w = df.copy()
    SEQ_LENGTH = window_size

    # Temporarily set index for time-based rolling window calculation
    temp_time_df = df_w.set_index("Timestamp")

    # ------------------------------------------------------------------------
    # PHYSIOLOGICAL VARIABILITY — Strict Time-Based Rolling
    # ------------------------------------------------------------------------
    print(f"\n[3/10] PHYSIOLOGICAL VARIABILITY anchored to {window_label} time...")
    physio_features_all, physio_hr, physio_eda, physio_bvp, physio_ibi, physio_skin = ([] for _ in range(6))

    def _add_std_anchored(col_src, target_list):
        if col_src in df_w.columns:
            col = f"{col_src.lower()}_std_{window_label}"
            rolled = temp_time_df.groupby("Patient_ID")[col_src].rolling(window_label).std()
            df_w[col] = rolled.values
            physio_features_all.append(col)
            target_list.append(col)

    _add_std_anchored("Heart_Rate", physio_hr)
    _add_std_anchored("EDA", physio_eda)
    _add_std_anchored("BVP", physio_bvp)
    _add_std_anchored("IBI", physio_ibi)
    _add_std_anchored("Skin_Temp", physio_skin)
    print(f"  Physio features (anchored to {window_label}): {len(physio_features_all)}")

    # ------------------------------------------------------------------------
    # ACTIVITY — Strict Time-Based Rolling
    # ------------------------------------------------------------------------
    print(f"\n[4/10] ACTIVITY FEATURES anchored to {window_label} time...")
    activity_features = []
    if "Acc_Vmu" in df_w.columns:
        col_mean = f"acc_vmu_mean_{window_label}"
        col_max = f"acc_vmu_max_{window_label}"
        df_w[col_mean] = temp_time_df.groupby("Patient_ID")["Acc_Vmu"].rolling(window_label).mean().values
        df_w[col_max] = temp_time_df.groupby("Patient_ID")["Acc_Vmu"].rolling(window_label).max().values
        activity_features = [col_mean, col_max]
    print(f"  Activity features (anchored to {window_label}): {len(activity_features)}")

    # ---> NEW: Lookup structure mapping Exact Timestamp + Patient to a reading
    lookup = df_w.set_index(["Patient_ID", "Timestamp"])["Glucose"]

    # ------------------------------------------------------------------------
    # GLUCOSE LAG — Exact Timestamp Shift (Not row shift)
    # ------------------------------------------------------------------------
    print(f"\n[5/10] GLUCOSE LAG (ablation-only) anchored to exact {window_label} shift...")
    lag_col = f"glucose_lag_{window_label}"
    roc_col = f"glucose_roc_{window_label}"
    
    # Subtract exactly the window time (e.g. 1 hour) from the current timestamp
    past_times = df_w["Timestamp"] - pd.Timedelta(window_label)
    df_w[lag_col] = pd.MultiIndex.from_arrays([df_w["Patient_ID"], past_times]).map(lookup)
    df_w[roc_col] = df_w["Glucose"] - df_w[lag_col]
    lag_features = [lag_col, roc_col]
    print(f"  Lag features (ablation-only): {lag_features}")

    # ------------------------------------------------------------------------
    # TARGET EXTRACTION — Exact Timestamp Shift (Not row shift)
    # ------------------------------------------------------------------------
    for h_label in HORIZONS.keys():
        future_times = df_w["Timestamp"] + pd.Timedelta(h_label.replace("min", "m"))
        df_w[f"Target_{h_label}"] = pd.MultiIndex.from_arrays([df_w["Patient_ID"], future_times]).map(lookup)

    # ------------------------------------------------------------------------
    # MAIN FEATURE SETS
    # ------------------------------------------------------------------------
    all_features = (["Glucose"] + temporal_features + sensor_cols + food_features_all + activity_features + physio_features_all)
    all_features = [f for f in all_features if f in df_w.columns]

    all_features_comparable = (["Glucose"] + temporal_features + sensor_cols + food_features_carbs + activity_features + physio_hr)
    all_features_comparable = [f for f in all_features_comparable if f in df_w.columns]

    # ------------------------------------------------------------------------
    # ABLATION GROUPS
    # ------------------------------------------------------------------------
    feature_groups = {
        "glucose": ["Glucose"],
        "physio_only": ["Glucose"] + physio_features_all,
        "EDA": ["Glucose"] + physio_eda,
        "heart_rate": ["Glucose"] + physio_hr,
        "BVP": ["Glucose"] + physio_bvp,
        "IBI": ["Glucose"] + physio_ibi,
        "food_only": ["Glucose"] + food_features_all,
        "carbs_only": ["Glucose"] + food_features_carbs,
        "activity": ["Glucose"] + activity_features,
        "comparable": all_features_comparable, 
        "all": all_features,
        "physio_food": ["Glucose"] + physio_features_all + food_features_all,
        "food_movement": ["Glucose"] + food_features_all + activity_features,
        "physio_movement": ["Glucose"] + physio_features_all + activity_features,
        "lag": lag_features,
    }
    feature_groups = {name: [f for f in feats if f in df_w.columns] for name, feats in feature_groups.items()}

    FEATURE_SETS = {"full": all_features, "comparable": all_features_comparable}

    print(f"\n[6/10] BUILDING WALK-FORWARD FOLDS ({window_label})...")
    wf_folds = build_walk_forward_folds(df_w, N_SPLITS)
    df_train_final, df_test_final = wf_folds[-1]
    for i, (tr, te) in enumerate(wf_folds):
        print(f"  Fold {i}: train={len(tr):,}  test={len(te):,}")

    print(f"\n[7/10] GRID SEARCH ({window_label})...")
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

    print(f"\n[8/10] MAIN TRAINING LOOP ({window_label})...")
    for fold_i, (df_tr, df_te) in enumerate(wf_folds):
        for subset, features in FEATURE_SETS.items():
            for h_label in HORIZONS:
                target_col = f"Target_{h_label}"
                seq = build_seq_dataset(df_tr, df_te, features, target_col, SEQ_LENGTH)
                weights = calculate_clinical_weights(seq["y_train"])

                model, val_loss, epochs_trained = train_weighted_lstm(
                    seq["X_train"], seq["y_train"], weights,
                    input_dim=len(features), hidden_dim=best_params["hidden_dim"], lr=best_params["lr"],
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

    print(f"\n[9/10] ABLATION STUDY ({window_label})...")
    for combo_name, combo_features in feature_groups.items():
        if not combo_features:
            continue
        for h_label in HORIZONS:
            target_col = f"Target_{h_label}"
            seq = build_seq_dataset(df_train_final, df_test_final, combo_features, target_col, SEQ_LENGTH)
            weights = calculate_clinical_weights(seq["y_train"])

            model, _, _ = train_weighted_lstm(
                seq["X_train"], seq["y_train"], weights,
                input_dim=len(combo_features), hidden_dim=best_params["hidden_dim"], lr=best_params["lr"],
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

    print(f"\n[LOPO] LEAVE-ONE-PATIENT-OUT ({window_label})...")
    for subset, features in FEATURE_SETS.items():
        for h_label in HORIZONS:
            target_col = f"Target_{h_label}"
            lopo_results, pooled_true, pooled_pred = run_lopo_dl(
                df_w, features, target_col, SEQ_LENGTH, best_params,
            )
            for r in lopo_results:
                r.update({"model": MODEL_NAME, "window": window_label, "subset": subset, "horizon": h_label})
            all_lopo_results.extend(lopo_results)

            if pooled_true:
                key = (window_label, subset, h_label)
                pooled_lopo_predictions.setdefault(key, {"y_true": [], "y_pred": []})
                pooled_lopo_predictions[key]["y_true"].extend(pooled_true)
                pooled_lopo_predictions[key]["y_pred"].extend(pooled_pred)

            n_patients_used = len(lopo_results)
            mean_rmse = np.mean([r["rmse"] for r in lopo_results]) if lopo_results else float("nan")
            print(f"  [{subset:10s}] {h_label:6s} LOPO RMSE (mean over {n_patients_used} patients) = {mean_rmse:.2f}")

# ============================================================================
# 14. SAVE RESULTS
# ============================================================================
print(f"\n[SAVE] SAVING RESULTS...")
results_df = pd.DataFrame(all_results)
results_df.to_csv(RESULTS_DIR / f"results_{MODEL_NAME}.csv", index=False)

ablation_df = pd.DataFrame(all_ablation_results)
ablation_df.to_csv(RESULTS_DIR / f"results_{MODEL_NAME}_ablation.csv", index=False)

lopo_df = pd.DataFrame(all_lopo_results)
lopo_df.to_csv(RESULTS_DIR / f"results_{MODEL_NAME}_lopo.csv", index=False)

with open(RESULTS_DIR / f"results_{MODEL_NAME}_pooled_preds.pkl", "wb") as f:
    pickle.dump(pooled_predictions, f)

print(f"  Saved results_{MODEL_NAME}.csv, results_{MODEL_NAME}_ablation.csv, "
      f"results_{MODEL_NAME}_lopo.csv, results_{MODEL_NAME}_pooled_preds.pkl")

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

lopo_summary = (lopo_df.groupby(["window", "subset", "horizon"])
                .agg(rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
                     mae_mean=("mae", "mean"), mae_std=("mae", "std"),
                     epochs_mean=("epochs_trained", "mean"),
                     n_patients=("rmse", "count"))
                .reset_index())
lopo_summary.to_csv(RESULTS_DIR / f"summary_{MODEL_NAME}_lopo.csv", index=False)

print(f"\n{'='*70}\nLOPO SUMMARY (mean +/- std across held-out patients)\n{'='*70}")
for window_label in WINDOW_SIZES:
    print(f"\n  Window: {window_label}")
    sub = lopo_summary[lopo_summary["window"] == window_label]
    for _, r in sub.iterrows():
        print(f"    [{r['subset']:10s}] {r['horizon']:6s}  "
              f"RMSE={r['rmse_mean']:.2f}+/-{r['rmse_std']:.2f}  "
              f"MAE={r['mae_mean']:.2f}+/-{r['mae_std']:.2f}  "
              f"epochs~{r['epochs_mean']:.0f}  n_patients={r['n_patients']:.0f}")

print(f"\n{'='*70}\nPERSONALISED (WALK-FORWARD) vs. LOPO — how much does personalisation matter?\n{'='*70}")
for window_label in WINDOW_SIZES:
    for subset in FEATURE_SETS:
        for h_label in HORIZONS:
            pers_row = summary[(summary["window"] == window_label) &
                                (summary["subset"] == subset) &
                                (summary["horizon"] == h_label)]
            lopo_row = lopo_summary[(lopo_summary["window"] == window_label) &
                                     (lopo_summary["subset"] == subset) &
                                     (lopo_summary["horizon"] == h_label)]
            if pers_row.empty or lopo_row.empty:
                continue
            pers_rmse = pers_row["rmse_mean"].values[0]
            lopo_rmse = lopo_row["rmse_mean"].values[0]
            diff = lopo_rmse - pers_rmse
            print(f"  [{window_label}][{subset:10s}][{h_label:6s}]  "
                  f"Personalised={pers_rmse:.2f}  LOPO={lopo_rmse:.2f}  Diff={diff:+.2f} mg/dL")

# ============================================================================
# 15. CLARKE ERROR GRID (pooled, walk-forward AND LOPO) + ABLATION BAR PLOTS
# ============================================================================
print(f"\n[PLOTS] CLARKE ERROR GRID + ABLATION PLOTS...")
clarke_grid_pooled(pooled_predictions, MODEL_NAME, RESULTS_DIR, N_SPLITS)
clarke_grid_pooled(pooled_lopo_predictions, f"{MODEL_NAME}_lopo", RESULTS_DIR, n_splits=1)

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

print("\nLSTM pipeline complete (walk-forward + ablation + LOPO).")