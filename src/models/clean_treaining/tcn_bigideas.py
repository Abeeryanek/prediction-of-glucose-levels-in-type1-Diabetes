"""
tcn.py — Temporal Convolutional Network, Window Size x Horizon Walk-Forward Pipeline
===================================================================================
Aligned to FINAL_EXPERIMENTS_PLAN.md:

- Clinically-weighted MSELoss (hypo/hyper sample weights)
- Grid: num_filters [32,64,128] x lr [1e-3, 5e-4] (§2.4)
- Window sizes: 1h/2h/3h/6h (12/24/36/72 steps) — grid sweep
- Per-patient row-count check before large windows
- LOPO removed (§10 item 3)
- Food columns: exact plain names, linear interpolation (§0A)
- max_epochs=150, patience=15, batch_size=32, shuffle=False,
  val_ratio=0.20, seed=42
- TCN: kernel_size=3, dilations=[1,2,4,8], dilated causal convolutions
- Raw sensor values removed from feature sets (redundant with rolling std)
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
from torch.nn.utils import weight_norm
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import clarke_error_grid as ceg

warnings.filterwarnings("ignore")

# ============================================================================
# CONSTANTS
# ============================================================================
DATA_PATH   = "data/bigideas"
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

MODEL_NAME = "tcn"
# "2h": 24, "3h": 36, "6h": 72
WINDOW_SIZES = {"1h": 12 }
HORIZONS     = {"15min": 3, "30min": 6, "45min": 9}
N_SPLITS     = 5

MAX_EPOCHS = 150
PATIENCE   = 15
BATCH_SIZE = 32
VAL_RATIO  = 0.20
SHUFFLE    = False

GRID_SEARCH_EPOCHS   = 30
GRID_SEARCH_PATIENCE = 5

# TCN architecture fixed params (per plan §2.4)
KERNEL_SIZE = 3
DILATIONS = [1, 2, 4, 8]

DL_PARAM_GRID = {"num_filters": [32, 64, 128], "lr": [1e-3, 5e-4]}

FOOD_COLS = ["calorie", "total_carb", "dietary_fiber", "sugar", "protein", "total_fat"]

print(f"\n{'='*70}\nTCN — WINDOW x HORIZON WALK-FORWARD PIPELINE\n{'='*70}")

# ============================================================================
# 1. LOAD DATA - FIXED for \N values and Patient_ID preservation
# ============================================================================
print("\n[1/10] LOADING DATA...")

file_list = glob.glob(os.path.join(DATA_PATH, "clean_patient_*.parquet"))
if not file_list:
    file_list = glob.glob("bent_*.parquet")
if not file_list:
    raise ValueError("No parquet files found!")

dfs = []
for file_path in file_list:
    temp_df = pd.read_parquet(file_path)
    
    # Ensure Patient_ID stays as string (your data has "001", "002", etc.)
    if "Patient_ID" in temp_df.columns:
        temp_df["Patient_ID"] = temp_df["Patient_ID"].astype(str).str.zfill(3)
    
    # CRITICAL: Handle \N in food columns - convert to actual NaN
    for col in FOOD_COLS:
        if col in temp_df.columns:
            temp_df[col] = temp_df[col].replace('\\N', np.nan)
            temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce')
    
    print(f"  Loaded: {os.path.basename(file_path)} -> {len(temp_df):,} rows, Patient: {temp_df['Patient_ID'].iloc[0]}")
    dfs.append(temp_df)

df = pd.concat(dfs, ignore_index=True)
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df = df.sort_values(["Patient_ID", "Timestamp"]).reset_index(drop=True)

# Validate Patient_ID loading
print(f"\n DATA VALIDATION:")
print(f"  Total rows: {len(df):,}")
print(f"  Unique patients: {df['Patient_ID'].nunique()}")
print(f"  Patient IDs: {sorted(df['Patient_ID'].unique())}")

if df['Patient_ID'].nunique() < 2:
    raise ValueError(f" CRITICAL: Only {df['Patient_ID'].nunique()} unique patient(s)!")

print(f" Successfully loaded {len(df):,} rows from {df['Patient_ID'].nunique()} patients")

# ============================================================================
# 2. BASE FEATURE ENGINEERING
# ============================================================================
print("\n[2/10] BASE FEATURE ENGINEERING...")

df["Hour"]            = df["Timestamp"].dt.hour
df["Minute"]          = df["Timestamp"].dt.minute
df["DayOfWeek"]       = df["Timestamp"].dt.dayofweek
df["MinFromMidnight"] = df["Hour"] * 60 + df["Minute"]
df["Hour_sin"]        = np.sin(2 * np.pi * df["Hour"] / 24)
df["Hour_cos"]        = np.cos(2 * np.pi * df["Hour"] / 24)
temporal_features = ["Hour", "Minute", "DayOfWeek", "MinFromMidnight",
                     "Hour_sin", "Hour_cos"]

# Sensor columns for physiological variability computation only
# (raw values not included in main features - redundant with rolling std)
sensor_cols = [c for c in ["Heart_Rate", "Acc_Vmu", "EDA", "Skin_Temp", "BVP", "IBI"]
               if c in df.columns]

# Interpolate sensors for rolling std calculation (limit=6 steps = 30 min)
for col in sensor_cols:
    df[col] = (
        df.groupby("Patient_ID")[col]
          .transform(lambda x: x.interpolate(method="linear", limit=6))
    )
print(f"  Sensors interpolated for physio features: {sensor_cols}")

food_features_all   = [c for c in df.columns if c in FOOD_COLS]
food_features_carbs = [c for c in df.columns if c == "total_carb"]

for col in food_features_all:
    df[col] = (
        df.groupby("Patient_ID")[col]
          .transform(lambda x: x.interpolate(method="linear"))
    )
print(f"  Food interpolated (no limit): {food_features_all}")
print(f"  Food carbs only: {food_features_carbs}")


# ============================================================================
# 3. WALK-FORWARD FOLD BUILDER
# ============================================================================
def build_walk_forward_folds(df_in, n_splits):
    fold_train_idx = [[] for _ in range(n_splits)]
    fold_test_idx  = [[] for _ in range(n_splits)]

    for pid, group in df_in.groupby("Patient_ID"):
        group = group.sort_values("Timestamp")
        idx   = group.index.values
        if len(idx) < n_splits + 1:
            continue
        tscv = TimeSeriesSplit(n_splits=n_splits)
        for fold_i, (tr, te) in enumerate(tscv.split(idx)):
            fold_train_idx[fold_i].extend(idx[tr])
            fold_test_idx[fold_i].extend(idx[te])

    folds = []
    for fold_i in range(n_splits):
        df_tr = (df_in.loc[fold_train_idx[fold_i]]
                      .sort_values(["Patient_ID", "Timestamp"])
                      .reset_index(drop=True))
        df_te = (df_in.loc[fold_test_idx[fold_i]]
                      .sort_values(["Patient_ID", "Timestamp"])
                      .reset_index(drop=True))
        folds.append((df_tr, df_te))
    return folds


# ============================================================================
# 4. CLINICAL SAMPLE WEIGHTING
# ============================================================================
def calculate_clinical_weights(y_true):
    weights = np.ones(len(y_true), dtype=np.float32)
    weights[y_true < 54]                      = 3.0
    weights[(y_true >= 54) & (y_true < 70)]   = 2.5
    weights[(y_true > 180) & (y_true <= 250)] = 1.5
    weights[y_true > 250]                     = 2.0
    return weights


# ============================================================================
# 5. 3D SEQUENCE BUILDER (Time-Locked)
# ============================================================================
def create_3d_sequences(X_scaled, y_series, patient_ids, timestamps, seq_length):
    patient_ids      = np.asarray(patient_ids)
    timestamps       = np.asarray(timestamps, dtype="datetime64[ns]")
    expected_minutes = (seq_length - 1) * 5

    Xs, ys = [], []
    for i in range(seq_length - 1, len(X_scaled)):
        if patient_ids[i - seq_length + 1] != patient_ids[i]:
            continue
        span_minutes = (
            (timestamps[i] - timestamps[i - seq_length + 1])
            / np.timedelta64(1, "m")
        )
        if abs(span_minutes - expected_minutes) > 0.01:
            continue
        Xs.append(X_scaled[i - seq_length + 1: i + 1])
        ys.append(y_series.iloc[i])

    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def build_seq_dataset(df_tr, df_te, features, target_col, seq_length):
    train_clean = df_tr.dropna(subset=[target_col] + features).reset_index(drop=True)
    test_clean  = df_te.dropna(subset=[target_col] + features).reset_index(drop=True)

    scaler     = StandardScaler()
    X_train_2d = scaler.fit_transform(train_clean[features])
    X_test_2d  = scaler.transform(test_clean[features])

    X_train_3d, y_train_raw = create_3d_sequences(
        X_train_2d, train_clean[target_col],
        train_clean["Patient_ID"], train_clean["Timestamp"], seq_length,
    )
    X_test_3d, y_test_raw = create_3d_sequences(
        X_test_2d, test_clean[target_col],
        test_clean["Patient_ID"], test_clean["Timestamp"], seq_length,
    )
    y_scaler = StandardScaler()
    y_train = (y_scaler.fit_transform(y_train_raw.reshape(-1, 1)).flatten().astype(np.float32)
               if len(y_train_raw) else y_train_raw)
    y_test  = (y_scaler.transform(y_test_raw.reshape(-1, 1)).flatten().astype(np.float32)
               if len(y_test_raw) else y_test_raw)
    return {
        "X_train": X_train_3d, "y_train": y_train, "y_train_raw": y_train_raw,
        "X_test":  X_test_3d,  "y_test":  y_test, "y_test_raw":  y_test_raw,
        "y_scaler": y_scaler,
        "n_train": len(train_clean), "n_test": len(test_clean),
    }


# ============================================================================
# 6. MODEL DEFINITION — Temporal Convolutional Network
# ============================================================================
class Chomp1d(nn.Module):
    """Removes the extra right-padding added for causal convolution."""
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous() if self.chomp_size > 0 else x


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, dropout=0.3):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class GlucoseDataset(Dataset):
    def __init__(self, x, y, weights):
        self.x = x
        self.y = y
        self.w = weights

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.w[idx]


class GlucoseTCN(nn.Module):
    def __init__(self, input_dim, num_filters=32, kernel_size=KERNEL_SIZE, 
                 dilations=None, dropout=0.3):
        super().__init__()
        dilations = dilations or DILATIONS
        layers = []
        in_channels = input_dim
        for d in dilations:
            layers.append(TemporalBlock(in_channels, num_filters, kernel_size, 
                                      dilation=d, dropout=dropout))
            in_channels = num_filters
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(num_filters, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = self.tcn(x)
        return self.fc(out[:, :, -1])


# ============================================================================
# 7. TRAINING LOOP — clinically weighted MSE
# ============================================================================
def train_tcn(
    x_train, y_train, sample_weights, input_dim, lr,
    epochs=MAX_EPOCHS, patience=PATIENCE, val_ratio=VAL_RATIO,
    **model_kwargs,
):
    split = int(len(x_train) * (1 - val_ratio))

    xt = torch.tensor(x_train[:split])
    yt = torch.tensor(y_train[:split]).unsqueeze(1)
    wt = torch.tensor(sample_weights[:split]).unsqueeze(1)

    xv = torch.tensor(x_train[split:])
    yv = torch.tensor(y_train[split:]).unsqueeze(1)
    wv = torch.tensor(sample_weights[split:]).unsqueeze(1)

    model     = GlucoseTCN(input_dim=input_dim, **model_kwargs)
    criterion = nn.MSELoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(
        GlucoseDataset(xt, yt, wt),
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE,
    )

    best_val_loss    = float("inf")
    best_weights     = None
    patience_counter = 0
    epochs_trained   = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb, wb in train_loader:
            optimizer.zero_grad()
            loss = (criterion(model(xb), yb) * wb).mean()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = (criterion(model(xv), yv) * wv).mean().item()

        epochs_trained = epoch + 1

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            best_weights     = {k: v.clone() for k, v in model.state_dict().items()}
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
    keys   = list(param_grid.keys())
    combos = list(product(*[param_grid[k] for k in keys]))

    best_score, best_params = float("inf"), {}
    for combo in combos:
        params = dict(zip(keys, combo))
        _, val_loss, _ = train_tcn(
            x_train, y_train, sample_weights, input_dim=input_dim,
            epochs=GRID_SEARCH_EPOCHS, patience=GRID_SEARCH_PATIENCE,
            num_filters=params["num_filters"], lr=params["lr"],
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
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "r2":   float(r2_score(y_true, y_pred)),
        "mape": float(
            np.mean(np.abs((np.asarray(y_true) - y_pred) / np.asarray(y_true))) * 100
        ),
    }


# ============================================================================
# 10. CLARKE ERROR GRID
# ============================================================================
def clarke_grid_pooled(pooled_predictions, model_name, out_dir, n_splits):
    for (window_label, subset, horizon), bucket in pooled_predictions.items():
        y_true = np.concatenate(bucket["y_true"])
        y_pred = np.concatenate(bucket["y_pred"])

        print(f"--- {window_label} {horizon.upper()} {model_name} "
              f"{subset.upper()} ZONES (pooled, {n_splits} folds) ---")
        zones = ceg.zone(y_true, y_pred)
        print(zones)

        fig = ceg.plot(y_true, y_pred)
        fig.update_layout(
            title_text=(f"Clarke Error Grid — {model_name} {subset} — "
                        f"{window_label} {horizon.upper()} (pooled)")
        )
        fig.write_html(
            str(out_dir / f"clarke_{model_name}_{subset}_{window_label}"
                          f"_{horizon}_pooled.html")
        )


# ============================================================================
# 11. ABLATION BAR PLOT
# ============================================================================
def create_bar_plot(ablation_results, title, save_path):
    df_results = pd.DataFrame(ablation_results).T
    horizons   = list(df_results.columns)
    n_groups   = len(df_results)

    fig_height = max(6, n_groups * 0.55) * len(horizons)
    fig, axes  = plt.subplots(len(horizons), 1,
                               figsize=(12, fig_height), layout="constrained")
    if len(horizons) == 1:
        axes = [axes]

    for ax, horizon in zip(axes, horizons):
        values = df_results[horizon].dropna().sort_values()
        y_pos  = np.arange(len(values))
        bars   = ax.barh(y_pos, values.values, height=0.65,
                         edgecolor="black", linewidth=0.6, color="purple")

        max_val = values.values.max() if len(values) else 1.0
        for bar, val in zip(bars, values.values):
            ax.text(bar.get_width() + max_val * 0.015,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}", ha="left", va="center",
                    fontsize=10, fontweight="bold")

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

all_results, all_ablation_results = [], []
pooled_predictions = {}

# ============================================================================
# 12. WINDOW SIZE LOOP
# ============================================================================
for window_label, window_size in WINDOW_SIZES.items():
    print(f"\n{'#'*70}\nWINDOW SIZE: {window_label} ({window_size} steps)\n{'#'*70}")

    df_w       = df.copy()
    SEQ_LENGTH = window_size

    # per-patient row-count check
    min_rows_needed   = SEQ_LENGTH + N_SPLITS + 1
    excluded_patients = []
    for pid, grp in df_w.groupby("Patient_ID"):
        if len(grp) < min_rows_needed:
            excluded_patients.append((pid, len(grp)))
    if excluded_patients:
        print(f" [{window_label}] {len(excluded_patients)} patient(s) below "
              f"{min_rows_needed} rows:")
        for pid, n in excluded_patients:
            print(f"      Patient {pid}: {n} rows")
    else:
        print(f" [{window_label}] All patients have ≥ {min_rows_needed} rows.")

    temp_time_df = df_w.set_index("Timestamp")

    # -------------------------------------------------------------------------
    # PHYSIOLOGICAL VARIABILITY — merge-based alignment
    # -------------------------------------------------------------------------
    print(f"\n[3/10] PHYSIOLOGICAL VARIABILITY anchored to {window_label}...")
    physio_features_all                                   = []
    physio_hr, physio_eda, physio_bvp, physio_ibi, physio_skin = [], [], [], [], []

    def _add_std_anchored(col_src: str, target_list: list) -> None:
        if col_src not in df_w.columns:
            return
        col = f"{col_src.lower()}_std_{window_label}"

        rolled = (
            temp_time_df
            .groupby("Patient_ID")[col_src]
            .rolling(window_label)
            .std()
            .reset_index()
            .rename(columns={col_src: col})
        )
        df_w[col] = (
            df_w[["Patient_ID", "Timestamp"]]
            .merge(rolled, on=["Patient_ID", "Timestamp"], how="left")[col]
            .values
        )
        physio_features_all.append(col)
        target_list.append(col)

    _add_std_anchored("Heart_Rate", physio_hr)
    _add_std_anchored("EDA",        physio_eda)
    _add_std_anchored("BVP",        physio_bvp)
    _add_std_anchored("IBI",        physio_ibi)
    _add_std_anchored("Skin_Temp",  physio_skin)

    # -------------------------------------------------------------------------
    # ACTIVITY — merge-based alignment
    # -------------------------------------------------------------------------
    print(f"\n[4/10] ACTIVITY FEATURES anchored to {window_label}...")
    activity_features = []
    if "Acc_Vmu" in df_w.columns:
        col_mean = f"acc_vmu_mean_{window_label}"
        col_max  = f"acc_vmu_max_{window_label}"

        for col, agg_fn in [(col_mean, "mean"), (col_max, "max")]:
            rolled = (
                temp_time_df
                .groupby("Patient_ID")["Acc_Vmu"]
                .rolling(window_label)
                .agg(agg_fn)
                .reset_index()
                .rename(columns={"Acc_Vmu": col})
            )
            df_w[col] = (
                df_w[["Patient_ID", "Timestamp"]]
                .merge(rolled, on=["Patient_ID", "Timestamp"], how="left")[col]
                .values
            )
        activity_features = [col_mean, col_max]

    # -------------------------------------------------------------------------
    # GLUCOSE LOOKUP
    # -------------------------------------------------------------------------
    glucose_lookup: dict = (
        df_w.set_index(["Patient_ID", "Timestamp"])["Glucose"].to_dict()
    )

    # -------------------------------------------------------------------------
    # GLUCOSE LAG (ablation only)
    # -------------------------------------------------------------------------
    print(f"\n[5/10] GLUCOSE LAG (ablation-only) anchored to {window_label}...")
    lag_col = f"glucose_lag_{window_label}"
    roc_col = f"glucose_roc_{window_label}"

    past_times    = df_w["Timestamp"] - pd.Timedelta(window_label)
    df_w[lag_col] = [
        glucose_lookup.get((pid, ts), np.nan)
        for pid, ts in zip(df_w["Patient_ID"], past_times)
    ]
    df_w[roc_col] = df_w["Glucose"] - df_w[lag_col]
    lag_features  = [lag_col, roc_col]

    # -------------------------------------------------------------------------
    # TARGET EXTRACTION
    # -------------------------------------------------------------------------
    for h_label in HORIZONS:
        td = pd.Timedelta(h_label.replace("min", "m"))
        future_times = df_w["Timestamp"] + td
        df_w[f"Target_{h_label}"] = [
            glucose_lookup.get((pid, ts), np.nan)
            for pid, ts in zip(df_w["Patient_ID"], future_times)
        ]

    # -------------------------------------------------------------------------
    # FEATURE SETS - NO RAW SENSOR VALUES (redundant with physio variability)
    # -------------------------------------------------------------------------
    all_features = (
        ["Glucose"] + temporal_features
        + food_features_all + activity_features + physio_features_all
    )
    all_features = [f for f in all_features if f in df_w.columns]

    all_features_comparable = (
        ["Glucose"] + temporal_features
        + food_features_carbs + activity_features + physio_hr
    )
    all_features_comparable = [f for f in all_features_comparable if f in df_w.columns]

    # -------------------------------------------------------------------------
    # ABLATION GROUPS
    # -------------------------------------------------------------------------
    feature_groups = {
        "glucose":         ["Glucose"],
        "physio_only":     physio_features_all,
        "EDA":             physio_eda,
        "heart_rate":      physio_hr,
        "BVP":             physio_bvp,
        "IBI":             physio_ibi,
        "food_only":       food_features_all,
        "carbs_only":      food_features_carbs,
        "activity":        activity_features,
        "comparable":      all_features_comparable,
        "all":             all_features,
        "physio_food":     physio_features_all + food_features_all,
        "food_movement":   food_features_all + activity_features,
        "physio_movement": physio_features_all + activity_features,
        "lag":             lag_features,
    }
    feature_groups = {
        name: [f for f in feats if f in df_w.columns]
        for name, feats in feature_groups.items()
    }

    FEATURE_SETS = {"full": all_features, "comparable": all_features_comparable}

    # -------------------------------------------------------------------------
    # WALK-FORWARD FOLDS
    # -------------------------------------------------------------------------
    print(f"\n[6/10] BUILDING WALK-FORWARD FOLDS ({window_label})...")
    wf_folds = build_walk_forward_folds(df_w, N_SPLITS)
    df_train_final, df_test_final = wf_folds[-1]
    for i, (tr, te) in enumerate(wf_folds):
        print(f"  Fold {i}: train={len(tr):,}  test={len(te):,}")

    # -------------------------------------------------------------------------
    # GRID SEARCH
    # -------------------------------------------------------------------------
    print(f"\n[7/10] GRID SEARCH ({window_label}, 30-min horizon only)...")
    gs_key = f"{MODEL_NAME}_{window_label}"
    if gs_key in grid_cache:
        best_params = grid_cache[gs_key]
        print(f"  Loaded cached params: {best_params}")
    else:
        gs_seq = build_seq_dataset(
            df_train_final, df_test_final,
            all_features, "Target_30min", SEQ_LENGTH,
        )
        if len(gs_seq["X_train"]) == 0:
            print(f"  No sequences for grid search at {window_label} — skipping")
            continue

        gs_weights  = calculate_clinical_weights(gs_seq["y_train_raw"])
        best_params, best_score = grid_search_dl(
            gs_seq["X_train"], gs_seq["y_train"], gs_weights,
            input_dim=len(all_features), param_grid=DL_PARAM_GRID,
        )
        grid_cache[gs_key] = best_params
        with open(grid_cache_path, "w") as f:
            json.dump(grid_cache, f, indent=2)
        print(f"  Best params: {best_params}  (val_loss={best_score:.4f})")

    # -------------------------------------------------------------------------
    # MAIN TRAINING LOOP
    # -------------------------------------------------------------------------
    print(f"\n[8/10] MAIN TRAINING LOOP ({window_label})...")
    for fold_i, (df_tr, df_te) in enumerate(wf_folds):
        for subset, features in FEATURE_SETS.items():
            for h_label in HORIZONS:
                target_col = f"Target_{h_label}"
                seq        = build_seq_dataset(df_tr, df_te, features,
                                               target_col, SEQ_LENGTH)

                if len(seq["X_train"]) == 0 or len(seq["X_test"]) == 0:
                    print(f"  [fold {fold_i}] {subset} {h_label} — "
                          f"no sequences, skipping")
                    continue

                weights = calculate_clinical_weights(seq["y_train_raw"])
                model, val_loss, epochs_trained = train_tcn(
                    seq["X_train"], seq["y_train"], weights,
                    input_dim=len(features),
                    num_filters=best_params["num_filters"],
                    lr=best_params["lr"],
                )
                model.eval()
                with torch.no_grad():
                    preds_scaled = model(torch.tensor(seq["X_test"])).numpy().flatten()
                preds = seq["y_scaler"].inverse_transform(preds_scaled.reshape(-1,1)).flatten()
                m = metrics_dict(seq["y_test_raw"], preds)
                all_results.append({
                    "model": MODEL_NAME, "window": window_label,
                    "horizon": h_label, "subset": subset,
                    "fold": fold_i, "epochs_trained": epochs_trained, **m,
                })

                key = (window_label, subset, h_label)
                pooled_predictions.setdefault(key, {"y_true": [], "y_pred": []})
                pooled_predictions[key]["y_true"].append(seq["y_test_raw"])
                pooled_predictions[key]["y_pred"].append(preds)

                print(f"  [fold {fold_i}] {subset:10s} {h_label:6s} "
                      f"RMSE={m['rmse']:.2f} MAE={m['mae']:.2f} "
                      f"epochs={epochs_trained}")

    # -------------------------------------------------------------------------
    # ABLATION STUDY
    # -------------------------------------------------------------------------
    print(f"\n[9/10] ABLATION STUDY ({window_label})...")
    for combo_name, combo_features in feature_groups.items():
        if not combo_features:
            continue
        for h_label in HORIZONS:
            target_col = f"Target_{h_label}"
            seq        = build_seq_dataset(
                df_train_final, df_test_final,
                combo_features, target_col, SEQ_LENGTH,
            )

            if len(seq["X_train"]) == 0 or len(seq["X_test"]) == 0:
                continue

            weights = calculate_clinical_weights(seq["y_train_raw"])
            model, _, _ = train_tcn(
                seq["X_train"], seq["y_train"], weights,
                input_dim=len(combo_features),
                num_filters=best_params["num_filters"],
                lr=best_params["lr"],
            )
            model.eval()
            with torch.no_grad():
                preds_scaled = model(torch.tensor(seq["X_test"])).numpy().flatten()
            preds = seq["y_scaler"].inverse_transform(preds_scaled.reshape(-1,1)).flatten()
            rmse = float(np.sqrt(mean_squared_error(seq["y_test_raw"], preds)))
        
            all_ablation_results.append({
                "model":   MODEL_NAME, "window": window_label,
                "combo":   combo_name, "horizon": h_label, "rmse": rmse,
                "n_train": seq["n_train"], "n_test": seq["n_test"],
            })

        window_rows = [x for x in all_ablation_results
                       if x["combo"] == combo_name and x["window"] == window_label]
        print(f"  [{combo_name:20s}] " + ", ".join(
            f"{h}={r['rmse']:.2f}" for h, r in zip(HORIZONS, window_rows)
        ))


# ============================================================================
# 13. SAVE RESULTS
# ============================================================================
print(f"\n[SAVE] SAVING RESULTS...")

results_df = pd.DataFrame(all_results)
results_df.to_csv(RESULTS_DIR / f"results_{MODEL_NAME}.csv", index=False)

ablation_df = pd.DataFrame(all_ablation_results)
ablation_df.to_csv(RESULTS_DIR / f"results_{MODEL_NAME}_ablation.csv", index=False)

with open(RESULTS_DIR / f"results_{MODEL_NAME}_pooled_preds.pkl", "wb") as f:
    pickle.dump(pooled_predictions, f)

print(f"  Saved results, ablation, pooled_preds for {MODEL_NAME}")

summary = (
    results_df
    .groupby(["window", "subset", "horizon"])
    .agg(
        rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
        mae_mean=("mae", "mean"),   mae_std=("mae", "std"),
        r2_mean=("r2", "mean"),     mape_mean=("mape", "mean"),
        epochs_mean=("epochs_trained", "mean"),
        n_folds=("rmse", "count"),
    )
    .reset_index()
)
summary.to_csv(RESULTS_DIR / f"summary_{MODEL_NAME}.csv", index=False)

print(f"\n{'='*70}\nWALK-FORWARD SUMMARY (mean ± std over {N_SPLITS} folds)\n{'='*70}")
for window_label in WINDOW_SIZES:
    print(f"\n  Window: {window_label}")
    for _, r in summary[summary["window"] == window_label].iterrows():
        print(f"    [{r['subset']:10s}] {r['horizon']:6s}  "
              f"RMSE={r['rmse_mean']:.2f}±{r['rmse_std']:.2f}  "
              f"MAE={r['mae_mean']:.2f}±{r['mae_std']:.2f}  "
              f"R2={r['r2_mean']:.4f}  MAPE={r['mape_mean']:.2f}%  "
              f"epochs~{r['epochs_mean']:.0f}")

# ============================================================================
# 14. CLARKE ERROR GRID + ABLATION PLOTS
# ============================================================================
print(f"\n[PLOTS] CLARKE ERROR GRID + ABLATION PLOTS...")
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

print("\nTCN pipeline complete (walk-forward + ablation).")