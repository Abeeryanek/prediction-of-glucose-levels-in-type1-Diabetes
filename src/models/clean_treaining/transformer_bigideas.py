"""
cnn_lstm.py — CNN-LSTM, Window Size x Horizon Walk-Forward Pipeline
================================================================================
Aligned to FINAL_EXPERIMENTS_PLAN.md:

- Plain MSELoss everywhere (§1, §8A.2) — clinically-weighted MSE removed
- Grid: hidden_dim [32,64,128] x lr [1e-3, 5e-4] (§2.8, §8A.5)
- Window sizes: 1h/2h/3h/6h (12/24/36/72 steps) — grid sweep (§2.6, §8A.6)
- Per-patient row-count check before large windows (§8A.6)
- LOPO removed (§10 item 3)
- Food columns: exact plain names, linear interpolation (§0A)
- max_epochs=150, patience=15, batch_size=32, shuffle=False,
  val_ratio=0.20, seed=42 (§1)
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
# CONSTANTS
# ============================================================================
DATA_PATH   = "data/bigideas"
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

MODEL_NAME = "cnnlstm"

# §2.6, §8A.6 — window size is now a grid sweep, not a single fixed value
WINDOW_SIZES = {"1h": 12, "2h": 24, "3h": 36, "6h": 72}

HORIZONS = {"15min": 3, "30min": 6, "45min": 9}
N_SPLITS  = 5

# §1 — standard training conditions
MAX_EPOCHS = 150
PATIENCE   = 15
BATCH_SIZE = 32
VAL_RATIO  = 0.20
SHUFFLE    = False

GRID_SEARCH_EPOCHS   = 30
GRID_SEARCH_PATIENCE = 5

LOPO_MIN_TRAIN_ROWS = 200   # kept for reference, LOPO not run (§10 item 3)
LOPO_MIN_TEST_ROWS  = 50

# §2.8, §8A.5 — CNN-LSTM grid: hidden_dim x lr only
DL_PARAM_GRID = {"hidden_dim": [32, 64, 128], "lr": [1e-3, 5e-4]}

# §8A.6 — exact plain food column names after fix_food_features.py
FOOD_COLS = ["calorie", "total_carb", "dietary_fiber", "sugar", "protein", "total_fat"]

print(f"\n{'='*70}\nCNN-LSTM — WINDOW x HORIZON WALK-FORWARD PIPELINE\n{'='*70}")

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
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df = df.sort_values(["Patient_ID", "Timestamp"]).reset_index(drop=True)

print(f"Loaded: {len(df):,} rows from {df['Patient_ID'].nunique()} patients")

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

# define sensor_cols BEFORE use
sensor_cols = [c for c in ["Heart_Rate", "Acc_Vmu", "EDA", "Skin_Temp", "BVP", "IBI"]
               if c in df.columns]

# §5, §6 — bounded linear interpolation for sensors, max gap 6 steps (30 min)
for col in sensor_cols:
    df[col] = (
        df.groupby("Patient_ID")[col]
          .transform(lambda x: x.interpolate(method="linear", limit=6))
    )
print(f"  Sensors interpolated (limit=6 steps / 30 min): {sensor_cols}")

# food column detection — exact plain names, no trailing underscore
food_features_all   = [c for c in df.columns if c in FOOD_COLS]
food_features_carbs = [c for c in df.columns if c == "total_carb"]

# §0A — plain linear interpolation for food, no limit
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

def calculate_clinical_weights(y_true):
    weights = np.ones(len(y_true), dtype=np.float32)
    weights[y_true < 54] = 3.0
    weights[(y_true >= 54) & (y_true < 70)] = 2.5
    weights[(y_true > 180) & (y_true <= 250)] = 1.5
    weights[y_true > 250] = 2.0
    return weights

# ============================================================================
# 4. 3D SEQUENCE BUILDER (Time-Locked)
# ============================================================================
def create_3d_sequences(X_scaled, y_series, patient_ids, timestamps, seq_length):
    """
    Builds (N, seq_length, n_features) arrays.
    Skips sequences spanning patient boundaries or timestamp gaps.
    Uses float-minute tolerance to avoid ns vs minute precision issues.
    """
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
    """
    Food already interpolated upstream — dropna removes only rows where
    target or rolling-derived features are genuinely unavailable.
    """
    train_clean = df_tr.dropna(subset=[target_col] + features).reset_index(drop=True)
    test_clean  = df_te.dropna(subset=[target_col] + features).reset_index(drop=True)

    scaler     = StandardScaler()
    X_train_2d = scaler.fit_transform(train_clean[features])
    X_test_2d  = scaler.transform(test_clean[features])

    X_train_3d, y_train = create_3d_sequences(
        X_train_2d, train_clean[target_col],
        train_clean["Patient_ID"], train_clean["Timestamp"], seq_length,
    )
    X_test_3d, y_test = create_3d_sequences(
        X_test_2d, test_clean[target_col],
        test_clean["Patient_ID"], test_clean["Timestamp"], seq_length,
    )

    return {
        "X_train": X_train_3d, "y_train": y_train,
        "X_test":  X_test_3d,  "y_test":  y_test,
        "n_train": len(train_clean), "n_test": len(test_clean),
    }


# ============================================================================
# 5. MODEL DEFINITION
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


class GlucoseCNNLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2,
                 dropout=0.3, cnn_filters=64, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, cnn_filters, kernel_size)
        self.relu  = nn.ReLU()
        self.lstm  = nn.LSTM(cnn_filters, hidden_dim, num_layers,
                             batch_first=True, dropout=dropout)
        self.fc    = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


# ============================================================================
# 6. TRAINING LOOP — plain MSELoss (§1, §8A.2)
# ============================================================================
def train_cnn_lstm(
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

    model     = GlucoseCNNLSTM(input_dim=input_dim, **model_kwargs)
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
# 7. GRID SEARCH — §2.8: hidden_dim x lr only
# ============================================================================
def grid_search_dl(x_train, y_train, sample_weights, input_dim, param_grid):
    keys   = list(param_grid.keys())
    combos = list(product(*[param_grid[k] for k in keys]))

    best_score, best_params = float("inf"), {}
    for combo in combos:
        params = dict(zip(keys, combo))
        _, val_loss, _ = train_cnn_lstm(
            x_train, y_train, sample_weights, input_dim=input_dim,
            epochs=GRID_SEARCH_EPOCHS, patience=GRID_SEARCH_PATIENCE,
            hidden_dim=params["hidden_dim"], lr=params["lr"],
        )
        if val_loss < best_score:
            best_score, best_params = val_loss, params
    return best_params, best_score


 # ---- grid search call ----
    gs_seq = build_seq_dataset(
        df_train_final, df_test_final,
        all_features, "Target_30min", SEQ_LENGTH,
    )
    if len(gs_seq["X_train"]) == 0:
        print(f"  ⚠ No sequences for grid search at {window_label} — skipping")
        continue
    gs_weights  = calculate_clinical_weights(gs_seq["y_train"])
    best_params, best_score = grid_search_dl(
        gs_seq["X_train"], gs_seq["y_train"], gs_weights,
        input_dim=len(all_features), param_grid=DL_PARAM_GRID,
    )

    # ---- main training loop call ----
    seq     = build_seq_dataset(df_tr, df_te, features, target_col, SEQ_LENGTH)
    if len(seq["X_train"]) == 0 or len(seq["X_test"]) == 0:
        print(f"  [fold {fold_i}] {subset} {h_label} — no sequences, skipping")
        continue
    weights = calculate_clinical_weights(seq["y_train"])
    model, val_loss, epochs_trained = train_cnn_lstm(
        seq["X_train"], seq["y_train"], weights,
        input_dim=len(features),
        hidden_dim=best_params["hidden_dim"],
        lr=best_params["lr"],
    )

    # ---- ablation loop call ----
    seq     = build_seq_dataset(
        df_train_final, df_test_final,
        combo_features, target_col, SEQ_LENGTH,
    )
    if len(seq["X_train"]) == 0 or len(seq["X_test"]) == 0:
        continue
    weights = calculate_clinical_weights(seq["y_train"])
    model, _, _ = train_cnn_lstm(
        seq["X_train"], seq["y_train"], weights,
        input_dim=len(combo_features),
        hidden_dim=best_params["hidden_dim"],
        lr=best_params["lr"],
    )

# ============================================================================
# 8. METRICS
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
# 9. CLARKE ERROR GRID
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
# 10. ABLATION BAR PLOT
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
                         edgecolor="black", linewidth=0.6, color="blue")

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
# 11. WINDOW SIZE LOOP — §2.6, §8A.6
# ============================================================================
for window_label, window_size in WINDOW_SIZES.items():
    print(f"\n{'#'*70}\nWINDOW SIZE: {window_label} ({window_size} steps)\n{'#'*70}")

    df_w       = df.copy()
    SEQ_LENGTH = window_size

    # §8A.6 — per-patient row-count check before running large windows
    min_rows_needed = SEQ_LENGTH + N_SPLITS + 1
    excluded_patients = []
    for pid, grp in df_w.groupby("Patient_ID"):
        if len(grp) < min_rows_needed:
            excluded_patients.append((pid, len(grp)))
    if excluded_patients:
        print(f"  ⚠ [{window_label}] {len(excluded_patients)} patient(s) have fewer "
              f"than {min_rows_needed} rows — will be silently excluded from folds:")
        for pid, n in excluded_patients:
            print(f"      Patient {pid}: {n} rows")
    else:
        print(f"  ✓ [{window_label}] All patients have ≥ {min_rows_needed} rows.")

    # Set Timestamp as index for time-based rolling
    temp_time_df = df_w.set_index("Timestamp")

    # -------------------------------------------------------------------------
    # PHYSIOLOGICAL VARIABILITY — time-based rolling (aligned via reindex)
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
            .reset_index(level=0, drop=True)
            .sort_index()
        )
        df_w[col] = rolled.reindex(df_w["Timestamp"]).values
        physio_features_all.append(col)
        target_list.append(col)

    _add_std_anchored("Heart_Rate", physio_hr)
    _add_std_anchored("EDA",        physio_eda)
    _add_std_anchored("BVP",        physio_bvp)
    _add_std_anchored("IBI",        physio_ibi)
    _add_std_anchored("Skin_Temp",  physio_skin)

    # -------------------------------------------------------------------------
    # ACTIVITY — time-based rolling (aligned via reindex)
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
                .reset_index(level=0, drop=True)
                .sort_index()
            )
            df_w[col] = rolled.reindex(df_w["Timestamp"]).values
        activity_features = [col_mean, col_max]

    # -------------------------------------------------------------------------
    # GLUCOSE LOOKUP — dict-based for lag + target extraction
    # -------------------------------------------------------------------------
    glucose_lookup: dict = (
        df_w.set_index(["Patient_ID", "Timestamp"])["Glucose"].to_dict()
    )

    # -------------------------------------------------------------------------
    # GLUCOSE LAG (ablation only) — exact timestamp shift
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
    # TARGET EXTRACTION — exact timestamp shift
    # §2.6: grid search uses 30-min horizon only; best params reused for 15/45
    # -------------------------------------------------------------------------
    for h_label in HORIZONS:
        td = pd.Timedelta(h_label.replace("min", "m"))
        future_times = df_w["Timestamp"] + td
        df_w[f"Target_{h_label}"] = [
            glucose_lookup.get((pid, ts), np.nan)
            for pid, ts in zip(df_w["Patient_ID"], future_times)
        ]

    # -------------------------------------------------------------------------
    # FEATURE SETS
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
    # ABLATION GROUPS (§8A.4 — individual signal isolation)
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
    # GRID SEARCH — §2.6: run at 30-min horizon, reuse for 15/45
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
            print(f"  ⚠ No training sequences for grid search at {window_label} — skipping")
            continue

        best_params, best_score = grid_search_dl(
            gs_seq["X_train"], gs_seq["y_train"],
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

                model, val_loss, epochs_trained = train_cnn_lstm(
                    seq["X_train"], seq["y_train"],
                    input_dim=len(features),
                    hidden_dim=best_params["hidden_dim"],
                    lr=best_params["lr"],
                )
                model.eval()
                with torch.no_grad():
                    preds = model(torch.tensor(seq["X_test"])).numpy().flatten()

                m = metrics_dict(seq["y_test"], preds)
                all_results.append({
                    "model": MODEL_NAME, "window": window_label,
                    "horizon": h_label, "subset": subset,
                    "fold": fold_i, "epochs_trained": epochs_trained, **m,
                })

                key = (window_label, subset, h_label)
                pooled_predictions.setdefault(key, {"y_true": [], "y_pred": []})
                pooled_predictions[key]["y_true"].append(seq["y_test"])
                pooled_predictions[key]["y_pred"].append(preds)

                print(f"  [fold {fold_i}] {subset:10s} {h_label:6s} "
                      f"RMSE={m['rmse']:.2f} MAE={m['mae']:.2f} "
                      f"epochs={epochs_trained}")

    # -------------------------------------------------------------------------
    # ABLATION STUDY — §8A.4
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

            model, _, _ = train_cnn_lstm(
                seq["X_train"], seq["y_train"],
                input_dim=len(combo_features),
                hidden_dim=best_params["hidden_dim"],
                lr=best_params["lr"],
            )
            model.eval()
            with torch.no_grad():
                preds = model(torch.tensor(seq["X_test"])).numpy().flatten()
            rmse = float(np.sqrt(mean_squared_error(seq["y_test"], preds)))

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
# 12. SAVE RESULTS
# ============================================================================
print(f"\n[SAVE] SAVING RESULTS...")

results_df = pd.DataFrame(all_results)
results_df.to_csv(RESULTS_DIR / f"results_{MODEL_NAME}.csv", index=False)

ablation_df = pd.DataFrame(all_ablation_results)
ablation_df.to_csv(RESULTS_DIR / f"results_{MODEL_NAME}_ablation.csv", index=False)

with open(RESULTS_DIR / f"results_{MODEL_NAME}_pooled_preds.pkl", "wb") as f:
    pickle.dump(pooled_predictions, f)

print(f"  Saved results, ablation, pooled_preds for {MODEL_NAME}")

# Summary
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
# 13. CLARKE ERROR GRID + ABLATION PLOTS
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

print("\nCNN-LSTM pipeline complete (walk-forward + ablation).")