"""
gradient_boost.py — Gradient Boosting, Window Size x Horizon Walk-Forward Pipeline
==============================================================================================
Aligned to FINAL_EXPERIMENTS_PLAN.md:

- Clinically-weighted sample weights (hypo/hyper sample weights)
- Grid: n_estimators [100,200,300] x max_depth [10,15,20] x min_samples_leaf [1,2,4]
- Window sizes: 1h/2h/3h/6h (12/24/36/72 steps) — grid sweep
- Per-patient row-count check before large windows
- LOPO removed (§10 item 3)
- Food columns: exact plain names, linear interpolation (§0A)
- Raw sensor values removed (redundant with physio variability features)
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

from sklearn.ensemble import GradientBoostingRegressor
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

MODEL_NAME = "gb"

WINDOW_SIZES = {"1h": 12, "2h": 24, "3h": 36, "6h": 72}
HORIZONS     = {"15min": 3, "30min": 6, "45min": 9}
N_SPLITS     = 5

GB_PARAM_GRID = {
    "n_estimators":     [100, 200, 300],
    "max_depth":        [10, 15, 20],
    "min_samples_leaf": [1, 2, 4],
}

FOOD_COLS = ["calorie", "total_carb", "dietary_fiber", "sugar", "protein", "total_fat"]

print(f"\n{'='*70}\nGRADIENT BOOSTING — WINDOW x HORIZON WALK-FORWARD PIPELINE\n{'='*70}")

# ============================================================================
# 1. LOAD DATA - FIXED for \N values and Patient_ID preservation
# ============================================================================
print("\n[1/9] LOADING DATA...")

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
print("\n[2/9] BASE FEATURE ENGINEERING...")

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

# Food columns - exact plain names, no prefixes
food_features_all   = [c for c in df.columns if c in FOOD_COLS]
food_features_carbs = [c for c in df.columns if c == "total_carb"]

# Linear interpolation for food (no limit)
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
# 5. GRID SEARCH
# ============================================================================
def grid_search_gb(X_train, y_train, sample_weight, param_grid, n_splits=3, **fixed_kwargs):
    tscv  = TimeSeriesSplit(n_splits=n_splits)
    keys  = list(param_grid.keys())
    combos = list(product(*[param_grid[k] for k in keys]))

    X_arr = X_train.values if hasattr(X_train, "values") else np.asarray(X_train)
    y_arr = y_train.values if hasattr(y_train, "values") else np.asarray(y_train)
    w_arr = np.asarray(sample_weight)

    best_score, best_params = float("inf"), {}
    for combo in combos:
        params = dict(zip(keys, combo))
        fold_scores = []
        for train_idx, val_idx in tscv.split(X_arr):
            model = GradientBoostingRegressor(**params, **fixed_kwargs)
            model.fit(X_arr[train_idx], y_arr[train_idx],
                      sample_weight=w_arr[train_idx])
            preds = model.predict(X_arr[val_idx])
            fold_scores.append(np.sqrt(mean_squared_error(y_arr[val_idx], preds)))
        mean_score = float(np.mean(fold_scores))
        if mean_score < best_score:
            best_score, best_params = mean_score, params
    return best_params, best_score


# ============================================================================
# 6. METRICS
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
# 7. CLARKE ERROR GRID
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
# 8. ABLATION BAR PLOT
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
                         edgecolor="black", linewidth=0.6, color="orange")

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
# 9. WINDOW SIZE LOOP
# ============================================================================
for window_label, window_size in WINDOW_SIZES.items():
    print(f"\n{'#'*70}\nWINDOW SIZE: {window_label} ({window_size} steps)\n{'#'*70}")

    df_w = df.copy()

    # per-patient row-count check
    min_rows_needed   = window_size + N_SPLITS + 1
    excluded_patients = []
    for pid, grp in df_w.groupby("Patient_ID"):
        if len(grp) < min_rows_needed:
            excluded_patients.append((pid, len(grp)))
    if excluded_patients:
        print(f"  [{window_label}] {len(excluded_patients)} patient(s) below "
              f"{min_rows_needed} rows:")
        for pid, n in excluded_patients:
            print(f"      Patient {pid}: {n} rows")
    else:
        print(f"  [{window_label}] All patients have ≥ {min_rows_needed} rows.")

    temp_time_df = df_w.set_index("Timestamp")

    # -------------------------------------------------------------------------
    # PHYSIOLOGICAL VARIABILITY — merge-based alignment
    # -------------------------------------------------------------------------
    print(f"\n[3/9] PHYSIOLOGICAL VARIABILITY anchored to {window_label}...")
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
    print(f"\n[4/9] ACTIVITY FEATURES anchored to {window_label}...")
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
    print(f"\n[5/9] GLUCOSE LAG (ablation-only) anchored to {window_label}...")
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
    print(f"\n[6/9] BUILDING WALK-FORWARD FOLDS ({window_label})...")
    wf_folds = build_walk_forward_folds(df_w, N_SPLITS)
    df_train_final, df_test_final = wf_folds[-1]
    for i, (tr, te) in enumerate(wf_folds):
        print(f"  Fold {i}: train={len(tr):,}  test={len(te):,}")

    # -------------------------------------------------------------------------
    # GRID SEARCH
    # -------------------------------------------------------------------------
    print(f"\n[7/9] GRID SEARCH ({window_label}, 30-min horizon only)...")
    gs_key = f"{MODEL_NAME}_{window_label}"
    if gs_key in grid_cache:
        best_params = grid_cache[gs_key]
        print(f"  Loaded cached params: {best_params}")
    else:
        gs_train = df_train_final.dropna(subset=["Target_30min"] + all_features)
        if len(gs_train) == 0:
            print(f"  No data for grid search at {window_label} — skipping")
            continue

        gs_weights = calculate_clinical_weights(gs_train["Target_30min"].values)
        best_params, best_score = grid_search_gb(
            gs_train[all_features], gs_train["Target_30min"], gs_weights,
            GB_PARAM_GRID, random_state=RANDOM_SEED, subsample=0.8,
        )
        grid_cache[gs_key] = best_params
        with open(grid_cache_path, "w") as f:
            json.dump(grid_cache, f, indent=2)
        print(f"  Best params: {best_params}  (RMSE={best_score:.2f})")

    gb_params = {**best_params, "random_state": RANDOM_SEED, "subsample": 0.8}

    # -------------------------------------------------------------------------
    # MAIN TRAINING LOOP
    # -------------------------------------------------------------------------
    print(f"\n[8/9] MAIN TRAINING LOOP ({window_label})...")
    for fold_i, (df_tr, df_te) in enumerate(wf_folds):
        for subset, features in FEATURE_SETS.items():
            for h_label in HORIZONS:
                target_col  = f"Target_{h_label}"
                train_clean = df_tr.dropna(subset=[target_col] + features)
                test_clean  = df_te.dropna(subset=[target_col] + features)

                if len(train_clean) == 0 or len(test_clean) == 0:
                    print(f"  [fold {fold_i}] {subset} {h_label} — no data, skipping")
                    continue

                scaler   = StandardScaler().fit(train_clean[features])
                x_tr_sc  = scaler.transform(train_clean[features])
                x_te_sc  = scaler.transform(test_clean[features])
                weights  = calculate_clinical_weights(train_clean[target_col].values)

                model = GradientBoostingRegressor(**gb_params)
                model.fit(x_tr_sc, train_clean[target_col], sample_weight=weights)
                preds = model.predict(x_te_sc)

                m = metrics_dict(test_clean[target_col].values, preds)
                all_results.append({
                    "model": MODEL_NAME, "window": window_label,
                    "horizon": h_label, "subset": subset,
                    "fold": fold_i, **m,
                })

                key = (window_label, subset, h_label)
                pooled_predictions.setdefault(key, {"y_true": [], "y_pred": []})
                pooled_predictions[key]["y_true"].append(test_clean[target_col].values)
                pooled_predictions[key]["y_pred"].append(preds)

                print(f"  [fold {fold_i}] {subset:10s} {h_label:6s} "
                      f"RMSE={m['rmse']:.2f} MAE={m['mae']:.2f}")

    # -------------------------------------------------------------------------
    # ABLATION STUDY
    # -------------------------------------------------------------------------
    print(f"\n[9/9] ABLATION STUDY ({window_label})...")
    for combo_name, combo_features in feature_groups.items():
        if not combo_features:
            continue
        for h_label in HORIZONS:
            target_col  = f"Target_{h_label}"
            train_clean = df_train_final.dropna(subset=[target_col] + combo_features)
            test_clean  = df_test_final.dropna(subset=[target_col] + combo_features)

            if len(train_clean) == 0 or len(test_clean) == 0:
                continue

            scaler  = StandardScaler()
            x_tr    = scaler.fit_transform(train_clean[combo_features])
            x_te    = scaler.transform(test_clean[combo_features])
            weights = calculate_clinical_weights(train_clean[target_col].values)

            model = GradientBoostingRegressor(**gb_params)
            model.fit(x_tr, train_clean[target_col], sample_weight=weights)
            preds = model.predict(x_te)
            rmse  = float(np.sqrt(mean_squared_error(test_clean[target_col], preds)))

            all_ablation_results.append({
                "model":   MODEL_NAME, "window": window_label,
                "combo":   combo_name, "horizon": h_label, "rmse": rmse,
                "n_train": len(train_clean), "n_test": len(test_clean),
            })

        window_rows = [x for x in all_ablation_results
                       if x["combo"] == combo_name and x["window"] == window_label]
        print(f"  [{combo_name:20s}] " + ", ".join(
            f"{h}={r['rmse']:.2f}" for h, r in zip(HORIZONS, window_rows)
        ))


# ============================================================================
# 10. SAVE RESULTS
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
              f"R2={r['r2_mean']:.4f}  MAPE={r['mape_mean']:.2f}%")

# ============================================================================
# 11. CLARKE ERROR GRID + ABLATION PLOTS
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

print("\nGradient Boosting pipeline complete (walk-forward + ablation).")