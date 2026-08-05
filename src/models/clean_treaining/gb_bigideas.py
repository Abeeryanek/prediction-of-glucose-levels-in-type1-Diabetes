"""
gradient_boost.py — Gradient Boosting, Window Size x Horizon Walk-Forward + LOPO Pipeline
==============================================================================================
Self-contained: no external config/utils modules. Everything in one file.
Mirrors random_forest.py exactly — same fixes applied, same structure. Only the
model class (GradientBoostingRegressor) and its grid search space differ
(GB has no n_jobs param; subsample=0.8 for stochastic regularization).

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
   the signal being tested — e.g. 'heart_rate' = ["Glucose"] + hr_std_features,
   not hr_std_features alone. Without this, a model given ONLY heart-rate
   variability (no glucose signal at all) can't beat a near-constant
   prediction, which is why every ablation group previously collapsed to
   similar, uninformative RMSE values regardless of window/horizon. The
   'lag' group remains the one exception, studying lag/roc in isolation with
   no glucose baseline added.

4. LOPO (Leave-One-Patient-Out), nested inside the window_size loop, reusing
   that window's cached grid-search-best GB params.

5. BIGGER, BETTER-SPACED ABLATION PLOTS (FIX). One horizontal subplot per
   horizon, figure width scales with number of feature groups, wider bar
   gaps, generous headroom above bars so RMSE labels never collide or clip.

6. STRICT TIME-BASED SHIFTING. Rolling windows, lag extraction, and target horizon 
   generation are all anchored strictly to timestamp math rather than row-counts 
   to prevent cross-gap contamination.
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
# CONSTANTS (all inline — no config.py)
# ============================================================================
DATA_PATH = "data/bigideas"
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

MODEL_NAME = "gb"

WINDOW_SIZES = {"1h": 12, "2h": 24, "3h": 36, "6h": 72}   # lookback
HORIZONS = {"15min": 3, "30min": 6, "45min": 9}            # prediction horizon
N_SPLITS = 5                                                # walk-forward folds per patient

GB_PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 15, 20],
    "min_samples_leaf": [1, 2, 4],
    "learning_rate": [0.01, 0.05, 0.1],
}

# LOPO minimum row thresholds (skip patients with too little data)
LOPO_MIN_TRAIN_ROWS = 200
LOPO_MIN_TEST_ROWS = 50

print(f"\n{'='*70}\nGRADIENT BOOSTING — WINDOW x HORIZON WALK-FORWARD + LOPO PIPELINE\n{'='*70}")

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

# ---- Temporal ----
df["Hour"] = df["Timestamp"].dt.hour
df["Minute"] = df["Timestamp"].dt.minute
df["DayOfWeek"] = df["Timestamp"].dt.dayofweek
df["MinFromMidnight"] = df["Hour"] * 60 + df["Minute"]
df["Hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
df["Hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)
temporal_features = ["Hour", "Minute", "DayOfWeek", "MinFromMidnight", "Hour_sin", "Hour_cos"]

# ---- Sensor gap interpolation (bounded, max 6 steps = 30min; NOT window_size-dependent) ----
sensor_cols = [c for c in ["Heart_Rate", "Acc_Vmu", "EDA", "Skin_Temp", "BVP", "IBI"] if c in df.columns]
for col in sensor_cols:
    df[col] = (df.groupby("Patient_ID")[col]
                 .transform(lambda x: x.interpolate(method="linear", limit=6)))
print(f"  Interpolated {len(sensor_cols)} sensors: {sensor_cols}")

# ---- Food rolling windows ----
food_features_all = [c for c in df.columns if any(
    c.startswith(n) for n in ["calorie_", "total_carb_", "dietary_fiber_", "sugar_", "protein_", "total_fat_"])]
food_features_carbs = [c for c in df.columns if c.startswith("total_carb_")]
print(f"  Food features (all): {len(food_features_all)}  |  (carbs only): {len(food_features_carbs)}")

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
# 4. CLINICAL SAMPLE WEIGHTING
# ============================================================================
def calculate_clinical_weights(y_true):
    weights = np.ones(len(y_true), dtype=np.float32)
    weights[y_true < 54] = 3.0
    weights[(y_true >= 54) & (y_true < 70)] = 2.5
    weights[(y_true > 180) & (y_true <= 250)] = 1.5
    weights[y_true > 250] = 2.0
    return weights


# ============================================================================
# 5. METRICS
# ============================================================================
def metrics_dict(y_true, y_pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": float(np.mean(np.abs((np.asarray(y_true) - y_pred) / np.asarray(y_true))) * 100),
    }


# ============================================================================
# 6. GRID SEARCH
# ============================================================================
def grid_search_gb(X_train, y_train, sample_weight, param_grid, n_splits=3, **fixed_kwargs):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    keys = list(param_grid.keys())
    combos = list(product(*[param_grid[k] for k in keys]))

    X_arr = X_train.values if hasattr(X_train, "values") else X_train
    y_arr = y_train.values if hasattr(y_train, "values") else y_train
    w_arr = np.asarray(sample_weight)

    best_score, best_params = float("inf"), {}
    for combo in combos:
        params = dict(zip(keys, combo))
        fold_scores = []
        for train_idx, val_idx in tscv.split(X_arr):
            model = GradientBoostingRegressor(**params, **fixed_kwargs)
            model.fit(X_arr[train_idx], y_arr[train_idx], sample_weight=w_arr[train_idx])
            preds = model.predict(X_arr[val_idx])
            fold_scores.append(np.sqrt(mean_squared_error(y_arr[val_idx], preds)))
        mean_score = float(np.mean(fold_scores))
        if mean_score < best_score:
            best_score, best_params = mean_score, params
    return best_params, best_score


# ============================================================================
# 7. LOPO (Leave-One-Patient-Out)
# ============================================================================
def run_lopo(df_w, features, target_col, gb_params, patient_col="Patient_ID"):
    patients = df_w[patient_col].unique()
    lopo_results = []
    pooled_true, pooled_pred = [], []

    for test_pid in patients:
        train_df = df_w.loc[df_w[patient_col] != test_pid].dropna(subset=[target_col] + features)
        test_df = df_w.loc[df_w[patient_col] == test_pid].dropna(subset=[target_col] + features)

        if len(train_df) < LOPO_MIN_TRAIN_ROWS or len(test_df) < LOPO_MIN_TEST_ROWS:
            continue

        scaler = StandardScaler().fit(train_df[features])
        x_tr = scaler.transform(train_df[features])
        x_te = scaler.transform(test_df[features])
        weights = calculate_clinical_weights(train_df[target_col].values)

        model = GradientBoostingRegressor(**gb_params)
        model.fit(x_tr, train_df[target_col], sample_weight=weights)
        preds = model.predict(x_te)

        y_true = test_df[target_col].values
        m = metrics_dict(y_true, preds)
        m["patient"] = test_pid
        m["n_train"] = len(train_df)
        m["n_test"] = len(test_df)
        lopo_results.append(m)

        pooled_true.append(y_true)
        pooled_pred.append(preds)

    return lopo_results, pooled_true, pooled_pred


# ============================================================================
# 8. CLARKE ERROR GRID (pooled across folds)
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
# 9. ABLATION BAR PLOT
# ============================================================================
def create_bar_plot(ablation_results, title, save_path):
    df_results = pd.DataFrame(ablation_results).T
    horizons = list(df_results.columns)
    n_groups = len(df_results)

    fig_width = max(18, n_groups * 1.6)
    fig_height = 7 * len(horizons)

    fig, axes = plt.subplots(len(horizons), 1, figsize=(fig_width, fig_height), layout="constrained")
    if len(horizons) == 1:
        axes = [axes]

    for ax, horizon in zip(axes, horizons):
        values = df_results[horizon].dropna().sort_values()
        x_pos = np.arange(len(values)) * 1.4

        bars = ax.bar(x_pos, values.values, width=0.85, edgecolor="black", linewidth=0.6, color="blue")

        max_val = values.values.max() if len(values) else 1.0
        for bar, val in zip(bars, values.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max_val * 0.02,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

        ax.set_xticks(x_pos)
        ax.set_xticklabels(values.index, rotation=45, ha="right", fontsize=11)
        ax.set_ylabel("RMSE (mg/dL)", fontweight="bold", fontsize=12)
        ax.set_title(f"Horizon: {horizon}", fontweight="bold", fontsize=14)
        ax.set_ylim(0, max_val * 1.25)
        ax.grid(axis="y", alpha=0.3, linewidth=0.8)
        ax.margins(x=0.02)

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
# 10. WINDOW SIZE LOOP — STRICT TIME-BASED MATH APPLIED HERE
# ============================================================================
for window_label, window_size in WINDOW_SIZES.items():
    print(f"\n{'#'*70}\nWINDOW SIZE: {window_label} ({window_size} steps)\n{'#'*70}")

    df_w = df.copy()

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
            # Time-based rolling directly groups the time index exactly by 1 hour (etc.)
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
    for h_label in HORIZONS.keys(): # e.g. "15min"
        # Add exactly 15/30/45 minutes to the current timestamp
        future_times = df_w["Timestamp"] + pd.Timedelta(h_label.replace("min", "m"))
        df_w[f"Target_{h_label}"] = pd.MultiIndex.from_arrays([df_w["Patient_ID"], future_times]).map(lookup)

    # ------------------------------------------------------------------------
    # MAIN FEATURE SETS 
    # ------------------------------------------------------------------------
    all_features = (["Glucose"] + temporal_features + food_features_all + activity_features + physio_features_all)
    all_features = [f for f in all_features if f in df_w.columns]

    all_features_comparable = (["Glucose"] + temporal_features +  food_features_carbs + activity_features + physio_hr)
    all_features_comparable = [f for f in all_features_comparable if f in df_w.columns]

    # ------------------------------------------------------------------------
    # ABLATION GROUPS
    # ------------------------------------------------------------------------
    feature_groups = {
        "glucose": ["Glucose"],
        "physio_only": physio_features_all,
        "EDA": physio_eda,
        "heart_rate": physio_hr,
        "BVP":  physio_bvp,
        "IBI":  physio_ibi,
        "food_only": food_features_all,
        "carbs_only": food_features_carbs,
        "activity":  activity_features,
        "comparable": all_features_comparable,
        "all": all_features,
        "physio_food": physio_features_all + food_features_all,
        "food_movement": food_features_all + activity_features,
        "physio_movement": physio_features_all + activity_features,
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
        target_col = "Target_30min"
        gs_train = df_train_final.dropna(subset=[target_col] + all_features)
        gs_weights = calculate_clinical_weights(gs_train[target_col].values)

        best_params, best_score = grid_search_gb(
            gs_train[all_features], gs_train[target_col], gs_weights,
            GB_PARAM_GRID, random_state=RANDOM_SEED, subsample=0.8,
        )
        grid_cache[gs_key] = best_params
        with open(grid_cache_path, "w") as f:
            json.dump(grid_cache, f, indent=2)
        print(f"  Best params: {best_params}  (RMSE={best_score:.2f})")

    gb_params = {**best_params, "random_state": RANDOM_SEED, "subsample": 0.8}

    print(f"\n[8/10] MAIN TRAINING LOOP ({window_label})...")
    for fold_i, (df_tr, df_te) in enumerate(wf_folds):
        for subset, features in FEATURE_SETS.items():
            for h_label in HORIZONS:
                target_col = f"Target_{h_label}"
                train_clean = df_tr.dropna(subset=[target_col] + features)
                test_clean = df_te.dropna(subset=[target_col] + features)

                scaler = StandardScaler().fit(train_clean[features])
                x_tr_sc = scaler.transform(train_clean[features])
                x_te_sc = scaler.transform(test_clean[features])
                weights = calculate_clinical_weights(train_clean[target_col].values)

                model = GradientBoostingRegressor(**gb_params)
                model.fit(x_tr_sc, train_clean[target_col], sample_weight=weights)
                preds = model.predict(x_te_sc)

                m = metrics_dict(test_clean[target_col].values, preds)
                all_results.append({
                    "model": MODEL_NAME, "window": window_label, "horizon": h_label,
                    "subset": subset, "fold": fold_i, **m,
                })

                key = (window_label, subset, h_label)
                pooled_predictions.setdefault(key, {"y_true": [], "y_pred": []})
                pooled_predictions[key]["y_true"].append(test_clean[target_col].values)
                pooled_predictions[key]["y_pred"].append(preds)

                print(f"  [fold {fold_i}] {subset:10s} {h_label:6s} RMSE={m['rmse']:.2f} MAE={m['mae']:.2f}")

    print(f"\n[9/10] ABLATION STUDY ({window_label})...")
    for combo_name, combo_features in feature_groups.items():
        if not combo_features:
            continue
        for h_label in HORIZONS:
            target_col = f"Target_{h_label}"
            train_clean = df_train_final.dropna(subset=[target_col] + combo_features)
            test_clean = df_test_final.dropna(subset=[target_col] + combo_features)

            scaler = StandardScaler()
            x_tr = scaler.fit_transform(train_clean[combo_features])
            x_te = scaler.transform(test_clean[combo_features])
            y_tr = train_clean[target_col]
            y_te = test_clean[target_col]
            weights = calculate_clinical_weights(y_tr.values)

            model = GradientBoostingRegressor(**gb_params)
            model.fit(x_tr, y_tr, sample_weight=weights)
            preds = model.predict(x_te)
            rmse = float(np.sqrt(mean_squared_error(y_te, preds)))

            all_ablation_results.append({
                "model": MODEL_NAME, "window": window_label, "combo": combo_name,
                "horizon": h_label, "rmse": rmse,
                "n_train": len(train_clean), "n_test": len(test_clean),
            })
        print(f"  [{combo_name:20s}] " + ", ".join(
            f"{h}={r['rmse']:.2f}" for h, r in
            zip(HORIZONS, [x for x in all_ablation_results if x["combo"] == combo_name and x["window"] == window_label])
        ))

    print(f"\n[LOPO] LEAVE-ONE-PATIENT-OUT ({window_label})...")
    for subset, features in FEATURE_SETS.items():
        for h_label in HORIZONS:
            target_col = f"Target_{h_label}"
            lopo_results, pooled_true, pooled_pred = run_lopo(
                df_w, features, target_col, gb_params,
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
# 11. SAVE RESULTS
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

# ---- Walk-forward summary table (mean +/- std across folds) ----
summary = (results_df.groupby(["window", "subset", "horizon"])
           .agg(rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
                mae_mean=("mae", "mean"), mae_std=("mae", "std"),
                r2_mean=("r2", "mean"), mape_mean=("mape", "mean"),
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
              f"R2={r['r2_mean']:.4f}  MAPE={r['mape_mean']:.2f}%")

# ---- LOPO summary table (mean +/- std across patients) ----
lopo_summary = (lopo_df.groupby(["window", "subset", "horizon"])
                .agg(rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
                     mae_mean=("mae", "mean"), mae_std=("mae", "std"),
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
              f"n_patients={r['n_patients']:.0f}")

# ---- Personalised (walk-forward) vs. LOPO comparison ----
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
# 12. CLARKE ERROR GRID (pooled, walk-forward AND LOPO) + ABLATION BAR PLOTS
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

print("\nGradient Boosting pipeline complete (walk-forward + ablation + LOPO).")