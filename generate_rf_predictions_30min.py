"""Standalone: retrain RF only across all 12 OhioT1DM patients and save the
pooled 30-min (ref, pred) arrays to disk, so we never have to retrain again
just to replot the Clarke Error Grid.
"""
import json
from pathlib import Path

import numpy as np

from src.preprocessing.ohio_loader import load_patient, COHORT_2018, COHORT_2020
from src.training.pipeline import make_splits
from src.models import random_forest as rf

DATA_ROOT   = Path("data/ohio")
RESULTS_DIR = Path("results/ohio")

WINDOW_SIZE = 12
HORIZON     = 9
STEP_30MIN  = 5   # index into the horizon axis corresponding to 30 min

CLINICAL_FEATURES = ["glucose", "bolus", "carbs"]
ALL_PATIENTS      = COHORT_2018 + COHORT_2020

with open(RESULTS_DIR / "grid_search_results.json") as f:
    gs_results = json.load(f)
rf_params = gs_results["rf"]
print("RF params:", rf_params)

ref_pooled  = []
pred_pooled = []

for pid in ALL_PATIENTS:
    cohort = "2018" if pid in COHORT_2018 else "2020"
    sub = "2018" if pid in COHORT_2018 else "2020"
    train_df = load_patient(DATA_ROOT / sub / "train" / f"{pid}-ws-training.xml")
    test_df  = load_patient(DATA_ROOT / sub / "test"  / f"{pid}-ws-testing.xml")

    splits_rf = make_splits(
        train_df, test_df, CLINICAL_FEATURES,
        horizon_steps=HORIZON, multi_step=True, flat=True,
    )
    splits_dl = make_splits(
        train_df, test_df, CLINICAL_FEATURES,
        horizon_steps=HORIZON, multi_step=True, flat=False,
    )

    glucose_idx = CLINICAL_FEATURES.index("glucose")
    y_mean = float(splits_dl["scaler"].mean_[glucose_idx])
    y_std  = float(splits_dl["scaler"].scale_[glucose_idx])
    y_test_raw = splits_dl["y_test_raw"]

    rf_model = rf.train(splits_rf["X_train"], splits_rf["y_train"], params=rf_params)
    rf_pred  = rf_model.predict(splits_rf["X_test"]) * y_std + y_mean

    ref_pooled.append(y_test_raw[:, STEP_30MIN])
    pred_pooled.append(rf_pred[:, STEP_30MIN])

    print(f"[{pid}] ({cohort}) done — n_test={len(y_test_raw)}")

ref_pooled  = np.concatenate(ref_pooled)
pred_pooled = np.concatenate(pred_pooled)

out_path = RESULTS_DIR / "rf_30min_predictions.npz"
np.savez(out_path, ref=ref_pooled, pred=pred_pooled)
print(f"\nSaved pooled predictions: {out_path}  (n={len(ref_pooled)})")

rmse = float(np.sqrt(np.mean((ref_pooled - pred_pooled) ** 2)))
print(f"RF pooled 30-min RMSE: {rmse:.2f} mg/dL")
