# Blood Glucose Forecasting for Insulin Dose Optimisation

Bachelor project — University of Duisburg-Essen  
*Using Blood Sugar Forecasts to Optimize Insulin Doses*

---

## Project Overview

This repository implements and compares six machine learning models for
predicting blood glucose levels 30 minutes ahead (6 × 5-minute steps),
tested on both Type 1 diabetes cohorts (OhioT1DM) and a general-population
cohort (Glucdict). Accurate short-horizon forecasts enable insulin dose
optimisation systems (closed-loop or advisory) to act before hypo- or
hyperglycaemic episodes occur.

**Models evaluated**

| Model | Type | Input format |
|---|---|---|
| Random Forest | Ensemble (sklearn) | Flat window `(n, seq × feat)` |
| Gradient Boosting | Ensemble (sklearn), clinical sample weighting | Flat window `(n, seq × feat)` |
| LSTM | Recurrent deep learning | `(n, seq, feat)` |
| Seq2Seq Autoencoder | Encoder-decoder LSTM | `(n, seq, feat)` |
| TCN | Dilated causal convolution | `(n, seq, feat)` |
| Transformer | Attention-based deep learning | `(n, seq, feat)` |

CNN-LSTM is also implemented, but only in the BIG IDEAs-specific pipeline
(`src/models/clean_treaining/`) — it has not been ported to the shared
`src/models/` interface listed above.

**Datasets**

| Dataset | Patients | Period | Modalities |
|---|---|---|---|
| OhioT1DM 2018 | 6 | ~8 wk train / ~2 wk test | CGM, bolus, meals, wristband |
| OhioT1DM 2020 | 10 | ~8 wk train / ~2 wk test | CGM, bolus, meals, wristband |
| BigIDEAS | multiple | varies | CGM, meals, insulin, activity |
| Glucdict | 13 of 15 (2 excluded for data quality); source's 12-participant version is general population, mostly non-diabetic | ~10 days, no official split | CGM (Dexcom G6), smartwatch heart rate + accelerometer, activity/meal logging |

---

## Repository Structure

```
├── data/
│   ├── ohio/
│   │   ├── 2018/train/   ← OhioT1DM 2018 XML training files
│   │   ├── 2018/test/    ← OhioT1DM 2018 XML test files
│   │   ├── 2020/train/
│   │   └── 2020/test/
│   └── bigideas/         ← BigIDEAS .parquet files, one per participant
├── src/
│   ├── preprocessing/
│   │   ├── ohio_loader.py      parse OhioT1DM XML → tidy 5-min DataFrame
│   │   ├── bigideas_loader.py  load BigIDEAS Parquet → tidy DataFrame
│   │   └── glucdict_loader.py  load Glucdict CGM + wearable data → tidy DataFrame
│   ├── models/
│   │   ├── random_forest.py     train() / evaluate()
│   │   ├── gradient_boosting.py train() / evaluate(), clinical sample weighting
│   │   ├── lstm.py              GlucoseLSTM, train_model(), evaluate()
│   │   ├── autoencoder.py       GlucoseSeq2Seq, train_model(), evaluate()
│   │   ├── tcn.py               GlucoseTCN, train_model(), evaluate()
│   │   ├── transformer.py       GlucoseTransformer, train_model(), evaluate()
│   │   └── clean_treaining/     BIG IDEAs dataset-specific pipeline (CNN-LSTM
│   │                            and the other models, ported from BIG IDEAs code)
│   ├── evaluation/
│   │   ├── metrics.py          rmse, mae, mape, clarke_zone, clarke_error_grid
│   │   └── plots.py            Clarke EGA, prediction comparison, feature importance
│   └── training/
│       ├── pipeline.py                  create_windows, walk_forward_splits, make_splits
│       ├── grid_search.py               grid_search_rf, grid_search_lstm
│       ├── losses.py                    shared loss functions, incl. clinical sample weighting
│       ├── grid.py                      Clarke Error Grid plotting for BIG IDEAs pipeline results
│       ├── rf_and_gb_pipeline.py        BIG IDEAs RF / Gradient Boosting training pipeline
│       └── lstm_and_cnnlstm_pipeline.py BIG IDEAs LSTM / CNN-LSTM training pipeline
├── experiments/
│   ├── ohio_experiments.ipynb
│   └── bigideas_experiments.ipynb
└── results/
    ├── ohio/       ← saved figures, metric CSVs
    └── bigideas/
```

BIG IDEAs uses its own pipeline (`src/models/clean_treaining/`,
`src/training/rf_and_gb_pipeline.py`, `lstm_and_cnnlstm_pipeline.py`)
instead of the shared `src/models/` interface, because its data format
and preprocessing needs differ enough from the other datasets to make a
shared interface impractical for now — this is a deliberate scoping
decision, not an oversight.

---

## Conventions

- The glucose column is always named **`glucose`** across both datasets so that all pipeline and model code works identically.
- `shuffle=False` everywhere — temporal order must be preserved.
- Event features (`carbs`, `bolus`): NaN filled with 0 (absence of event).
- Sensor features and glucose: rows with NaN are **dropped**, never interpolated.
- Normalisation (`StandardScaler`) is always fit on the **training** portion only and then applied to validation and test.

---

## Installation

```bash
pip install -r requirements.txt
```

Python ≥ 3.10 recommended. GPU support requires a CUDA-compatible PyTorch build.

---

## Data Setup

**OhioT1DM** (requires registration at <http://smarthealth.cs.ohio.edu/OhioT1DM-dataset.html>):
```
data/ohio/2018/train/559-ws-training.xml
data/ohio/2018/test/559-ws-testing.xml
... (repeat for all patients)
```

**BigIDEAS** — place participant `.parquet` files directly in `data/bigideas/`.

---

## Quickstart

```python
from pathlib import Path
from src.preprocessing.ohio_loader import load_patient
from src.training.pipeline import make_splits
from src.models import random_forest as rf

# Load one patient
df_train = load_patient(Path("data/ohio/2018/train/559-ws-training.xml"))
df_test  = load_patient(Path("data/ohio/2018/test/559-ws-testing.xml"))

feature_cols = ["glucose", "carbs", "bolus", "heartrate"]

splits = make_splits(
    df_train, df_test, feature_cols, horizon_steps=6,
    flat=True, multi_step=True,   # flat X for RF; multi-step y to match rf.evaluate
)

# Random Forest baseline
model = rf.train(splits["X_train"], splits["y_train"])

glucose_idx = feature_cols.index("glucose")
y_mean = splits["scaler"].mean_[glucose_idx]
y_std  = splits["scaler"].scale_[glucose_idx]

rmse_val, mae_val, preds = rf.evaluate(
    model, splits["X_test"], splits["y_test_raw"], y_mean, y_std
)
print(f"RMSE {rmse_val:.2f} mg/dL  |  MAE {mae_val:.2f} mg/dL")
```

Walk-forward cross-validation (single-patient, for model selection):
```python
from src.training.pipeline import walk_forward_splits

glucose_idx = feature_cols.index("glucose")

for X_tr, y_tr, X_val, y_val, scaler in walk_forward_splits(
    df_train, feature_cols, horizon_steps=6, n_splits=3,
    flat=True, multi_step=True,
):
    model = rf.train(X_tr, y_tr)

    # walk_forward_splits only yields the scaled y_val, so invert it to
    # mg/dL first — rf.evaluate() expects raw targets, not scaled ones.
    y_mean = scaler.mean_[glucose_idx]
    y_std  = scaler.scale_[glucose_idx]
    y_val_raw = y_val * y_std + y_mean

    score, _, _ = rf.evaluate(model, X_val, y_val_raw, y_mean, y_std)
    print(f"Fold RMSE: {score:.2f} mg/dL")
```

---

## Evaluation

All deep-learning `evaluate()` functions return predictions in the **original mg/dL scale** after inverse-transforming through the training-set scaler:

```python
# glucose is feature index 0 in feature_cols
glucose_idx = feature_cols.index("glucose")
y_mean = scaler.mean_[glucose_idx]
y_std  = scaler.scale_[glucose_idx]

from src.models import lstm
rmse_v, mae_v, preds = lstm.evaluate(model, splits["X_test"], splits["y_test"], y_mean, y_std)
```

Clinical accuracy is measured with the **Clarke Error Grid Analysis** (zones A–E):

```python
from src.evaluation.metrics import clarke_error_grid
from src.evaluation.plots import plot_clarke_error_grid

ceg = clarke_error_grid(splits["y_test"][:, -1], preds[:, -1])
print(ceg["percentages"])   # {'A': 95.2, 'B': 4.1, ...}
plot_clarke_error_grid(splits["y_test"][:, -1], preds[:, -1])
```

---

## References

1. **Cerqueira, V., Torgo, L., & Mozetič, I. (2020).** Evaluating time series
   forecasting models: an empirical study on performance estimation methods.
   *Machine Learning*, 109, 1997–2028.
   — Justification for walk-forward (TimeSeriesSplit) cross-validation.

2. **Bergmeir, C., & Benítez, J. M. (2012).** On the use of cross-validation
   for time series predictor evaluation. *Information Sciences*, 191, 192–213.
   — Demonstrates that blocked CV avoids future leakage for autocorrelated data.

3. **Probst, P., Wright, M. N., & Boulesteix, A.-L. (2019).** Tunability:
   Importance of hyperparameters of machine learning algorithms.
   *Journal of Machine Learning Research*, 20(53), 1–32.
   — Basis for the RF hyperparameter grid (n_estimators, max_depth, min_samples_leaf).

4. **Kingma, D. P., & Ba, J. (2015).** Adam: A method for stochastic
   optimization. *ICLR 2015*.
   — Justification for Adam optimiser and default lr=0.001 for LSTM/TCN/Autoencoder.

5. **Srivastava, N., Mansimov, E., & Salakhudinov, R. (2015).** Unsupervised
   learning of video representations using LSTMs. *ICML 2015*.
   — Architecture basis for the Seq2Seq LSTM Autoencoder.

6. **Clarke, W. L. et al. (2005).** Evaluating clinical accuracy of systems
   for self-monitoring of blood glucose. *Diabetes Care*, 28(5), 1373–1374.
   — Definition of the Clarke Error Grid zones A–E used for clinical evaluation.
