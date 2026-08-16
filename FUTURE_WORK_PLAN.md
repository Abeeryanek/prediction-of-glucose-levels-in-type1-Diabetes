# Future Work Plan
Blood Glucose Forecasting — University of Duisburg-Essen
Khalil & Abeer | July 2026

## What We Have Achieved

- 5 models benchmarked on OhioT1DM and Glucdict
  (RF, LSTM, Autoencoder, TCN, Transformer)
- All 5 models on BIG IDEAs (Abeer)
- Grid Search tuned for all 5 models
- Feature ablation across 3 datasets and both cohorts
- Leave-One-Patient-Out cross-validation
- Cross-dataset transfer learning (OhioT1DM → Glucdict)
- Unified Clarke Error Grid implementation

## Key Findings So Far

1. Wearable features consistently degrade 30-min prediction
   across all 3 datasets
2. Personalisation barely matters — LOPO gap only +1.17 mg/dL
3. Transfer across populations costs only ~2 mg/dL
4. Autoencoder is best or joint-best on both datasets
5. TCN is consistently weakest — short window limits
   dilated convolution advantage

## Remaining Experiments

### Must Complete
- Longer window sizes: test 24, 36, 72 steps (2h, 3h, 6h)
  — especially important for TCN which needs longer sequences
- Additional horizons: 45-min and 60-min
- Interpolation experiment: test bounded linear interpolation
  (gap ≤ 15 min) instead of row dropping — Zhu et al. (2018)
  showed this gives ~2.1 mg/dL improvement
- Unified Clarke Error Grid across all 3 datasets

### Important for Completeness
- LOPO on Glucdict — test whether population generalisation
  holds on prediabetic data
- Patient 540 sensitivity analysis — report results with and
  without this outlier to quantify its effect on mean RMSE
- Extended Grid Search including window size as a parameter
- CNN-LSTM currently runs in the BIG IDEAs pipeline
  (`src/models/clean_treaining/`). Porting it into the shared
  `src/models/` interface is planned future work, not yet done.

## Known Limitations (Not Yet Implemented)

- Interpolation experiments (glucose-gap interpolation pre-test and full
  matrix) are not implemented. The current pipeline drops NaN rows
  rather than interpolating glucose gaps. Planned future work.
- Gradient Boosting exists as a model module
  (`src/models/gradient_boosting.py`) but is not yet wired into the
  main run scripts; it currently runs only in the BIG IDEAs pipeline.
- CNN-LSTM runs in the BIG IDEAs pipeline (`clean_treaining/`) and is
  not yet ported to the shared `src/models/` interface.
- Glucdict experiments currently run at the 30-minute horizon only;
  the 15/45-minute sweep implemented for OhioT1DM is not yet applied
  to Glucdict.

## What We Aim to Achieve

A reproducible multi-model, multi-dataset benchmark covering
the full range of modern ML architectures across three clinical
populations. The main contribution is identifying which models,
feature sets, and training strategies generalise across datasets
and patient populations — with honest evaluation of both
predictive accuracy (RMSE, MAE) and clinical safety
(Clarke Error Grid).

The key open question we want to resolve: does a longer lookback
window (2h–6h) make TCN competitive, and does interpolation of
missing sensor data close the remaining gap to the literature?
