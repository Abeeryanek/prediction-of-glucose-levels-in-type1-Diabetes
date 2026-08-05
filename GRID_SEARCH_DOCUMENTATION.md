# Grid Search Documentation

Blood Glucose Forecasting — University of Duisburg-Essen

## Methodology

- RF: exhaustive grid with TimeSeriesSplit 3-fold cross-validation
- DL models (LSTM, AE, TCN, Transformer): grid search on fixed
  chronological validation split
- Search subject: OhioT1DM patient 559 (representative — complete data,
  good glucose variability)
- Best params applied to all patients (personalised models share the
  same architecture)

## Parameter Grids Tested (with justification)

### Random Forest — 27 combinations

n_estimators [50, 100, 200] → best 200
Probst et al. (2019): "number of trees is the most important hyperparameter"
max_depth [10, 15, 20] → best 10
Biau & Scornet (2016): controls overfitting vs underfitting
min_samples_leaf [1, 2, 4] → best 4
Probst et al. (2019): key parameter after n_estimators

### LSTM — 6 combinations

hidden_size [32, 64, 128] → best 128
Anchored to Kalita & Mirza (2025) who used comparable dimensions
lr [1e-3, 5e-4] → best 5e-4
Kingma & Ba (2015): 1e-3 is Adam default, test one step below

### Autoencoder — 6 combinations

latent_size [16, 32, 64] → best 16
Srivastava et al. (2015): bottleneck compression ratio
lr [1e-3, 5e-4] → best 5e-4

### TCN — 6 combinations

num_filters [32, 64, 128] → best 32
Bai et al. (2018): convolutional capacity
lr [1e-3, 5e-4] → best 5e-4

### Transformer — 6 combinations

d_model [32, 64, 128] → best 128, nhead fixed at 4
Vaswani et al. (2017): embedding dimension, must be divisible by nhead
lr [1e-3, 5e-4] → best 5e-4

Total: 51 combinations explored.

## Justification for Parameter Reuse Across Datasets

Our transfer learning experiment (OhioT1DM → Glucdict, zero-shot) showed
only ~2 mg/dL RMSE degradation despite different populations and CGM
devices. This demonstrates glucose dynamics share a common structure
across datasets, making hyperparameter reuse reasonable. Per-dataset
grid search is noted as a possible refinement but is unlikely to
change the model ranking given the demonstrated transfer robustness.

## Known Limitation

Grid search was performed once on a single representative patient rather
than per-patient, and DL models used a fixed validation split rather than
full cross-validation (chosen for computational feasibility — each DL
training takes minutes). This is documented as a scope decision.
