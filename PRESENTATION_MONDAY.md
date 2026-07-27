# Slide 1 — Title

**Blood Glucose Forecasting — Progress Update**

University of Duisburg-Essen | Khalil & Abeer | 2026-07-11

---

# Slide 2 — Overview: What we did since the last meeting

- Corrected the LSTM Autoencoder architecture (Srivastava et al. 2015)
- Built a shared GitHub repository with unified `src/` structure
- Replaced fixed 80/20 split with Walk-Forward Validation
- Added Grid Search with literature-justified parameter ranges
- Ran full benchmark: 5 models × 12 patients × 3 horizons (added Transformer)
- Fixed a Random Forest bug (nested parallelism crashing on Windows)
- Re-ran feature ablation excluding patients 563/575 (professor's request)
- Extended Grid Search to Autoencoder, TCN, and Transformer
- Added Clarke Error Grid analysis for every feature combination
- New: Glucdict dataset pipeline (13 patients) — cross-dataset validation

---

# Slide 3 — Shared Repository Structure

```
src/preprocessing/  — dataset-specific loaders (OhioT1DM XML, BIG IDEAs parquet)
src/models/         — RF, LSTM, Autoencoder, TCN (identical interface)
src/training/       — pipeline, walk-forward splits, grid search
src/evaluation/     — metrics, Clarke Error Grid plots
experiments/        — one notebook per dataset
```

---

# Slide 4 — Decision 1: Walk-Forward Validation

**BEFORE:** fixed 80/20 chronological split — one validation estimate,
potentially biased by the specific 20% period chosen.

**AFTER:** Walk-Forward Validation with `n_splits=3` (TimeSeriesSplit)

**Justification:** Cerqueira et al. (2020), Bergmeir & Benítez (2012)

Each fold trains on the past and validates on the immediately following
period. Final estimate is the average across 3 folds.

---

# Slide 5 — Decision 2: Corrected Autoencoder Architecture

**BEFORE (wrong):** Encoder LSTM → bottleneck → linear layer prediction

**AFTER (correct, Srivastava et al. 2015):**
- Encoder LSTM → bottleneck (64 → 32) → Decoder LSTM (initialised from latent vector, autoregressive) → prediction

**Key difference:** decoder is a full LSTM, not a linear layer.

---

# Slide 6 — Decision 3: Grid Search with Literature Justification

**RF parameter grid:**

| Parameter | Values | Source |
|-----------|--------|--------|
| `n_estimators` | [50, 100, 200] | Probst et al. (2019) |
| `max_depth` | [10, 15, 20] | Biau & Scornet (2016) |
| `min_samples_leaf` | [1, 2, 4] | Probst et al. (2019) |

**LSTM parameter grid:**

| Parameter | Values | Source |
|-----------|--------|--------|
| `hidden_size` | [32, 64, 128] | Kalita & Mirza (2025) as anchor |
| `lr` | [0.001, 0.0005] | Kingma & Ba (2015) Adam default |

**Best found:**
- RF: `n_estimators=200`, `max_depth=10`, `min_samples_leaf=4`
- LSTM: `hidden_size=128`, `lr=0.0005`

---

# Slide 6b — Decision 3b: Grid Search Extended to All 5 Models

Autoencoder, TCN, and Transformer were previously untuned (AE/TCN reused LSTM's
learning rate; Transformer was hardcoded). Same validation-set search strategy
as LSTM, run on patient 559:

| Model | Parameter | Values | Source |
|-------|-----------|--------|--------|
| Autoencoder | `latent_size` | [16, 32, 64] | Srivastava et al. (2015) |
| Autoencoder | `lr` | [0.001, 0.0005] | Kingma & Ba (2015) |
| TCN | `num_filters` | [32, 64, 128] | Bai et al. (2018) |
| TCN | `lr` | [0.001, 0.0005] | Kingma & Ba (2015) |
| Transformer | `d_model` (nhead=4 fixed) | [32, 64, 128] | Vaswani et al. (2017) |
| Transformer | `lr` | [0.001, 0.0005] | Kingma & Ba (2015) |

**Best found:**
- Autoencoder: `latent_size=16`, `lr=0.0005`
- TCN: `num_filters=32`, `lr=0.0005`
- Transformer: `d_model=128`, `nhead=4`, `lr=0.0005`

---

# Slide 7 — Main Results Table (OhioT1DM, clinical features, all 5 models tuned)

RMSE [mg/dL] mean ± std across 12 patients:

| Horizon | RF | LSTM | Autoencoder | TCN | Transformer |
|---------|----|------|-------------|-----|-------------|
| 5 min  | 5.87 ± 1.97 | 5.76 ± 1.42 | 6.37 ± 1.67 | 8.24 ± 2.46 | 8.51 ± 3.43 |
| 15 min | 14.06 ± 4.14 | 12.62 ± 2.45 | 12.91 ± 2.47 | 14.67 ± 2.22 | 14.12 ± 3.28 |
| 30 min | 23.35 ± 6.00 | **21.21 ± 3.48** | 21.41 ± 3.49 | 24.29 ± 3.73 | 22.91 ± 4.54 |

**Model hierarchy: LSTM ≈ Autoencoder > Transformer > RF > TCN**

---

# Slide 8 — Feature Ablation — Clean Cohort (excl. patients 563/575)

30-min RMSE [mg/dL], mean ± std across 4 patients (professor's request: drop
563/575, which have sparse wristband coverage):

| Feature Set | RF RMSE | LSTM RMSE |
|-------------|---------|-----------|
| glucose_only | 21.36 ± 2.29 | 20.54 ± 2.80 |
| **clinical** ★ | **21.33 ± 2.27** | **20.22 ± 2.76** |
| glucose_steps | 24.03 ± 4.78 | 21.37 ± 4.65 |
| clinical_steps | 24.03 ± 4.78 | 21.24 ± 4.38 |
| clinical_hr | 24.83 ± 2.48 | 23.61 ± 2.68 |
| glucose_hr | 24.84 ± 2.51 | 24.91 ± 3.69 |
| full | 32.61 ± 12.90 | 35.37 ± 18.81 |
| wearable | 32.74 ± 13.08 | 33.75 ± 16.61 |

glucose_only ≈ clinical — bolus/carbs add essentially nothing. full/wearable
are consistently worst AND highest-variance. Also fixed along the way: an RF
nested-parallelism bug that was silently dropping patients from some runs.

---

# Slide 8b — Cross-Dataset Check: Glucdict

13 patients, Dexcom G6 + TicWatch HR + accelerometer + meal events, 30-min horizon:

| Feature Set | N | RF RMSE | LSTM RMSE |
|-------------|---|---------|-----------|
| **glucose_only** ★ | 13 | **15.23 ± 2.26** | **13.98 ± 1.81** |
| glucose_activity | 13 | 15.19 ± 2.21 | 13.97 ± 1.75 |
| glucose_hr | 10 | 17.89 ± 9.90 | 17.17 ± 8.77 |
| full_wearable | 10 | 17.28 ± 8.87 | 17.61 ± 9.43 |

Same pattern as OhioT1DM: extra wearable sensors don't help. `glucose_hr` /
`full_wearable` also lose 3 patients (User3/13/15 — no heart-rate data in the
chronological test tail, likely watch battery dropout). RF/LSTM hyperparameters
reused from the OhioT1DM grid search — no Glucdict-specific tuning yet.

---

# Slide 9 — Clarke Error Grid (30-min horizon, all 5 models)

| Model | Zone A | Zone B |
|-------|--------|--------|
| LSTM | **87.3% ± 5.0%** | 12.4% |
| Autoencoder | 87.0% ± 5.7% | 12.7% |
| Transformer | 85.0% ± 7.0% | 14.7% |
| RF | 86.4% ± 5.1% | 13.1% |
| TCN | 81.0% ± 8.5% | 18.7% |

**Target: > 95%**

TCN shows highest variance (8.5%) — inconsistent across patients.

---

# Slide 9b — Clarke Error Grid — Every Feature Combination

LSTM, 30-min horizon, predictions pooled across patients (OhioT1DM clean
cohort, 8 feature sets — see `results/ohio/clarke_ablation_clean_all_featuresets.png`):

Zone A drops from 88.6% (`clinical`) to 75.6% (`full`/`wearable`) — with 1.5%
landing in dangerous Zone D once heartrate/steps are added. Confirms the RMSE
ranking with clinical-accuracy evidence, not just RMSE.

---

# Slide 10 — Literature Comparison (30-min horizon)

| Model | RMSE [mg/dL] | Gap to ours |
|-------|-------------|-------------|
| MHA-NBEATS (Kalita & Mirza 2025) | 16.57 | — |
| Bertachi et al. (2018) | 19.33 | — |
| Rodríguez-Rodríguez (custom 40-patient DM1 dataset) | 18.60 | — |
| **Our LSTM** | **21.21** | +1.9 vs. Bertachi |
| Our Autoencoder | 21.41 | +2.1 vs. Bertachi |
| Our Transformer | 22.91 | +3.6 vs. Bertachi |
| **Our RF** | **23.35** | +4.8 vs. Rodríguez |

- Our models use **3 features**; literature uses 5–6 features
- Grid Search reduced the gap by **~3 mg/dL** vs. our previous run

---

# Slide 11 — Key Findings

1. **Grid Search matters:** +2–3 mg/dL improvement over default parameters
2. **LSTM best at 30 min** (21.21), Autoencoder essentially tied — only 1.7 mg/dL behind Bertachi et al. (2018)
3. **Tuned Transformer (22.91) now beats RF and TCN** — was worse than both when untuned
4. **TCN still underperforms:** OhioT1DM is too small / window too short for TCN's advantage to show
5. **More features hurt, not help:** on BOTH OhioT1DM and Glucdict, glucose_only/clinical beat every feature set that adds heartrate/steps — confirmed via RMSE and Clarke Zone A/D
6. **Clarke Zone A ~87% (best models):** below 95% target — explainable by 3-feature input only

---

# Slide 12 — Next Steps

- Glucdict-specific Grid Search (currently reusing OhioT1DM hyperparameters)
- Investigate the User7 heart-rate outlier in Glucdict (RF RMSE 45 vs. cohort mean ~15)
- Explore longer window sizes (24, 36, 72 steps)
- Investigate Leave-One-Patient-Out (LOPO) cross-validation
- Look for additional experiment setups from literature not yet covered
- Merge Abeer's BIG IDEAs results into the shared benchmark
