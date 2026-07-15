# Detailed Final Experiments Plan

Blood Glucose Forecasting — University of Duisburg-Essen
Khalil & Abeer | 17 July 2026

## 1. Purpose and Scope

The previous plan (`FINAL_EXPERIMENTS_PLAN.md`) listed experiments by name and
dataset but did not specify exact training conditions, exact hyperparameter
values, or the literature reasoning behind them — this is why it was
rejected as too superficial. This document replaces it. Every remaining
experiment below states the exact training configuration, the exact
parameter values or grid, the exact data subset it runs on, and the
reasoning (usually a citation) behind each choice, so that implementation
can begin immediately upon approval. Where our current code or a prior
presentation slide diverges from what is stated here, the divergence is
called out explicitly rather than silently corrected, since some of these
are deliberate proposed changes that still need sign-off.

## 2. Standard Training Conditions (applies to ALL experiments unless stated)

| Setting | Value | Reasoning |
|---|---|---|
| Optimizer | Adam | Kingma & Ba (2015) |
| Loss function | MSE (`nn.MSELoss`) | Standard regression loss for continuous glucose target |
| max_epochs | 150 | See note below — current code default is 100 |
| Early stopping patience | 15 epochs | Prechelt (1998); current code default is 10 |
| Early stopping monitor | Validation loss, restore best weights | Prevents overfitting past the generalisation minimum |
| Batch size | 32 | See note below — current code default is 64 |
| Shuffle | False | Temporal order must be preserved for autocorrelated CGM series |
| Train/val split | Last 20% of training data, chronological (`val_ratio=0.20`) | Matches `make_splits` in `src/training/pipeline.py`; avoids future leakage |
| Normalisation | `StandardScaler`, fit on training data only | Prevents test/val leakage into scaler statistics |
| Random seed | 42 | See note below — currently only set for Random Forest |
| Hardware | NVIDIA GPU (CUDA) if available, else CPU (`torch.device("cuda" if torch.cuda.is_available() else "cpu")`) | All prior runs (`run_log_epochs.txt`) executed on CUDA |

**Three code changes are required before Sections 4–10 can run under these
conditions, and should be confirmed with the supervisor before implementation:**

1. **`max_epochs` 100 → 150, `patience` 10 → 15.** All four DL models
   (`src/models/lstm.py`, `autoencoder.py`, `tcn.py`, `transformer.py`)
   currently default to `max_epochs=100, patience=10`. The Autoencoder hit
   the 100-epoch cap on 3 of 12 patients in the completed run (570, 575,
   588 — see table below), which means its training was capped rather than
   converged for those patients. Raising the cap to 150 with patience 15
   gives it (and any model near the cap) room to actually converge instead
   of being cut off arbitrarily.
2. **`batch_size` 64 → 32.** Currently hard-coded in each model's
   `_make_loader()` helper. Smaller batches increase gradient noise, which
   can help generalisation on the relatively small per-patient datasets
   (~12,000–14,000 training rows) used here; this also aligns compute cost
   more closely across models before the grid-search expansion in §3.6.
3. **Add `torch.manual_seed(42)` (+ `np.random.seed(42)`) at the start of
   each DL training run.** Currently `random_state=42` is set only for
   Random Forest (`src/models/random_forest.py`); none of the four PyTorch
   models fix a seed, so weight initialisation and (once batch size drops
   to 32) minibatch composition are not reproducible run-to-run. This must
   be fixed before Sections 4–10 are run, since reproducibility will be
   questioned at the defense.

**Observed epoch counts from the completed run** (`results/ohio/run_log_epochs.txt`,
12 patients, current 100-epoch cap, patience 10 — will be re-measured once
the cap changes to 150/15):

| Model | Mean ± SD | Range | Notes |
|---|---|---|---|
| LSTM | 54.5 ± 15.8 | 32–76 | Never hit the cap |
| Autoencoder | 72.4 ± 24.3 | 36–100 | Hit the 100-epoch cap on patients 570, 575, 588 (3/12) |
| TCN | 51.6 ± 19.6 | 27–79 | Never hit the cap |
| Transformer | 42.1 ± 24.3 | 16–97 | Never hit the cap |

## 3. Grid Search — Full Documentation (all models, all datasets)

The grid search (`src/training/grid_search.py`) was run once, on OhioT1DM
patient 559 (2018 cohort) only, using `TimeSeriesSplit` walk-forward
validation. The resulting best parameters (`results/ohio/grid_search_results.json`)
were then applied globally to every patient and every dataset. This is a
known limitation, addressed in §3.6.

### 3.1 Random Forest

| Parameter | Values tested | Best found | Why this range |
|---|---|---|---|
| n_estimators | [50, 100, 200] | 200 | Probst et al. (2019): "number of trees is the most important hyperparameter" for RF variance reduction |
| max_depth | [10, 15, 20] | 10 | Biau & Scornet (2016): shallower trees reduce variance on datasets of this size (~12k rows/patient) |
| min_samples_leaf | [1, 2, 4] | 4 | Probst et al. (2019): leaf-size regularisation is the second most impactful RF hyperparameter |

### 3.2 LSTM

| Parameter | Values tested | Best found | Why this range |
|---|---|---|---|
| hidden_size | [32, 64, 128] | 128 | Anchored to Kalita & Mirza (2025) |
| lr | [1e-3, 5e-4] | 5e-4 | Kingma & Ba (2015) default is 1e-3; halved value tested as a lower-variance alternative for small per-patient datasets |

### 3.3 Autoencoder (Seq2Seq)

| Parameter | Values tested | Best found | Why this range |
|---|---|---|---|
| latent_size | [16, 32, 64] | 16 | Srivastava et al. (2015): a tight bottleneck forces the encoder to retain only the dominant glucose trend, reducing overfitting on ~12k-row per-patient data |
| lr | [1e-3, 5e-4] | 5e-4 | Same reasoning as §3.2 |

### 3.4 TCN

| Parameter | Values tested | Best found | Why this range |
|---|---|---|---|
| num_filters | [32, 64, 128] | 32 | Bai et al. (2018): fewer filters per layer with dilation depth doing the representational work, rather than channel width |
| lr | [1e-3, 5e-4] | 5e-4 | Same reasoning as §3.2 |

### 3.5 Transformer

| Parameter | Values tested | Best found | Why this range |
|---|---|---|---|
| d_model | [32, 64, 128] | 128 | Vaswani et al. (2017); must be divisible by `nhead=4` (fixed) |
| lr | [1e-3, 5e-4] | 5e-4 | Same reasoning as §3.2 |

### 3.6 Planned Grid Search Expansion

Not yet covered by the existing grid search, to be added:

- **`window_size` as a grid parameter**: [12, 24, 36, 72] steps (1h/2h/3h/6h)
  — currently fixed at 12 and never searched jointly with model
  hyperparameters (see §5, which runs window size as a standalone sweep at
  the *current* best hyperparameters; a joint search is a stretch goal if
  time permits).
- **`num_layers`** for LSTM/Transformer: [1, 2, 3] — currently hard-coded
  to a single layer in both models; not yet exposed as a parameter.
- **`nhead`** for Transformer: [2, 4, 8] — currently fixed at 4; must stay
  a divisor of whichever `d_model` is tested (so `d_model=32` excludes
  `nhead=8`).
- **`dropout`**: [0.1, 0.2, 0.3] — not currently implemented in any of the
  four DL models; needs to be added as a constructor argument first.
- **Grid Search on all 3 datasets** (currently OhioT1DM patient 559 only):
  run on one representative patient per dataset — OhioT1DM (559, already
  done), Glucdict (User5 — median-performing user by RF RMSE, 11.4 mg/dL),
  BIG IDEAs (one representative patient from `data/bigideas/`, TBD by
  Abeer). Reusing OhioT1DM's best params for Glucdict/BIG IDEAs, as we do
  today, is not justified without checking this — different CGM devices,
  sampling behaviour, and feature sets could shift the optimum.

## 4. Multi-Horizon Analysis (15, 30, 45 min — NOT only 30)

All experiments in Sections 5–9 will report **15-min (3 steps), 30-min (6
steps), and 45-min (9 steps)** horizons. The shared OhioT1DM/Glucdict
pipeline (`src/training/pipeline.py`) currently predicts a single fixed
`horizon=6` (30 min); it will be extended to also run `horizon=9` for the
45-min report, reusing the already-implemented multi-step target machinery
(`multi_step=True` path in `create_windows`).

**Horizon degradation already measured on OhioT1DM** (`results/ohio/run_log_epochs.txt`,
12 patients, mean RMSE, mg/dL):

| Model | 5 min | 15 min | 30 min | Growth 5→30 min |
|---|---|---|---|---|
| RF | 5.87 | 14.06 | 23.35 | +17.48 |
| LSTM | 6.11 | 12.70 | 21.22 | +15.11 |
| Autoencoder | 6.39 | 13.06 | 21.59 | +15.20 |
| TCN | 8.28 | 15.20 | 24.70 | +16.42 |
| Transformer | 8.15 | 13.91 | 22.58 | +14.43 |

This is a steep, roughly linear degradation — about +0.6–0.7 mg/dL RMSE per
additional minute of lead time, consistent with the literature (`FINAL_EXPERIMENTS_PLAN.md`
§5 already flagged this expectation).

**Abeer's BIG IDEAs finding is confirmed with numbers, not just an
impression.** The BIG IDEAs walk-forward benchmark (`experiments/results/bigideas/summary_metrics_walk_forward.csv`,
5-fold walk-forward CV, pooled/population model, `full` feature subset,
mean RMSE, mg/dL):

| Model | 15 min | 30 min | 45 min | Growth 15→45 min |
|---|---|---|---|---|
| RF | 20.20 | 20.17 | 20.52 | +0.32 |
| GB | 20.76 | 20.83 | 21.50 | +0.74 |
| LSTM | 22.75 | 22.33 | 23.73 | +0.98 |
| CNN-LSTM | 22.48 | 22.56 | 23.28 | +0.80 |
| Transformer | 23.65 | 23.01 | 24.18 | +0.53 |

Compare +0.3 to +1.0 mg/dL total growth over a 30-minute lead-time increase
on BIG IDEAs against +14 to +17 mg/dL over the same 25-minute increase on
OhioT1DM — roughly a **20–40× smaller horizon effect**. This needs
investigation, and both of the proposed hypotheses have partial support in
data already collected:

- **H1 (BIG IDEAs glucose is less volatile / prediabetic population)** —
  plausible but not yet directly tested; needs a glucose-variability
  comparison (e.g. patient-level SD or CV of glucose) between BIG IDEAs and
  OhioT1DM cohorts, not yet computed.
- **H2 (multi-output head predicts all steps from the same hidden state,
  so later steps are under-differentiated)** — partial evidence against a
  pure-volatility explanation: the BIG IDEAs R² values in the same CSV are
  weak-to-negative across the board (RF `full`: 0.19/0.19/0.16; LSTM
  `full`: −0.03/0.00/−0.11; Transformer `full`: −0.12/−0.07/−0.18). Near-zero
  or negative R² suggests the models are close to a persistence-like
  baseline at every horizon rather than genuinely differentiating longer
  lead times — this favours H2 or a third explanation (weak overall
  learnable signal in this feature set) over H1 alone.

**Planned experiment**: compute the per-horizon RMSE degradation slope
(mg/dL per minute of lead time) for every (model, dataset) pair once
OhioT1DM and Glucdict also have 45-min results from the shared pipeline,
and pair each with a patient-level glucose-SD figure per dataset to
directly test H1 vs. H2.

**Known caveat**: BIG IDEAs currently uses a different model set (RF, GB,
LSTM, CNN-LSTM, Transformer via Abeer's `experiments/` pipeline) than the
shared OhioT1DM/Glucdict pipeline (RF, LSTM, Autoencoder, TCN,
Transformer). This mismatch (GB/CNN-LSTM vs. Autoencoder/TCN) must be
resolved — either by running the shared 5-model set on BIG IDEAs or by
explicitly scoping the cross-dataset comparison to the 3 models common to
both (RF, LSTM, Transformer) — before any 3-dataset, 5-model claim can be
made in the final report.

## 5. Window Size Experiments

- **Window sizes to test**: 12 (1h, current default), 24 (2h), 36 (3h), 72
  (6h) steps.
- **Condition**: each window size tested at all 3 horizons (§4), all 5
  models, using the grid-search-best hyperparameters from §3 (not a joint
  search — see §3.6 for the joint-search stretch goal).
- **Rationale**: Bai et al. (2018) — TCN's dilated-convolution receptive
  field is designed to exploit long input sequences; our current 12-step
  window may be starving it of the context it needs.
- **Motivating evidence already in hand** (`results/ohio/run_log_epochs.txt`,
  30-min horizon, mean RMSE): TCN is the weakest model at 24.70 mg/dL,
  clearly behind LSTM (21.22) and Autoencoder (21.59) — a 3–4 mg/dL gap
  that the window-size hypothesis is meant to explain.
- **Hypothesis**: TCN improves the most as window size increases; LSTM
  saturates early (diminishing returns beyond ~24–36 steps) since its
  recurrent state already summarises long history, just less efficiently
  than TCN's dilated convolutions would if given enough input.
- **Metric**: RMSE per (model, window, horizon), plus Clarke Zone A % at
  the 30-min horizon for the clinical-safety angle.

## 6. Preprocessing Comparison and Planned Changes

Summary of the existing comparison (`PREPROCESSING_COMPARISON.md`, full
detail and citations there): of 12 preprocessing decisions compared against
the OhioT1DM literature (Marling & Bunescu 2020; Martinsson et al. 2020;
Zhu et al. 2018; Xie & Wang 2020), **10 match the dominant literature
convention** (5-min resampling, event zero-fill, sentinel handling,
train-only scaling, per-patient models, 60-min window, 30-min horizon,
official train/test split, walk-forward CV, no extra clipping/smoothing).
Two decisions are deliberate departures:

| Preprocessing step | Current | Literature | Assessment |
|---|---|---|---|
| Missing CGM values | Dropped | Split: Martinsson et al. also drop; Zhu et al. interpolate (linear), −2.1 mg/dL RMSE vs. no imputation | One literature branch matches ours; the other (interpolation) is untested here |
| Missing wearable sensor rows | Dropped, never interpolated | Where addressed, usually linearly interpolated with a bounded gap limit | Stricter than most published pipelines; costs sample count on `full`/`wearable` feature sets |

### Planned Preprocessing Changes

| Discrepancy | Current | Literature | Planned change | Expected effect |
|---|---|---|---|---|
| Missing sensor rows | Drop | Interpolate (bounded) | Test bounded linear interpolation, max gap 30 min (6 steps) | Reduce data loss on `full`/`wearable` feature sets, which currently lose the most rows (e.g. clean-cohort `full`/`wearable`: n=1,796 vs. `glucose_only`: n=10,742 — an 83% row loss) |
| Missing glucose | Drop | Mixed (Martinsson: drop; Zhu: interpolate) | Test linear interpolation for gaps ≤ 3 steps (15 min) only — never fabricate glucose across longer gaps | More training data without contaminating ground truth over clinically meaningful gaps |
| CGM smoothing | None | Median filter (used in some imputation-focused follow-up work) | Test median filter, window=5 (25 min) | Reduce sensor noise; test whether this recovers some of the RMSE gap to literature (H4, §10) |

## 7. Interpolation Experiments (multiple combinations)

Full matrix — interpolation method × max gap length × filtering:

- **Methods**: none (drop, current baseline), linear, spline (cubic),
  forward-fill.
- **Max gap**: 3 steps (15 min), 6 steps (30 min), 12 steps (60 min).
- **Filtering**: with median filter (window=5) vs. without.
- **Total configurations**: 4 methods × 3 gap lengths × 2 filter settings =
  24 configs. ("none" only has one meaningful config since gap length and
  filtering don't apply when nothing is imputed — reduces to 1 + 3×3×2 = 19
  distinct runs in practice, not 24; the 24-config framing is the full
  factorial before removing that redundancy.)
- **Run order**: OhioT1DM patient 559 first (fast iteration), then the
  full 12-patient cohort for the single best-performing configuration.
- **Metric**: RMSE at the 30-min horizon, plus Clarke Zone A % for the
  best configuration.
- **Citation**: Zhu, Li, Herrero & Georgiou (2018), *A Deep Learning
  Algorithm for Personalized Blood Glucose Prediction*, KDH@IJCAI-ECAI —
  reports −2.1 mg/dL RMSE from linear interpolation vs. no imputation. This
  is the full, disambiguated citation (there are other 2018 glucose-forecasting
  papers with a "Zhu et al." author list); use it verbatim in the final
  report rather than the shorthand "Zhu et al. (2018)" used in earlier
  presentation slides, to avoid ambiguity.

## 8. Feature Ablation — Complete Matrix

### 8.1 OhioT1DM 2018 (heart rate + steps available)

Clean cohort only (patients 559, 570, 588, 591 — 563 and 575 excluded for
sparse wristband coverage, per `FINAL_EXPERIMENTS_PLAN.md` §2). Mean RMSE
across the 4 patients, 30-min horizon (`results/ohio/results_feature_ablation_clean.csv`):

| Feature set | n (windows, pooled) | RF RMSE | LSTM RMSE | Clarke Zone A % |
|---|---|---|---|---|
| glucose_only | 10,742 | 21.36 | 20.54 | 88.51 |
| clinical (+bolus, carbs) | 10,742 | 21.33 | 20.22 | 88.56 |
| glucose_steps | 1,883 | 24.03 | 21.37 | 85.61 |
| clinical_steps | 1,883 | 24.03 | 21.24 | 86.83 |
| glucose_hr | 3,027 | 24.84 | 24.91 | 85.17 |
| clinical_hr | 3,027 | 24.83 | 23.61 | 85.93 |
| wearable (hr + steps, no clinical) | 1,796 | 32.74 | 33.75 | 75.61 |
| full (clinical + hr + steps) | 1,796 | 32.61 | 35.37 | 75.61 |

**Data-quality note**: `full` and `wearable` show identical Clarke Zone
percentages (75.61247216035635 to 15 decimal places) despite different RF
RMSE (32.61 vs. 32.74) — this needs to be checked before the final report;
either it's a coincidence of an identical row subset (both feature sets
drop to the same n=1,796 windows once hr/steps NaNs are dropped, since
`full` = `wearable` + always-present clinical columns) or a pipeline bug
that reused the same predictions for both Clarke computations.

Planned additions:
- **clinical_hr-only vs. clinical_steps-only** to isolate which wearable
  signal (heartrate vs. steps) drives the degradation — currently only the
  combined `wearable`/`full` sets and the individual `glucose_hr`/`glucose_steps`
  sets exist; the isolation is between `clinical_hr` and `clinical_steps`,
  already present above, but not yet discussed as an isolated comparison in
  any report (clinical_hr RF=24.83 vs. clinical_steps RF=24.03 — heartrate
  hurts slightly more than steps).
- **time-of-day feature** (sin/cos-encoded hour, or hour-of-day bucket):
  not yet implemented in `src/preprocessing/`; test whether circadian
  information recovers some of the accuracy the wearable features cost.

### 8.2 OhioT1DM 2020 (acceleration available)

All 6 patients (540, 544, 552, 567, 584, 596), 30-min horizon
(`results/ohio/results_ablation_2020.csv`):

| Feature set | RF RMSE | LSTM RMSE |
|---|---|---|
| glucose_only | 25.01 | 22.29 |
| clinical | 25.00 | 22.04 |
| glucose_accel | 30.57 | 28.98 |
| clinical_accel | 30.55 | 28.85 |

Confirms the same pattern as §8.1 with a third, independent wearable
signal (accelerometer instead of Basis wristband hr/steps) — accelerometer
degrades RMSE by ~5.5–7 mg/dL, comparable in size to the hr/steps
degradation on the 2018 cohort.

### 8.3 Glucdict (heartrate + acc_magnitude + eat/drink events)

13 users, 30-min horizon (`results/glucdict/results_glucdict_ablation.csv`):

| Feature set | n users | RF RMSE | LSTM RMSE | Clarke Zone A % |
|---|---|---|---|---|
| glucose_only | 13 | 15.23 | 14.04 | 90.87 |
| glucose_activity (acc_magnitude) | 13 | 15.19 | 13.86 | 91.05 |
| glucose_hr | 10 | 17.89 | 17.26 | 88.12 |
| full_wearable | 10 | 17.28 | 17.49 | 87.99 |

Note the third-dataset confirmation is *partial*: `glucose_activity` is
essentially tied with `glucose_only` (15.19 vs. 15.23 RF, 13.86 vs. 14.04
LSTM) — accelerometer-derived activity does not clearly hurt on Glucdict,
unlike heartrate, which does (17.89/17.26). This is a genuine
inconsistency with the "all wearable features hurt" narrative from
`SESSION_NOTES.md` and needs to be stated carefully rather than
overclaimed: the finding should be qualified as "heartrate consistently
hurts across all 3 datasets; accelerometer/steps hurt on OhioT1DM but not
clearly on Glucdict."

Planned addition:
- **eat_event/drink_event vs. glucose alone**: does adding logged
  eat/drink events (currently zero-filled, present in the `full_wearable`
  set but never tested in isolation) add value over `glucose_only`? Needs a
  new `glucose_events` feature set (glucose + eat_event + drink_event, no
  hr/activity).

### 8.4 BIG IDEAs (28 features)

Not yet run through the shared ablation protocol — Abeer's
`experiments/results/bigideas/` pipeline currently reports `comparable` vs.
`full` subsets (not a full 8-combination ablation matching §8.1–8.3). At
the `full` 30-min horizon (`experiments/results/bigideas/summary_metrics_walk_forward.csv`):
RF RMSE 20.17, LSTM RMSE 22.33 — both noticeably worse than either OhioT1DM
or Glucdict `glucose_only`, despite 28 available features, which is itself
worth stating as a finding (more features ≠ better here either).

Planned: nutritional rolling-window feature subsets (Abeer) — e.g.
carbs/calories summed over the last 1h/3h/6h — to test whether *aggregated*
nutritional signal helps where raw wearable channels do not.

## 9. LOPO Extension (all 5 models, not just RF+LSTM)

The completed LOPO run (`results/ohio/results_lopo.csv`) covers only RF and
LSTM, on all 12 OhioT1DM patients as held-out test folds. This will be
extended to Autoencoder, TCN, and Transformer, reusing the same fold
structure.

- **Number of folds**: 12 (one per OhioT1DM patient, both cohorts pooled).
- **Training condition per fold**: pooled training data from the other 11
  patients (chronological per-patient windows, concatenated; scaler
  refit per fold on the pooled training set only). Total pooled rows
  across all 12 patients ≈153,000 (`run_log_epochs.txt` §2 loader table);
  each fold therefore trains on roughly 139,000–142,000 rows depending on
  which patient is held out (exact count = 153,046 minus the excluded
  patient's row count).
- **Epochs per fold**: not currently persisted. `run_lopo.py` computes and
  prints `lstm_epochs` per fold to the console (line 174/201) but does not
  write it to `results_lopo.csv` — this needs a one-line fix (append an
  `epochs` column, matching the convention already used in
  `results_all_models.csv` and `run_log_epochs.txt`) before the AE/TCN/Transformer
  extension is run, so epoch counts are available for the LOPO write-up
  without re-running everything.

**Report**: per-model personalised-vs-LOPO RMSE gap, all 5 models. Existing
RF/LSTM numbers already show the gap is model-dependent — LSTM degrades
only slightly from personalised to pooled (mean personalised 21.22 mg/dL
§4 vs. mean LOPO 22.79 mg/dL across the 12 folds in `results_lopo.csv` — a
1.57 mg/dL, ~7% gap), while RF sometimes *improves* with pooling on
individual patients (e.g. patient 596: personalised 19.03 vs. LOPO 19.38 —
close; patient 559: personalised 23.35 vs. LOPO 26.73 — RF degrades more
than LSTM here). Extending to AE/TCN/Transformer will show whether this
model-dependence pattern holds architecture-wide.

**Also flagged, not yet started**: LOPO on Glucdict (tests whether
population generalisation holds on prediabetic data, per `FUTURE_WORK_PLAN.md`)
is a separate, larger extension — no Glucdict LOPO script exists yet. Given
the timeline (§11), this is deprioritised behind the OhioT1DM 5-model LOPO
extension unless the supervisor flags it as required (see §12).

## 10. Why Our Results Are Worse Than Literature — Hypotheses to Test

| Hypothesis | How we test it | Expected finding |
|---|---|---|
| H1: fewer features (3 vs. 6–7 in literature) | Add features incrementally (§8 ablation, already largely done) — but note §8 shows *adding* features consistently hurts here, the opposite of what H1 predicts | Partial closure at best; may need to be reframed as "our minimal feature set is already near-optimal, and the literature gap is not a feature-count gap" |
| H2: simpler architecture | Already tested — 5-model benchmark spans RF through Transformer | Architectural gap remains; no model in the current set closes it |
| H3: no interpolation | Interpolation experiment (§7) | ~1–2 mg/dL improvement, per Zhu et al. (2018)'s 2.1 mg/dL figure as an upper-bound reference |
| H4: no CGM smoothing | Median-filter experiment (§6/§7) | ~0.3–1 mg/dL improvement |
| H5: patient 540 outlier | Report OhioT1DM 2020 results with and without patient 540 (RF 30-min RMSE 39.49 mg/dL — by far the worst of all 12 patients, ~15 mg/dL above the cohort mean) | ~1.5 mg/dL improvement on the 2020-cohort mean once excluded |
| H6: shorter window (12 steps = 1h vs. literature's use of longer context in some studies) | Window experiment (§5) | TCN improves the most; LSTM saturates early |

Note H1 as currently framed is already in tension with our own §8 data —
this should be raised explicitly with the supervisor (§12) rather than
quietly resolved, since it changes the shape of the final narrative.

## 11. Timeline

**Week 1 (17–23 July 2026)**
- §4 Multi-horizon extension (45-min on OhioT1DM/Glucdict shared pipeline)
- §5 Window size experiments
- §9 LOPO extension (AE, TCN, Transformer) + epoch-logging fix

**Week 2 (24–30 July 2026)**
- §6 Preprocessing changes (implement interpolation/median-filter code paths)
- §7 Interpolation experiment matrix
- §8 Feature ablation additions (time-of-day, event isolation, BIG IDEAs
  nutritional subsets)

**Week 3 (31 July – 6 August 2026)**
- §10 Synthesis of worse-than-literature hypotheses
- Poster and final report writing

## 12. Open Questions for Supervisor

1. Confirm 45-min as the longest horizon for the final report, or extend
   to 60-min? (60-min was in the original OhioT1DM BGLP-challenge
   convention per `PREPROCESSING_COMPARISON.md`, but is not currently
   planned here given the Week 1–3 timeline.)
2. Should the interpolation experiments (§7) run on all 3 datasets or
   OhioT1DM only, given the 19–24-configuration matrix is already
   substantial on one dataset?
3. Priority order if time runs short: LOPO 5-model extension (§9),
   interpolation matrix (§7), or window-size sweep (§5)?
4. Is Glucdict LOPO (flagged in §9, not yet started) in scope for this
   final round, or deferred entirely?
5. How should we frame H1 in §10, given our own feature-ablation data
   contradicts the "fewer features explain the gap" hypothesis as
   originally stated?
6. Is resolving the BIG IDEAs model-set mismatch (§4 — GB/CNN-LSTM vs. the
   shared Autoencoder/TCN set) worth the implementation time, or is it
   acceptable to scope the 3-dataset comparison to the 3 models common to
   both pipelines (RF, LSTM, Transformer)?
