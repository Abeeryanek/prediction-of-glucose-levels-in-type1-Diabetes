# Detailed Final Experiments Plan

Blood Glucose Forecasting — University of Duisburg-Essen
Khalil & Abeer | 17 July 2026

## 0. Unification First

**Step 0 (prerequisite for everything else): unify all 7 models (RF,
LSTM, Autoencoder, TCN, Transformer, GB, CNN-LSTM) across all 3 datasets
under one pipeline and identical training conditions.**

**Current state**:

- OhioT1DM + Glucdict: shared `src/` pipeline, 5 models (RF, LSTM,
  Autoencoder, TCN, Transformer). GB and CNN-LSTM — BIG IDEAs' two extra
  models — are not yet implemented here.
- BIG IDEAs: separate `experiments/` pipeline, 5 models (RF, GB, LSTM,
  CNN-LSTM, Transformer) — Autoencoder and TCN are coded (§8A.1) but not
  yet run end-to-end — and different training conditions (`max_epochs=100`,
  `batch_size=64`, `shuffle=True` for some models, 15% val split, no fixed
  seed). Clinically-weighted MSE is used for 3 of these 5 models but not
  applied consistently across all 5 (§8A.2).
- Across both pipelines, 7 distinct models exist in total. This full set
  of 7 — not just the 5 already shared between OhioT1DM/Glucdict — is now
  the unification target for all 3 datasets.

**Unification actions**:

1. Run the 2 models missing from BIG IDEAs (Autoencoder, TCN — already
   coded, §8A.1) end-to-end there.
2. Implement the 2 models missing from the shared OhioT1DM/Glucdict
   pipeline (GB, CNN-LSTM) — new work, not previously scoped, reusing the
   architectures/grids already validated for BIG IDEAs (§8A.5, §2.7, §2.8).
3. Align BIG IDEAs training conditions to the shared standard (§1): both
   plain MSE and clinically-weighted MSE loss, applied uniformly across
   all 5 models (per the supervisor's recommendation — §8A.2),
   `max_epochs=150`, `patience=15`, `batch_size=32`, `shuffle=False`, 20%
   chronological val split, `torch.manual_seed(42)`.
4. Result: identical 7-model set + identical training conditions on all 3
   datasets, so every subsequent experiment (interpolation, ablation)
   produces directly comparable cross-dataset results.

**No new experiment in the sections below is run until Step 0 is
complete.**

**Expected outcome**: BIG IDEAs RMSE for RF/LSTM/Transformer should stay
close to the numbers already in `experiments/results/bigideas/` (§3),
since those 3 models are already common to both pipelines — the main
unknowns are the first Autoencoder/TCN numbers on BIG IDEAs and the first
GB/CNN-LSTM numbers on OhioT1DM/Glucdict, none of which exist yet.

**Feeds into**: every later cross-dataset claim in this document (§3
horizon comparison, §6 interpolation, §7 ablation) — none of those are
directly comparable across all 3 datasets until this step is done.

## 0A. Interpolation Pre-Test (before grid search)

Per the supervisor's explicit instruction, interpolation is prioritized
and gets a small pre-test **before** the full grid search (§2):

Run 3–4 proven interpolation setups on **one representative patient per
dataset**:

1. Drop baseline (current default, no imputation).
2. Linear interpolation, max gap ≤ 30 min (6 steps).
3. Linear interpolation + median filter (window = 5).
4. _(optional 4th setup, time permitting)_ — spline or forward-fill, to
   bracket the linear-interpolation result.

**Decision rule**: if the interpolation benefit confirms — expected
direction from Hameed & Kleinberg (2020): 42.71 → 19.80 mg/dL unfiltered
LSTM RMSE, though that magnitude was measured on noisier free-living data
and is not a direct estimate for our OhioT1DM baseline (§6, §9 H3) —
adopt interpolation as the **default preprocessing** for all subsequent
experiments and drop the non-interpolated variants from the full matrix
(§6), saving complexity there.

**Grid search ordering (explicit, so this doesn't contradict §2)**: the
existing grid-search best parameters (§2.1–§2.5) were already tuned, on
non-interpolated data, before interpolation was reprioritized to run
first. Adopting interpolation as default preprocessing does **not**
trigger a grid-search re-run — the cached parameters in
`grid_search_results.json` are reused as-is on interpolated data, exactly
like the existing global-reuse-across-patients/datasets pattern already
documented as a known limitation in §2. Re-tuning grid search specifically
on interpolated data is an **optional stretch goal** if time permits, not
a requirement, given the Week 1–3 timeline (§10, item 1) — and if
pursued, it would run _after_ the interpolation decision below, not
before it.

**Reprioritized experiment order** (per supervisor, supersedes the
priority question previously raised in §10):

1. **Interpolation** (§6) — all 3 datasets.

(The standalone window-size experiment that previously sat at priority 3
here has been removed entirely — see §4.)

**Expected outcome**: the pre-test confirms a small, positive RMSE
improvement from interpolation (consistent in direction with Hameed &
Kleinberg 2020, though likely much smaller in magnitude on OhioT1DM's
cleaner CGM stream — see §9 H3's ~1–2 mg/dL estimate).

**Feeds into**: if confirmed, this collapses the full §6 interpolation
matrix from 19 configurations down to a single adopted configuration
applied everywhere, and that adopted configuration becomes the new
default preprocessing for §7 ablation as well.

## 1. Standard Training Conditions (applies to ALL experiments unless stated)

| Setting                 | Value                                                                                                     | Reasoning                                                                  |
| ----------------------- | --------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| Optimizer               | Adam                                                                                                      | Kingma & Ba (2015)                                                         |
| Loss function           | Both plain MSE (`nn.MSELoss`) and clinically-weighted MSE (hypo/hyper weighted), evaluated separately, applied uniformly across all 5 models and all 3 datasets | Per the supervisor's recommendation: plain MSE is the standard regression baseline; clinically-weighted MSE tests whether up-weighting hypo/hyperglycemic errors improves clinically-relevant accuracy (e.g. Clarke Zone A/D/E) without hurting overall RMSE |
| max_epochs              | 150                                                                                                       | See note below — current code default is 100                               |
| Early stopping patience | 15 epochs                                                                                                 | Prechelt (1998); current code default is 10                                |
| Early stopping monitor  | Validation loss, restore best weights                                                                     | Prevents overfitting past the generalisation minimum                       |
| Batch size              | 32                                                                                                        | See note below — current code default is 64                                |
| Shuffle                 | False                                                                                                     | Temporal order must be preserved for autocorrelated CGM series             |
| Train/val split         | Last 20% of training data, chronological (`val_ratio=0.20`)                                               | Matches `make_splits` in `src/training/pipeline.py`; avoids future leakage |
| Normalisation           | `StandardScaler`, fit on training data only                                                               | Prevents test/val leakage into scaler statistics                           |
| Random seed             | 42                                                                                                        | See note below — currently only set for Random Forest                      |
| Hardware                | NVIDIA GPU (CUDA) if available, else CPU (`torch.device("cuda" if torch.cuda.is_available() else "cpu")`) | All prior runs (`run_log_epochs.txt`) executed on CUDA                     |

**The split is three-way, not two.** Per the supervisor's note that
"validation only exists if there's a further split": every patient's data
is divided into (1) the **official train set**, (2) a **validation set**
carved out of the last 20% of that train set, chronological
(`val_ratio=0.20`), used only for early stopping and hyperparameter
selection, and (3) the **official held-out test set**, used only for the
final reported metrics. The model never sees the test set during training
or early stopping — validation and test are distinct sets from distinct
sources (an internal chronological split vs. the dataset's own official
split), not the same 20% used twice.

**Three code changes are required before Sections 3–9 can run under these
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
   more closely across models before the grid-search expansion in §2.6.
3. **Add `torch.manual_seed(42)` (+ `np.random.seed(42)`) at the start of
   each DL training run.** Currently `random_state=42` is set only for
   Random Forest (`src/models/random_forest.py`); none of the four PyTorch
   models fix a seed, so weight initialisation and (once batch size drops
   to 32) minibatch composition are not reproducible run-to-run. This must
   be fixed before Sections 3–9 are run, since reproducibility will be
   questioned at the defense.

**Observed epoch counts from the completed run** (`results/ohio/run_log_epochs.txt`,
12 patients, current 100-epoch cap, patience 10 — will be re-measured once
the cap changes to 150/15):

| Model       | Mean ± SD   | Range  | Notes                                                  |
| ----------- | ----------- | ------ | ------------------------------------------------------ |
| LSTM        | 54.5 ± 15.8 | 32–76  | Never hit the cap                                      |
| Autoencoder | 72.4 ± 24.3 | 36–100 | Hit the 100-epoch cap on patients 570, 575, 588 (3/12) |
| TCN         | 51.6 ± 19.6 | 27–79  | Never hit the cap                                      |
| Transformer | 42.1 ± 24.3 | 16–97  | Never hit the cap                                      |

## 2. Grid Search — Full Documentation (all models, all datasets)

**Note on model count**: §2.1–§2.5 cover the 5 models already in the
shared pipeline. §2.7 (GB) and §2.8 (CNN-LSTM), added after §2.6 to avoid
renumbering existing cross-references, cover the 2 models being newly
added to the shared pipeline as part of the 7-model unification (§0).

The grid search (`src/training/grid_search.py`) was run once, on OhioT1DM
patient 559 (2018 cohort) only, using `TimeSeriesSplit` walk-forward
validation. The resulting best parameters (`results/ohio/grid_search_results.json`)
were then applied globally to every patient and every dataset. This is a
known limitation, addressed in §2.6.

**Note on interpolation ordering**: this grid search was already run, on
non-interpolated data, before interpolation was reprioritized to run
first (§0A appears earlier in this document, but not chronologically
before this search). If §0A's pre-test adopts interpolation as default
preprocessing, these cached parameters are reused as-is rather than
re-tuned — see §0A's explicit ordering note for why this is not a
contradiction.

**Note on hyperparameter value selection**: the specific numeric _values_
tested below (e.g., `hidden_size` ∈ {32, 64, 128}, `num_filters` ∈ {32,
64, 128}, `d_model` ∈ {32, 64, 128}, `latent_size` ∈ {16, 32, 64}) are
standard defaults — powers of two spanning small/medium/large capacity —
chosen to bracket the conventional range at feasible compute cost, not
derived from a specific source. The citations in the tables below justify
_which_ hyperparameter matters and _why_ a search over capacity/learning
rate is worthwhile, not the specific numbers chosen. Only the Adam base
learning rate (1e-3) is a cited default (Kingma & Ba 2015); the halved
5e-4 is tested as a lower-variance alternative for these small
per-patient datasets, not itself a literature value.

### 2.1 Random Forest

| Parameter        | Values tested  | Best found | Why this range                                                                                         |
| ---------------- | -------------- | ---------- | ------------------------------------------------------------------------------------------------------ |
| n_estimators     | [50, 100, 200] | 200        | Probst et al. (2019): "number of trees is the most important hyperparameter" for RF variance reduction |
| max_depth        | [10, 15, 20]   | 10         | Biau & Scornet (2016): shallower trees reduce variance on datasets of this size (~12k rows/patient)    |
| min_samples_leaf | [1, 2, 4]      | 4          | Probst et al. (2019): leaf-size regularisation is the second most impactful RF hyperparameter          |

### 2.2 LSTM

| Parameter   | Values tested | Best found | Why this range                                                                                                                |
| ----------- | ------------- | ---------- | ----------------------------------------------------------------------------------------------------------------------------- |
| hidden_size | [32, 64, 128] | 128        | Standard small/medium/large capacity bracket (see note above) — not derived from Kalita & Mirza (2025) or any specific source |
| lr          | [1e-3, 5e-4]  | 5e-4       | Kingma & Ba (2015) default is 1e-3; halved value tested as a lower-variance alternative for small per-patient datasets        |

### 2.3 Autoencoder (Seq2Seq)

| Parameter   | Values tested | Best found | Why this range                                                                                                                                                                                                                                                                                                               |
| ----------- | ------------- | ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| latent_size | [16, 32, 64]  | 16         | Standard small/medium/large capacity bracket (see note above). Srivastava et al. (2015) motivates _why_ a tight bottleneck is worth testing (forces the encoder to retain only the dominant glucose trend, reducing overfitting on ~12k-row per-patient data) — the specific values 16/32/64 are not derived from that paper |
| lr          | [1e-3, 5e-4]  | 5e-4       | Same reasoning as §2.2                                                                                                                                                                                                                                                                                                       |

### 2.4 TCN

| Parameter   | Values tested | Best found | Why this range                                                                                                                                                                                                                                               |
| ----------- | ------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| num_filters | [32, 64, 128] | 32         | Standard small/medium/large capacity bracket (see note above). Bai et al. (2018) motivates _why_ fewer filters may suffice (dilation depth doing the representational work, rather than channel width) — the specific values are not derived from that paper |
| lr          | [1e-3, 5e-4]  | 5e-4       | Same reasoning as §2.2                                                                                                                                                                                                                                       |

### 2.5 Transformer

| Parameter | Values tested | Best found | Why this range                                                                                                                                          |
| --------- | ------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| d_model   | [32, 64, 128] | 128        | Standard small/medium/large capacity bracket (see note above), constrained to be divisible by `nhead=4` (fixed); not derived from Vaswani et al. (2017) |
| lr        | [1e-3, 5e-4]  | 5e-4       | Same reasoning as §2.2                                                                                                                                  |

### 2.6 Planned Grid Search Expansion

**Prediction horizon (confirmed with the supervisor): grid search is
performed only at the 30-min horizon. The resulting best parameters are
then reused unchanged across the 15/30/45-min horizons reported in the
experiments (§3) — grid search itself is not repeated per horizon.**

Not yet covered by the existing grid search, to be added:

- **`window_size` as a grid parameter**: [12, 24, 36, 72] steps (1h/2h/3h/6h)
  — currently fixed at 12 and never searched jointly with model
  hyperparameters. This joint search is now the only place window size is
  tested at all, since the standalone window-size experiment (§4) has been
  removed.
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

**Expected outcome**: joint window_size × hyperparameter search likely
shifts TCN's best `num_filters`/dilation combination the most, since TCN
is the model hypothesized to be most sensitive to window size; RF
and LSTM parameters are expected to be comparatively stable across window
sizes.

**Feeds into**: this joint search is now the sole source of window-size
findings in the final report, since the standalone window-size experiment
(§4) has been removed as redundant with it — resolved per the supervisor,
see §10.

### 2.7 Gradient Boosting (GB)

Not yet run — GB exists in the BIG IDEAs pipeline (§8A.5) but must be
implemented in the shared `src/` pipeline as part of the 7-model
unification (§0). Once implemented, grid search reuses the grid already
validated on BIG IDEAs (§8A.5):

| Parameter        | Values to test  | Best found | Why this range                                     |
| ---------------- | --------------- | ---------- | -------------------------------------------------- |
| n_estimators     | [100, 200, 300] | TBD        | Same rationale as RF (§2.1) — Probst et al. (2019) |
| max_depth        | [10, 15, 20]    | TBD        | Same rationale as RF (§2.1)                        |
| min_samples_leaf | [1, 2, 4]       | TBD        | Same rationale as RF (§2.1)                        |

### 2.8 CNN-LSTM

Not yet run — CNN-LSTM exists in the BIG IDEAs pipeline (§8A.5) but must
be implemented in the shared `src/` pipeline as part of the 7-model
unification (§0). Grid, matching BIG IDEAs (§8A.5):

| Parameter  | Values to test | Best found | Why this range                               |
| ---------- | -------------- | ---------- | -------------------------------------------- |
| hidden_dim | [32, 64, 128]  | TBD        | Same capacity-bracket reasoning as §2.2/§2.5 |
| lr         | [1e-3, 5e-4]   | TBD        | Same reasoning as §2.2                       |

## 3. Multi-Horizon Analysis (15, 30, 45 min — NOT only 30)

All experiments in Sections 4–8 will report **15-min (3 steps), 30-min (6
steps), and 45-min (9 steps)** horizons, **on all 3 datasets** —
OhioT1DM, Glucdict, and now BIG IDEAs, once Step 0 (§0) unification makes
BIG IDEAs directly comparable to the other two. The shared
OhioT1DM/Glucdict pipeline (`src/training/pipeline.py`) currently predicts
a single fixed `horizon=6` (30 min); it will be extended to also run
`horizon=9` for the 45-min report, reusing the already-implemented
multi-step target machinery (`multi_step=True` path in `create_windows`).
The input window is **12 steps (1h)** as the default across all 3
datasets (the grid-search window_size sweep, §2.6, varies this; everywhere
else in this document, 12 steps is assumed unless stated otherwise).

**Horizon degradation already measured on OhioT1DM** (`results/ohio/run_log_epochs.txt`,
12 patients, mean RMSE, mg/dL):

| Model       | 5 min | 15 min | 30 min | Growth 5→30 min |
| ----------- | ----- | ------ | ------ | --------------- |
| RF          | 5.87  | 14.06  | 23.35  | +17.48          |
| LSTM        | 6.11  | 12.70  | 21.22  | +15.11          |
| Autoencoder | 6.39  | 13.06  | 21.59  | +15.20          |
| TCN         | 8.28  | 15.20  | 24.70  | +16.42          |
| Transformer | 8.15  | 13.91  | 22.58  | +14.43          |

This is a steep, roughly linear degradation — about +0.6–0.7 mg/dL RMSE per
additional minute of lead time, consistent with the literature (`FINAL_EXPERIMENTS_PLAN.md`
§4 already flagged this expectation).

**Abeer's BIG IDEAs finding is confirmed with numbers, not just an
impression.** The BIG IDEAs walk-forward benchmark (`experiments/results/bigideas/summary_metrics_walk_forward.csv`,
5-fold walk-forward CV, pooled/population model, `full` feature subset,
mean RMSE, mg/dL):

| Model       | 15 min | 30 min | 45 min | Growth 15→45 min |
| ----------- | ------ | ------ | ------ | ---------------- |
| RF          | 20.20  | 20.17  | 20.52  | +0.32            |
| GB          | 20.76  | 20.83  | 21.50  | +0.74            |
| LSTM        | 22.75  | 22.33  | 23.73  | +0.98            |
| CNN-LSTM    | 22.48  | 22.56  | 23.28  | +0.80            |
| Transformer | 23.65  | 23.01  | 24.18  | +0.53            |

Compare +0.3 to +1.0 mg/dL total growth over a 30-minute lead-time increase
on BIG IDEAs against +14 to +17 mg/dL over the same 25-minute increase on
OhioT1DM — roughly a **20–40× smaller horizon effect**. This needs
investigation, and both of the proposed hypotheses have partial support in
data already collected:

- **H1 (BIG IDEAs glucose is less volatile / prediabetic population)** —
  plausible but not yet directly tested; needs a glucose-variability
  comparison between BIG IDEAs and
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

**Known caveat (resolved by §0)**: BIG IDEAs currently uses a different
model set (RF, GB, LSTM, CNN-LSTM, Transformer via Abeer's `experiments/`
pipeline) than the shared OhioT1DM/Glucdict pipeline (RF, LSTM,
Autoencoder, TCN, Transformer). This mismatch is exactly what Step 0 (§0)
unification addresses — not only by adding Autoencoder/TCN to BIG IDEAs,
but also by adding GB/CNN-LSTM to the shared pipeline — before any
experiment in this section runs. No 3-dataset, 7-model claim is made
until that unification is complete.

**Expected outcome**: once BIG IDEAs is unified (§0) and run through
15/30/45-min horizons, its horizon-degradation slope is expected to
remain far shallower than OhioT1DM's (consistent with the 20–40× gap
already observed here), supporting H2 (persistence-like ceiling) over H1
(population volatility) unless the glucose-SD comparison below says
otherwise.

**Feeds into**: the H1-vs-H2 test in §9 (via the planned per-horizon
slope + glucose-SD comparison above), and directly determines whether the
"worse than literature" narrative in §9 needs a BIG IDEAs-specific
caveat.

## 4. Window Size Experiments — REMOVED

**Decided with the supervisor (28 July 2026 meeting): this standalone
experiment is cut entirely.** It overlapped with the grid-search
`window_size` sweep (§2.6), which already varies window size ([12, 24,
36, 72] steps) jointly with model hyperparameters — a separate sweep at
fixed hyperparameters would not have added information beyond that joint
search. Window size is therefore tested only as part of grid search
(§2.6); no standalone window-size × horizon × model matrix will appear in
the final report, and the TCN-context hypothesis originally motivating
this section (§9, formerly H6) is not being tested as its own experiment.
This resolves the open question previously raised in §10.

## 5. Preprocessing Comparison and Planned Changes

Summary of the existing comparison (`PREPROCESSING_COMPARISON.md`, full
detail and citations there): of 12 preprocessing decisions compared against
the OhioT1DM literature (Marling & Bunescu 2020; Martinsson et al. 2020;
Zhu et al. 2018; Xie & Wang 2020), **10 match the dominant literature
convention** (5-min resampling, event zero-fill, sentinel handling,
train-only scaling, per-patient models, 60-min window, 30-min horizon,
official train/test split, walk-forward CV, no extra clipping/smoothing).
Two decisions are deliberate departures:

| Preprocessing step           | Current                     | Literature                                                                                                                                                                               | Assessment                                                                                   |
| ---------------------------- | --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| Missing CGM values           | Dropped                     | Split: Martinsson et al. also drop; Hameed & Kleinberg (2020) interpolate (linear), 22.91 mg/dL RMSE improvement vs. no imputation (on free-living/patient-generated data, not OhioT1DM) | One literature branch matches ours; the other (interpolation) is untested here on our data   |
| Missing wearable sensor rows | Dropped, never interpolated | Where addressed, usually linearly interpolated with a bounded gap limit                                                                                                                  | Stricter than most published pipelines; costs sample count on `full`/`wearable` feature sets |

### Planned Preprocessing Changes

| Discrepancy         | Current | Literature                                                                                                                                                                           | Planned change                                                                                          | Expected effect                                                                                                                                                                      |
| ------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Missing sensor rows | Drop    | Interpolate (bounded); Hameed & Kleinberg (2020) use this exact bound — their Table 4 specifies "linear interpolation... for gaps ≤ 30 min"                                          | Test bounded linear interpolation, max gap 30 min (6 steps)                                             | Reduce data loss on `full`/`wearable` feature sets, which currently lose the most rows (e.g. clean-cohort `full`/`wearable`: n=1,796 vs. `glucose_only`: n=10,742 — an 83% row loss) |
| Missing glucose     | Drop    | Mixed (Martinsson: drop; Hameed & Kleinberg: interpolate)                                                                                                                            | Test linear interpolation for gaps ≤ 3 steps (15 min) only — never fabricate glucose across longer gaps | More training data without contaminating ground truth over clinically meaningful gaps                                                                                                |
| CGM smoothing       | None    | Median filter, window=5 samples — Hameed & Kleinberg (2020) use this exact value (their Table 4: "Median filter window size = 5"), which they in turn attribute to Zhu et al. (2018) | Test median filter, window=5 (25 min)                                                                   | Reduce sensor noise; test whether this recovers some of the RMSE gap to literature (H4, §9)                                                                                          |

**Expected outcome**: bounded interpolation recovers most of the row loss
on `full`/`wearable` feature sets (currently an 83% loss, §7.1) without
materially changing RMSE, since the gaps being filled are short relative
to glucose's autocorrelation timescale.

**Feeds into**: the interpolation pre-test (§0A) and full matrix (§6)
directly test the first two rows of this table; the median-filter test
feeds H4 in §9.

## 6. Interpolation Experiments (multiple combinations)

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
- **Citation**: Hameed & Kleinberg (2020), _Comparing Machine Learning
  Techniques for Blood Glucose Forecasting Using Free-living and Patient
  Generated Data_, MLHC, PMLR 126 — reports linear interpolation reduced
  unfiltered LSTM RMSE from 42.71 to 19.80 mg/dL (a 22.91 mg/dL
  improvement). **Correction**: this experiment was previously justified by
  citing Zhu, Li, Herrero, Chen & Georgiou (2018) for a −2.1 mg/dL
  interpolation effect; that attribution was wrong — Zhu et al. (2018)'s
  contribution is a CNN/RNN glucose prediction model, not an
  interpolation-effect study, and should not be cited for this claim.
  Note also that Hameed & Kleinberg's 22.91 mg/dL figure was measured on
  noisier, unfiltered free-living/patient-generated data, not OhioT1DM, so
  treat it as an upper-bound reference from a much noisier baseline rather
  than a directly transferable estimate for our ~1–2 mg/dL expectation
  below (§9, H3).

**Expected outcome**: consistent with the §0A pre-test, a modest (~1–2
mg/dL) RMSE improvement from bounded linear interpolation — smaller than
Hameed & Kleinberg's (2020) 22.91 mg/dL figure since that was measured on
noisier free-living data (§9, H3).

**Feeds into**: if the pre-test (§0A) confirms the benefit, this
experiment's scope collapses from the full 19-configuration matrix to
validating the single adopted configuration on the full 12-patient
cohort; the adopted configuration then becomes the default preprocessing
for §7 ablation.

## 7. Feature Ablation — Complete Matrix

**Setup**: ablation uses RF and LSTM (the fastest tree-based and
recurrent models, respectively) at the 30-min horizon, with the
grid-search-best hyperparameters (§2), on the unified pipeline (§0).
Autoencoder/TCN/Transformer are not run per feature set here to keep the
ablation matrix tractable — RF and LSTM are used as representative
proxies for tree-based and sequence models respectively.

### 7.1 OhioT1DM 2018 (heart rate + steps available)

Clean cohort only (patients 559, 570, 588, 591 — 563 and 575 excluded for
sparse wristband coverage, per `FINAL_EXPERIMENTS_PLAN.md` §2). Mean RMSE
across the 4 patients, 30-min horizon (`results/ohio/results_feature_ablation_clean.csv`):

| Feature set                        | n (windows, pooled) | RF RMSE | LSTM RMSE | Clarke Zone A % |
| ---------------------------------- | ------------------- | ------- | --------- | --------------- |
| glucose_only                       | 10,742              | 21.36   | 20.54     | 88.51           |
| clinical (+bolus, carbs)           | 10,742              | 21.33   | 20.22     | 88.56           |
| glucose_steps                      | 1,883               | 24.03   | 21.37     | 85.61           |
| clinical_steps                     | 1,883               | 24.03   | 21.24     | 86.83           |
| glucose_hr                         | 3,027               | 24.84   | 24.91     | 85.17           |
| clinical_hr                        | 3,027               | 24.83   | 23.61     | 85.93           |
| wearable (hr + steps, no clinical) | 1,796               | 32.74   | 33.75     | 75.61           |
| full (clinical + hr + steps)       | 1,796               | 32.61   | 35.37     | 75.61           |

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

### 7.2 OhioT1DM 2020 (acceleration available)

All 6 patients (540, 544, 552, 567, 584, 596), 30-min horizon
(`results/ohio/results_ablation_2020.csv`):

| Feature set    | RF RMSE | LSTM RMSE |
| -------------- | ------- | --------- |
| glucose_only   | 25.01   | 22.29     |
| clinical       | 25.00   | 22.04     |
| glucose_accel  | 30.57   | 28.98     |
| clinical_accel | 30.55   | 28.85     |

Confirms the same pattern as §7.1 with a third, independent wearable
signal (accelerometer instead of Basis wristband hr/steps) — accelerometer
degrades RMSE by ~5.5–7 mg/dL, comparable in size to the hr/steps
degradation on the 2018 cohort.

### 7.3 Glucdict (heartrate + acc_magnitude + eat/drink events)

13 users, 30-min horizon (`results/glucdict/results_glucdict_ablation.csv`):

| Feature set                      | n users | RF RMSE | LSTM RMSE | Clarke Zone A % |
| -------------------------------- | ------- | ------- | --------- | --------------- |
| glucose_only                     | 13      | 15.23   | 14.04     | 90.87           |
| glucose_activity (acc_magnitude) | 13      | 15.19   | 13.86     | 91.05           |
| glucose_hr                       | 10      | 17.89   | 17.26     | 88.12           |
| full_wearable                    | 10      | 17.28   | 17.49     | 87.99           |

Note the third-dataset confirmation is _partial_: `glucose_activity` is
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

### 7.4 BIG IDEAs (28 features)

Not yet run through the shared ablation protocol — Abeer's
`experiments/results/bigideas/` pipeline currently reports `comparable` vs.
`full` subsets (not a full 8-combination ablation matching §7.1–7.3). At
the `full` 30-min horizon (`experiments/results/bigideas/summary_metrics_walk_forward.csv`):
RF RMSE 20.17, LSTM RMSE 22.33 — both noticeably worse than either OhioT1DM
or Glucdict `glucose_only`, despite 28 available features, which is itself
worth stating as a finding (more features ≠ better here either).

Planned: nutritional rolling-window feature subsets (Abeer) — e.g.
carbs/calories summed over the last 1h/3h/6h — to test whether _aggregated_
nutritional signal helps where raw wearable channels do not.

**Expected outcome**: the "wearable features hurt" pattern already
confirmed on OhioT1DM 2018/2020 and partially on Glucdict (§7.3) is
expected to replicate on BIG IDEAs once ablation runs there post-
unification (§0) — though BIG IDEAs' already-poor `full`-set RMSE (§7.4)
suggests the effect may be harder to isolate from a weak overall signal.

**Feeds into**: directly answers H1 in §9 (feature count vs. gap-to-
literature), and the isolated clinical_hr/clinical_steps comparison feeds
the "which wearable signal hurts more" question raised for future work.

## 8A. BIG IDEAs Pipeline — Current State and Alignment (Abeer)

Abeer provided a detailed status update on the BIG IDEAs pipeline
(`experiments/` codebase). This section documents the current state, the
gaps against the shared OhioT1DM/Glucdict pipeline (§1–§2), and the work
needed to reconcile them.

### 8A.1 Model Set Difference

| Pipeline                            | Models                                                           |
| ----------------------------------- | ---------------------------------------------------------------- |
| BIG IDEAs (current)                 | RF, GB, LSTM, CNN-LSTM, Transformer                              |
| Shared (OhioT1DM/Glucdict, current) | RF, LSTM, Autoencoder, TCN, Transformer                          |
| Common to both                      | RF, LSTM, Transformer                                            |
| Unification target (all 3 datasets) | RF, LSTM, Autoencoder, TCN, Transformer, GB, CNN-LSTM (7 models) |

Autoencoder and TCN have now been **coded** for BIG IDEAs but **not yet run
end-to-end** — no result numbers exist for them yet. GB and CNN-LSTM, in
the other direction, do not yet exist in the shared OhioT1DM/Glucdict
pipeline at all and need to be implemented there (§0, unification action 2) using the same architectures already validated on BIG IDEAs (§8A.5).
This is what makes full 7-model unification (§0) feasible as a committed
first step rather than an open question: each of the 4 missing
model/dataset combinations (Autoencoder+BIG IDEAs, TCN+BIG IDEAs,
GB+shared, CNN-LSTM+shared) already has either working code or a directly
reusable architecture to port — none require research from scratch.

### 8A.2 Training Condition Differences (must be reconciled)

| Setting             | Shared pipeline (§1) | BIG IDEAs current                                                                                                    | Action                                                                      |
| ------------------- | -------------------- | -------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| Loss (all 5 models) | Both plain MSE and clinically-weighted MSE | Clinically weighted MSE (hypo/hyper weights) for RF, GB, LSTM, CNN-LSTM, Transformer; plain MSE for Autoencoder, TCN | Apply both plain MSE and clinically-weighted MSE uniformly across all 5 models, matching the shared pipeline (§1) |
| max_epochs          | 150                  | 100                                                                                                                  | Raise to 150 or justify keeping 100                                         |
| Batch size          | 32                   | 64                                                                                                                   | Lower to 32 to match §1                                                     |
| Shuffle             | False                | True (LSTM, CNN-LSTM, Transformer); False (Autoencoder, TCN)                                                         | Run the shuffle experiment below before deciding                            |
| Train/val split     | Last 20%             | Last 15%                                                                                                             | Align to 20%, per §1                                                        |
| Random seed         | 42                   | Not set                                                                                                              | Add `torch.manual_seed(42)` (+ `np.random.seed(42)`), same fix as §1 item 3 |

**Per the supervisor's recommendation: both plain MSE and clinically-
weighted MSE are used, not just one or the other.** Clinical weighting
(hypo/hyperglycemia mispredictions weighted more heavily, since they cost
more clinically than in-range ones) is relevant enough to keep — but
applying it to only 3 of BIG IDEAs' 5 models was an inconsistency, so the
fix is to apply *both* loss functions uniformly across all 5 models here,
matching the same dual-loss standard adopted project-wide in §1 (all 5
models, all 3 datasets, under the unified pipeline). Each model is
trained/evaluated under plain MSE and separately under clinically-
weighted MSE, so the comparison stays fair across models and datasets.
This resolves the open question previously raised in §10: models are
evaluated under both loss functions to measure whether clinical
weighting improves clinically-relevant accuracy (e.g. Clarke Zone A/D/E)
without hurting overall RMSE.

### 8A.3 Shuffle Experiment (BIG IDEAs specific)

Because consecutive windows overlap by 11 of 12 timesteps, with
`shuffle=False` consecutive minibatches are highly similar, which can bias
gradients. Since BIG IDEAs currently shuffles 3 of its 5 models but not the
other 2, this inconsistency needs to be tested rather than resolved by
assumption.

- **Plan**: train each model twice — `shuffle=True` vs. `shuffle=False` —
  holding fold, horizon, features, and hyperparameters fixed. Compare RMSE
  and epochs-to-convergence.
- **Why it matters**: if shuffle changes RMSE meaningfully, the current
  per-model inconsistency (§8A.2) is actively affecting which model looks
  better in today's BIG IDEAs results, and this must be resolved before any
  cross-model comparison on BIG IDEAs is treated as fair.

**Expected outcome**: `shuffle=True` is expected to produce a small RMSE
change (likely a modest improvement, since it breaks the highly-
overlapping consecutive-minibatch structure), plausibly explaining part
of why BIG IDEAs currently shuffles 3 of 5 models rather than being an
arbitrary inconsistency.

**Feeds into**: directly informs shuffle policy in §8A.2 — if shuffle
matters, it must be applied uniformly across all 5 BIG IDEAs models
before any cross-model comparison (§8A.2) is treated as fair. (The
clinically-weighted MSE question that this used to also feed into is
resolved — see §8A.2.)

### 8A.4 BIG IDEAs Ablation — More Granular Than OhioT1DM

BIG IDEAs already isolates individual wearable signals (EDA, heart_rate,
BVP, IBI separately) — more granular than the shared pipeline's
wearable/full grouping in §7.1–7.3. This is an advantage worth keeping and
extending, not replacing. Planned revisions:

- Extend ablation from single-fold to all 5 walk-forward folds (report
  mean ± std, matching the convention used elsewhere in this plan).
- Add Clarke Zone A % per feature group (currently only RMSE is reported).
- Log row count `n` per feature group, to catch silent dropped-row bugs —
  the same class of issue flagged for OhioT1DM in §7.1's data-quality note.
- Run the ablation for Autoencoder and TCN once they have their first
  end-to-end run (§8A.1).
- Isolate nutritional rolling-window granularities individually:
  `carbs_30min` vs. `carbs_1h` vs. `carbs_4h`, rather than only as a
  combined nutritional block — feeds into the planned nutritional-subset
  experiment already noted in §7.4.

**Expected outcome**: individual wearable-signal isolation (EDA,
heart_rate, BVP, IBI) is expected to show heterogeneous effects rather
than a uniform "all wearable signals hurt" pattern, consistent with the
heartrate-vs-accelerometer split already found on Glucdict (§7.3).

**Feeds into**: the granular BIG IDEAs ablation directly extends the
cross-dataset feature-effect comparison in §7 once combined with
OhioT1DM/Glucdict results.

### 8A.5 BIG IDEAs Grid Search

| Model                         | Grid                                                                                            |
| ----------------------------- | ----------------------------------------------------------------------------------------------- |
| RF / GB                       | `n_estimators` [100, 200, 300] × `max_depth` [10, 15, 20] × `min_samples_leaf` [1, 2, 4]        |
| LSTM / CNN-LSTM / Transformer | `hidden_dim` [32, 64, 128] × `lr` [1e-3, 5e-4]                                                  |
| Autoencoder                   | `hidden_size` [32, 64, 128] × `lr` [1e-3, 5e-4]; latent size auto-set to `hidden_size // 2`     |
| TCN                           | `num_filters` [32, 64, 128] × `lr` [1e-3, 5e-4]; kernel size fixed at 3; dilations [1, 2, 4, 8] |

Same limitation as the shared pipeline's grid search (§2): searched once on
the largest fold, 30-min horizon, `full` feature set, then reused
everywhere else. Needs the same fold/horizon invariance check flagged in
§2.6 for OhioT1DM/Glucdict/BIG IDEAs.

### 8A.6 BIG IDEAs Window Size Note

The BIG IDEAs parquet files **already apply** an exponentially weighted
moving average (halflife 2h / 8h / 24h) to food macro columns at
_ingestion_ time, before any model sees the data. This is a separate,
already-applied preprocessing step and must not be confused with
`SEQ_LENGTH` (the model's input lookback window, analogous to `window_size`
in §2.6) — the grid-search window_size sweep (§2.6, the only remaining
window-size test now that the standalone §4 experiment has been cut)
varies `SEQ_LENGTH` ∈ [12, 24, 36, 72] only, and does not touch the EWM
halflives.

At `SEQ_LENGTH=72`, patients with fewer than 73 rows in a given fold are
silently excluded from that fold. A per-patient row-count check is needed
before running the `SEQ_LENGTH=72` condition, to know in advance how many
BIG IDEAs patients would actually be dropped rather than discovering it
after the fact.

## 9. Why Our Results Are Worse Than Literature — Hypotheses to Test

| Hypothesis                                                                                | How we test it                                                                                                                                                                                       | Expected finding                                                                                                                                                                                                              |
| ----------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| H1: fewer features (3 vs. 6–7 in literature)                                              | Add features incrementally (§7 ablation, already largely done) — but note §7 shows _adding_ features consistently hurts here, the opposite of what H1 predicts                                       | Partial closure at best; may need to be reframed as "our minimal feature set is already near-optimal, and the literature gap is not a feature-count gap"                                                                      |
| H2: simpler architecture                                                                  | Will be tested once §0's 7-model unification completes — benchmark spans RF, LSTM, Autoencoder, TCN, Transformer, GB, and CNN-LSTM                                                                   | Architectural gap expected to remain; if GB/CNN-LSTM also fail to close it, that further weakens a pure model-choice explanation                                                                                              |
| H3: no interpolation                                                                      | Interpolation experiment (§6)                                                                                                                                                                        | ~1–2 mg/dL improvement expected on OhioT1DM; Hameed & Kleinberg (2020) found a much larger 22.91 mg/dL improvement, but on noisier free-living data, so treat their figure only as a loose upper bound, not a direct estimate |
| H4: no CGM smoothing                                                                      | Median-filter experiment (§5/§6)                                                                                                                                                                     | ~0.3–1 mg/dL improvement                                                                                                                                                                                                      |
| H5: patient 540 outlier                                                                   | Report OhioT1DM 2020 results with and without patient 540 (RF 30-min RMSE 39.49 mg/dL — by far the worst of all 12 patients, ~15 mg/dL above the cohort mean)                                        | ~1.5 mg/dL improvement on the 2020-cohort mean once excluded                                                                                                                                                                  |
| H6: shorter window (12 steps = 1h vs. literature's use of longer context in some studies) | **Not tested separately** — the standalone window experiment (§4) was cut per the supervisor as redundant with the grid-search `window_size` sweep (§2.6); H6 cannot be tested as originally planned | Not applicable — no standalone finding will be reported for H6                                                                                                                                                                |

Note H1 as currently framed is already in tension with our own §7 data —
this should be raised explicitly with the supervisor (§10, item 5) rather
than quietly resolved, since it changes the shape of the final
narrative.

## 10. Open Questions for Supervisor

1. Confirm 45-min as the longest horizon for the final report, or extend
   to 60-min? (60-min was in the original OhioT1DM BGLP-challenge
   convention per `PREPROCESSING_COMPARISON.md`, but is not currently
   planned here given the Week 1–3 timeline.)
2. ~~Should the interpolation experiments (§6) run on all 3 datasets or
   OhioT1DM only~~ — **RESOLVED per supervisor**: all 3 datasets (§0A),
   with a small pre-test first to decide whether the full matrix is even
   needed.
3. ~~Priority order if time runs short~~ — **RESOLVED per supervisor**:
   interpolation (§6) is the top priority; window-size sweep is cut
   entirely — see item 5. (LOPO — both the OhioT1DM extension formerly at
   §8 and BIG IDEAs LOPO formerly at §8A.7 — has been removed from this
   plan entirely and is no longer a priority-order item.)
4. How should we frame H1 in §9, given our own feature-ablation data
   contradicts the "fewer features explain the gap" hypothesis as
   originally stated?
5. ~~Window-size experiment (§4): cut, keep as-is, or rescope to what
   grid search doesn't cover?~~ — **RESOLVED per supervisor**: cut
   entirely. It overlapped with the grid-search `window_size` sweep
   (§2.6); §4 now records the removal, and H6 in §9 can no longer be
   tested as originally planned.
6. ~~Clinically-weighted MSE (§8A.2): adopt uniformly across all 5 models
   and all 3 datasets, or keep it scoped to BIG IDEAs?~~ — **RESOLVED per
   supervisor's recommendation**: both plain and clinically-weighted MSE
   are used, applied uniformly across all 5 models and all 3 datasets
   under the unified pipeline (§8A.2, §1) — not scoped to BIG IDEAs only.

## References

All entries below were confirmed via web search against publisher pages
(IEEE Xplore, Springer, SAGE, arXiv, CEUR-WS) rather than written from
memory. Verification status is noted per entry.

- Bertachi, A., Biagi, L., Contreras, I., Luo, N., & Vehí, J. (2018).
  Prediction of blood glucose levels and nocturnal hypoglycemia using
  physiological models and artificial neural networks. _3rd International
  Workshop on Knowledge Discovery in Healthcare Data (KDH@IJCAI-ECAI 2018)_, 85–90. **[Verified — the 19.33 mg/dL @ 30-min RMSE figure used
  in the literature-comparison slides was previously misattributed to
  Xiong et al. (2025); it belongs to Bertachi et al. (2018), who used
  glucose, insulin, carbohydrate, and activity data. Corrected in
  `PRESENTATION_MONDAY.md` and `make_pptx_final.py`.]**

- Bai, S., Kolter, J. Z., & Koltun, V. (2018). _An empirical evaluation of
  generic convolutional and recurrent networks for sequence modeling_.
  arXiv:1803.01271. **[Verified — provided pre-verified]**

- Biau, G., & Scornet, E. (2016). A random forest guided tour. _TEST,
  25_(2), 197–227. https://doi.org/10.1007/s11749-016-0481-7
  **[Verified]**

- Cerqueira, V., Torgo, L., & Mozetič, I. (2020). Evaluating time series
  forecasting models: An empirical study on performance estimation
  methods. _Machine Learning, 109_(11), 1997–2028.
  https://doi.org/10.1007/s10994-020-05910-7 **[Verified]**

- Hameed, H., & Kleinberg, S. (2020). Comparing machine learning
  techniques for blood glucose forecasting using free-living and patient
  generated data. _Machine Learning for Healthcare (MLHC)_, PMLR 126.
  **[Verified — provided pre-verified]**

- Kalita, D., & Mirza, K. B. (2025). Multivariate glucose forecasting
  using deep multihead attention layers inside neural basis expansion
  networks. _IEEE Journal of Biomedical and Health Informatics, 29_(5),
  3654–3663. https://doi.org/10.1109/JBHI.2025.3530461
  **[Verified — extracted directly from the source PDF. Reported result:
  16.57 ± 2.56 mg/dL RMSE (MARD 6.81 ± 1.39%) on OhioT1DM at a 30-min
  prediction horizon; also reports 29.25 ± 6.02 mg/dL at 60-min.]**

- Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic
  optimization. _3rd International Conference on Learning Representations
  (ICLR 2015)_. arXiv:1412.6980. **[Verified]**

- Marling, C., & Bunescu, R. (2020). The OhioT1DM dataset for blood
  glucose level prediction: Update 2020. _5th International Workshop on
  Knowledge Discovery in Healthcare Data (KDH@ECAI 2020)_, CEUR Workshop
  Proceedings, 2675, 71–74. **[Verified]**

- Martinsson, J., Schliep, A., Eliasson, B., & Mogren, O. (2020). Blood
  glucose prediction with variance estimation using recurrent neural
  networks. _Journal of Healthcare Informatics Research, 4_(1), 1–18.
  https://doi.org/10.1007/s41666-019-00059-y **[Verified]**

- Prechelt, L. (1998). Early stopping — but when? In G. B. Orr & K.-R.
  Müller (Eds.), _Neural Networks: Tricks of the Trade_ (Lecture Notes in
  Computer Science, Vol. 1524, pp. 55–69). Springer.
  https://doi.org/10.1007/3-540-49430-8_3 **[Verified]**

- Probst, P., Wright, M. N., & Boulesteix, A.-L. (2019). Hyperparameters
  and tuning strategies for random forest. _WIREs Data Mining and
  Knowledge Discovery, 9_(3), e1301. https://doi.org/10.1002/widm.1301
  **[Verified — provided pre-verified]**

- Rodríguez-Rodríguez, I., Campo-Valera, M., Rodríguez, J.-V., & Woo, W. L.
  (2024). IoMT innovations in diabetes management: Predictive models
  using wearable data. _Expert Systems with Applications, 238_, Article 121994. https://doi.org/10.1016/j.eswa.2023.121994 **[Verified —
  extracted directly from the source PDF. Reported result: Random Forest
  model, 18.60 mg/dL RMSE at a 30-min horizon (six-hour input window),
  and 26.21 mg/dL at 45-min, on the authors' own passively-monitored
  cohort of 40 DM1 patients (also corroborated on 10 Type 2 DM patients
  as a proof of concept) — not OhioT1DM.]**

- Srivastava, N., Mansimov, E., & Salakhutdinov, R. (2015). Unsupervised
  learning of video representations using LSTMs. _Proceedings of the 32nd
  International Conference on Machine Learning (ICML)_, PMLR 37, 843–852.
  **[Verified — provided pre-verified]**

- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez,
  A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need.
  _Advances in Neural Information Processing Systems 30 (NeurIPS 2017)_,
  5998–6008. **[Verified]**

- Xie, J., & Wang, Q. (2020). Benchmarking machine learning algorithms on
  blood glucose prediction for type I diabetes in comparison with
  classical time-series models. _IEEE Transactions on Biomedical
  Engineering, 67_(11), 3101–3124.
  https://doi.org/10.1109/TBME.2020.2975959 **[Verified — this citation
  appeared in `PREPROCESSING_COMPARISON.md` §5 without full bibliographic
  detail; now confirmed and should replace the shorthand "Xie & Wang
  (2020)" wherever it is cited.]**

- Xiong, X., Yang, X., Cai, Y., Xue, Y., He, J., & Su, H. (2025).
  Exploring the potential of deep learning models integrating transformer
  and LSTM in predicting blood glucose levels for T1D patients. _DIGITAL
  HEALTH, 11_, Article 20552076251328980.
  https://doi.org/10.1177/20552076251328980 **[Verified — cited here for
  the authors' own Transformer+LSTM architecture. The 19.33 mg/dL @
  30-min RMSE figure previously attributed to this paper in the
  literature-comparison slides was a misattribution; that figure belongs
  to Bertachi et al. (2018) — see the Bertachi entry above. Do not
  reattach 19.33 to Xiong et al.]**

- Zhu, T., Li, K., Herrero, P., Chen, J., & Georgiou, P. (2018). A deep
  learning algorithm for personalized blood glucose prediction. _3rd
  International Workshop on Knowledge Discovery in Healthcare Data
  (KDH@IJCAI-ECAI 2018)_, CEUR Workshop Proceedings, 2148, 64–78.
  **[Verified — this paper's own contribution is a CNN/RNN glucose
  prediction model. It was previously miscited in §5/§6/§9 (as "Zhu,
  Li, Herrero & Georgiou," dropping fourth author Jianwei Chen) as the
  source of a −2.1 mg/dL linear-interpolation effect; that claim has now
  been removed and replaced with the verified Hameed & Kleinberg (2020)
  interpolation figure. Zhu et al. (2018) should only be cited going
  forward for its actual CNN/RNN contribution, not for interpolation.]**
