# Final Experiments Plan

Blood Glucose Forecasting for Insulin Dose Optimisation — University of
Duisburg-Essen. Khalil & Abeer. Last updated 2026-07-11.

## 1. Overview

By the end of the project we aim to deliver a benchmark of **5 forecasting
models** (Random Forest, LSTM, Seq2Seq Autoencoder, TCN, Transformer) across
**3 datasets** (OhioT1DM, Glucdict, BIG IDEAs), covering:

- Which model architecture generalises best across datasets and patients,
  not just which wins on one cohort.
- Which feature combination is actually useful — our results so far show
  added wearable sensors (heartrate, steps, accelerometer) consistently
  *hurt* accuracy rather than help, on both OhioT1DM and Glucdict.
- Clinical safety, not just RMSE — every model/feature combination is
  scored with the Clarke Error Grid (Zone A/B/D), since a model can have a
  competitive RMSE while still producing dangerous outlier predictions.
- Whether population-level generalisation (across-patient, across-dataset)
  is feasible, or whether per-patient personalisation remains necessary —
  this is the open question the remaining experiments (LOPO, cross-dataset
  transfer) are designed to answer.

## 2. Completed experiments

| Experiment | Scope | Status |
|---|---|---|
| OhioT1DM full benchmark | 5 models × 12 patients × 3 horizons (5/15/30 min), clinical features, Grid Search tuned | ✅ Committed (`results/ohio/results_all_models.csv`) |
| OhioT1DM feature ablation | 8 feature combinations (glucose_only, clinical, glucose_steps, clinical_steps, clinical_hr, glucose_hr, full, wearable) × RF/LSTM, clean 2018 cohort (excl. patients 563/575 — sparse wristband coverage) | ✅ Committed (`results/ohio/results_feature_ablation_clean.csv`) |
| Glucdict pipeline + ablation | 4 feature combinations (glucose_only, glucose_activity, glucose_hr, full_wearable) × RF/LSTM, 13 users, 30-min horizon | ✅ Committed (`results/glucdict/results_glucdict_ablation.csv`) |
| BIG IDEAs RF baseline | RF only, Abeer's work — not yet merged into the shared `src/` pipeline or benchmark | ⏳ Done standalone, integration pending |
| Clarke Error Grid (all feature combinations, both datasets) | OhioT1DM + Glucdict, every feature set, 30-min horizon | ✅ Committed |
| Preprocessing comparison vs. literature | See `PREPROCESSING_COMPARISON.md` | ✅ Committed |

## 3. Remaining experiments

| Experiment | Dataset | Models | Feature sets | Horizon | Priority | Status |
|---|---|---|---|---|---|---|
| a. Extended feature sets, 2020 cohort | OhioT1DM 2020 | RF, LSTM | glucose_only, clinical, glucose_accel, clinical_accel, full (accel replaces heartrate/steps — 2020 cohort uses Empatica accelerometer, not Basis wristband) | 30 min | High | Not started |
| b. Leave-One-Patient-Out (LOPO) CV | OhioT1DM (12 patients) | RF, LSTM, best DL model from Slide 7 | clinical | 30 min | High | Not started |
| c. Glucdict — all 5 models | Glucdict (13 users) | Autoencoder, TCN, Transformer (RF/LSTM already done) | glucose_only, glucose_activity (top-2 from existing ablation) | 30 min | High | Not started |
| d. BIG IDEAs — all 5 models | BIG IDEAs | LSTM, Autoencoder, TCN, Transformer (RF already done by Abeer) | clinical-equivalent feature set | 30 min | Medium | Not started |
| e. Cross-dataset transfer | Train: OhioT1DM → Test: Glucdict | RF, LSTM (best 2 from OhioT1DM) | glucose_only (only features common to both datasets) | 30 min | Medium | Not started |
| f. Longer window sizes | OhioT1DM | RF, LSTM (best 2 architectures) | clinical | 30 min | Medium | Not started |
| g. Additional prediction horizons | OhioT1DM | All 5 models (reuse tuned hyperparameters) | clinical | 45 min, 60 min | Low–Medium | Not started |
| h. Unified Clarke Error Grid, 3-dataset comparison | OhioT1DM + Glucdict + BIG IDEAs | All 5 models | best feature set per dataset | 30 min | Medium | Not started |

Notes on scope:
- (a) window size stays at 12 steps (60 min) / horizon 6 steps (30 min) — only the feature set changes (`acceleration` substitutes `heartrate`/`steps`, since 2020-cohort XML files don't contain a `basis_heart_rate` node).
- (f) "longer window sizes" means `window_size ∈ {24, 36, 72}` (2 h / 3 h / 6 h of history) at the fixed 30-min horizon, to test whether more history compensates for the 3-feature limitation flagged in Slide 10.
- (h) depends on (c) and (d) being far enough along to have at least one DL model result per dataset; if either slips, this becomes a 2-dataset comparison (OhioT1DM + Glucdict) instead.

## 4. Timeline

**Week 1 (2026-07-13 to 2026-07-17)**
- (a) OhioT1DM 2020 cohort extended feature sets
- (b) LOPO cross-validation
- (c) Glucdict — remaining 3 models (Autoencoder, TCN, Transformer)

**Week 2 (2026-07-20 to 2026-07-24)**
- (d) BIG IDEAs — remaining 4 models, merge Abeer's RF baseline into shared benchmark format
- (e) Cross-dataset transfer (OhioT1DM → Glucdict)
- (f) Longer window sizes (24/36/72 steps)

**Week 3 (2026-07-27 to 2026-07-31)**
- (g) 45-min and 60-min horizons
- (h) Unified 3-dataset Clarke Error Grid comparison
- Final presentation and written report

## 5. Expected outcomes

- **(a) 2020 cohort extended features:** expect the same pattern already
  seen twice (Slide 8, Slide 8b) — `glucose_only`/`clinical` outperforming
  any feature set that adds a wearable sensor, this time with accelerometer
  instead of heartrate/steps. A third confirmation would make this a robust,
  citable finding rather than a two-dataset coincidence.
- **(b) LOPO:** expect a measurable RMSE increase vs. per-patient
  personalised models (Slide 7's 21.21 mg/dL for LSTM), since personalised
  models currently outperform population approaches in most cited literature
  (`PREPROCESSING_COMPARISON.md` — BGLP-challenge convention is
  per-patient). LOPO quantifies exactly how much personalisation is worth,
  which is a required discussion point for the "clinical deployability"
  angle of the thesis.
- **(c)/(d) Remaining model coverage:** completes the 5×3 model/dataset
  grid, which is necessary before any cross-dataset claim can be made —
  right now Glucdict and BIG IDEAs are not comparable to OhioT1DM on
  architecture coverage.
- **(e) Cross-dataset transfer:** expect a substantial RMSE degradation
  (different CGM devices — Medtronic Enlite vs. Dexcom G6 — different
  patient populations, different sampling of daily life). A large gap here
  would support LOPO/personalisation as the more promising direction over
  transfer learning; a small gap would be a stronger, more novel result.
- **(f) Longer windows:** expect diminishing or negative returns beyond
  60–120 min of history for RF/LSTM on this data (glucose dynamics are
  dominated by the last ~30–60 min per the existing ablation results), but
  this directly tests whether TCN's underperformance (Slide 11, point 4)
  is caused by too little context rather than the architecture itself.
- **(g) 45/60-min horizons:** expect roughly linear RMSE growth with
  horizon (matches the 5→15→30 min progression already observed in Slide 7:
  ~5.8 → ~13 → ~21 mg/dL), useful for characterising the accuracy/lead-time
  trade-off for insulin-dosing decision support.
- **(h) Unified Clarke EGA:** the single figure most directly usable in the
  thesis's clinical-safety chapter — one glance at Zone A/D across all three
  datasets and five models.

**Overall contribution:** a reproducible, literature-grounded benchmark
showing that (1) minimal feature sets (glucose ± bolus/carbs) are
sufficient and additional wearable sensors measurably hurt accuracy across
independent datasets and cohorts, and (2) a quantified answer to how much
per-patient personalisation is worth relative to population and
cross-dataset generalisation — a gap not clearly addressed by the
literature comparison in Slide 10, whose cited models don't report LOPO or
transfer results.

## 6. Open questions for the professor

1. Should **LOPO cross-validation** be run on all three datasets, or is
   OhioT1DM alone sufficient given the time budget (Week 1 vs. a
   multi-week extension)?
2. Should **cross-dataset transfer learning** (fine-tuning on a target
   dataset after pre-training on another, rather than the simpler
   zero-shot transfer in experiment (e)) be explored, or is LOPO considered
   sufficient evidence for the generalisation question?
3. Is the **60-minute horizon** a required deliverable, or is 30-minute the
   final target for the thesis, with 45/60-min treated as optional
   supplementary results if time permits?
