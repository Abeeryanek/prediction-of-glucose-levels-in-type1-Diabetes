# Finer Preprocessing Analysis: Discrepancies vs Literature

Blood Glucose Forecasting — University of Duisburg-Essen

## Our Current Preprocessing

1. CGM resampled to 5-min grid via `resample('5min').mean()`
2. Event features (carbs, bolus, eat/drink) NaN filled with 0
3. Sensor NaN rows dropped entirely (no interpolation)
4. No smoothing/filtering applied to CGM
5. StandardScaler fit per-patient on training data only
6. Sliding window of 12 steps (1h), no gap-length check

## Discrepancy 1 — No CGM Smoothing/Filtering (HIGH impact)

Ours: none — raw resampled glucose goes into windowing.
Literature: Hameed & Kleinberg (2020) apply a median filter (window=5) and
show it improves RMSE regardless of imputation approach. Kalita & Mirza
(2025) apply smoothing.
Likely effect: HIGH. Unfiltered CGM noise propagates into every window.
Planned test: median filter window=5, compare RMSE.

## Discrepancy 2 — No Gap-Length Distinction (MODERATE)

Ours: every empty 5-min bin becomes NaN and the row is dropped, whether the
gap is 5 min or 3 hours.
Literature: standard practice interpolates short gaps (<30 min per
Hameed & Kleinberg) and splits sequences at long gaps.
Likely effect: MODERATE — combined with Discrepancy 3.

## Discrepancy 3 — Temporal Continuity Broken by Row-Dropping (HIGH)

Ours: after `dropna()`, windowing treats remaining rows as contiguous, so a
12-step window can silently span a real-time gap of hours while the model
assumes 12 consecutive 5-min steps.
Literature: interpolation-based pipelines preserve true temporal spacing;
sequences are split at large gaps so windows never cross them.
Likely effect: HIGH — some training windows encode physically impossible
glucose jumps, adding noise to the learning signal.
Planned test: split sequences at gaps > 6 steps (30 min) so no window
crosses a large gap; compare RMSE.

## Discrepancy 4 — No Derived Features (MODERATE, needs literature check)

Ours: raw glucose, bolus, carbs only — no engineered features.
Literature: derived features such as glucose rate-of-change (first
difference) and time-of-day are commonly used in glucose forecasting
preprocessing pipelines (e.g. patent US11426102 describes rate-of-change
after carbohydrate consumption and time-of-day statistical measures as
standard extracted features). We have NOT yet confirmed the exact derived
features used by our specific comparison papers (Kalita & Mirza 2025,
Xiong 2025) — this must be verified against those sources before any
claim about their feature engineering is made.
Likely effect: MODERATE — derived features explicitly encode trend and
circadian patterns the model would otherwise infer implicitly.
Planned test: add glucose velocity (first difference) and hour-of-day,
compare RMSE — this is a self-contained experiment independent of what
any specific paper did.

## Discrepancy 5 — Normalisation Scope (LOW)

Ours: per-patient StandardScaler on training data.
Literature: mixed — some global z-score, some fixed physiological range.
Likely effect: LOW for RMSE, but relevant to transfer learning.

## Summary Ranking

| Discrepancy            | Likely impact | Planned experiment             |
| ---------------------- | ------------- | ------------------------------ |
| 1. No smoothing        | HIGH          | median filter window=5         |
| 3. Temporal continuity | HIGH          | split sequences at gaps >30min |
| 2. Gap-length handling | MODERATE      | interpolate <30min gaps        |
| 4. Derived features    | MODERATE      | add velocity + time-of-day     |
| 5. Normalisation scope | LOW           | global vs per-patient          |

## Connection to Interpolation Experiment

Discrepancies 1, 2, 3 are tested together in the interpolation experiment
matrix (see `FINAL_EXPERIMENTS_PLAN_DETAILED.md` §6, "Interpolation
Experiments" — method × max gap length × median filtering), which already
varies imputation method, gap length, and median filtering. This document
adds derived features (4) and normalisation scope (5) as additional
factors worth isolating.

## Caveat on Literature Citations

The Hameed & Kleinberg (2020) claims used above (median filter window=5,
<30min gap threshold — Discrepancies 1 and 2) have been verified against
the source paper. The Kalita & Mirza (2025) derived-features attribution
originally used for Discrepancy 4 could NOT be verified against that
source and has been reworded above to avoid an unverified attribution —
the general practice is now supported by a patent citation (US11426102)
instead, and the specific comparison papers (Kalita & Mirza 2025,
Xiong 2025) are flagged as still needing that check before any claim
about their feature engineering is made in the thesis or presentation.
