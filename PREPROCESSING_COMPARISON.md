# Preprocessing Comparison: This Project vs. Literature

Comparison of the preprocessing pipeline (`src/preprocessing/ohio_loader.py`,
`src/preprocessing/glucdict_loader.py`, `src/training/pipeline.py`) against
published blood-glucose-forecasting studies, most of which also use the
OhioT1DM dataset.

## Reference studies

| Ref | Study | Dataset | Notes |
|---|---|---|---|
| [MB20] | Marling & Bunescu (2020), *The OhioT1DM Dataset for Blood Glucose Level Prediction: Update 2020*, KDH@ECAI | OhioT1DM (source paper) | Defines the dataset itself: 12 patients, 8 weeks, 5-min CGM, wristband sensors |
| [MSEM20] | Martinsson, Schliep, Eliasson, Meijner et al. (2020), *Blood Glucose Prediction with Variance Estimation Using Recurrent Neural Networks*, J. Healthcare Informatics Research | OhioT1DM | RNN/LSTM, BGLP-challenge style personalized models |
| [ZLHG18] | Zhu, Li, Herrero, Georgiou (2018), *A Deep Learning Algorithm for Personalized Blood Glucose Prediction*, KDH@IJCAI-ECAI | OhioT1DM | Dilated-CNN (WaveNet-style); explicitly compares missing-data strategies |
| [XW20] | Xie & Wang (2020), *Benchmarking Machine Learning Algorithms on Blood Glucose Prediction for Type I Diabetes in Comparison With Classical Time-Series Models*, IEEE TBME | OhioT1DM | Large benchmark of ML vs. classical (ARX) models, standard BGLP protocol |

## Comparison table

| Preprocessing step | This project | Literature | Assessment |
|---|---|---|---|
| **CGM resampling frequency** | 5-min grid, built with `resample("5min").mean()` on CGM timestamps (tolerates sub-second jitter) | Universal 5-min grid — matches the native CGM sampling rate of the OhioT1DM Dexcom sensors [MB20][MSEM20][XW20] | **Matches literature standard.** No deviation. |
| **Missing CGM values** | Rows with NaN glucose are **dropped**, never interpolated (`_preprocess` in `pipeline.py`) | Split in the literature: Martinsson et al. also skip windows that would require imputed glucose, training only on complete `(x, y)` pairs [MSEM20]. Zhu et al. instead **interpolate** — first-order (linear) interpolation reduced mean RMSE by 2.1 mg/dL vs. no imputation, but large gaps still hurt performance [ZLHG18]. Several later works (e.g. Transformer/imputation-smoothing studies) compare linear/cubic/spline/PCHIP interpolation. | **One valid literature branch matches ours** (drop, no interpolation — MSEM20); the other branch (interpolate short gaps) is a documented alternative we deliberately did not take, to avoid injecting synthetic glucose values into ground truth. |
| **Missing wearable sensor values** (heartrate, GSR, skin_temp, acceleration) | Dropped along with the row if used as a feature (never interpolated) — same policy as glucose | Not universally addressed; where addressed, gaps are usually **linearly interpolated** with a bounded gap limit (e.g. our own earlier, unused BigIDEAS exploration script used `limit=6` samples = 30 min) | We are stricter than most published pipelines for wearable features — deliberate choice, consistent with never fabricating sensor readings; documented as a limitation, since it costs sample count on the `full`/`wearable` feature sets. |
| **Missing event features** (carbs, bolus, eat/drink events) | NaN → 0 (absence of a logged event, not a missing sensor) | Standard practice; event channels are implicitly zero-filled in all reviewed studies since "no bolus logged" is a valid state, not a gap [MB20][ZLHG18] | **Matches literature standard.** |
| **Sentinel / out-of-range values** | Glucdict Dexcom "Low"/"נמוך" (<40 mg/dL text) → filled with 39.0 mg/dL | Consistent with the Dexcom G6 published detection floor (40 mg/dL); OhioT1DM's Medtronic Enlite sensor has a similar device-defined range and is not separately clipped in the source papers | **Matches device-defined convention**, not an arbitrary choice. |
| **Outlier / physiological clipping** | None — no additional clipping beyond device range | Not commonly applied in the reviewed studies either; a few later imputation-focused papers add smoothing (Kalman, smoothing splines) on top of interpolation, which we do not use since we don't interpolate | **Matches literature standard** (no extra clipping/smoothing layer). |
| **Normalization / scaling** | `StandardScaler` (z-score), fit on **training data only**, applied to val/test | Mixed: Martinsson et al. use a fixed scalar (glucose × 0.01) rather than a fitted scaler [MSEM20]; most ML-benchmark papers (RF/SVR/ensemble-heavy studies, incl. XW20-style pipelines) use z-score/min-max scaling fit on train only | **Matches the dominant convention** (train-only fit to prevent leakage); differs from MSEM20's simpler fixed-constant scaling, which is RNN-specific and doesn't generalize to RF/TCN input scales the way this project needs. |
| **Model personalization** | Per-patient: `make_splits` is called separately per patient; scaler, model, and evaluation are all patient-specific | Standard BGLP-challenge protocol is per-patient personalized models [MB20][MSEM20][XW20]; pooled/population models are a minority, more recent (transfer-learning / meta-learning) approach | **Matches the dominant literature convention** for this dataset. |
| **History window length** | 12 steps = 60 minutes (`window_size=12`) | 60 minutes is the most commonly reported optimal history length for 30/60-min-ahead prediction, confirmed by hyperparameter search in [MSEM20] | **Matches literature-optimal value.** |
| **Prediction horizon(s)** | 6 steps = 30 minutes (`horizon=6`) | 30 min and 60 min are the two standard BGLP-challenge horizons across nearly all reviewed studies [MB20][MSEM20][XW20] | **Matches literature standard** (we report 30-min; 60-min not yet run). |
| **Train / validation / test split** | Official OhioT1DM train/test XML split; validation carved from the **chronological end** of the training split (`val_ratio=0.20`), never shuffled | Chronological splits are standard for CGM time series; Martinsson et al. use a 60/20/20 chronological split of the *combined* record rather than the dataset's official train/test files [MSEM20]; other studies use the official OhioT1DM train/test division directly, as we do [MB20][XW20] | **Matches literature standard**; we prefer the official train/test division (comparable to published OhioT1DM benchmark numbers) over an arbitrary re-split. |
| **Cross-validation strategy** | Walk-forward CV via `TimeSeriesSplit` for hyperparameter search (`walk_forward_splits`), scaler refit per fold | Walk-forward / blocked CV is the accepted method for autocorrelated time series (Cerqueira et al. 2020; Bergmeir & Benítez 2012 — already cited in `pipeline.py`), used across the glucose-forecasting literature to avoid future leakage | **Matches literature standard.** |

## Summary

Of the 12 preprocessing decisions compared, **10 align directly with the
dominant approach in the literature** (5-min resampling, event zero-fill,
sentinel handling, train-only scaling, per-patient models, 60-min window,
30-min horizon, official train/test split, walk-forward CV, no extra
clipping/smoothing). One decision — **no interpolation of missing glucose or
sensor values** — follows one documented branch of the literature
(Martinsson et al.) rather than the more common interpolation-based branch
(Zhu et al. and later imputation-focused papers); this is a deliberate
choice to keep all reported glucose/sensor values real sensor readings, at
the cost of some sample count on gap-heavy patients and feature sets. The
remaining decision — dropping (rather than interpolating) missing wearable
sensor rows — is stricter than most published pipelines and should be
called out explicitly as a limitation in the thesis, since it is the most
likely explanation for reduced sample counts on the `full`/`wearable`
feature sets relative to `glucose_only`/`clinical`.

## References

- [MB20] Marling, C., & Bunescu, R. (2020). The OhioT1DM Dataset for Blood
  Glucose Level Prediction: Update 2020. *KDH@ECAI 2020*.
- [MSEM20] Martinsson, J., Schliep, A., Eliasson, B., & Mogren, O. (2020).
  Blood Glucose Prediction with Variance Estimation Using Recurrent Neural
  Networks. *Journal of Healthcare Informatics Research*, 4, 1–18.
- [ZLHG18] Zhu, T., Li, K., Herrero, P., & Georgiou, P. (2018). A Deep
  Learning Algorithm for Personalized Blood Glucose Prediction.
  *KDH@IJCAI-ECAI 2018*.
- [XW20] Xie, J., & Wang, Q. (2020). Benchmarking Machine Learning
  Algorithms on Blood Glucose Prediction for Type I Diabetes in Comparison
  With Classical Time-Series Models. *IEEE Transactions on Biomedical
  Engineering*, 67(11), 3101–3124.
