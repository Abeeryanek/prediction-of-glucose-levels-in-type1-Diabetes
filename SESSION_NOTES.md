## All 4 Remaining Tasks — COMPLETED

### Task 1: Training epochs documented
All 4 DL model train_model() functions return actual_epochs.
run_experiments.py reports mean +/- std epochs per model.

### Task 2: OhioT1DM 2020 cohort acceleration features
run_ablation_2020.py — acceleration hurts exactly like heartrate/steps.
Also fixed critical loader bug: empty <basis_heart_rate> placeholder
in 2020 XML files was misclassifying cohort. Fixed in ohio_loader.py.

### Task 3: Leave-One-Patient-Out cross-validation
run_lopo.py — LSTM degrades only 1.17 mg/dL (5.5%) from personalised
to population. RF improves by 0.54 mg/dL with pooling.

### Task 4: Glucdict all 5 models
run_glucdict_experiments.py extended — AE=14.01, LSTM=14.03,
Transformer=14.09, RF=15.23, TCN=15.90 mg/dL.

### Task 5: Cross-dataset transfer learning
run_transfer.py — domain shift costs only +1.44/+1.92 mg/dL despite
different populations and CGM devices.

### Key Scientific Narrative (3 findings for presentation)
1. Wearable features consistently hurt at 30-min horizon (OhioT1DM
   2018, OhioT1DM 2020, Glucdict — all three datasets)
2. Personalisation barely matters — LOPO gap only 1.17 mg/dL
3. Transfer across datasets costs only ~1-2 mg/dL — glucose
   autocorrelation dominates regardless of population or device

### Next: BUILD PRESENTATION
All experiments done. Now need presentation for next meeting.
