## Session Summary — Latest Update

### Completed today
- Transformer model implemented (src/models/transformer.py)
  Vaswani et al. 2017 / Xiong et al. 2025 / Kalita & Mirza 2025
- Glucdict dataset loader implemented (src/preprocessing/glucdict_loader.py)
  13 users, Dexcom G6 CGM + TicWatch heart rate + accelerometer + activity logs
  Smoke test passed: User1 shape (2838, 6)
- Full 5-model OhioT1DM experiment completed and committed (7d50788)
- Abeer's branch merged cleanly into main

### 5-Model Results (30-min horizon, 12 patients)
  Autoencoder:  21.45 ± 3.52  Zone A: 87.4%
  LSTM:         ~21.25         Zone A: 87.2%
  Transformer:  23.01 ± 4.05  Zone A: 84.7%
  RF:           ~23.35         Zone A: ~86.4%
  TCN:          24.82 ± 3.65  Zone A: 81.3%

### Tomorrow — in this exact order
1. Run run_feature_ablation_clean.py
   (re-run without patients 563+575 — professor's request)
   Create the file first, then: python -u run_feature_ablation_clean.py

2. Create run_glucdict_experiments.py and run it
   Feature sets to test: glucose_only, glucose_hr,
   glucose_activity, full_wearable

3. Expand Grid Search to all 5 models (AE, TCN, Transformer)

4. Clarke Error Grid for ALL feature combinations

5. Prepare presentation for next meeting
