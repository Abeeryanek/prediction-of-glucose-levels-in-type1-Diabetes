## Session Summary — Latest

### Everything completed and committed
- 5 models: RF, LSTM, Autoencoder, TCN, Transformer
- Grid Search: all 5 models tuned (results in grid_search_results.json)
- OhioT1DM full experiment: results_all_models.csv committed
- Clean feature ablation (without 563+575): committed
- Glucdict: loader + experiments committed
- Clarke Error Grid for every feature combination, both datasets: committed
  (results/ohio/clarke_ablation_clean_all_featuresets.png,
  results/glucdict/clarke_glucdict_all_featuresets.png)
- Fixed a Random Forest bug: nested joblib parallelism (MultiOutputRegressor
  n_jobs=-1 wrapping RandomForestRegressor n_jobs=-1) was crashing on Windows
  with OS resource exhaustion — outer wrapper now n_jobs=1

### Tuned hyperparameters
  RF:          n_estimators=200, max_depth=10, min_samples_leaf=4
  LSTM:        hidden_size=128, lr=5e-4
  Autoencoder: latent_size=16, lr=5e-4
  TCN:         num_filters=32, lr=5e-4
  Transformer: d_model=128, nhead=4, lr=5e-4

### Key findings this session
- glucose_only ≈ clinical on both datasets — bolus/carbs/meal-events add
  almost nothing at the 30-min horizon
- full/wearable feature sets are consistently worst AND highest-variance on
  both OhioT1DM and Glucdict — extra heartrate/accelerometer features hurt,
  not help; confirmed via RMSE and Clarke Zone A/D (Zone A drops from 88.6%
  to 75.6% on Ohio, with 1.5% landing in dangerous Zone D)
- Tuned Transformer (22.91 mg/dL @ 30min) now beats RF and TCN — was worse
  than both when untuned/hardcoded

### Not yet committed
- PRESENTATION_MONDAY.md / .pptx / make_pptx.py — content is fully updated
  with this session's results (15 slides), just needs `git add` + commit
  when ready

### Tomorrow — in this exact order
1. Preprocessing comparison table vs literature
   Create a markdown/CSV table comparing our steps to the papers

2. Detailed final experiments plan document

3. Commit presentation files (see "Not yet committed" above)
