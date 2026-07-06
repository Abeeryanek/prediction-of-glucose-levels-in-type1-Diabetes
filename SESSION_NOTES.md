## Session Summary — Latest

### What was completed
- Transformer model implemented (src/models/transformer.py)
- Glucdict dataset loader implemented (src/preprocessing/glucdict_loader.py)
- Pipeline _EVENT_COLS updated for Glucdict event columns
- run_experiments.py extended to 5 models (RF/LSTM/AE/TCN/Transformer)
- All committed and pushed

### Still todo before next meeting
1. Re-run full OhioT1DM experiment (5 models now including Transformer)
   python -u run_experiments.py
2. Create run_glucdict_experiments.py (same structure as run_experiments.py)
3. Remove patients 563+575, re-run feature ablation
4. Expand Grid Search to Autoencoder, TCN, Transformer
5. Clarke Error Grid for ALL feature combinations
6. Document training epochs in results
7. Preprocessing comparison table vs literature

### Key paths
- Glucdict raw data: C:\Users\mefte\...\Glucdict Dataset\Glucdict Dataset
- Experiment results: results/ohio/
- Models: src/models/
