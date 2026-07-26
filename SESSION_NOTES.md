## Session Summary — Latest

### ALL experiments complete
- 5 models (RF, LSTM, Autoencoder, TCN, Transformer)
- Clarke Error Grid unified with Abeer via clarke-error-grid library v0.1.4
- 45-min horizon added (results at 15/30/45 for all models)
- LOPO complete: 5 models, 3 horizons, 2 modes (pooled + per_patient_scaled)

### ALL email TODOs done
- Grid search documented (GRID_SEARCH_DOCUMENTATION.md)
- Finer preprocessing analysis (PREPROCESSING_DISCREPANCIES_DETAILED.md)
- Detailed experiment plan sent to professor (awaiting his OK)

### Key LOPO findings
- Personalisation gap small (0.3-1.2 mg/dL) across ALL 5 models
- per_patient_scaled beats pooled — answers professor's variation request
- TCN highlight: pooled gap +6.51 at 45min → +0.53 with per-patient 
  scaling. TCN's weakness was inter-patient baseline differences.

### TOMORROW — the only thing left: THE PRESENTATION
BLOCKER: need the Moodle presentation template (professor requires it).
Step 1: get template from Moodle course, add to repo
Step 2: build deck in that template with all results

Presentation must include:
- Best feature combo per model (data ready)
- Multi-horizon 15/30/45 results
- Clarke at all horizons (unified library)
- LOPO all 5 models + pooled vs per_patient_scaled
- Transfer slide with "Personalised" clarification
- Grid search all datasets
- Preprocessing discrepancies (5)
- Three findings WITH Glucdict caveat (wearables "do not improve" 
  not "consistently hurt" — Glucdict glucose_activity is exception)

### Poster: Abeer done. Abeer's horizon investigation: Abeer done.
