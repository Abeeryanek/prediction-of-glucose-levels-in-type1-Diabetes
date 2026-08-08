## Weekend solo work — progress (Ohio/Glucdict side only)

Goal: finish unifying MY shared pipeline (src/models/) and re-run Ohio +
Glucdict at final settings. BIG IDEAs left for after Monday (needs Abeer
+ architecture decision).

DONE this session:
- loss_fn hook added + verified non-breaking (bit-identical with
  loss_fn=None) in: lstm.py (earlier), autoencoder.py, tcn.py

STILL TODO on hooks:
- transformer.py — loss_fn hook not yet added (last of the 4 neural models)

REMAINING weekend steps (in order):
1. Add loss_fn hook to transformer.py (verify non-breaking)
2. Wire RF weighting into run_experiments.py (capability exists in
   random_forest.py, not yet called)
3. Wire weighted loss into neural models in the run scripts
4. Verify Glucdict runs with all changes
5. Re-run Ohio + Glucdict at final settings

## ⚠️ UNRESOLVED DECISION — weighted vs unweighted reporting

Question: how to report against literature. Literature RMSE (Bertachi
19.33, Kalita 16.57, Rodriguez 18.60) is UNWEIGHTED. Reporting
weighted-trained RMSE against unweighted literature makes our numbers
look artificially worse (weighting trades RMSE for hypo/hyper accuracy).
LEANING: run BOTH — unweighted for literature comparison (apples-to-
apples), weighted for clinical-safety story (Clarke Zone A).
→ Also a good question to raise with professor Monday.
DO NOT commit to weighted-only-vs-unweighted-literature — it invites
"why is your RMSE worse" with no good answer.

## For Monday presentation:
- "Finished": unified pipeline re-run cleanly on Ohio + Glucdict (goal)
- Open/next: BIG IDEAs unification (needs Abeer + architecture decision),
  interpolation experiments (not started), CNN-LSTM port
- Decisions to get from professor: (1) architecture-match requirement
  across datasets, (2) weighted vs unweighted reporting framing

## Unification status — ACCURATE (per this session's evidence)

Confirmed with supervisor: unification target is model-family/training/
weighting/metrics level; dataset-specific loaders are intended and
approved.

VERIFIED done (evidence in this session's transcript):
- Training conditions 150 epochs / patience 15 / batch 32 / seed 42:
  confirmed at the actual call sites, not just constant declarations —
  for Ohio (src/models/*.py) AND BIG IDEAs (clean_treaining/*.py). Grid-
  search's reduced budget (30 epochs/patience 5) proven NOT to leak into
  final results: the searched model is discarded in code, the cache
  holds only hyperparameters (plain JSON, structurally can't hold a
  trained model), and a fresh model is instantiated on every call.
- Shared calculate_clinical_weights DEFINED in src/training/losses.py,
  verified numerically identical to Abeer's original scheme (all bands +
  boundaries, exact match).
- GB ported to src/models/gradient_boosting.py with weighting (via the
  shared function), matching RF's train()/evaluate() interface. Sanity-
  run on real OhioT1DM data: RMSE ~23 mg/dL, sample_weight confirmed
  non-trivial and affecting results.
- RF weighting capability added to src/models/random_forest.py (same
  shared-function import, same pattern as GB).
- BIG IDEAs neural models units-correct (mg/dL metrics: y_test_raw vs
  inverse-transformed preds) — independently re-verified for all 5
  (lstm, tcn, transformer, autoencoder, cnn_lstm), not just the 2 fixed
  in the y-scaling commit.
- Weighted-MSE loss (clinically_weighted_mse_scaled) refactored to call
  the shared function — non-breaking, verified identical loss values
  before/after across 1D and 2D (multi-step) shapes.

STILL OPEN (do not claim "unification complete" while these stand):
- CNN-LSTM NOT yet ported into src/models/ — no shared counterpart
  exists at all, nothing to point a BIG IDEAs runner at.
- BIG IDEAs' 7 clean_treaining scripts each hold their own DUPLICATE
  copy of calculate_clinical_weights, not an import of the shared one —
  content verified identical, but not actually single-source (8 copies
  of the same logic exist in the repo).
- RF's weighting capability is not yet wired into run_experiments.py
  (or any run script) — it exists on the model but isn't invoked with
  weighting on anywhere yet.
- The loss_fn hook exists only on lstm.train_model among the 4 shared
  neural models — autoencoder.py, tcn.py, transformer.py have no such
  parameter yet, so they can't be switched to weighted loss without
  further changes.
- Model IMPLEMENTATIONS diverge across the two pipelines, same model
  family but different code: Autoencoder (BIG IDEAs = LSTM encoder +
  linear bottleneck + linear head; shared = full Seq2Seq with an
  autoregressive decoder LSTM, Srivastava et al. 2015 — this is the
  largest divergence, arguably a different architecture, not just
  different sizes). TCN (BIG IDEAs = weight_norm + Chomp1d; shared =
  causal conv + BatchNorm1d — different normalization mechanism). RF/GB
  (BIG IDEAs = one single-output model per horizon; shared =
  MultiOutputRegressor, one multi-step model). LSTM/Transformer are the
  closest matches (same core layers/defaults, differ only in the output
  head: single-step vs multi-step).
- Glucdict (run_glucdict_experiments.py, results/glucdict/) was NEVER
  checked this session against any of the four criteria (models,
  training conditions, weighting, metrics) — "all 3 datasets" is not a
  claim this session's evidence supports; only Ohio and BIG IDEAs were
  verified.

## Session Summary — Latest

## MILESTONE: normalization blocker cleared ✅
All 5 BIG IDEAs neural models now behave like the shared pipeline:
train on scaled glucose, weights from mg/dL (y_train_raw), metrics in mg/dL.
- lstm.py + transformer_bigideas.py: FIXED (units mismatch + preds_scaled
  typo). Both sanity-run on real data: lstm 14.41, transformer 17.26 mg/dL
  (would be ~116 if bug present). Committed.
- cnn_lstm (Abeer fixed), autoencoder, tcn: verified already correct.
The normalization mismatch that blocked the port for several sessions is
GONE. Her models and mine now do the same thing.

## Consolidation state
DONE:
- Training conditions unified (150/15/32 + seed)
- Weighted MSE ported — Option A (un-scales to mg/dL), verified vs Abeer's
- loss_fn hook in lstm.py (non-breaking)
- All BIG IDEAs neural models units-correct + proven

NEXT (port now unblocked):
- Port Abeer's 2 models (CNN-LSTM, GB) from clean_treaining/ into
  src/models/ — GB is scale-invariant (easy), CNN-LSTM now units-correct
- Make BIG IDEAs runner call the shared src/ pipeline
- Wire weighted loss into the other 4 models (hook only in lstm so far)
- THEN full re-run at final settings

⚠️ STILL TRUE — results are stale:
- All committed results + presentation numbers are from OLD settings
  (100/10/64, no seed). Full re-run happens ONCE, after port complete.
- Presentation still custom style, not Moodle template.

## Coordination note
Abeer gave permission to fix her files; confirm she's off lstm.py +
transformer_bigideas.py before assuming they're settled. She's been
actively editing her models.
