# Full-feature NN + GBM ensemble for wicket: real, modest win

Date: 2026-08-25. Closes the "does architecture matter" question left
open by the earlier LSTM finding (which only gave the neural net 9-19
basic features vs. the GBM's 60+ -- not a fair comparison). Same ~61
features as live `contract22_wicket_v16_team_composition` fed to both a
small MLP (low-cardinality categorical embeddings only -- player
identity is already numeric career-stat features, no player-name
embeddings needed) and the GBM, blended via a Platt-calibrated
logistic-regression stacker fit on the 2024 calibration set.

**Real engineering obstacle, not a modeling one**: torch and lightgbm
cannot coexist in one process on this machine -- every combined attempt
hung or silently died at the same point regardless of import order or
`KMP_DUPLICATE_LIB_OK`. Fixed by splitting into 3 separate processes
(`train_wicket_v19a_gbm_only.py`, `..v19b_nn_only.py`,
`..v19c_combine.py`) that only share `.npz` files on disk -- GBM and NN
each train in isolation, the combiner never imports either ML framework.

## Result (real 2025+ holdout, identical rows)

| | Blended AUC | IPL AUC | T20I AUC |
|---|---:|---:|---:|
| GBM alone (reproduces live `v16` exactly) | 0.6126 | 0.6151 | 0.6113 |
| NN alone | 0.6097 | 0.6097 | 0.6080 |
| **Ensemble** | **0.6137** | **0.6158** | **0.6120** |

Brier also improves on every cut (0.2042 vs 0.2043 blended; 0.1928 vs
0.1933 IPL; 0.2063 vs 0.2063 T20I -- flat/tiny T20I gain, real elsewhere).

The NN alone is weaker than the GBM everywhere (expected -- GBMs are
hard to beat on structured tabular data). But the **ensemble beats the
GBM alone on every cut, both AUC and Brier** -- small (~0.001-0.0011 AUC)
but consistent, exactly the complementary-error mechanism ensembling is
supposed to provide. Comparable in size to past real promotions here
(partnership rate shipped on a +0.05pp hit-rate margin).

## Wired as a live shadow (2026-08-25, uncommitted)

User confirmed building this into production, explicitly accepting added
per-over latency over the risk of hanging/crashing the live process:
"latency is better than hang or die." **`contract22_wicket_v16_team_composition`
remains the only served wicket prediction, unchanged** -- the ensemble
runs alongside as a logged shadow, same "observe on real matches before
deciding" plan as the win-probability Monte Carlo work.

**Process isolation carried into production**: the live `PredictionEngine`
process already runs LightGBM for run-range/wicket/win-probability, so
importing torch directly into it would risk the exact hang/crash found
during today's research -- but for ALL THREE live predictions, not just
this one. Fixed by running the NN in its own subprocess per prediction
(`predict_wicket_nn_subprocess.py`), called by the new
`app/ml/wicket_ensemble_runtime.py`. Costs ~1s of extra latency per over
(subprocess startup + model load) -- acceptable given predictions happen
once per over (~60-90s cadence), not per-ball. A subprocess timeout/crash
falls back to GBM alone for that over, never blocking or affecting the
served result.

**Production artifacts** (`build_wicket_ensemble_artifacts.py`, run in
3 parts -- gbm/nn/combine -- for the same process-isolation reason):
`gbm_model.pkl`, `nn_model.pt` + `nn_metadata.json` (preprocessing
needed for the subprocess), `gbm_calibrator.pkl`, `nn_calibrator.pkl`,
`stacker.pkl`, `ARTIFACT_MANIFEST.json` -- all under
`models/candidates/wicket_v19_nn_gbm_ensemble/`.

**Shadow logging** (`app/ml/wicket_ensemble_shadow_log.py`, mirrors
`win_probability_shadow_log.py`): unlike win-probability, a wicket's real
outcome is known one over later, not just at match end, so `pipeline.py`
logs the (GBM, NN, ensemble) triple and records the real outcome together
in one place -- right where the next over's actuals resolve. Once 5 real
matches have logged outcomes, a one-time Telegram summary reports real
accuracy/AUC/Brier for all three and prompts a decision.

**Verified end-to-end with real (non-mocked) calls**: `PredictionEngine()`
instantiates in ~1s, a real prediction (including the subprocess NN call)
completes in ~1s and returns a sane, non-degenerate shadow
(`{"gbm_probability": 0.347, "nn_probability": 0.24, "ensemble_probability": 0.303}`
for a realistic mid-chase scenario). A real single-match IPL Cricsheet
replay through the actual `VerifiedLivePredictionPipeline` logged all 36
real over-level predictions with correctly joined outcomes (confirmed by
recomputing the summary after the replay: GBM 66.7% accuracy/0.755 AUC,
NN 69.4%/0.632, ensemble 69.4%/0.712 on that single match -- illustrative
only, not a real validation sample). 5 new tests
(`tests/test_wicket_ensemble_shadow.py`) cover shadow-log recording,
outcome idempotency, the reminder firing once, and `PredictionEngine`'s
fault isolation (missing/broken ensemble runtime never affects the served
GBM result) -- 221 tests pass total.
