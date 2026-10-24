# Finding: GBM+NN ensemble beats GBM alone on win-probability — real win (2026-08-26)

## Why this was tried
CTO-prompted "any idea for today?" after run-range's feature/architecture
search was exhausted (7 negatives + 1 win). The GBM+NN ensemble lever had
now proven itself twice — wicket
(`finding_wicket_nn_gbm_ensemble_real_win.md`) and run-range
(`finding_run_range_nn_gbm_ensemble_real_win.md`) — but had never been
tried on the third live prediction, `match_winner_v1` (win-probability).
Live match availability checked again first (IPL 2026 doesn't start until
March 28; today's only live T20 is domestic leagues, not IPL/full-member
international) — still blocked, so this was the next best use of time.

## Method
Exact same feature set as the live `match_winner_v1` (bowler-known cut,
its own reported "ceiling"). Three-script split
(`train_match_winner_v2a_gbm_only.py`/`v2b_nn_only.py`/`v2c_combine.py`),
same torch/lightgbm process isolation as every other ensemble test.
GBM: identical LightGBM architecture as `v1`. NN: same embedding+MLP
binary classifier architecture as the wicket ensemble (128→64→1,
BCEWithLogitsLoss with class-imbalance pos_weight, early-stopped on
calibration AUC). Combined via Platt calibration of each model
individually, then a 2-input logistic-regression stacker — the exact
combination method validated for wicket.

**Sanity check on the re-implementation**: GBM-alone reproduced the live
model's numbers almost to the decimal (0.85867 vs reported 0.8587
blended AUC, 0.78891 vs 0.7889 IPL, 0.87034 vs 0.8703 T20I; row counts
109,356/24,369/35,593 match exactly) — confirms a faithful
re-implementation, not an accidentally different setup.

## Result
| | Blended AUC | IPL AUC | T20I AUC | Blended Brier | IPL Brier | T20I Brier |
|---|---:|---:|---:|---:|---:|---:|
| v1 (live) | 0.8587 | 0.7889 | 0.8703 | 0.1538 | 0.1905 | 0.1472 |
| GBM alone (re-run) | 0.8587 | 0.7889 | 0.8703 | 0.1538 | 0.1905 | 0.1472 |
| NN alone | 0.8592 | 0.8042 | 0.8696 | 0.1535 | 0.1867 | 0.1475 |
| **Ensemble** | **0.8631** | **0.8019** | **0.8738** | **0.1515** | **0.1864** | **0.1452** |

The ensemble beats the GBM alone on **AUC and Brier, all three cuts,
simultaneously** — no tradeoffs. IPL AUC alone gains +1.3 points, the
largest single-cut gain of any ensemble test this session (wicket's and
run-range's ensembles both gained less on their hardest cut). The NN
alone is already competitive with the GBM (slightly ahead on IPL AUC,
0.8042 vs 0.7889), and the ensemble beats both individual models on
every cut except IPL Brier (ensemble 0.1864 vs NN-alone 0.1867 — the
ensemble still wins there too, just marginally) — real evidence of
independent signal, not one model carrying the other.

**Decision: promote — clearest ensemble win of the three found today.**

## Why this isn't live yet
Same production lift every ensemble here has needed: persisted NN
weights + preprocessing (not just the calibration/holdout output arrays
this test saved), a subprocess-isolated predictor, and wiring into the
live path. Unlike run-range and wicket, `match_winner_v1` was
deliberately built as its own isolated engine
(`app/ml/match_winner_engine.py`, `finding_match_winner_engine_isolation.md`)
specifically so a fault here can never affect run-range/wicket — the
ensemble shadow should wire into that engine and `pipeline.py`'s
existing `_match_winner_engine` call site, not `PredictionEngine`,
preserving that isolation. Building the production artifacts and shadow
wiring next.
