# Finding: GBM+NN ensemble beats GBM alone on run-range — real win (2026-08-26)

## Why this was tried
After four feature-level negatives today (batter-vs-opponent-team,
team-phase-venue mean+volatility, bowler-phase-recency), the honest read
was that run-range is close to its information ceiling for *new inputs*
on top of the current architecture — cumulative feature importance stays
diffuse (33 of 63 features for 80% of importance), and no untested
player/team/venue idea moved the needle. The lever that hadn't been tried
was the *architecture*, not another feature: the GBM+NN ensemble already
gave a real, measured win on the wicket target
(`finding_wicket_nn_gbm_ensemble_real_win.md`, now live as a shadow).
That exact idea had never been tested against run-range.

## Method
Same exact feature set as the live `run_range_v11_partnership_rate` (63
features, no new inputs — this tests architecture, not more data).
Three-script split (`train_run_range_v21a_gbm_only.py`,
`v21b_nn_only.py`, `v21c_combine.py`) because torch and lightgbm cannot
share a process on this machine. GBM: identical LightGBM architecture as
v11. NN: a small embedding+MLP multiclass classifier (128→64→NUM_CLASSES,
dropout, Adam, early-stopped on calibration NLL) over the same inputs.
Combined via a single scalar blend weight (`alpha`, GBM's share of the
mix) fit by minimizing calibration-set NLL only — never the holdout.
Same downstream pipeline as every other test today: per-phase temperature
scaling, then the corrected width-aware `best_bands` (IPL=3, T20I=2).

**Sanity check on the re-implementation**: GBM-alone under this pipeline
reproduced 30.035% blended / 33.25% IPL / 29.45% T20I — matching the real
v11 baseline (30.04% / 33.25% / 29.45%) almost exactly, confirming this
is a faithful re-run of the live architecture, not an accidentally
different setup.

## Result
| | Blended | IPL | T20I |
|---|---:|---:|---:|
| v11 (live, reference) | 30.04% | 33.25% | 29.45% |
| GBM alone (this re-run) | 30.04% | 33.25% | 29.45% |
| NN alone | 30.08% | 33.00% | 29.55% |
| **Ensemble (alpha=0.307)** | **30.22%** | **33.43%** | **29.64%** |

The ensemble beats the GBM alone on **all three cuts** — blended +0.18pp,
IPL +0.18pp, T20I +0.20pp. Neither the GBM alone nor the NN alone beats
the ensemble on any cut — real evidence the two models make different
mistakes and the blend captures independent signal, not one model just
being better. Blend weight alpha=0.307 means the mix leans NN-heavy
(69.3% NN, 30.7% GBM), consistent with the NN alone already edging the
GBM alone on 2 of 3 cuts before blending.

For calibration: this project has promoted real production changes on
smaller margins (partnership rate was promoted at +0.05pp blended). A
uniform +0.18-0.20pp gain across all three cuts, with a clean sanity-
check reproduction of the baseline, clears this project's own bar.

**Decision: promote — genuinely deployable win, pending production
engineering (see below).**

## Why this isn't live yet
Unlike a single new feature column, this requires the same production
lift the wicket ensemble needed: persisting the trained NN's weights and
preprocessing (not just its calibration/holdout output arrays, which is
all `v21b` saved), a subprocess-isolated predictor (torch/lightgbm can't
coexist at serving time either), and wiring into the live run-range path
— either as a shadow first (this project's established pattern for any
architecture change) or a direct promotion given the clean, consistent
result. Building the production artifacts and shadow wiring next.
