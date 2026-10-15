# Finding: run-range's momentum lag is not fixed by a shorter recency window

**Date**: 2026-09-18
**Status**: Tested and rejected. Live model (`run_range_v11_partnership_rate`) unchanged.

## Background

`docs/live_field_test_2026-09-15.md` flagged a real observed behavior:
during a live match, `run_range`'s prediction stayed at "8-10" through two
consecutive massive overs (20 runs, then 27 runs) before finally jumping
to "15-17" on the third over — a real, scoped future improvement
candidate, not classified as a bug at the time.

## Diagnosis

Reproduced the exact pattern on real historical data (not just the one
live observation): New Zealand vs Bangladesh, 2026-04-27
(`data/raw/cricsheet/t20i/1528275.json`, innings 2), which has a genuine
18-run over followed by a 25-run over. Replayed through the actual
current `PredictionEngine` (`diagnose_momentum_lag.py`):

| Over | Actual runs | Predicted (before that over) | `recent_runs_per_ball` |
|-----:|------------:|------------------------------:|------------------------:|
| 16   | 18          | 7.00                           | 1.33 |
| 17   | 25          | 7.00                           | 2.33 |
| 18   | 5           | 8.00                           | 3.58 |

`recent_runs_per_ball` (computed over the last 12 legal balls / 2 overs)
correctly and quickly reaches an extreme value (3.58, i.e. ~21.5
runs/over pace) once both massive overs are inside its window. But the
model's own `predicted_runs` barely reacts — 7.00 → 8.00 — even when fed
that extreme signal directly. This rules out "the feature updates too
slowly" as the root cause.

Checked feature importance in the real live model
(`models/candidates/run_range_v11_partnership_rate/sharp_range_model.pkl`,
a `LGBMClassifier`): `recent_runs_per_ball` ranks 28th of 63 features
(1.9% importance). All momentum-related features combined
(`recent_runs_per_ball`, `recent_dot_rate`, `recent_single_rate`,
`recent_boundary_rate`, `recent_wicket_rate`, `state_regime`,
`recent_legal_balls`) total under 6% combined importance, versus
individual career/context features like `score_before_over` (3.6%),
`current_run_rate` (3.5%), `venue_par_score` (3.5%), and multiple
prior-stat features each ranking above 2.5%. The model structurally
under-weights in-the-moment scoring momentum relative to historical and
match-state context.

## Candidate tested: shorten the recency window from 12 balls to 6

Hypothesis: a 12-ball (2-over) window dilutes a single massive over with
whatever preceded it, so the model never sees the full magnitude of a
surge until the window is entirely populated by it (mechanically
consistent with the ~2-over lag observed). A 6-ball (1-over) window would
let one massive over immediately and fully dominate the signal.

Built via `app/ml/candidate_v3_dataset.py`'s new `recent_window` parameter
(default 12, unchanged for all other callers) +
`build_verified_dataset_recent6.py` + `train_run_range_v25_faster_momentum.py`
— identical architecture, features, split, calibration, and IPL/T20I
band-width policy to the live `run_range_v11_partnership_rate`, changing
only the window length for `recent_*`/`phase_*_pressure`/
`phase_rotation_value`.

**Result: rejected, worse on every cut**, versus the real live baseline
(30.04%/33.25%/29.45% blended/IPL/T20I, the model's own current numbers
post the 2026-08-26 IPL band-width fix — not a stale reference):

| Cut | Live (12-ball window) | Candidate (6-ball window) | Delta |
|---|---:|---:|---:|
| Blended | 30.04% | 29.83% | −0.21pp |
| IPL | 33.25% | 32.53% | −0.72pp |
| T20I | 29.45% | 29.34% | −0.11pp |

`recent_runs_per_ball`'s own importance also *dropped* slightly under the
shorter window (1.9% → 1.6%, rank 28th → 36th) rather than increasing —
the model didn't even start relying on it more once it was less diluted;
if anything a 6-ball sample is noisier (higher variance, fewer balls) and
the model correctly learned to trust it even less.

## Conclusion

This is real, tested evidence — not an assumption — that the model's
muted reaction to short-term scoring bursts is not an engineering bug
fixable by a window-size tweak. A plausible honest interpretation: T20
in-match momentum has limited genuine incremental predictive power for
the *very next* over specifically, once phase, match-state, and
player/venue context are already accounted for — so a model that doesn't
overreact to a noisy 1-2 over sample may already be close to correctly
calibrated, even though it looks surprising to a human watching live.

**Not promoted.** `run_range_v11_partnership_rate` remains live, unchanged.
Infrastructure kept (`app/ml/candidate_v3_dataset.py`'s `recent_window`
parameter, `build_verified_dataset_recent6.py`,
`train_run_range_v25_faster_momentum.py`) since it's real, correct, and
reusable if a different momentum-signal hypothesis is worth testing later
— e.g. a nonlinear "surge indicator" (max single-over rate in the recent
window, distinct from its average) rather than a shorter linear window,
which this finding does not rule out.

Large generated artifacts
(`data/candidates/v3/verified_training_overs_recent6.csv`,
`models/candidates/run_range_v25_faster_momentum/*.pkl`) are gitignored,
not committed — only this finding and the reusable scripts are.
