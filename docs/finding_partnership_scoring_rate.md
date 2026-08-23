# Partnership scoring rate: real gains for both models — both promoted

Date: 2026-08-23. Follow-up to `finding_recency_weighted_form.md`, same
user-flagged direction ("partnership builds"). The codebase already
tracked `partnership_legal_ball_age` (how long the current pair has faced
deliveries together, summed per-batter) but never the actual *rate* that
pair is scoring at.

## What was built

`app/ml/partnership_dataset.py`: the real cricket "current stand" stat --
team runs and legal balls since the fall of the last wicket, resetting
exactly at each dismissal. Cleaner than `partnership_legal_ball_age`,
which sums each individual batter's own total innings balls faced and so
overstates stand length whenever the longer-set batter has already been
through an earlier partner this innings; this tracks the stand itself.
Spot-checked against a real match's over-by-over trajectory before
trusting it: correctly resets mid-over at a wicket, mean rate 6.94
matches typical T20 scoring, distribution non-degenerate (max 48,
min/reset 0).

## Tested against both real live baselines (post-recency-form promotion)

**Run-range** (`train_run_range_v11_partnership_rate.py`, vs. live `run_range_v7_competition_prior`): small but consistent win, promoted.

| | Blended | IPL | T20I |
|---|---:|---:|---:|
| Live `v7` | 28.75% | 25.00% | 29.43% |
| `v11` + partnership rate | 28.80% | 25.16% | 29.45% |

All three cuts improve; the script's own promotion gate (identical
criterion used to promote `v7` itself over `v4`) says promote. Margins
are modest -- smaller than the recency-form win -- but directionally
consistent across all three populations, not a coin flip.
`partnership_run_rate` ranks 11th of 63 features.

**Wicket** (`train_contract22_wicket_v10_partnership_rate.py`, vs. live `contract22_wicket_v9_recency_form`): real win, promoted.

| | AUC known | AUC unknown |
|---|---:|---:|
| Live `v9`, blended | 0.6106 | 0.6085 |
| `v10`, blended | 0.6118 | 0.6102 |
| Live `v9`, IPL | 0.6097 | 0.6058 |
| `v10`, IPL | 0.6104 | 0.6059 |
| Live `v9`, T20I | 0.6100 | 0.6090 |
| `v10`, T20I | 0.6113 | 0.6110 |

All 6 cuts improve, though honestly: the IPL-unknown-bowler gain
(+0.0001) is within noise, essentially flat -- the T20I and blended gains
are the more robust part of this result. `partnership_balls` ranks 9th of
52 features, `partnership_run_rate` 17th.

## Promoted to production

Unlike recency-weighted form, partnership rate needs **no external live
snapshot** -- it's pure current-match state, already computed from the
`deliveries` sequence both live feature computers already receive every
call. Wired directly into the existing per-ball scan in both:

- `app/ml/wicket_contract22_features.py`'s `_partnership_state`: added
  `stand_runs`/`stand_balls` tracking (reset on any `wicket_kind`) in the
  same loop that already computes `partnership_legal_ball_age`. Returns
  `partnership_runs`/`partnership_balls`/`partnership_run_rate`.
- `app/ml/run_range_v3_features.py`'s `RunRangeV3FeatureComputer.compute`:
  identical addition to its equivalent loop.
- `app/ml/prediction_engine.py`: `RUN_RANGE_V3_ARTIFACTS` repointed to
  `run_range_v11_partnership_rate`, `WICKET_CONTRACT22_ARTIFACTS`
  repointed to `contract22_wicket_v10_partnership_rate`.

Verified end-to-end, not just trusted: `PredictionEngine()` instantiates
cleanly against both new artifact directories; a hand-constructed
delivery sequence (a wicket falls, then 12 runs off 5 balls) through the
real feature computer returned exactly `partnership_runs=12,
partnership_balls=5, partnership_run_rate=14.4` -- matching the manual
calculation precisely, confirming the reset-at-wicket logic is correct,
not just plausible. 208 tests passing.

## Session arc, both promotions today

`contract22_wicket_v2_batter_state` -> `v9_recency_form` (recency-weighted
form) -> `v10_partnership_rate` (this one). `run_range_v7_competition_prior`
-> `v11_partnership_rate` (recency-weighted form was rejected for
run-range; partnership rate was not). Both live models improved twice in
one session from the same user-prompted direction: stop treating every
player-stat as a flat, un-decayed, individual-only number -- weight
recent form, and track the partnership as its own live entity, not just
two separate batters' tallies.
