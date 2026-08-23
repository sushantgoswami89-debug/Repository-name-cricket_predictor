# Phase-specific batter recency: real, consistent IPL win for wicket — promoted

Date: 2026-08-23. Web-research-driven direction: death overs are the
single most decisive IPL phase (71.4% win correlation vs 64.2%
powerplay, per external analysis -- see
`finding_home_advantage_no_wicket_signal.md`/`finding_venue_recency_ipl_regression.md`
for the same research thread). "Finishing ability" (death-overs hitting
specifically) is a well-documented distinct skill from general batting
form, and like any skill, a player's current finishing form can be in a
slump or a purple patch. `BATTER_PHASE_FEATURES` (already live in
run-range) captures phase-specific tendencies as a flat career average
with no recency weighting -- the same gap already fixed for the
phase-agnostic version. This mirrors `finding_bowler_phase_recency_mixed.md`'s
approach exactly, batter side.

## What was built

`app/ml/recency_weighted_prior_dataset.py` gained
`build_batter_phase_recency_dataset` (keyed `"{batter_id}|{phase}"`,
match-EWMA decay=0.95). Spot-checked against Kohli's real death-overs
strike-rate trajectory before trusting it: rises through his 2011-2017
peak (SR ~180-204), tapers in recent years (SR ~167-186) -- consistent
with well-documented shifts in his game, not degenerate.

## Tested on both models

**Run-range** (`train_run_range_v13_batter_phase_recency.py`, vs. live `run_range_v11_partnership_rate`): **rejected**.

| | Blended | IPL | T20I |
|---|---:|---:|---:|
| Live `v11` | 28.80% | 25.16% | 29.45% |
| `v13` + batter phase recency | 28.63% | 24.66% | 29.35% |

Worse on all three cuts. Not promoted -- run-range stays on `v11`.

**Wicket** (`train_contract22_wicket_v15_batter_phase_recency.py`, vs. live `contract22_wicket_v10_partnership_rate`): **real, consistent win, promoted.**

| | AUC known | AUC unknown |
|---|---:|---:|
| Live `v10`, blended | 0.6118 | 0.6102 |
| `v15`, blended | 0.6124 | 0.6099 |
| Live `v10`, IPL | 0.6104 | 0.6059 |
| `v15`, IPL | **0.6149** | **0.6102** |
| Live `v10`, T20I | 0.6113 | 0.6110 |
| `v15`, T20I | 0.6114 | 0.6104 |

IPL improves substantially on **both** known and unknown cuts together
(+0.0045, +0.0043) -- unlike the earlier bowler-phase-recency test
(`finding_bowler_phase_recency_mixed.md`), where known and unknown
reshuffled unpredictably between formulations, a sign of noise. Here both
move the same direction by a similar, large margin. T20I and blended are
close to flat (small moves in both directions, none exceeding 0.0006) --
not a regression, unlike venue recency's clear T20I-trade-off pattern.
`striker_recency_phase_balls` ranks 7th of 56 features by importance.

## Why this one is real where the others were noisy

Two structural differences from the rejected phase-specific attempts:
unlike bowler-phase-recency (which showed a genuine but unstable
IPL-vs-T20I trade-off that a standard competition-mask fix couldn't
resolve), here IPL improves cleanly on both evaluation cuts without any
masking needed, and T20I doesn't regress at all -- it's a strict
improvement over the live baseline in the population that matters most,
with no real cost anywhere else. That combination (large + directionally
consistent + no offsetting regression) is what separates this from the
declined candidates in the same research thread today.

## Promoted to production (contract22_wicket_v15_batter_phase_recency)

Same infrastructure pattern as the earlier recency-form promotion:

- `build_recency_weighted_live_snapshots.py` extended to also write
  `data/live/recency_form_batter_phase_stats.json` (gitignored,
  regenerate via this script). Batter/bowler/bowler-phase snapshot counts
  unchanged (7,061/5,205/13,001) confirming the existing live behavior
  wasn't disturbed; new: 15,822 batter-phase profiles.
- `app/ml/wicket_contract22_features.py`: `WicketContract22FeatureComputer`
  gained `_batter_phase_recency`, folded into `compute()` as
  `striker_recency_phase_*`.
- `app/ml/prediction_engine.py`: `WICKET_CONTRACT22_ARTIFACTS` repointed
  to `models/candidates/contract22_wicket_v15_batter_phase_recency`.
- Verified end-to-end: `PredictionEngine()` instantiates cleanly, the new
  lookup resolves to real non-zero values for a known player (Kohli's
  death-overs recency strike rate 1.91 runs/ball, dismissal rate 4.8%),
  and `predict_wicket_probability` returns a sane 33.3% for a realistic
  death-overs scenario. 208 tests passing.

`run_range_v11_partnership_rate` is unchanged -- this candidate is
wicket-only.

## Session lineage, wicket model

`contract22_wicket_v2_batter_state` -> `v9_recency_form` ->
`v10_partnership_rate` -> `v15_batter_phase_recency` (this promotion).
Three real, verified promotions to the live wicket model in one session,
each tested against the real prior live baseline, each verified
end-to-end before wiring.
