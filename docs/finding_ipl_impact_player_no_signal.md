# IPL Impact Player state was implemented and tested; it adds no signal to either live model

Date: 2026-08-23. CTO-agreed priority #1 (smallest, most scoped item in the
2026-08-23 strategic-direction discussion): `EngineRouter`'s IPL
`EngineSpecification` (`app/ml/engine_router.py`) had declared
`extension_features=("impact_player_state", "strategic_timeout_state",
"ipl_team_strategy")` since it was written, but none of the three were ever
implemented anywhere -- pure placeholder strings, not read by
`FeatureBuilder` or `PredictionEngine`.

## What already existed, unwired

`app/ml/ipl_impact_dataset.py` (real code, already committed) correctly
parses Cricsheet's real impact-player substitution events --
`delivery["replacements"]["match"]` entries with `reason ==
"impact_player"` -- into a leakage-safe per-over state (as of the *start*
of each over, using only replacements applied before that over began):
`ipl_impact_{batting,bowling}_{used,available,role,overs_since}`, gated to
`year >= 2023` (the rule's actual era). Verified directly against a raw
match file; e.g. `{"in": "RA Tripathi", "out": "TM Head", "team":
"Sunrisers Hyderabad", "reason": "impact_player"}`.

A prior training script (`train_ipl_engine_v1.py`) had tested this once,
but against a **stale architecture** -- the old V3 feature set, fixed band
width=2, no phase temperature calibration -- not comparable to what's
actually live now. That earlier result (0.09pp improvement, p=0.87, gates
failed) wasn't real evidence either way for the current models.

## Re-tested against the actual live architectures

Per standing practice (compare against the real current production
baseline, not a stale one -- see `finding_blended_holdout_masks_ipl_accuracy.md`),
built two new candidates that add the same impact-player features onto the
*current* live model architectures, same split, same calibration:

**Run-range** (`train_run_range_v8_impact_player.py`, vs. live
`run_range_v7_competition_prior`):

| | Blended | IPL | T20I |
|---|---:|---:|---:|
| Live `v7` (reconfirmed baseline, reproduces exactly) | 28.75% | 25.00% | 29.43% |
| `v8` + impact-player features | 28.51% | 24.61% | 29.22% |

Impact features ranked 56th-67th of 69 by importance (dead last cluster).
Slightly *below* baseline, not better -- consistent with noise/overfitting
from added feature count, not a real effect.

**Wicket** (`train_contract22_wicket_v6_impact_player.py`, vs. live
`contract22_wicket_v2_batter_state`):

| | AUC (known) | AUC (unknown) | Brier (Platt) |
|---|---:|---:|---:|
| Live `v2` (reference) | 0.6081 | 0.6072 | 0.2049 |
| `v6` + impact-player features | 0.6075 | 0.6053 | 0.2049 |

Flat to slightly worse. Impact features ranked 32nd-49th of 49 (bottom
third); `overs_since` ranked highest of the group (32/35) but still not
meaningfully predictive.

## Why this plausibly reads as a real negative, not a testing artifact

Two different hypotheses were tested (impact-player state as a run-scoring
signal, and as a wicket-risk signal analogous to "new batter at crease" --
the same state-representation pattern, `used`/`available`/`role`/
`overs_since`, that *did* work for the batter-state features already live
in `v2`). Both came back flat-to-negative on real 2025+ holdout data
against the real production baseline. The likely reason: Impact Player
substitutions are rare (at most one per team per match) and happen at a
small number of fixed strategic points (innings break, or in response to
match state); by the time a substitute is at the crease or bowling, the
model already has that player's own prior-stat features, current match
state, and phase context -- the "just subbed in" flag itself doesn't add
information beyond what's already there.

## Recommendation

**Don't re-try this exact feature formulation on either model.** Removed
`impact_player_state` from `EngineSpecification.extension_features` in
`app/ml/engine_router.py` -- it's no longer an untested placeholder, it's a
tested-and-rejected hypothesis, and leaving it in the declared tuple would
misrepresent that. `strategic_timeout_state` and `ipl_team_strategy` remain
untested and still legitimately open.

If Impact Player is revisited later, a match-level or narrative framing
(e.g. runs scored by the substitute specifically in the overs immediately
following substitution, rather than a per-over state flag folded into the
existing over-level models) is a genuinely different formulation, not
covered by this test -- but there's no evidence today that it's worth
pursuing over the other two roadmap items (toss integration, the v7.x
wicket lever, live-pipeline validation).

Candidate artifacts: `models/candidates/run_range_v8_impact_player/`,
`models/candidates/contract22_wicket_v6_impact_player/`. Neither promoted;
`production_changed: false` in both reports.
