# Toss state was implemented and tested; it adds no signal to either live model

Date: 2026-08-23. Roadmap priority #2 from the 2026-08-23 strategic
discussion. `ToiSnapshot` already fetches `toss_won_by`/`toss_decision`
live but `app/live/pipeline.py`'s two `MatchContext` construction sites
silently drop it. Of the four dropped fields (toss/pitch/weather/lineup),
toss was picked to go first specifically because it's the only one with a
historical source (Cricsheet's `info.toss`) and therefore the only one
backtestable before shipping blind.

## What was built

`app/ml/toss_dataset.py`: leakage-safe, known before a ball is bowled.
`batting_team_won_toss` is derived from `toss.decision` + the over's own
`innings` number alone -- no team-name matching needed (toss winner either
bats first, innings 1, or elects to field, sending themselves in second,
innings 2). Verified 100% Cricsheet coverage across IPL+T20I (6,767/6,767
raw match files have a valid `toss.decision`).

## Tested against the actual live architectures

Same methodology as the Impact Player test
(`finding_ipl_impact_player_no_signal.md`): built candidates that add
toss features onto the *current* live model architectures, same split,
same calibration, compared against the real production baseline.

**Run-range** (`train_run_range_v9_toss.py`, vs. live `run_range_v7_competition_prior`):

| | Blended | IPL | T20I |
|---|---:|---:|---:|
| Live `v7` (baseline) | 28.75% | 25.00% | 29.43% |
| `v9` + toss features | 28.65% | 24.57% | 29.39% |

Toss features ranked 53rd-54th of 62 by importance. Flat-to-worse.

**Wicket** (`train_contract22_wicket_v7_toss.py`, vs. live `contract22_wicket_v2_batter_state`):

| | AUC (known) | AUC (unknown) | Brier (Platt) |
|---|---:|---:|---:|
| Live `v2` (baseline) | 0.6081 | 0.6072 | 0.2049 |
| `v7` + toss features | 0.6075 | 0.6072 | 0.2050 |

The unknown-bowler AUC is identical to three decimal places; the
known-bowler AUC is marginally lower. Toss features ranked 33rd/36th of
42 -- low-middle, not meaningfully predictive.

## Interpretation

Two features (`toss_decision`, `batting_team_won_toss`) tested, both flat
to slightly negative on both models, on real 2025+ holdout, against real
production baselines. Plausible reason: by the time any over is bowled,
the model already knows the match state directly (score, wickets,
required rate, chase_pressure, phase) -- toss only matters insofar as it
predicts *how the match state evolves* (e.g. via a dew/pitch effect), and
if that effect exists, it's apparently too small or already absorbed by
existing state features to show up here.

## Recommendation

**Don't wire toss into `MatchContext`/`pipeline.py` for the run-range or
wicket models** -- there's no validated benefit, and adding unused
plumbing (fields threaded through but not consumed by anything) is exactly
the kind of stale/dead-code pattern this project's standing audits look
for.

**Re-scope the rest of roadmap priority #2.** The original framing treated
toss as the "easy, provable" first step before committing to
weather/pitch/lineup (which can't be backtested at all -- no historical
source, would have to ship blind and be observed on a real live match).
Toss was the best-case scenario for this whole direction: perfect
historical coverage, clean leakage-safe derivation, zero data-quality
excuses -- and it still came back flat on both models. That's evidence
against the broader hypothesis, not just against toss specifically:
weather/pitch/lineup are less certain to help and cost more to validate
(no offline backtest, real-match-only observation). **Recommend
deprioritizing weather/pitch/lineup integration below roadmap priority #3
(merging the v7.x wicket line's one proven lever into the live model)
until there's a more specific reason to expect it'll help.**

Candidate artifacts: `models/candidates/run_range_v9_toss/`,
`models/candidates/contract22_wicket_v7_toss/`. Neither promoted;
`production_changed: false` in both.
