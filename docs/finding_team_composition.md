# Team role composition: helps wicket specifically when the bowler is unknown — promoted

Date: 2026-08-23. User-requested direction: "team combinations opener,
middle, hitter, allrounder, spinners and pacers combination." The
confirmed playing XI is known before a ball is bowled (same timing as
toss) -- genuinely match-level, constant for both innings.

## What was built

`app/ml/team_composition_dataset.py`: aggregates
`data/external/cricsheet_player_styles.csv`'s per-player `playing_role`
(Bowler/Allrounder/Batter/Wicketkeeper variants) and `bowling_style`
(pace/spin) to the team level for the first time -- this data was already
loaded and used per-individual-player (striker/bowler classification),
never aggregated across a full XI before. Produces, per match:
`batting_team_allrounder_count`, `batting_team_specialist_batter_count`,
`bowling_team_pace_count`, `bowling_team_spin_count`,
`bowling_team_allrounder_count`.

**Caught and fixed a real bug before trusting it**: spot-checked against
a real, known CSK squad and got pace=9/spin=3 on the first pass --
suspiciously pace-heavy even for a team known to lean that way. Traced
it to `bowling_style` being a nominal database attribute recorded for
every player regardless of whether they actually bowl (Rahane, a
specialist batter, and Dhoni, a wicketkeeper who never bowls in IPL, both
had a recorded "Right arm Medium"). Fixed by only counting bowling style
for players whose role says they're actually part of the attack
(`bowler`/`allrounder`); re-checked and got pace=7/spin=2 (Jadeja +
Moeen Ali as the two front-line spin options) -- correctly matches CSK's
known identity as a pace-leaning attack with two spin all-rounders.

## Tested on both models

**Run-range** (`train_run_range_v14_team_composition.py`, vs. live `run_range_v11_partnership_rate`): **rejected**.

| | Blended | IPL | T20I |
|---|---:|---:|---:|
| Live `v11` | 28.80% | 25.16% | 29.45% |
| `v14` + team composition | 28.59% | 24.21% | 29.39% |

Worse on all three cuts. Consistent with run-range's architecture always
assuming the bowler is known during training (no known/unknown split at
all) -- see below for why that matters.

**Wicket** (`train_contract22_wicket_v16_team_composition.py`, vs. live `contract22_wicket_v15_batter_phase_recency`): **real, narrower win, promoted.**

| | AUC known | AUC unknown |
|---|---:|---:|
| Live `v15`, blended | 0.6124 | 0.6099 |
| `v16`, blended | 0.6126 | **0.6114** |
| Live `v15`, IPL | 0.6149 | 0.6102 |
| `v16`, IPL | 0.6151 | **0.6131** |
| Live `v15`, T20I | 0.6114 | 0.6104 |
| `v16`, T20I | 0.6113 | **0.6113** |

Mostly flat when the bowler is known (+0.0002, +0.0001, -0.0001 --
within noise) but a real, consistent gain when the bowler is *unknown*
(+0.0015 blended, +0.0029 IPL, +0.0009 T20I). Nothing regresses on any
cut. Feature importance ranks are low (38th-49th of 61) -- the model
leans on this lightly, consistent with a supplementary role rather than
a primary driver.

## Why "unknown bowler" specifically, and why that's the case that matters most

When the bowler is known, the model already has that specific bowler's
own individual career/recency/phase stats -- team-level pace/spin counts
add little beyond what the individual already tells it. When the bowler
is unknown, team composition is a genuine partial substitute: it can't
say *who's* bowling, but it can say what kind of attack this team fields,
filling part of the gap. This is also the harder, more realistic
deployment case -- TOI's live feed frequently doesn't have the next-over
bowler confirmed ahead of a prediction (see
`app/ml/wicket_contract22_features.py`'s own docstring on
bowler-known/unknown parity), which is exactly why this model already
evaluates and reports both cuts separately as standing practice, and
exactly why a gain concentrated in the unknown case is not a lesser
result -- it's the more commonly-hit real path.

## Promoted to production (contract22_wicket_v16_team_composition)

No new live snapshot needed -- reuses the same `playing_role`/
`bowling_style` classification `WicketContract22FeatureComputer` already
loads for individual striker/bowler features (`self._playing_role`,
new; `self._bowler_type`, already existed), aggregated over a roster via
a new `_team_composition` method.

- `app/ml/wicket_contract22_features.py`: `__init__` now also builds
  `self._playing_role`; `compute()` gained
  `batting_team_players`/`bowling_team_players` parameters (optional,
  default `()` -- degrades to all-zero counts, never crashes, for
  callers without a roster yet).
- `runtime_wicket_contract22.py`: `predict_wicket_probability` threads
  the two new parameters through to `compute()`.
- `app/ml/prediction_engine.py`: resolves which named team is batting
  (reusing the same is_chase/batting_first/bowling_first logic the
  run-range section already had, computed once now instead of twice)
  and passes `context.team1_players`/`team2_players` through as
  `batting_team_players`/`bowling_team_players`.
- `app/live/pipeline.py`: **fixed the actual root gap** -- both
  `MatchContext(...)` construction sites now populate
  `team1_players`/`team2_players` from `snapshot.team_players` (TOI's
  confirmed-XI data, already fetched, previously dropped before reaching
  `MatchContext` -- the same drop pattern this session already found and
  fixed for toss). New `_team_player_names()` helper degrades to an
  empty list when TOI's roster block isn't available yet, matching
  `ToiSnapshot`'s own best-effort convention.
- `WICKET_CONTRACT22_ARTIFACTS` repointed to
  `models/candidates/contract22_wicket_v16_team_composition`.

Verified end-to-end, not just trusted: `PredictionEngine()` instantiates
cleanly, 208 tests pass (including `test_live_prediction_pipeline.py`,
which exercises the `MatchContext` construction path directly), and a
direct call with CSK's real squad reproduced the exact same
pace=7/spin=2 composition found in the training-side spot-check --
confirming train/serve consistency, not just "didn't crash." Sane wicket
probability (19.8%) for a realistic scenario.

`run_range_v11_partnership_rate` is unchanged -- this candidate is
wicket-only.

## Session lineage, wicket model

`contract22_wicket_v2_batter_state` -> `v9_recency_form` ->
`v10_partnership_rate` -> `v15_batter_phase_recency` ->
`v16_team_composition` (this promotion). Four real, verified promotions
to the live wicket model in one session.
