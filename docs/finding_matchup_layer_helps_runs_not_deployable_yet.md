# The matchup layer shows a real, positive result for run prediction — but has no current data path to deploy it

Date: 2026-08-23. Direct follow-up to the wicket-prediction investigation
(`docs/finding_matchup_score_batter_weak_zone_bowler_mix.md`, 4 consecutive
negative attempts). User asked whether the same layer had been tested
against runs, not just wickets — it hadn't. This closes that gap.

## Runs are a more natural fit than wickets, and the data agreed

Line/length is literally a proxy for "how hard is this ball to score
off" — a more direct mechanical connection to runs scored than to
wicket risk (harder to score off doesn't automatically mean higher
dismissal chance). Tested the same profile-building approach (batter's
shrunk runs-per-ball rate by delivery length / vs. bowler pace-type,
built from IPL+T20I+ODI career data through 2022) against a
crude 3-feature context proxy first:

| | MAE | R2 |
|---|---:|---:|
| Granular length layer: context only | 3.7932 | -0.0173 |
| Granular length layer: context + layer | 3.7527 | -0.0057 (better) |
| Pace-type layer: context only | 3.9331 | -0.0089 |
| Pace-type layer: context + layer | 3.9226 | +0.0016 (better, crosses positive) |

Directionally positive both ways — a real contrast to every wicket test,
which was flat-to-negative in all four attempts.

## Tested against the REAL run_range_v11 architecture, not a proxy

The crude-baseline result alone wasn't enough to trust (same lesson
learned from the wicket investigation's early false-positive). Rebuilt
using the actual production feature-building calls from
`train_run_range_v11_partnership_rate.py` (63 real features: MOE/venue/
bowler/batter-phase/competition-prior/partnership-rate), not a toy
proxy. Found and fixed a real name-resolution mismatch along the way:
the production pipeline's `striker`/`bowler`-adjacent columns are
canonical registry IDs (`player:xxxx`), not raw Cricsheet names --
had to join raw striker/bowler names from source JSON separately to
match the Kaggle-derived (raw-name-keyed) profiles.

Split shifted back one year (train<=2022/cal=2023/holdout=2024) for the
same reason as every other test in this thread -- the Kaggle data's real
coverage ends in 2024.

| | Hit rate | Holdout rows |
|---|---:|---:|
| Real 63 features, no layer (same subset) | 29.01% | 12,306 |
| Real 63 features + pace-type matchup layer | **29.45%** (+0.44pp) | 12,306 |

For calibration: this project has promoted real production changes on
smaller margins than this (partnership rate was promoted at +0.05pp).
A +0.44pp gain on a 12,306-row sample, using the real feature
architecture, is a genuinely meaningful result by this project's own
established bar -- the first clearly positive finding across five real
attempts in this whole investigation thread (four on wicket, this one
on runs).

## Why it's not deployable yet, and it isn't a modeling problem

Three real, independent constraints:

1. **Not the exact live holdout.** The 29.01%/29.45% numbers use the
   shifted split, not run_range_v11's actual reported 28.80% (different
   holdout year) -- the *improvement* is real and apples-to-apples on
   the same subset, but not directly comparable to the live baseline
   number.
2. **~50% coverage even historically** (12,306 of 24,714 holdout rows) --
   roughly half of deliveries wouldn't have this feature available at
   all without a fallback.
3. **The frozen-data problem is the real blocker.** For genuinely
   current matches (2025+), real coverage drops to ~22% (established in
   the earlier xG finding doc). Deploying this as-is would mean relying
   on player profiles frozen at 2022 data even for a 2026 match --
   untested whether pace-type tendencies are stable enough over that
   long a gap to still be valid.

## Searched for an updated data source -- found none suitable

Checked whether a newer dataset with the same enriched fields exists.
The most promising lead, ["IPL 2026 ball by ball dataset"](https://www.kaggle.com/datasets/sahiltailor/ipl-2024-ball-by-ball-dataset)
(updated daily, real 2025/2026 IPL seasons included), turned out to be
a dead end for this purpose: verified its full 20-column schema directly
(`match_id, season, match_no, date, venue, batting_team, bowling_team,
innings, over, striker, bowler, runs_of_bat, extras, wide, legbyes,
byes, noballs, player_dismissed, wicket_type, fielder`) -- basic
Cricsheet-equivalent data only, no line/length/shot/direction fields at
all, and we already effectively have this via our own
`data/raw/cricsheet/ipl/`. Other 2025-dated results (Champions Trophy
commentary datasets) are one-off tournament snapshots, not ongoing
sources -- same frozen-in-time limitation, just narrower scope.

## Conclusion

Real, positive, properly-validated finding -- not a false positive like
the initial crude-baseline check would have been if trusted alone. But
genuinely not deployable today: no current data source exists (checked
directly, not assumed) with the enriched fields needed, free or
otherwise found. Not wired into any live path. Worth revisiting if:
(a) a live-commentary-parsing pipeline is ever built (the same blocker
flagged in the original xG finding doc), or (b) a better-maintained
enriched dataset appears in the future -- re-search rather than assume
today's negative result about data availability is permanent.

No code or model changes. Career data and production-pipeline
comparison test kept in session scratchpad only, not added to the repo.
