# Batter weak-zone × bowler delivery-mix matchup score: real idea, no signal, not promoted

Date: 2026-08-23. Direct follow-up to the xG delivery-quality finding
-- user's own idea to make it deployable: instead of a real-time
per-ball feature (needs live commentary data we don't have), build
**static, precomputable player profiles** from historical data, looked
up by identity at serving time (same architecture as every other
live-snapshot feature in this project). Concretely: does a bowler's own
historical length-of-delivery mix, weighted against a specific batter's
historical dismissal rate at each length, predict wicket risk for that
matchup beyond generic match context?

## Construction

- **Batter weak-zone profile**: for each batter, shrunk dismissal rate
  per `ball_length` bucket (shrinkage=200 balls toward the global rate
  for that bucket -- same style of shrinkage this project already uses
  for H2H/venue features).
- **Bowler delivery-mix profile**: for each bowler, shrunk % of
  deliveries in each `ball_length` bucket.
- **Matchup score** = sum over length buckets of
  `bowler_mix[bucket] * batter_dismissal_rate[bucket]` -- high when a
  bowler's typical attack concentrates in a batter's known weak zone.
- Built from **full IPL+T20I+ODI international career data** (user's
  explicit scope, excluding Test), 920,926 profile-building rows
  through 2022 -- 2,348 batters, 1,836 bowlers with real profiles.
- Deliberately identity-only, not per-ball: this is precomputable
  offline and looked up by player name, unlike raw `ball_length` which
  needs live commentary for the CURRENT ball (the blocker that killed
  the earlier flat delivery-quality test's deployability).

## Illustrative check, before the real test

The conversation's own example (Rohit Sharma vs Pat Cummins): real data
showed Rohit's actual weak zone is short balls/bouncers (3.74%/7.59%
dismissal rate vs 3.20% baseline), not "good length" as originally
guessed -- and Cummins' and Mohit Sharma's real length mixes turned out
nearly identical (~30% short+bouncer each), undercutting the "Mohit is
more mixed" framing. Real head-to-head (Rohit vs Cummins, 297 balls, 7
wickets, 2.36%) was actually *below* Rohit's baseline, not elevated.
None of this proves the mechanism wrong on its own (one pairing is not
statistical evidence, and 297 balls is a modest sample) -- it's exactly
why this needed a real population-level test rather than trusting or
rejecting one anecdote.

## Real test, real result

Same leakage-safe discipline as this whole research thread:
train<=2022 (profiles + model) / calibration=2023 / holdout=2024, our
own real IPL over-level `wicket_in_over` target (not a proxy), fair
apples-to-apples comparison (identical row subset across all three
tests -- an earlier draft accidentally compared different-sized
populations and was corrected before trusting it).

| | AUC (holdout=2024, n=2,129) |
|---|---:|
| Context only (over/phase/wickets-down) | 0.5737 |
| Context + matchup_score | 0.5629 (**worse**) |
| matchup_score alone | 0.5057 (**no signal, ~random**) |

**Not promoted.** The feature carries no detectable predictive value on
its own and actively hurts when combined with context, most likely by
adding pure noise to a small model.

## Why it likely didn't work

- **Length alone is probably too coarse.** A real weak zone is plausibly
  a length x line combination ("short AND outside off"), not length in
  isolation -- collapsing to one dimension may have thrown away the
  real signal.
- **Individual dismissal-rate-by-length is inherently noisy.** Even with
  200-ball shrinkage, a ~3-6% base-rate event split across 11 length
  buckets per player leaves most players with genuinely thin, noisy
  per-bucket estimates -- the "true" skill differential against
  different lengths may be smaller than the noise floor for most
  players, with only rare, extreme cases (like Rohit's bouncer number)
  showing a real, large effect.
- **Likely selection effects.** Bowlers already tactically target
  perceived weaknesses -- a bowler's *overall* length mix isn't
  independent of who they're bowling to, which muddies a naive
  population-level test built this way.

## Follow-up v2: a real feature layer, not one scalar, informed by actual production research

User's direct pushback on the v1 result ("we are just not digging deep") led to two further, real
attempts rather than defending the first negative result.

**External validation found first**: [Gurpinar-Morgan et al., "You Cannot Do That Ben Stokes"](https://arxiv.org/pdf/2102.01952)
(a real, published, production-grade system using Opta ball-tracking data) confirms the underlying
*mechanism* is real -- a personalized deep model using 46 batter/bowler features nearly doubled a naive
baseline (10.9% -> 19.9% accuracy) for shot-type prediction. It explicitly diagnoses the exact failure
mode v1 hit: *"the best analytics currently generated in cricket rely on either broad averages that
ignore context or ever-diminishing sample size."* Their shrinkage method (blend a player's own data
with the global average, weighted by sample size) matches this project's own H2H/venue shrinkage
convention -- validates that part of the methodology was already right.

**v2 test, built accordingly**: instead of one hand-built scalar, fed a real 26-feature layer -- shrunk
batter dismissal rate per length bucket (11), batter dismissal rate vs. bowler pace-type (3), bowler's
own shrunk delivery-length mix (11), and bowler pace-type as a categorical (1) -- letting LightGBM find
the interaction itself, exactly per the user's direction. Same leakage-safe split
(train<=2022/cal=2023/holdout=2024), same real IPL `wicket_in_over` target.

| | AUC (holdout=2024) |
|---|---:|
| Context only | 0.5809 (n=798) |
| Context + full 26-feature layer | 0.5740 -- **worse** |
| Full layer alone | 0.5131 -- no signal |

Requiring full coverage across all 26 features shrank the holdout to 798 rows (both batter and
bowler need enough career history simultaneously) -- thinner than ideal, but the result is consistent
with v1, not an improvement from richer features.

## Follow-up v3: the field's own recommended simplification, tested directly

Searched further for how practitioners actually handle this. Real cricket analysts
([Cricmetric](https://www.cricmetric.com/blog/2012/05/the-batsman-versus-bowler-matchup-tool/),
[Dan Weston](https://danweston.substack.com/p/quantifying-match-ups)) explicitly warn that
individual batter-vs-bowler samples are too small to trust (even 50 balls is called unreliable) and
recommend the opposite of a granular per-bowler profile: *"a batsman's record against pace or spin
bowling... would generate a more robust sample from a size perspective"* -- broad pace-type
aggregation across many bowlers, not one bowler at a time.

Isolated exactly this (the literature's own most-robust recommendation, not our own invention):
batter's shrunk dismissal rate vs. fast/medium/spin broadly (built from 2,076,860 career rows,
4,685 batters profiled) plus the specific bowler's own pace-type.

| | AUC (holdout=2024, n=2,123) |
|---|---:|
| Context only | 0.5726 |
| Context + bowler_pace_type | 0.5698 |
| Context + expert-recommended pace matchup | 0.5696 |
| batter_rate_vs_bowler_pace alone | 0.5266 -- weak, directionally better than the granular version (0.51), still not usable |

Confirms the "coarser is more robust" principle directionally (0.5266 > 0.5131/0.5057 from the
granular attempts) -- but still doesn't clear a usable bar, and still doesn't beat context when
combined.

## Final conclusion after three real attempts

Four consecutive tests -- a hand-built scalar, a rich 26-feature layer, and the field's own
recommended simplification -- consistently show no promotable signal on **our** data (IPL, ESPNcricinfo
commentary-derived line/length, ~920K-2M career rows depending on scope). The underlying *mechanism*
is real (confirmed by a genuine production system using professional Opta ball-tracking data and a
neural network with far more capacity to share statistical strength across related situations than a
GBM with hand-aggregated features). The binding constraint here is very likely **data quality/source**
-- commentary-scraped categorical line/length is meaningfully noisier than sensor-based ball-tracking
-- not the underlying idea or the feature-engineering approach. Not promoted. Closing this
investigation thread; would need a fundamentally better data source (real ball-tracking, not
commentary-derived) to be worth reopening.

## What's kept vs. discarded

No code or model changes across any of the three attempts -- all research-only, none wired into any
live path. Downloaded career data (up to 294MB extracted subset per attempt) kept in session
scratchpad only, not added to the repo.
