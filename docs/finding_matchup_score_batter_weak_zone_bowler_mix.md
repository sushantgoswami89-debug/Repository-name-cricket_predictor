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

## What's kept vs. discarded

No code or model changes -- this was a research-only test, not wired
into any live path. The player-profile-building approach (precomputable
by identity, no live-commentary dependency) remains the right
*architecture* if a future refinement (length x line combined, or
feeding raw profile components into the model instead of one hand-built
scalar, letting the GBM find the interaction itself) is ever worth
testing -- but that's a new hypothesis to test on its own merits, not
assumed to work because this specific formulation didn't. Downloaded
career data (294MB extracted subset) kept in session scratchpad only,
not added to the repo.
