# Batter pressure-response (two formulations + combined): real signal, closed, no net win

Date: 2026-08-25. User's idea, prompted by "any thoughts on model that
need update": does a specific batter's own historical tendency to get
dismissed under pressure predict wicket risk beyond what the live model
already has? Tested three ways on `contract22_wicket_v16_team_composition`
(currently live), same architecture/split/IPL-T20I methodology as every
other wicket candidate.

## Why this was worth testing (unlike the "volatility" tests earlier)

This project's own history shows a clear pattern: features that helped
the wicket model (recency-weighted form, partnership rate, phase-specific
batter recency, team composition) were all **player-specific**; features
that failed (toss, home advantage, team H2H, venue recency, this
session's volatility tests) were all **generic match-context**. This idea
is squarely player-specific, giving it a real structural reason to expect
a different outcome -- not just optimism.

## v20: one-over stall trigger

`pressure_trigger` = the immediately preceding over's runs were at least
half below the run rate established up to that point (matches the user's
own example: RR=8.0, an over yields 4). `striker_pressure_dismissal_rate_shrunk`
= this batter's career-to-date dismissal rate specifically following such
a trigger, shrunk toward the global rate.

**Spot-checked first**: V Kohli 9.8%, MS Dhoni 11.2% vs AD Russell 30.9%
-- Russell (famously high-risk) nearly 3x Kohli/Dhoni (famously composed).
Real, sane signal before any model was touched.

**Result**: `striker_pressure_dismissal_rate_shrunk` ranked 18th of 63.
Blended AUC barely moved (+0.0002 both known/unknown); IPL regressed on
both cuts (-0.0016 known, -0.0025 unknown); T20I saw a tiny genuine gain.

## v24: sustained multi-over survival (the "survival/hazard framing" flagged 2026-08-24, applied as a feature)

User's refinement: not a one-over blip but genuine chase pressure
(`current_run_rate < required_run_rate`, sustained). Two features:
`overs_under_pressure_this_spell` (live, real-time spell length) and
`striker_extended_pressure_dismissal_rate_shrunk` (this batter's
career-to-date dismissal rate once 2+ overs into such a spell).

**Spot-checked again**: V Kohli 11.4%, MS Dhoni 14.0% vs AD Russell 22.7%,
HH Pandya 24.7% -- composed players still low, aggressive players still
higher (Pandya's ordering shifted vs v20, a genuinely different but still
sane signal -- sustained pressure and single-over pressure aren't
identical psychological tests).

**Result**: `striker_extended_pressure_dismissal_rate_shrunk` ranked 17th
of 63 -- almost identical importance to v20's feature. Blended AUC barely
moved (+0.0004/-0.0003); IPL regressed on both cuts again (-0.0016/-0.0024)
-- nearly the same magnitude as v20's IPL regression.

## v25: combined -- the decisive check

Two independently-motivated formulations landing at nearly identical
individual results is suspicious (same signal measured twice, not two
complementary ones) -- tested directly by combining both in one
candidate, per this project's standing practice of testing a combination
before closing an investigation (same as the matchup-score closure).

**Result: combining them made things WORSE than either alone on most
cuts** (known blended 0.6125 vs 0.6128/0.6130 individually; unknown
blended 0.6105 vs 0.6116/0.6111 individually), even though both features'
importance rank *rose* when combined (13th, 14th) -- the model leans on
them more, yet holdout performance drops. This is the clearest version of
the "high importance != helpful" pattern already seen once this project
(venue recency, 2026-08-23): the two formulations are redundant enough
that combining them adds instability, not complementary signal.

## Verdict

**Closed, not promoted, any formulation.** Real, per-player signal exists
(confirmed twice by spot-check and by real, non-trivial feature
importance) -- but it appears to be substantially the same underlying
signal the model's existing recency/dismissal-rate features already
capture, and slicing "pressure" a different way doesn't add anything net,
while consistently costing IPL specifically. The combined test's
backfire is strong evidence this is genuine redundancy, not an unlucky
holdout split.

No code/model changes to the live wicket model. New infrastructure kept
(research-only): `app/ml/batter_pressure_response_dataset.py`,
`app/ml/pressure_survival_dataset.py`,
`backend/train_contract22_wicket_v20_batter_pressure_response.py`,
`backend/train_contract22_wicket_v24_pressure_survival.py`,
`backend/train_contract22_wicket_v25_combined_pressure.py`, three
`models/candidates/` directories.

Also this session: `batting_team_venue_context` (home/away) re-tested on
the current architecture (`v22`), confirmed rejected consistent with the
original 2026-08-23 test; tested combined with a target-size bucket
(`v23`), made things worse than venue alone -- both closed, no
interaction found. `h2h_wicket_rate` (bowler's history dismissing this
specific batter, `v21`) also tested: real feature (rank 28/62, previously
tracked internally in `build_enriched()` but never surfaced), small
real IPL gain on the known-bowler cut but a larger loss on unknown-bowler
IPL -- not promoted either.
