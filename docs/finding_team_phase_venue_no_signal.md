# Finding: team x phase x venue scoring average + volatility adds no signal (2026-08-26)

## Request
"use RR, wicket in hands, Target, avg of team in each phase and venue" —
then corrected: "not hardcoded, we needed a random or volatile index."

RR (`current_run_rate`, `required_run_rate`), wickets-in-hand
(`wickets_in_hand`), and Target-derived state (`runs_required`, `is_chase`)
were already live inputs (`BASE_FEATURES`) — not rebuilt. What was
genuinely missing was a TEAM x PHASE x VENUE scoring number: existing
venue features are venue-wide (`venue_par_score`) or team x venue as a
home/away label only (`batting_team_venue_context`), never an actual team
scoring average at a venue in a given phase.

The "volatile index" instruction was implemented literally as a real
volatility statistic (std-dev of the team's own per-over run totals in
that bucket, career-to-date) — not a fixed formula, not random noise —
alongside the mean, both hierarchically shrunk
(team, venue, phase) → (team, phase) → global (phase). No hand-coded
blend: both features feed the GBM's own learned splits alongside
everything else already live.

## What was built
`app/ml/team_phase_venue_dataset.py`: two outputs per pre-over row —
`team_phase_venue_runs_per_over_shrunk` (mean) and
`team_phase_venue_volatility_shrunk` (std-dev), both career-to-date and
leakage-safe (updated only between matches).

## Spot-check (before training)
CSK @ Chepauk death overs: 9.92 runs/over, std 5.04 (387 overs of history).
MI @ Wankhede death overs: 10.37 runs/over, std 5.68 (433 overs). Wankhede
is a smaller, more boundary-friendly ground than Chepauk — both the mean
*and* the volatility come out higher there, matching real cricketing
knowledge. Global phase means (powerplay 7.72, middle 7.72, death 9.58)
also match the expected shape. Passed sanity check.

## Result (train_run_range_v19_team_phase_venue.py, corrected width-aware bands)
| | v11 (live, reference) | v19 (candidate) |
|---|---|---|
| Blended | 30.04% | 30.03% (30.027%, effectively flat) |
| IPL | 33.25% | 32.89% (worse) |
| T20I | 29.45% | 29.51% (negligibly better) |

**Decision: reject, keep as research only. No production changes.**

## The notable part: importance rank vs. real lift
`team_phase_venue_volatility_shrunk` ranked **#1 of 65 features** by GBM
importance — the single most-split-on feature in the whole model. The
mean feature ranked #5. Despite this, real holdout hit rate did not
improve and IPL got measurably worse. This is the sharpest instance yet
of a pattern seen repeatedly this session (batter pressure-scoring
ranked #4 and was still rejected; see
[finding_batter_pressure_scoring closed]): the GBM finds real structure to
split on in a feature (it correlates with something — likely just
restating each phase's baseline run level, which the model was already
getting some of through `phase` and `venue_par_score`), but that structure
doesn't translate into sharper predictions once temperature-calibration
and best-fit banding are applied. Feature importance measures how much a
feature is used, not how much using it helps — this result is the
clearest demonstration of that gap in the whole project so far.

## Why this likely doesn't help
The venue-wide par score and phase bucket already carry most of the
"how does scoring vary by ground and by phase" signal generically; adding
a team-specific slice on top is a fine-grained re-statement of information
the model can already approximate from team quality (via player-level
priors) x venue par x phase, without enough independent variance sparse
team-venue-phase samples (many buckets built on well under 100 overs of
history) can add beyond what's already there.
