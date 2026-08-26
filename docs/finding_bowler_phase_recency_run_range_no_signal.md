# Finding: bowler phase-specific recency adds no signal to run-range (2026-08-26)

## Context
Batter-side recency-weighted form is a clean, promoted win for run-range
(`finding_recency_weighted_form.md`, live since v9). Bowler-side phase-
specific recency (the analogous lever, other side of the crease) was
tested against the *wicket* target before (`finding_bowler_phase_recency_mixed.md`,
2026-08-23) — real IPL gain but an unstable T20I regression, not promoted
there. It had never been tested against run-range specifically. This
closes that gap, reusing the existing `build_bowler_phase_recency_dataset`
(app/ml/recency_weighted_prior_dataset.py) unchanged — no new dataset
code needed, since the feature already exists and was already validated
(leakage-safe, `canonical_player_id`-based) for the earlier wicket test.

## Result (train_run_range_v20_bowler_phase_recency.py, corrected width-aware bands)
| | v11 (live, reference) | v20 (candidate) |
|---|---|---|
| Blended | 30.04% | 30.04% (30.038%, a true wash — 0.002pp) |
| IPL | 33.25% | 33.16% (slightly worse, -0.09pp) |
| T20I | 29.45% | 29.47% (negligibly better) |

Feature importance ranks: `bowler_recency_phase_wicket_rate` 13th,
`bowler_recency_phase_balls` 19th, `bowler_recency_phase_economy` 30th of
66 — genuinely mid-tier, not the suspiciously high rank/no-lift pattern
seen in `finding_team_phase_venue_no_signal.md`.

**Decision: reject, keep as research only. No production changes.**

## Why this is a cleaner negative than the other tests today
This is the closest-to-neutral result of the four run-range candidates
tested today (batter-vs-team, team-phase-venue, this one) — genuinely
flat rather than a clear IPL regression. Plausible reason: run-range
already has `bowl_phase_avg_runs` (career-to-date, phase-specific, just
not recency-weighted) plus the general match-state features (RR,
wickets-in-hand). A bowler's phase-specific *economy* has less headroom
to move the needle on "how many runs land in the NEXT over" than a
batter's recent form does, because batters have much more agency over an
over's outcome (which ball to attack, strike rotation) than the specific
bowler bowling it — consistent with the wicket-side test where the batter
lever also outranked the bowler lever (`bowler_recency_economy` 28th vs
`striker_recency_balls` 4th, per the mixed finding's opening comparison).

## Standing pattern after today's four tests
Every non-batter-specific idea tried today for run-range (batter-vs-
opponent-team, team-phase-venue mean+volatility, bowler-phase-recency)
came back flat-to-negative. Batter-side, career-to-date, recency-weighted
signal remains the only lever that has actually moved run-range accuracy
this project has found. Not fully exhausted (batter-side ideas not yet
tried today weren't revisited), but bowler-side and team/venue-context
levers both look tapped out for this target specifically.
