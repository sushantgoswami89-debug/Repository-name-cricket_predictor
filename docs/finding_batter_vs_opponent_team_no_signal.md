# Finding: batter-vs-opponent-team history adds no signal to run-range (2026-08-26)

## Request
"check recent form and historical form against a team, it should in weighed
but not hard coded, recent form in totality" — test whether a batter's own
historical scoring rate specifically against the current opponent TEAM
(distinct from the existing bowler-specific H2H and from team-vs-team H2H)
helps run-range prediction, with weighting left to the model rather than a
hand-coded blend formula.

## What was built
`app/ml/batter_vs_team_dataset.py`: career-to-date, chronological,
leakage-safe (updated only between matches) batter scoring rate against
each opponent team collectively, shrunk toward the batter's own overall
prior rate (`SHRINKAGE_BALLS=60`). Player identity via `canonical_player_id`
(registry-based) from the start — applying the lesson from the
pressure-scoring bug earlier this session, where raw Cricsheet names
fragmented player history and produced a false positive.

Added as a single extra input feature
(`striker_vs_opponent_team_runs_per_ball_shrunk`) on top of the live
`run_range_v11_partnership_rate` feature set, letting the GBM's own splits
decide how to weight it against the existing recency-weighted general-form
features — no hard-coded blend, per the request.

## Spot-check (before training)
V Kohli's shrunk rate across his highest-sample opponents: Sunrisers
Hyderabad 1.51 runs/ball, Australia 1.44, Kolkata Knight Riders 1.41,
Mumbai Indians 1.33, Pakistan 1.25. Sane, non-degenerate spread matching
known form — passed sanity check.

## Result (train_run_range_v18_batter_vs_team.py, corrected width-aware bands)
| | v11 (live, reference) | v18 (candidate) |
|---|---|---|
| Blended | 30.04% | 30.04% (30.035%, effectively flat) |
| IPL | 33.25% | 32.86% (worse) |
| T20I | 29.45% | 29.52% (negligibly better) |

Feature importance rank: 21st of 64 — meaningfully lower than the
pressure-scoring feature's rank (4th) that was also rejected, indicating
genuinely weak signal rather than "important but not helpful."

**Decision: reject, keep as research only. No production changes.**

## Why this likely doesn't help
The batter's overall recency-weighted form and competition-specific priors
already live in the model account for most of what "form against a team"
would capture — a batter who's in good form or historically strong tends to
score at a similar rate against most opponents, and the opponent-specific
slice is usually too data-sparse (median 8 balls) to add information beyond
the player's general rate it's already shrunk toward. This is consistent
with the standing pattern this session: TEAM-level context (team H2H, venue
recency, toss, cascade features) tends not to help; this feature, while
technically player-specific, is really asking a team-level question
("does this player do better against team X") and inherits that same
weak-signal pattern.
