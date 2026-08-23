# Team-vs-team head-to-head history: no signal on either model, despite strong external grounding

Date: 2026-08-23. User request: "search for more." Web research on
cricket analytics specifically flagged this as unusually predictive for
IPL: "IPL's consistency of venue -- each team plays at the same home
ground every season -- and the relative stability of squad identity
across multiple seasons means that head-to-head records carry more
predictive value in IPL than in most other cricket competitions." Every
H2H feature already in this codebase (`MATCHUP_FEATURES`,
`h2h_avg_runs`) is player-vs-player (batter vs bowler); this tested
franchise-vs-franchise for the first time.

## What was built

`app/ml/team_h2h_dataset.py`: does the currently-batting team tend to
beat this specific opponent, shrunk toward a neutral 0.5 prior for
sparse pairings (most franchise pairings have played only a handful of
times). IPL-only -- T20I national-side pairings don't have the
fixed-franchise/fixed-venue structure this is actually about. Fixed a
design issue during construction: the first draft shrunk toward a
"global batting-first-team win rate," which doesn't match the quantity
being estimated (team-specific win rate against a specific opponent) --
corrected to shrink toward neutral 0.5, the only defensible prior for an
unseen pairing. Spot-checked against the real CSK-vs-MI rivalry (41
matches found, win rate correctly hovers near 0.5, consistent with their
well-known evenly-matched history) before trusting it.

## Tested on both models: clean negative, both

**Run-range** (`train_run_range_v15_team_h2h.py`, vs. live `run_range_v11_partnership_rate`):

| | Blended | IPL | T20I |
|---|---:|---:|---:|
| Live `v11` | 28.80% | 25.16% | 29.45% |
| `v15` + team H2H | 28.68% | 24.82% | 29.38% |

Worse on all three cuts.

**Wicket** (`train_contract22_wicket_v17_team_h2h.py`, vs. live `contract22_wicket_v16_team_composition`):

| | AUC known | AUC unknown |
|---|---:|---:|
| Live `v16`, blended | 0.6126 | 0.6114 |
| `v17`, blended | 0.6123 | 0.6107 |
| Live `v16`, IPL | 0.6151 | 0.6131 |
| `v17`, IPL | 0.6137 | 0.6100 |
| Live `v16`, T20I | 0.6113 | 0.6113 |
| `v17`, T20I | 0.6110 | 0.6103 |

Worse on every single cut, IPL unknown-bowler most notably (-0.0031).
Feature importance ranks low on both models (33rd-43rd of 60+ features).

## Why a well-grounded, externally-validated signal still didn't help

Plausible explanation, consistent with this session's other rejections
of pre-match contextual signals (toss, home advantage): team-level
head-to-head win/loss history is largely a *proxy* for relative squad
strength -- which team has the better players. This model already has
much richer, more direct signals about exactly that (career stats,
recency-weighted form, phase-specific tendencies, individual matchup
history, team role composition) computed at the player level, not the
team level. Once those are already in the feature set, a coarser
team-level summary of "who tends to win this pairing" adds redundant,
noisier information rather than new signal -- the finer-grained data
already explains whatever the head-to-head record was capturing.

## Recommendation

**Not promoted, either model.** `run_range_v11_partnership_rate` and
`contract22_wicket_v16_team_composition` remain live, unchanged. This is
now the third pre-match/contextual (as opposed to player- or
match-state-level) signal tested and rejected this session (toss, home
advantage, team H2H) -- a consistent enough pattern to deprioritize
further contextual/team-level feature ideas in favor of player-level or
in-match-state ones, which have been this session's actual source of
real improvements.

Candidate artifacts: `models/candidates/run_range_v15_team_h2h/`,
`models/candidates/contract22_wicket_v17_team_h2h/`. Neither promoted.
