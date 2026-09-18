# Finding: batting-order position — tested, rejected for both models

**Date**: 2026-09-18
**Status**: Tested and rejected. Neither live model changed.

## Hypothesis

User asked for another feature candidate to test for run-range/wicket
accuracy. Checked first (2026-09-18) that batting-order position — a
real, well-known cricket skill/role proxy (a number-9 batter is far more
likely to get out and scores fewer runs than a number-3) — was used by
**neither** the live run-range nor wicket model at all, despite being one
of the most obvious facts about a batter's role. Genuinely untested,
unlike most other recently-tried ideas.

## Built

`app/ml/batting_position_dataset.py`: derives each batter's position as
the order of their first appearance in the innings (1-indexed), the same
leakage-safe convention as every other per-over feature in this codebase
(the row for over N reflects positions established strictly before over
N's own first ball). Spot-checked against a real match
(`data/raw/cricsheet/t20i/1528275.json`) before trusting it: positions
increased monotonically as wickets fell (1, 2, 3, 4, 5, 6, 7, 8),
matching the real chronological order of appearance exactly.

## Tested on both models

Identical architecture/features/split/calibration to each live model,
adding only `striker_batting_position`/`non_striker_batting_position`.

**Wicket** (`train_contract22_wicket_v26_batting_position.py`, reference:
`contract22_wicket_v16_team_composition`'s own real numbers):

| | Live (known) | Candidate (known) | Live (unknown) | Candidate (unknown) |
|---|---:|---:|---:|---:|
| Blended AUC | 0.6126 | 0.6128 | 0.6114 | 0.6109 |
| IPL AUC | 0.6151 | 0.6162 | 0.6131 | 0.6115 |
| T20I AUC | 0.6113 | 0.6113 | 0.6113 | 0.6110 |

Essentially flat on the "known" ceiling case (small IPL uptick within
noise), **worse on the "unknown" realistic case** — the actually-common
live-serving scenario — across every cut.

**Run-range** (`train_run_range_v27_batting_position.py`, reference:
`run_range_v11_partnership_rate`'s own real numbers, 30.04/33.25/29.45%
blended/IPL/T20I):

| Cut | Live | Candidate | Delta |
|---|---:|---:|---:|
| Blended | 30.04% | 29.93% | −0.11pp |
| IPL | 33.25% | 32.46% | −0.79pp |
| T20I | 29.45% | 29.47% | +0.02pp (noise) |

Worse on blended and IPL, flat on T20I.

Feature importance was low in both models (ranked 44th/40th of 63 for
wicket, 41st/43rd of 65 for run-range) — consistent with the negative
holdout results, not contradicting them.

## Conclusion

Rejected for both models. Most likely explanation: player-identity-based
prior/recency stats (career average, strike rate, dismissal rate — all
already in both models, per-player) already capture most of what batting
position would tell the model for a known player, and the incremental
signal for a genuinely new/sparse-history player is too thin to move
holdout accuracy once shrinkage priors already handle that case. Not
promoted — `contract22_wicket_v16_team_composition` and
`run_range_v11_partnership_rate` remain live, unchanged.

`app/ml/batting_position_dataset.py` kept as real, reusable
infrastructure — the leakage-safe derivation itself is correct and
spot-checked, even though this specific hypothesis didn't pay off; worth
reusing directly if a future feature needs "which numbered batter is
this" (e.g., a phase-specific tailender-death-overs interaction, not yet
tried).
