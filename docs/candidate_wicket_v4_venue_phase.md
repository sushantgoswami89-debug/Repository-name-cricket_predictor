# Wicket v4 — Batter/Bowler Venue x Phase Profiles

Date: 2026-08-22

## Motivation

Neither the batter nor bowler side of any model in this codebase had a
per-player venue split. `ipl_venue_regime_dataset.py` pools all players at
a venue into one par score; the richest per-player split
(`wicket_contract22_bowler_phase_stats.json`) is player x phase, no venue.
No dataset anywhere answered "how does this specific batter/bowler perform
at this specific venue in this specific phase."

## Sparsity check, before building anything

Bowler-phase (2-way, no venue) cells: median 42 balls across 8,618 cells,
1,351 (16%) under 12 balls. Adding a third dimension (315 venues) divides
that further -- a raw per-(player, venue, phase) groupby would mostly be
noise, the same failure mode already documented for the phase-MoE
specialist line (`ipl_catboost_phase_moe_v1_report.md`'s "unsupported
segments: zero held-out rows").

## Approach

Built `app/ml/player_venue_phase_dataset.py`: each (player, venue, phase)
cell is shrunk toward the player's own venue-agnostic (player, phase) rate,
weighted `cell_balls / (cell_balls + 24)` -- same style of shrinkage
`ipl_venue_regime_dataset.py` already uses for venue par scores, applied
one level down to the player. Leakage-safe: profiles updated only after
every row in a match is emitted (mirrors `build_ipl_phase_moe_features`).

Merged onto `contract22_wicket_v2_batter_state`'s exact feature set (the
currently-live wicket model) and retrained CatBoost/LightGBM the same way,
same chronological split (train<=2023/calib=2024/holdout>=2025), single
holdout evaluation.

## Result: rejected -- within noise, not a real gain

| | AUC (known) | AUC (unknown) | Brier (Platt) |
|---|---:|---:|---:|
| v2 (reference, currently live) | 0.6081 | 0.6072 | 0.2049 |
| v4 (this candidate) | 0.6085 | 0.6089 | 0.2048 |

+0.0004 to +0.0017 AUC, Brier flat to the fourth decimal -- smaller than
several movements already treated as noise elsewhere this session (e.g.
the phase_penalty holdout-AUC reproducibility check landed within 0.0002).
Feature importance confirms it: the seven new venue-phase columns ranked
11th, 15th, 16th, 18th, 20th, 26th, and 27th of 47 total features -- present,
not dominant.

**Conclusion**: shrinkage successfully avoided making things worse (unlike
a raw groupby would have on this sparse a signal), but there's no
exploitable venue-specific-per-player signal beyond what phase and
pooled-venue already capture separately. This closes the "venue-level
pitch classification" thread flagged as a next direction in
`candidate_ipl_wicket_v7_2_spell_features.md` -- tried, real, honestly
neutral. Not wired into production; kept as `models/candidates/contract22_wicket_v4_venue_phase/`
for reference. Don't re-try per-player venue splits on this data volume
without a new data source (more seasons/venues) to reduce cell sparsity.

## v5 follow-up: 1st-innings (setting) vs 2nd-innings (chasing) split

Date: 2026-08-22, user-flagged hypothesis: a batter's/bowler's average in
a given phase can genuinely differ between setting a total and chasing
one -- distinct from the venue split above (a 2-way innings split, not a
315-way venue split; sparsity check confirmed powerplay/middle/death row
counts split close to 50/50 between innings, so this isn't the same
sparsity failure mode). Built `batter_phase_innings_*` /
`bowler_phase_innings_*` in `player_venue_phase_dataset.py`
(`SHRINKAGE_BALLS_INNINGS=12`, shrunk toward the innings-agnostic phase
rate), tested both sides:

| | metric | live baseline | + innings x phase |
|---|---|---:|---:|
| Runs (`train_run_range_v5_innings_phase.py`) | holdout hit rate | 28.60% | **28.37% (-0.82% relative, worse)** |
| Wicket (`train_contract22_wicket_v5_innings_phase.py`) | AUC known/unknown | 0.6081/0.6072 | 0.6087/0.6073 (flat, noise) |

**Rejected on both sides** -- bowler side is neutral (same noise-level
verdict as the venue test); batter side is a genuine regression, not just
flat (`batter_phase_innings_runs_per_ball_shrunk` ranked last, 28th of 57
features). Row-count balance wasn't the risk here -- overlapping signal
was: `chase_pressure`/`state_regime` (already live features) already
capture most of how 2nd-innings match state differs from 1st via required
rate and wickets-in-hand, so splitting the player's own historical rate
by innings on top of that adds a thinner, mostly-redundant slice rather
than new information. Same "largely redundant with what's already
captured" pattern `contract22_wicket_v3_pressure_state` found for a
related over-enrichment attempt. Not wired in; kept as
`models/candidates/run_range_v5_innings_phase/` and
`models/candidates/contract22_wicket_v5_innings_phase/` for reference.
