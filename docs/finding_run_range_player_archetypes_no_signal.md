# Finding: batter/bowler role-archetype clustering adds no signal to run-range (2026-08-26)

## Why this was tried
Follow-up to explicit web research (user: "check with internet what else
we can do") into published cricket-analytics techniques not yet tried in
this project. Found: IPL batting-archetype studies using K-means
clustering on batter phase/style profiles, then using the cluster as a
role feature. Genuinely different mechanism from anything tested today —
a generalization/role layer, not another raw stat, interaction, or
architecture change.

## What was built
`app/ml/player_archetype_dataset.py`: career-to-date, leakage-safe
3-phase (powerplay/middle/death) scoring-rate profile per batter and
economy profile per bowler, shrunk toward the global per-phase rate.
`train_run_range_v24_player_archetypes.py`: fit K-means (K=4) on these
profile vectors using TRAIN-split rows only (no calibration/holdout
leakage into the clustering itself), assign every row's `striker_archetype`/
`bowler_archetype` via nearest-centroid, add both as new categorical
features on top of the live `run_range_v11_partnership_rate` set.

## Spot-check (before training)
MS Dhoni: powerplay 1.18, middle 1.12, death 1.71 runs/ball — a real
finisher shape (large late-innings acceleration), matching his known
reputation. Kohli/Rohit showed stronger, flatter profiles consistent with
accumulator/opener roles. Passed sanity check.

## Result (corrected width-aware bands)
| | v11 (live, reference) | v24 (candidate) |
|---|---|---|
| Blended | 30.04% | 29.997% (flat/negligibly worse) |
| IPL | 33.25% | 32.84% (worse) |
| T20I | 29.45% | 29.48% (negligibly better) |

Feature importance: `striker_archetype` 55th of 65, `bowler_archetype`
56th — genuinely low, not a "high importance, no lift" case like the
team-phase-venue volatility feature earlier today.

**Decision: reject, keep as research only. No production changes.**

## The centroids themselves are real, just not useful to this model
Batter clusters found real, distinguishable shapes — e.g. cluster 0
(powerplay 1.22, death 1.64 — a finisher-type curve) vs cluster 1 (1.07/
1.36 — flatter, weaker throughout) vs cluster 3 (1.23/1.42 — strong early,
flatter — anchor/opener-shaped). This isn't a degenerate clustering.

## Why it doesn't help anyway
The archetype label is built entirely from phase-specific scoring/economy
rates the GBM already receives directly as continuous features (via the
existing MOE/phase-prior features). Collapsing that information into a
4-way categorical bucket is a compression, not new information — a tree
model can already split flexibly on the underlying continuous values with
more precision than a 4-cluster label allows. The technique's real value
in the published research is for *team-composition analysis and human
interpretability* (e.g. "does this team have the right balance of
anchors and finishers"), not for sharpening an already-flexible model's
per-over predictions. Would plausibly matter more for a simpler model
(e.g. linear regression) that can't already learn nonlinear splits on the
raw rates — not relevant here.

## Where this leaves the "push for more" thread
Five real feature/architecture ideas tested today for run-range
(batter-vs-team, team-phase-venue, bowler-phase-recency, bigger NN,
NN bagging + phase-blend, player archetypes) are now flat-to-negative
apart from the one clear win (the GBM+NN ensemble itself, architecture
not feature). The honest read: this specific target, with this specific
feature/architecture family, is well-explored. Future gains most likely
require either new data (not more feature engineering on what's already
here) or a genuinely different target formulation (e.g. per-ball rather
than per-over prediction), not further tuning of this line.
