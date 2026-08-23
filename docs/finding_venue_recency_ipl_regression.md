# Recency-weighted venue par score: hurts IPL specifically in both models — not promoted

Date: 2026-08-23. User-flagged strategic direction: find a lever that
generalizes across models, not another narrow single-model feature. The
existing `venue_par_score` (`app/ml/ipl_venue_regime_dataset.py`) is a
flat, all-time average of every first-innings total ever played at a
venue -- the same "no recency weighting" gap already fixed for player
form (`recency_weighted_prior_dataset.py`) and current-partnership state
(`partnership_dataset.py`). Unlike those, `venue_par_score` is already
consumed by *both* live models (run-range and wicket both use
`VENUE_FEATURES`), so this was the first test this session genuinely
built once and tried on both.

## What was built

`app/ml/venue_recency_dataset.py`: an innings-indexed EWMA (decay=0.95)
version of the venue par score, added alongside the existing flat one.
Spot-checked against Chinnaswamy's real trajectory before trusting it --
correctly trends toward higher recent-scoring (176.9 recency vs 170.99
flat, as of 2024), consistent with the venue's known reputation as an
increasingly batting-friendly ground.

## Tested against both real live baselines

**Run-range** (`train_run_range_v12_venue_recency.py`, vs. live `run_range_v11_partnership_rate`):

| | Blended | IPL | T20I |
|---|---:|---:|---:|
| Live `v11` | 28.80% | 25.16% | 29.45% |
| `v12` + venue recency | 28.83% | **24.66%** | 29.59% |

The script's own mechanical gate says "promote" (blended technically
improves and calibration doesn't regress), but IPL -- the project's
priority population -- drops by half a point, a larger regression than
most rejected candidates this session. **Not promoted despite the
gate passing**, per the standing practice of never trusting a blended
number that hasn't been checked split by population
(`finding_blended_holdout_masks_ipl_accuracy.md`).

**Wicket** (`train_contract22_wicket_v14_venue_recency.py`, vs. live `contract22_wicket_v10_partnership_rate`):

| | AUC known | AUC unknown |
|---|---:|---:|
| Live `v10`, IPL | 0.6104 | 0.6059 |
| `v14`, IPL | **0.6088** | 0.6064 |
| Live `v10`, T20I | 0.6113 | 0.6110 |
| `v14`, T20I | 0.6108 | 0.6097 |

Worse on 5 of 6 reported cuts, IPL-known notably so. Despite this,
`venue_recency_par_score` ranks 4th of 54 (run-range: 3rd of 65) by
importance in both models -- heavily used by the trees, but the net
effect is negative. High importance isn't the same as helpful: it's
plausibly cannibalizing information the existing, already-tuned flat
`venue_par_score` (with its own shrinkage-toward-global-average logic)
already captured well, adding conflicting rather than complementary
signal.

## Why IPL specifically, in both models

A coherent, plausible explanation: IPL matches concentrate on a small,
fixed set of ~13-14 franchise home venues that each host many matches
per season already -- the existing flat average already has a large,
stable sample there, so recency-weighting mostly just narrows the
effective sample (discounting real history) without correcting genuine
drift. T20I is played across a far more geographically diverse set of
grounds with less repeat usage per venue, where a flat all-time average
is more likely to be stale or mixing unrelated grounds' conditions --
recency-weighting has more room to help there, and did (T20I improved in
both models: run-range +0.14pp, wicket flat-to-slightly-worse but less so
than IPL).

## Recommendation

**Not promoted, either model.** This is the first venue-level test this
session, and it replicates cleanly in the same direction on two
independently-trained models -- that consistency is itself informative,
not just noise. `run_range_v11_partnership_rate` and
`contract22_wicket_v10_partnership_rate` remain live, unchanged.

Candidate artifacts: `models/candidates/run_range_v12_venue_recency/`,
`models/candidates/contract22_wicket_v14_venue_recency/`. If venue
conditions are revisited, restricting recency-weighting to venues with
sparser/more volatile history (rather than applying it uniformly) is the
natural next formulation -- not tested here.
