# The reported run-range holdout accuracy has been a blended IPL+T20I number all along

Date: 2026-08-22, user-flagged: "most of the stadiums in the data are new
[to the current rotation] -- run the test against the valid data set."

## What was actually checked

The literal venue-staleness question checked out fine: 328 of 1,243 IPL
matches (26.4%) are at now-discontinued venues (UAE relocated seasons --
Dubai/Abu Dhabi/Sharjah, 111 matches, last used 2021; Pune MCA/Mumbai
Brabourne/DY Patil, 115 matches, last used 2022; plus older 2009-2018
defunct grounds). But the 2025-2026 holdout period (what the model is
actually evaluated against) already only contains currently-active
venues -- 0 of 5,576 IPL holdout rows are at a discontinued venue. IPL's
own venue rotation had already consolidated by the time the holdout
window starts. Not a live problem.

## What the check actually surfaced instead

Re-evaluating `run_range_v4_batter_phase`'s already-trained holdout
predictions split by competition (IPL vs. T20I), rather than the blended
number every report this session has quoted:

| Scope | Holdout rows | Hit rate |
|---|---:|---:|
| IPL only | 5,576 (15.3%) | **24.19%** |
| T20I only | 30,808 (84.7%) | 29.22% |
| Blended (the number reported everywhere) | 36,384 | 28.45% |

The T20I population -- 5.5x larger than the IPL population in this
holdout -- pulls the headline number up by roughly 4-5 points. This isn't
specific to v4: v2 (28.37%), v3.3 (28.10%), v3 (28.49%) all used the same
IPL+T20I population (`male_source_files()`, `scopes=("ipl","t20i")`) and
reported one blended figure throughout the entire run-range candidate
line documented in `candidate_run_range_enriched_v2.md`. None of those
numbers have ever been IPL-isolated before this check.

Per-IPL-venue breakdown (2025-2026 holdout, `n` = rows):

| venue | rows | hit rate |
|---|---:|---:|
| ahmedabad_narendra_modi | 657 | 22.8% |
| lucknow_ekana | 580 | 24.3% |
| mumbai_wankhede | 537 | 22.2% |
| delhi_arun_jaitley | 529 | 24.8% |
| chennai_chepauk | 504 | 27.2% |
| kolkata_eden_gardens | 493 | 26.0% |
| hyderabad_rajiv_gandhi | 486 | 24.3% |
| new_chandigarh_mullanpur | 454 | 24.0% |
| jaipur_sawai_mansingh | 426 | 24.6% |
| bengaluru_chinnaswamy | 374 | 23.8% |
| dharamsala_hpca | 210 | 17.6% (thin) |
| guwahati_barsapara | 171 | 24.6% |
| raipur_shaheed_veer_narayan | 80 | 32.5% (thin, brand-new venue) |
| visakhapatnam_acavdca | 75 | 22.7% (thin) |

Most venues cluster tightly around 22-27%; Dharamsala and Raipur are the
only real outliers, and both have thin samples (210 and 80 rows) --
plausible noise, not (yet) evidence of a venue-specific problem.

## Why T20I scores higher than IPL on the same fixed-width-band metric

Not investigated in depth here (out of scope for this check), but worth
naming as an open question: IPL is a higher, more competitive standard
(better bowlers, sharper death-overs execution, tighter margins) than the
median global T20I fixture in the training population (many associate-
nation and lower-intensity bilateral matches) -- run outcomes in the
average T20I row may simply be more clustered/predictable than IPL's,
independent of anything about venues.

## Recommendation, going forward

**Report IPL-only and T20I-only holdout accuracy separately in every
future validation, not just a blended figure.** The blended number has
been silently overstating accuracy on the competition that most likely
matters most (IPL is the project's namesake and primary focus throughout
`docs/`). This applies to any future candidate built on the
`scopes=("ipl","t20i")` population -- both wicket and run-range lines.

Not a code change -- a measurement/reporting practice change. No
production model swap follows from this finding by itself; it's a
transparency fix, and a flag that the *true* IPL-specific accuracy
ceiling (~24%) is a more honest number to anchor future improvement work
against than the blended ~28.5% figure this session repeatedly cited.

## Follow-up: root-cause dig, and the training-imbalance hypothesis was wrong

Bias check (`diagnose_ipl_t20i_bias.py`) found the model systematically
under-predicts IPL runs-per-over (-0.376 overall, -0.971 in the
powerplay specifically -- true IPL powerplay mean 9.86, predicted 8.88),
while T20I is nearly unbiased (-0.073). The obvious hypothesis: T20I
outnumbers IPL 1.83:1 in training with equal per-row weight, so the
shared model's decision boundaries get pulled toward T20I's lower center
-- the same failure mode the wicket v7.x line found and fixed by
down-weighting T20I.

**Tested directly (`train_run_range_v6_ipl_weighted.py`) and the
hypothesis does not hold:**

| Config | IPL hit rate | T20I hit rate | IPL bias |
|---|---:|---:|---:|
| Pooled, T20I weight=1.0 (current live) | 24.19% | 29.22% | -0.376 |
| Pooled, T20I weight=0.5 | 24.19% | 29.28% | -0.405 |
| Pooled, T20I weight=0.3 | 23.96% | 29.20% | -0.467 |
| Pooled, T20I weight=0.15 | 24.09% | 28.90% | -0.422 |
| IPL-only (T20I excluded from training) | 23.83% | 27.90% | -0.237 |
| T20I-only (IPL excluded from training) | 24.64% | 29.22% | -0.421 |

Down-weighting T20I doesn't move IPL hit rate at all (flat to slightly
worse) and makes IPL's bias *worse*, not better. Training an IPL-only
model (39,442 rows instead of 111,533 pooled) makes IPL hit rate worse,
not better -- its raw phase temperatures (1.20-1.43) are far higher than
the pooled model's (1.04-1.15), a clear overfitting signature from the
smaller sample, same pattern already documented elsewhere this session.
Most tellingly: the T20I-only model, which has never seen a single IPL
training row, scores *best* on IPL holdout (24.64%) of every
configuration tested.

**Revised conclusion**: the wicket-line analogy doesn't transfer. That
fix solved a sample-size problem for a sparse binary target; here every
configuration lands IPL in the same 23.8-24.6% band regardless of
training population, which points to something closer to the original
intrinsic-variance hypothesis (IPL's higher std, 4.72 vs 4.58) rather
than a fixable training-imbalance bug. A fixed-width-2 band structurally
covers less of a wider distribution -- this may be close to IPL's real
ceiling for this exact metric shape, not a bug to reweight away.
**Reweighting and full separation were both fair, real tests, and both
came back negative -- don't re-try either lever on this exact
architecture.** If IPL accuracy is worth pushing further, the next real
lever is band *width* (does IPL need a wider inclusive band to reach
comparable hit rates, given its wider true distribution), not population
mixing -- untested, flagged as the honest next direction.
