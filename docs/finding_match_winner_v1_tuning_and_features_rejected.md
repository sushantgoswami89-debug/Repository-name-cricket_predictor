# match_winner_v1: hyperparameter tuning and a new feature both rejected; a real accuracy-masking gap found instead

Date: 2026-08-23. Follow-up to the win-probability engine audit thread
(venue-wiring fix, chase team-swap bug fix, innings-1 audit, reversal-
lead-time analysis). User asked to pursue both open items from that
thread -- real hyperparameter tuning (the model had reused
`contract22_wicket`'s settings directly, untuned for this target) and a
partnership-acceleration feature (motivated by the finding that the
model is a confirming indicator, not a leading one) -- then recheck
accuracy.

## 1. Hyperparameter tuning: rejected

`train_match_winner_v1.py` was refactored to expose `build_dataset()`
and parameterize `main(lgbm_params=..., extra_features=..., version=...)`
without changing its default behavior -- verified by reproducing the
exact original numbers (AUC 0.8587/0.8544 known/unknown) byte-for-byte
before building anything on top of it.

Ran a 30-config random search over `num_leaves`/`max_depth`/
`learning_rate`/`n_estimators`/`min_child_samples`/`reg_lambda`/
`subsample`/`colsample_bytree`, using an internal `train(<=2022)` /
`val(2023)` split so the real `calibration(2024)`/`holdout(2025+)` sets
were never touched during selection (avoids overfitting the choice of
hyperparameters to the holdout by repeatedly checking it). Best config
found: val AUC 0.8359 vs the current live settings' 0.8340 (+0.0019).

**Confirmed on the real, untouched holdout: flat.** 0.8582 vs the live
model's 0.8587 known-bowler AUC -- the small validation-set gain didn't
generalize; noise, not signal. Not promoted. The wicket line's settings,
despite being borrowed rather than tuned for this target, turn out to
already be a reasonable choice.

## 2. Partnership-acceleration feature: rejected -- and a real methodology catch along the way

Built `partnership_recent_run_rate`/`partnership_trend`
(`app/ml/partnership_dataset.py`'s `build_partnership_trend_dataset`):
run rate of the last 12 legal balls of the *current stand only* (bounded
at the last wicket, unlike the existing whole-team last-12-ball window
which can span across a wicket and mix in a dismissed pair's
contribution) minus the stand's own overall rate -- is this partnership
speeding up or slowing down relative to its own history. Spot-checked
against the real 2022 IND-vs-PAK chase before trusting it: correctly
shows a sharp acceleration spike at over 13 (recent rate 14.5 vs overall
7.4, right as the Kohli/Ashwin stand took off) and a real deceleration
during the later squeeze (overs 17-18) -- the feature computes exactly
what it's supposed to.

**First result looked like a real, consistent win**: AUC improved on
every cut (known 0.8587->0.8596, unknown 0.8544->0.8555, IPL known
0.7889->0.7938). But checking feature importance directly showed
`partnership_trend`/`partnership_recent_run_rate` ranked **69th and 70th
of 70** -- exactly zero importance, never used in a single tree split
across the whole model. An "improvement" from a feature the model never
actually uses is a contradiction, not a real result.

**Root cause, confirmed directly**: `colsample_bytree=0.8` makes
LightGBM randomly subsample which columns are even considered at each
split. Adding two new (unused) columns changes the total column count,
which perturbs *which of the other, real features* get randomly
included at each split -- a pure side effect of the random seed
interacting with column count, not signal from the new feature.
Verified by re-running both the baseline and the candidate with
`colsample_bytree=1.0` (no random subsampling, so this specific artifact
channel is closed): the "improvement" vanished and mildly reversed
(0.8583 without the feature vs 0.8581 with it). **Not promoted.**

**Standing lesson for this project**: a feature that improves an
aggregate metric but shows near-zero or bottom-ranked feature importance
is a red flag, not a win -- verify importance before trusting a metric
delta, especially with `colsample_bytree < 1.0` in play. Kept
`build_partnership_trend_dataset` in `partnership_dataset.py` as
documented, verified, but rejected research code (same precedent as the
bowler-phase-recency infrastructure kept after its own rejection) --
removed the corresponding wiring from `app/ml/match_winner_features.py`
(a `MatchWinnerFeatureComputer.compute()` addition would have run,
unused, on every single live prediction; that's dead-code cost with no
research value once rejected, unlike the offline dataset builder).

## 3. Real finding along the way: T20I blended accuracy masks a genuine gap

Checked whether the win-probability model's T20I holdout population
skews toward associate/minor-nation fixtures rather than genuine
full-member internationals -- same "don't trust a blended number"
practice as the earlier IPL run-range finding this session. It does,
heavily: **688 of 829 T20I holdout matches (83%) involve at least one
associate nation; only 141 are genuine full-member internationals**
(India, Pakistan, Australia, England, New Zealand, South Africa, West
Indies, Sri Lanka, Bangladesh, Zimbabwe, Afghanistan, Ireland).

Splitting accuracy by this line found a real ~4.3-point gap:

| | rows | accuracy | AUC |
|---|---:|---:|---:|
| Full-member international | 5,274 | **73.99%** | 0.835 |
| Associate-involved | 24,879 | 78.28% | 0.877 |
| Blended T20I (what was reported before) | 30,153 | 77.53% | -- |
| IPL (for comparison) | 5,440 | 69.63% | -- |

Associate fixtures are systematically easier to call -- plausibly more
lopsided/mismatched contests (a bigger skill gap that team-composition
and recency features pick up cleanly), while genuine full-member
internationals are closer, more competitive games -- the same shape as
why IPL is harder than the blended T20I average. Full-member-
international accuracy by phase: powerplay 66.65% (down from the
blended 71.16%), middle 76.11%, death 80.41%.

**Made this permanent, not a one-off check**: added
`t20i_full_member_vs_associate` to `evaluate_match_winner_v1_accuracy.py`'s
standing report (`models/candidates/match_winner_v1/accuracy_report.json`)
so this split is reported every time the accuracy check runs, not
something that has to be independently rediscovered.

## What changed, net

Nothing about the live model. `match_winner_v1`'s artifacts
(model/calibrator/feature list) are byte-identical to before this
investigation -- both candidates tested (tuned hyperparameters,
partnership-trend feature) were built in separate `models/candidates/`
directories, evaluated, and removed after rejection. Real changes kept:

- `train_match_winner_v1.py`: refactored (`build_dataset()` +
  parameterized `main()`) -- reusable infrastructure for future
  candidate testing, reproduces the exact original output by default.
- `app/ml/partnership_dataset.py`: new `build_partnership_trend_dataset`
  (verified, documented, rejected for this target -- available if a
  future different target wants to test it).
- `evaluate_match_winner_v1_accuracy.py`: permanent
  full-member/associate T20I split, a real, standing improvement to how
  this model's accuracy gets reported.

208+2 tests pass throughout (210 total).
