# The v7.x wicket line's T20I-downweighting lever does not transfer to the live model; it makes IPL accuracy worse there

Date: 2026-08-23. Roadmap priority #3: reconcile
`ipl_wicket_v7_7_t20i_augmented` -- the only lever that measurably helped
in the CatBoost-based, IPL-only-holdout `ipl_wicket_v7.x` research line --
into `contract22_wicket_v2_batter_state`, which is the model actually
live today. The two lineages were never directly comparable before this:
v7.x uses CatBoost, an IPL-only holdout, and only the ~13 match-state
features common to both competitions; v2 uses LightGBM, a full feature
set (batter/bowler/matchup/venue/partnership-age), and trains on the full
blended IPL+T20I population at equal weight already.

## What v7.x actually found

`ipl_wicket_v7_7_t20i_augmented` started from a small, IPL-only model and
*added* extra T20I rows (down-weighted to 0.2) as pure augmentation on
top of a data-scarce IPL-only training set. That's a genuinely different
setup from v2: v2 was never IPL-only to begin with, and already trains on
every available T20I row at full weight (1.0).

## Re-tested: down-weighting T20I on v2's actual architecture

Built `train_contract22_wicket_v8_t20i_downweighted.py`: same features,
same split as `contract22_wicket_v2_batter_state`, two training runs --
baseline (T20I weight=1.0, an exact reproduction of live v2) and
candidate (T20I weight=0.2, matching the proven v7.x value) -- with
holdout AUC/Brier reported split IPL-only / T20I-only / blended for the
first time on this model line.

| | Blended (known) | IPL (known) | IPL (unknown) | IPL Brier |
|---|---:|---:|---:|---:|
| Baseline, T20I weight=1.0 (= live v2) | 0.6083 | 0.6079 | 0.6043 | 0.1944 |
| Candidate, T20I weight=0.2 | 0.6050 | 0.6050 | 0.6018 | 0.1960 |

Down-weighting T20I made IPL accuracy *worse* on every metric, not
better -- the opposite of what the v7.x lineage found. Rejected on all
three promotion gates.

## Why the lever doesn't transfer, and why that makes sense

v7.x's baseline was thin (13 shared match-state features, IPL-only
training, ~40k rows) -- adding *any* extra rows, even down-weighted
T20I ones limited to those same 13 features, gave that specific model
more instances of general match-state patterns it hadn't seen enough of.
v2 starts from a much richer position: full batter/bowler/venue/matchup
features, and the *entire* T20I population already included in training
at full weight, not as light augmentation. Down-weighting rows that were
already fully weighted just throws away real training signal without
adding anything new -- there's no scarcity for the lever to fix here.
Same shape of finding as `finding_blended_holdout_masks_ipl_accuracy.md`'s
run-range conclusion ("the wicket-line analogy doesn't transfer"), just
in the reverse direction between the two wicket lineages themselves.

## Free finding: wicket's IPL/T20I gap is much smaller than run-range's

Also worth recording since this is the first time it's been checked:
unlike run-range (IPL 25.00% vs T20I 29.43%, a real ~4.4pp gap), wicket's
known-bowler AUC is close between competitions -- IPL 0.6079 vs T20I
0.6077 (baseline). The blended figure isn't masking a hidden gap here the
way it was for run-range. No action needed; recorded so a future session
doesn't have to re-derive it.

## Recommendation

**Don't apply T20I down-weighting to `contract22_wicket_v2_batter_state`.**
The v7.x research line's proven lever is specific to that line's
data-scarce IPL-only setup and doesn't generalize. This closes out
roadmap priority #3 as ruled out, same as priorities #1 (Impact Player)
and #2 (toss) -- three real hypotheses tested this session, three honest
negatives, `contract22_wicket_v2_batter_state` and
`run_range_v7_competition_prior` remain the best validated models for
each target.

Candidate artifacts: `models/candidates/contract22_wicket_v8_t20i_downweighted/`.
Not promoted; `production_changed: false`.
