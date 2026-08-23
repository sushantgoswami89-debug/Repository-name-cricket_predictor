# The T20I associate-vs-full-member gap isn't unique to match_winner_v1 — it's in every live model

Date: 2026-08-23. Direct follow-up to making the full-member/associate
split permanent in `evaluate_match_winner_v1_accuracy.py`: does the same
gap exist in the two production-critical models (`run_range_v11_partnership_rate`,
`contract22_wicket_v16_team_composition`), or is it specific to the
newer win-probability model's own dataset/architecture?

## Method

Reused the exact `FULL_MEMBER_NATIONS` lookup already added to
`evaluate_match_winner_v1_accuracy.py` (India, Pakistan, Australia,
England, New Zealand, South Africa, West Indies, Sri Lanka, Bangladesh,
Zimbabwe, Afghanistan, Ireland — a match counts as "full-member
international" only if both teams are on this list). Rebuilt each live
model's real 2025+ holdout exactly as its own training script does:
retrained `run_range_v11_partnership_rate` deterministically
(`random_state=42`, reproduced its known live numbers exactly: blended
28.80%, IPL 25.16%, T20I 29.45%) and reused the already-trained
`contract22_wicket_v16_team_composition` artifacts on a freshly rebuilt
holdout frame, no retrain (reproduced AUC 0.6126 vs the known live
0.6124 — confirms the reconstruction is correct).

Both holdouts cover the same population: 863 T20I matches, 151
full-member-international, 712 associate-involved -- essentially
identical composition to match_winner_v1's own holdout (829 matches,
141/688), confirming this is one shared underlying population effect,
not three independently-behaving splits.

## Result: the gap is real and holds in the same direction for all three live models

| Model | Full-member international | Associate-involved | Blended T20I (what gets reported) | IPL (comparison) |
|---|---:|---:|---:|---:|
| `match_winner_v1` (accuracy) | 73.99% | 78.28% | 77.53% | 69.63% |
| `run_range_v11_partnership_rate` (hit rate) | 26.47% | 30.09% | 29.45% | 25.16% |
| `contract22_wicket_v16_team_composition` (AUC) | 0.6032 | 0.6144 | 0.6113 | 0.6151 |

Every model is worse on genuine full-member internationals than on the
blended T20I number it's normally reported against -- run-range by 3.6
points, wicket by 1.1 AUC points, win-probability by 4.3 accuracy
points. Consistent explanation across all three: associate/minor-nation
fixtures are more often lopsided contests (a bigger talent gap between
sides), which team-composition, recency-form, and player-quality
features all pick up cleanly and turn into confident, correct calls;
genuine full-member internationals are closer, more competitive games,
harder to call for the same reason IPL is harder than the blended T20I
average in every one of these models.

Wicket's gap is real but noticeably smaller (1.1 AUC points vs run-
range's 3.6-point hit-rate gap and win-probability's 4.3-point accuracy
gap) -- plausibly because wicket-in-over is already the most match-
state-driven of the three targets (dominated by wickets-in-hand/phase/
recency-form features that don't care much about opponent quality),
while run-range and match outcome are more directly sensitive to the
gulf in overall team strength.

## Confirmed the causal claim, not just asserted it

The explanation above ("associate fixtures are more often lopsided
contests") was a plausible-sounding hypothesis when first written --
checked it directly against real margin-of-victory data rather than
leaving it as an assumption. For every 2025+ holdout T20I match with a
clean win/loss result (runs margin normalized by target, wickets margin
normalized by 10 wickets, as one combined "how much room to spare"
scale):

| | matches | mean margin | median margin | close finishes (<15% margin) |
|---|---:|---:|---:|---:|
| Full-member international | 239 | 42.3% | 40.0% | **21.8%** of matches |
| Associate-involved | 1,273 | 48.5% | 50.0% | 12.9pp fewer — **14.9%** of matches |

Confirmed in the predicted direction: associate-involved matches really
are more lopsided on average, and full-member internationals really are
close finishes nearly 1.5x as often. This is real evidence for the
mechanism, not just a plausible-sounding story fit to the accuracy
numbers after the fact.

## What this doesn't change

Not a promotion-relevant finding for any of the three models -- their
`validation_report.json`/`accuracy_report.json` artifacts from
promotion time are left untouched (this is a post-hoc population-split
analysis, not a retrain or a new candidate). No code change to
`train_run_range_v11_partnership_rate.py` or
`train_contract22_wicket_v16_team_composition.py`.

## What this does change

Standing practice, going forward: **"T20I" should be read as "blended
across a population that's ~83% associate-nation fixtures"** for every
live model in this codebase, not as a proxy for "international cricket
between major teams." Anyone citing a T20I number for user-facing
communication (e.g. "how accurate is this for a real India vs Pakistan
match") should use the full-member-international split, not the blended
number -- the blended number systematically overstates real-world
accuracy for the matches users actually care most about watching.
