# CricketBaba Candidate v3.2 — Two-Run Sharp Range

Date: 22 July 2026

## Purpose

Candidate v3.2 addresses narrow-range selection only. It does not change the
locked point prediction, wicket probability, empirical confidence, or 90%
safety range.

The model predicts a complete probability distribution over next-over totals
from 0 through 30+ runs. It then selects the inclusive three-outcome window
with the greatest probability. For example, `8-10` has a range width of two
runs and contains the integer outcomes 8, 9, and 10.

## Unseen complete-match results

The chronological holdout contains 60,988 overs from 1,704 matches and 3,405
innings. There were zero sequence failures.

| Sharp range | Holdout hit rate | Naive centred hit rate | Relative gain |
| --- | ---: | ---: | ---: |
| One run, e.g. 8-9 | 21.42% | 20.06% | 6.75% |
| Two runs, e.g. 8-10 | 31.38% | 29.98% | 4.65% |

The two-run hit rate was 27.75% in the Powerplay, 34.07% in the middle overs,
and 31.14% at the death. Calibration-year hit rate was 32.31%, showing a small
0.93 percentage-point decline on the independent holdout.

## Decision

The two-run model passed every offline Sharp Range gate and is checksum-
packaged at `models/candidates/v3.2_sharp_range` for live shadow testing. Its
31.38% hit rate must be presented honestly; it is a sharp call, not a 90%
prediction interval. Production remains unchanged.
