# CricketBaba Candidate v3.1 — Historical Analogue Intelligence

Date: 22 July 2026

## Decision

Candidate v3.1 passed all offline gates and is packaged for live shadow
validation. Production remains unchanged. The locked Candidate v3 continues to
provide the point prediction; historical analogues improve the conditional
range, evidence, and calibrated confidence.

## Time-safe method

- Calibration-year states from 2024 search references through 2023 only.
- Independent holdout states from 2025 onward search references through 2024.
- Comparisons require the same match phase and the same innings/chase status.
- Each prediction uses 384 nearest historical states across 14 pre-over fields.
- No event from the predicted over is included in its search state.

Calibration rejected analogue blending for both runs and wicket point
predictions: the selected maximum weight was zero. This protects the stronger
locked model rather than forcing the new idea into the point forecast.
Historical outcomes are used for conditional quantiles, confidence, and
explanation.

## Unseen-match results

The holdout contains 60,988 overs from 1,704 complete matches and 3,405 innings.
There were zero sequence or score-reconciliation failures.

| Metric | Locked v3 | Candidate v3.1 |
| --- | ---: | ---: |
| Runs MAE | 3.2255 | 3.2255 |
| Wicket Brier | 0.20416 | 0.20416 |
| Range coverage | 91.02% | 91.69% |
| Average range width | 13.57 | 13.35 |

Coverage was 91.26% in the Powerplay, 91.39% in the middle overs, and 93.06%
at the death. Every phase exceeded its 88% gate.

## Empirical confidence

Confidence is calibrated as the historical probability that the point
prediction finishes within four runs of the actual over. Its unseen expected
calibration error was 1.26%, well inside the 5% gate.

- Lowest-confidence third: runs MAE 3.85.
- Highest-confidence third: runs MAE 2.59.
- Predictions in the 80–90% band achieved an observed 84.44% success rate.
- Predictions in the 90–100% band achieved an observed 90.23% success rate.

## Runtime output

The runtime enhancement returns the conditional range, calibrated confidence,
384-neighbour sample size, effective sample size, similarity, historical runs
and wicket expectations, and the most common runs bucket with its probability.

The candidate is checksum-packaged at
`models/candidates/v3.1_historical_analogue`. It is ready for live shadow use,
not production promotion.
