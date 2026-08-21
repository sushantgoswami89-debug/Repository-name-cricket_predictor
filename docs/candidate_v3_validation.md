# CricketBaba Candidate v3 Validation

Date: 22 July 2026

## Scope

Candidate v3 is a shadow-only next-over model trained from verified Cricsheet
deliveries. No production model artifact was replaced.

## Dataset

- 6,767 source matches inspected.
- 6,761 matches included after the six frozen anomaly exclusions.
- 1,543,463 deliveries verified.
- 246,846 unique pre-over rows generated with explicit match and innings keys.
- Zero duplicate keys and zero missing values.
- Every feature is calculated before the target over begins.

The chronological split is 148,607 rows through 2023 for training, 37,251
rows from 2024 for calibration, and 60,988 rows from 2025 onward for the
independent holdout.

## Phase-impact backtest

The controlled ablation uses identical model settings and time splits. The
base includes over, score, wickets, and phase. Candidate v3 adds only
prediction-time situation and recent-delivery signals, allowing the model to
learn how boundary, dot-ball, rotation, and wicket effects differ by phase.

| Metric | Base | Candidate v3 | Result |
| --- | ---: | ---: | --- |
| Runs MAE | 3.2761 | 3.2255 | Improved 1.54% |
| Wicket Brier | 0.20530 | 0.20416 | Improved 0.56% |
| Wicket AUC | 0.5889 | 0.6006 | Improved |
| 90% interval coverage | — | 91.02% | Passed |

Both runs MAE and wicket Brier improved independently in the Powerplay,
middle overs, and death overs. The largest gains were at the death: 2.57% for
runs MAE and 1.18% for wicket Brier.

## Complete-match test

All 1,704 unseen matches from 2025 onward were replayed, covering 3,405 innings
and 60,988 over predictions. There were zero sequence or score-reconciliation
failures, all outputs were finite, and every wicket probability was bounded.

## Promotion decision

All recorded Candidate v3 validation gates passed, so the candidate is eligible
for a separate promotion review. It remains isolated in
`models/candidates/v3_phase_impact_time_split`; production models were not
promoted or overwritten during this work.

## Ranked-country scope decision

A dated ICC ranking snapshot from 20 June 2026 was used to test two additional
international-only scopes. Both teams had to belong to the frozen list.

| Scope | Training rows | Holdout rows | Runs MAE | Wicket Brier |
| --- | ---: | ---: | ---: | ---: |
| Unrestricted verified v3 | 246,846 | 60,988 | 3.2255 | 0.20416 |
| Ranked top 10 | 56,955 | 7,248 | 3.5579 | 0.20719 |
| Ranked top 12 | 66,492 | 9,143 | 3.5030 | 0.20486 |

The top-10 model also missed the runs-bias and interval-coverage gates. The
top-12 model completed 252 unseen matches without sequence failures, but it did
not beat the unrestricted candidate on either primary metric and narrowly
missed 90% interval coverage. Restricting the database therefore reduced useful
training diversity more than it improved match quality.

The unrestricted verified Candidate v3 was selected and copied to the locked,
checksum-protected release at `models/locked/cricketbaba_candidate_v3`. All
locked files are read-only. Production remains unchanged.
