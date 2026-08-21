# IPL CatBoost Phase Mixture-of-Experts v1

## Decision

- Phase-specialist point model: **retain for research**.
- Phase-specialist fixed-width range: **reject**.
- Global CatBoost model: promising research candidate, but not the requested
  mixture-of-experts and not approved for production.
- No production path or locked artifact was changed.

The specialists beat the 3.8930 venue candidate but fail the essential
complexity gate because the global CatBoost model is better.

## Verified working tree

The repository was already heavily dirty: 32 tracked files were modified and
many experiment/data/model files were untracked before this work. Existing
changes were preserved. This experiment added only new candidate-specific
source, test, dataset, model and report paths. No commit, push, reset, stash,
deletion, production promotion or locked-artifact rewrite occurred.

## Data and split

| Split | Dates | Rows | Matches |
|---|---:|---:|---:|
| Training | through 2023 | 39,442 | 1,024 |
| Calibration | 2024 | 2,738 | 71 |
| Held-out | 2025+ | 5,576 | 148 |

Match-level separation is exact. Super Overs are excluded. Seed is 42.
CatBoost 1.2.10, LightGBM 4.6.0, pandas 3.0.3, NumPy 2.5.1 and Python
3.12.13 were recorded.

## Overall held-out comparison

| Model | MAE | Bias | Fixed width-3 coverage |
|---|---:|---:|---:|
| Locked shared T20 | 3.9721 | -1.6309 | 33.66% |
| Venue-only IPL | 3.8907 | -0.0535 | 32.50% |
| Venue + experimental features | 3.8931 | -0.1033 | 32.19% |
| CatBoost global | **3.8485** | -0.0372 | 33.25% |
| CatBoost phase specialists | 3.8699 | -0.0267 | 32.64% |

The global model beats the specialists by 0.0214 MAE. The specialists improve
on the venue-plus-experimental candidate by 0.0232 MAE, but do not justify
their added routing and artifact complexity.

The fixed range has numeric width 3 and therefore contains four inclusive
integer outcomes. Phase-specialist coverage improves over the venue candidate
but remains below the locked model's 33.66%; range promotion is rejected.

## Phase-specialist error matrix

All rows below are held out. Minimum supported sample size is 100.

| Dimension | Segment | Rows | MAE | Bias | Coverage |
|---|---|---:|---:|---:|---:|
| Innings | First innings | 2,901 | 3.8828 | -0.0865 | 33.20% |
| Innings | Chase | 2,675 | 3.8558 | 0.0382 | 32.04% |
| Phase | Powerplay | 1,751 | 4.0323 | -0.2351 | 30.27% |
| Phase | Middle | 2,582 | 3.6474 | -0.1111 | 35.59% |
| Phase | Death | 1,243 | 4.1034 | 0.4422 | 29.85% |
| Venue | High | 660 | 4.0257 | -0.2668 | 26.67% |
| Venue | Medium | 4,916 | 3.8490 | 0.0055 | 33.44% |
| State | Accelerating | 944 | 3.8765 | 0.2093 | 32.94% |
| State | Stable | 3,972 | 3.8904 | -0.1348 | 32.20% |
| State | Wicket pressure | 660 | 3.7370 | 0.2864 | 34.85% |
| Batters | New batter | 1,782 | 3.7430 | -0.1443 | 34.40% |
| Batters | Established pair | 3,794 | 3.9295 | 0.0285 | 31.81% |
| Chase pressure | High | 892 | 4.0184 | 0.1559 | 28.70% |
| Chase pressure | Medium | 425 | 3.8129 | 0.0263 | 34.82% |
| Chase pressure | Low | 1,358 | 3.7625 | -0.0354 | 33.36% |
| Chase pressure | Not chasing | 2,901 | 3.8828 | -0.0865 | 33.20% |

Supported wickets-remaining buckets:

| Wickets remaining | Rows | MAE | Bias | Coverage |
|---:|---:|---:|---:|---:|
| 10 | 1,243 | 3.9792 | -0.3633 | 31.13% |
| 9 | 1,062 | 3.9712 | -0.4341 | 30.70% |
| 8 | 935 | 3.6174 | 0.0856 | 34.97% |
| 7 | 804 | 3.5377 | 0.2682 | 35.57% |
| 6 | 592 | 3.8025 | 0.2377 | 35.47% |
| 5 | 426 | 4.2965 | 0.1513 | 30.99% |
| 4 | 265 | 4.0623 | 0.7150 | 27.17% |
| 3 | 115 | 4.2949 | -0.3269 | 31.30% |

No supported phase or venue segment regressed by more than 2% against the
venue-plus-experimental candidate.

## Wicket and quantile diagnostics

The phase specialists' wicket Brier score is 0.2539 versus 0.2151 for the
global CatBoost head, another argument against specialist complexity.
Independent 20th/80th quantile heads produced 55.38% diagnostic coverage with
mean width 7.19. These quantile widths are diagnostic only and were not used
to widen the production range.

## Five calibration replays

| Match | Date | Rows | MAE | Bias | Coverage |
|---|---|---:|---:|---:|---:|
| 1426307.json | 2024-05-19 | 40 | 3.7815 | -0.7404 | 30.00% |
| 1426309.json | 2024-05-21 | 34 | 3.6519 | -1.2114 | 29.41% |
| 1426310.json | 2024-05-22 | 39 | 4.3571 | 0.0877 | 23.08% |
| 1426311.json | 2024-05-24 | 40 | 3.7482 | 1.3056 | 27.50% |
| 1426312.json | 2024-05-26 | 30 | 4.2184 | 0.3393 | 30.00% |

Offsets were calibrated only on 2024 data: death -2, middle -2 and powerplay
-3.

## Five untouched held-out replays

| Match | Date | Rows | MAE | Bias | Coverage |
|---|---|---:|---:|---:|---:|
| 1529313.json | 2026-05-24 | 39 | 3.7168 | 0.6273 | 33.33% |
| 1535462.json | 2026-05-26 | 40 | 4.9336 | -1.0949 | 12.50% |
| 1535463.json | 2026-05-27 | 40 | 5.1406 | -1.6536 | 17.50% |
| 1535464.json | 2026-05-29 | 39 | 4.3537 | -1.0009 | 30.77% |
| 1535465.json | 2026-05-31 | 38 | 3.5188 | 0.6161 | 28.95% |

The poor three-match sequence near the end of 2026 is a material stability
risk despite acceptable aggregate metrics.

## Leakage and live parity

- Every state feature is calculated before the predicted over.
- Active batter identities and partnership age are available before
  publication.
- Batter historical profiles are frozen before each match.
- Venue regime and home/away context use prior-match information.
- Recent runs, dots, boundaries and wickets use completed legal balls only.
- Phase, wickets in hand and chase pressure use current pre-over state.
- Cricsheet's first-delivery bowler is not used. `known_bowler` is always the
  safe `__UNKNOWN__` category because the next bowler is not reliably known
  when the prediction is published.
- Training, calibration and held-out evaluation use the same constructed
  feature file. End-to-end live runtime parity remains unproven because this
  candidate was intentionally not integrated into the production runtime.

## Unsupported segments

- Low-scoring venue regime: zero held-out rows.
- One wicket remaining: 46 rows.
- Two wickets remaining: 88 rows.

These are reported as unsupported and do not influence non-regression gates.

## Tests and warnings

- Focused candidate/venue/router tests: 10 passed.
- Complete backend suite from repository root: 146 passed.
- 18 warnings investigated: 14 NumPy/joblib deprecations and four
  scikit-learn locked-pickle version mismatches. Locked artifacts were not
  rewritten or retrained.

## Remaining risks

- The aggregate improvement is small relative to match-to-match variance.
- Phase specialists underperform the global model on MAE and wicket Brier.
- High-venue coverage is weak; low venues lack held-out support.
- Latest held-out replays include concentrated MAE and bias failures.
- The global model needs a separate candidate cycle and live inference
  contract before any promotion discussion.
- End-to-end training/live feature parity has not yet been demonstrated.
