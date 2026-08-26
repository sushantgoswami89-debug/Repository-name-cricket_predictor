# Finding: NN bagging and per-phase blend weight also don't beat the v21 ensemble (2026-08-26)

## Why this was tried
Second follow-up push after "bigger single NN" failed
(`finding_run_range_bigger_nn_no_further_gain.md`). Two different levers,
both legitimate and distinct from "make the network bigger": (1) bag 3
NNs of the *same* smaller architecture that already won, different random
seeds, average their softmax outputs (variance reduction via multiple
training runs); (2) fit a separate blend weight per phase
(powerplay/middle/death) instead of one global scalar, since those three
already get separate temperature calibration in this pipeline.

## Result
| | Blended | IPL | T20I |
|---|---:|---:|---:|
| v11 (live) | 30.04% | 33.25% | 29.45% |
| **v21 (single NN, global blend — current shadow)** | **30.22%** | 33.43% | **29.64%** |
| Bagged NN (3 seeds) alone | 29.90% (worse than v21's single NN alone) | 32.87% | 29.36% |
| Bagged NN + global blend | 30.15% | 33.54% (best IPL) | 29.53% |
| Bagged NN + per-phase blend | 30.18% | 33.50% | 29.58% |

Fitted per-phase blend weights (GBM's share): powerplay 0.21, middle
0.23, death 0.27 — a real, mild trend (more GBM weight as the innings
progresses), but using it didn't beat the simpler global blend on the
metric that matters (blended hit rate).

**Neither new variant beats the v21 ensemble on blended hit rate.**
IPL improves further in both (33.50-33.54% vs v21's 33.43%), but T20I and
blended both regress. Bagging alone actually made the NN *worse* than a
single well-trained run, not better — with only 25 epochs and early
stopping per seed, 3 independent runs average out some of each one's
sharper (possibly useful, possibly noise) individual fit rather than
reinforcing a shared signal.

**Decision: keep `run_range_v21_nn_gbm_ensemble` (single NN, global
scalar blend) as the live shadow.** No production change.

## What this closes out
Two independent "push the NN further" attempts (bigger single network,
bagging + finer-grained blending) both failed to beat the simplest
version of the ensemble. Combined with today's four rejected new
features and this, the honest read is that `v21`'s specific
configuration — not under- or over-fit, single scalar blend — sits at a
real local optimum for this architecture on this data. Further gains
would need either more/different training data (not more model
complexity) or a genuinely different architecture family, not more
tuning of this one.
