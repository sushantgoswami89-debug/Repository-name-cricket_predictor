# Finding: a bigger NN doesn't improve the run-range ensemble further (2026-08-26)

## Why this was tried
Direct follow-up to the GBM+NN ensemble win
(`finding_run_range_nn_gbm_ensemble_real_win.md`, +0.18-0.20pp over live
`v11` on all three cuts). User felt the gain was "still low" and asked to
expand the NN specifically ("bigger NN architecture"). Tested whether NN
capacity was the limiting factor: same GBM (reused from `v21a`, not
retrained), same features, same downstream calibration/banding pipeline
— only the NN body changed.

## What changed
| | v21 (smaller NN) | v22 (bigger NN) |
|---|---|---|
| Body | 128→64→NUM_CLASSES | 256→128→64→NUM_CLASSES, + BatchNorm |
| Embedding dim | 4 | 8 |
| LR schedule | fixed 1e-3 | 1.5e-3 + ReduceLROnPlateau |
| Label smoothing | none | 0.02 |
| Training budget | 25 epochs / patience 4 | 60 epochs / patience 6 |

## Result
| | v11 (live) | v21 ensemble (smaller NN) | v22 ensemble (bigger NN) |
|---|---:|---:|---:|
| Blended | 30.04% | **30.22%** | 30.18% (slightly worse than v21) |
| IPL | 33.25% | 33.43% | **33.59%** (best of the three) |
| T20I | 29.45% | **29.64%** | 29.56% (slightly worse than v21) |

**The bigger NN does not beat the smaller-NN ensemble overall** — mixed
result, IPL improves further but blended and T20I both regress slightly.
Both ensembles still clearly beat the live `v11` baseline on every cut;
this is a comparison between two already-winning candidates, not a
regression to baseline.

**Decision: keep the v21 (smaller NN) ensemble as the live shadow.**
Simpler, faster to run (no BatchNorm/scheduler), and at least as good in
aggregate. No production change from this test — `models/candidates/run_range_v21_nn_gbm_ensemble/`
remains the artifact set wired into `PredictionEngine`.

## Why bigger didn't help
Consistent with the standing pattern from today's four rejected feature
tests: run-range's cumulative feature importance is diffuse (33 of 63
features for 80%), and the earlier `v11`→ensemble gain came from the NN
finding a genuinely *different* error pattern than the GBM (evidenced by
neither model alone beating the blend), not from the NN being
under-powered. A tabular problem with ~112k training rows and 63 features
has a real information ceiling that more parameters doesn't push past —
the bottleneck here is signal in the data/features, not model capacity.
This mirrors why none of today's four new features moved the needle
either: the lever that worked (ensembling two different model families)
already captured what there was to gain from architectural diversity;
throwing more capacity at either half of that ensemble doesn't unlock
more.
