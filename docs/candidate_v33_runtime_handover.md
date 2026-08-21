# CricketBaba v3.3 Runtime Handover

Date: 2026-07-24

## Completed

- Verified the locked male-only v3.3 run-range manifest and reproduced its
  chronological holdout result exactly.
- Holdout: 1,011 matches and 36,408 overs.
- Baseline hit rate: 27.2825%.
- v3.3 hit rate: 28.1037%.
- Relative improvement: 3.0102%.
- Added `backend/runtime_v33.py` with:
  - artifact integrity verification before deserialization;
  - exact 17-feature contract validation;
  - phase-specific temperature calibration;
  - configurable inclusive sharp-range widths;
  - calibrated range probability output.
- Added the candidate artifact digest manifest:
  `models/candidates/v3.3_phase_calibrated_sharp_range/ARTIFACT_MANIFEST.json`.
- Added focused tests in `backend/tests/test_runtime_v33.py`.
- Exact runtime smoke test passed with 10 predictions.
- Focused tests: 2 passed.
- Locked model artifacts were not modified.

## Known issue

The v3.3 artifact was serialized with scikit-learn 1.6.1 while the local
runtime uses 1.9.0. Loading succeeds, but emits an `InconsistentVersionWarning`.
This must be resolved before production promotion.

## Next production-readiness step

Make model serialization reproducible and version-safe without changing the
accepted model's predictions. Freeze the training/runtime dependency versions
or produce a controlled, equivalence-tested repackaging. Then run:

1. artifact-load and integrity tests;
2. prediction-equivalence testing against the accepted v3.3 artifact;
3. chronological holdout verification;
4. representative live-state and missing/stale-feature tests;
5. replay integration and latency checks.

Do not modify or retrain the accepted wicket model. Preserve all existing
repository changes.
