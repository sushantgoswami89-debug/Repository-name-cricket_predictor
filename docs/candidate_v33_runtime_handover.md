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

## Known issue — RESOLVED 2026-08-22

The v3.3 artifact was serialized with scikit-learn 1.6.1 while the local
runtime uses 1.9.0 (later 1.8.0). Loading succeeded but emitted an
`InconsistentVersionWarning`. Fixed by exactly the remediation this doc
already prescribed: `backend/train_phase_calibrated_sharp_range_v33.py`
is deterministic (fixed `random_state=42`, fixed input data), so it was
simply re-run in the current environment to produce freshly-serialized
artifacts — a controlled, equivalence-tested repackaging, not a retrain
of anything material.

Verification performed:
- `validation_report.json` after re-running: holdout hit rate 28.1037%,
  relative improvement 3.0102% — identical to the numbers in this doc's
  "Completed" section above (one phase-temperature value differs at the
  11th decimal place, `scipy.optimize.minimize_scalar` floating-point
  noise between library versions, not a behavior change).
- Prediction-equivalence test: loaded the new artifact and the
  backed-up original side by side, ran `predict_frame` on 2,000 sampled
  holdout rows — **100% exact match on predicted low/high bands, 0.0 max
  absolute probability difference.**
- `ARTIFACT_MANIFEST.json` digests updated to match the freshly-serialized
  files (`feature_cols.pkl`'s digest is unchanged, as expected — it's a
  plain list, no version-sensitive objects).
- Full backend test suite: 202 passed, and the v3.3
  `InconsistentVersionWarning` no longer appears (one unrelated
  `InconsistentVersionWarning` remains, from a different artifact —
  Candidate v3's `IsotonicRegression` calibrator, serialized with
  sklearn 1.9.0 vs. the current 1.8.0 runtime — not addressed here, same
  class of issue, separate component).

This candidate is now version-clean for the artifact-load and
integrity-test parts of the promotion checklist below. Steps 2-5 (formal
sign-off on prediction-equivalence documentation, live-state tests, replay
integration, latency) were not re-verified as a formal gate in this pass
beyond what's described above — do that before treating this as a full
promotion sign-off, not just a version fix.

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
