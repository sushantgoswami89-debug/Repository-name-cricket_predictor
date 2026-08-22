"""Diagnostic experiment, not a production fix: does the confidence
calibrator's near-constant output (see docs/CTO_HANDOVER_2026-07-24.md's
2026-08-22 addendum -- fitted domain [0.56, 0.91] vs. runtime [0.50, 0.92],
most of the range collapsing to a single 0.3368 value) improve with a much
larger calibration sample, or is it a genuine property of the underlying
heuristic regardless of sample size?

Found and fixed a second, more serious bug while building this: the
original fit_confidence_calibrator.py (and this script's first draft) used
`prediction.confidence` as the calibration INPUT feature -- but that field
is the ALREADY-CALIBRATED value once models/confidence_calibrator.pkl
exists on disk (PredictionEngine.predict() applies the existing calibrator
before returning). Since that artifact now exists (and is git-tracked),
any future "refit" using the original script's approach silently trains
against its own prior output -- a feedback loop, not real recalibration.
The genuinely-raw, pre-calibration heuristic score is exposed separately
at `prediction.metadata["raw_confidence"]` (see
app/ml/prediction_engine.py:298,334) -- this script uses that instead.

Same methodology as fit_confidence_calibrator.py otherwise (replay real
IPL matches, fit isotonic regression of raw confidence -> range_hit), but:
- uses prediction.metadata["raw_confidence"], not prediction.confidence
- N_MATCHES raised from 20 to 200 (~10x, roughly 16% of all IPL matches
  instead of ~1.6%)
- saved to models/candidates/confidence_calibrator_larger_sample/ --
  NEVER touches models/confidence_calibrator.pkl (the real production
  artifact)

Does not modify prediction_engine.py or any production code path.
"""

import json
import random
from pathlib import Path

import joblib
from sklearn.isotonic import IsotonicRegression

from app.data_ingestion.json_loader import JsonLoader
from app.data_ingestion.match_parser import MatchParser
from app.ml.historical_feature_store import match_exclude_key
from app.ml.prediction_engine import PredictionEngine
from app.models.match_context import MatchContext
from app.services.match_replay import MatchReplay

N_MATCHES = 200
random.seed(123)
match_dir = Path("../data/raw/cricsheet/ipl")
sample = random.sample(sorted(match_dir.glob("*.json")), N_MATCHES)

confidences = []
range_hits = []

for i, path in enumerate(sample):
    if i % 25 == 0:
        print(f"  ... {i}/{N_MATCHES} matches processed", flush=True)
    try:
        match = MatchParser().parse(JsonLoader().load(str(path)))
    except Exception:
        continue
    exclude_key = match_exclude_key(match)
    for innings in match.innings:
        engine = PredictionEngine()
        for frame in MatchReplay().frames(innings):
            context = MatchContext(
                team1=match.info.teams[0],
                team2=match.info.teams[1],
                venue=match.info.venue or "Unknown",
                format=match.info.match_type,
                live=frame.state,
                metadata={"exclude_match_key": exclude_key},
            )
            try:
                prediction = engine.predict(context)
            except Exception:
                continue
            low, high = (int(x) for x in prediction.expected_range.split("-"))
            hit = low <= frame.actual_runs <= high
            confidences.append(prediction.metadata["raw_confidence"])
            range_hits.append(1 if hit else 0)
            engine.update_actuals(
                actual_runs=frame.actual_runs,
                actual_wickets=frame.actual_wickets,
                bowler_name=frame.state.bowler,
            )

print(f"\nn = {len(confidences)}")
print(f"raw confidence range: {min(confidences):.3f} - {max(confidences):.3f}")
print(f"overall range-hit rate: {sum(range_hits)/len(range_hits)*100:.1f}%")

calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
calibrator.fit(confidences, range_hits)

calibrated = calibrator.predict(confidences)
print("\nBucketed check (raw confidence -> calibrated confidence):")
buckets = []
for lo in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90):
    hi = lo + 0.05
    idx = [i for i, c in enumerate(confidences) if lo <= c < hi]
    if not idx:
        continue
    raw_mean = sum(confidences[i] for i in idx) / len(idx)
    cal_mean = sum(calibrated[i] for i in idx) / len(idx)
    real_hit_rate = sum(range_hits[i] for i in idx) / len(idx)
    buckets.append({
        "range": [lo, hi], "n": len(idx), "raw_mean": raw_mean,
        "calibrated_mean": cal_mean, "actual_hit_rate": real_hit_rate,
    })
    print(
        f"  [{lo:.2f}-{hi:.2f}) n={len(idx):4d}  raw_mean={raw_mean:.3f}  "
        f"calibrated_mean={cal_mean:.3f}  actual_hit_rate={real_hit_rate:.3f}"
    )

print(f"\nFitted domain: X_min_={calibrator.X_min_:.4f}, X_max_={calibrator.X_max_:.4f}")
n_distinct_outputs = len(set(round(v, 4) for v in calibrator.y_thresholds_))
print(f"Distinct output levels: {n_distinct_outputs} (original 20-match version had 3)")
print(f"X_thresholds_: {calibrator.X_thresholds_}")
print(f"y_thresholds_: {calibrator.y_thresholds_}")

output_dir = Path("../models/candidates/confidence_calibrator_larger_sample")
output_dir.mkdir(parents=True, exist_ok=True)
joblib.dump(calibrator, output_dir / "confidence_calibrator.pkl")
(output_dir / "report.json").write_text(
    json.dumps({
        "n_matches": N_MATCHES,
        "n_overs": len(confidences),
        "raw_confidence_range": [min(confidences), max(confidences)],
        "overall_range_hit_rate": sum(range_hits) / len(range_hits),
        "buckets": buckets,
        "fitted_domain": [float(calibrator.X_min_), float(calibrator.X_max_)],
        "distinct_output_levels": n_distinct_outputs,
    }, indent=2),
    encoding="utf-8",
)
print(f"\nSaved to {output_dir} (candidate only, production untouched)")
