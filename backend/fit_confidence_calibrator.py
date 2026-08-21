"""Fit an isotonic-regression calibrator mapping the engine's heuristic
"confidence" score to the actual empirical probability that the predicted
range covers the real outcome (range_hit).

Diagnosed earlier this session: raw confidence buckets (50-99%) showed a
flat ~25-37% real hit rate -- the displayed number doesn't track accuracy
at all. Same fix pattern as the wicket calibrator: don't invent precision
that isn't there, but do correct the number to mean what it claims to mean.

Uses the same tuning sample (seed 123, 20 IPL matches) as
tune_momentum_blend.py / the original wicket calibrator, so it's
consistent with -- and distinct from -- every reporting sample used this
session. Saves models/confidence_calibrator.pkl.
"""

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

N_MATCHES = 20
random.seed(123)
match_dir = Path("../data/raw/cricsheet/ipl")
sample = random.sample(sorted(match_dir.glob("*.json")), N_MATCHES)

confidences = []
range_hits = []

for path in sample:
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
            confidences.append(prediction.confidence)
            range_hits.append(1 if hit else 0)
            engine.update_actuals(
                actual_runs=frame.actual_runs,
                actual_wickets=frame.actual_wickets,
                bowler_name=frame.state.bowler,
            )

print(f"n = {len(confidences)}")
print(f"raw confidence range: {min(confidences):.3f} - {max(confidences):.3f}")
print(f"overall range-hit rate: {sum(range_hits)/len(range_hits)*100:.1f}%")

calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
calibrator.fit(confidences, range_hits)

calibrated = calibrator.predict(confidences)
print("\nBucketed check (raw confidence -> calibrated confidence):")
for lo in (0.5, 0.6, 0.7, 0.8, 0.9):
    hi = lo + 0.1
    idx = [i for i, c in enumerate(confidences) if lo <= c < hi]
    if not idx:
        continue
    raw_mean = sum(confidences[i] for i in idx) / len(idx)
    cal_mean = sum(calibrated[i] for i in idx) / len(idx)
    real_hit_rate = sum(range_hits[i] for i in idx) / len(idx)
    print(
        f"  [{lo:.1f}-{hi:.1f}) n={len(idx):4d}  raw_mean={raw_mean:.3f}  "
        f"calibrated_mean={cal_mean:.3f}  actual_hit_rate={real_hit_rate:.3f}"
    )

out_path = Path("../models/confidence_calibrator.pkl")
joblib.dump(calibrator, out_path)
print(f"\nSaved to {out_path}")
