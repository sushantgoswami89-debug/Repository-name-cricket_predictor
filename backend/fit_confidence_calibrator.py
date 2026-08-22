"""Fit an isotonic-regression calibrator mapping the engine's heuristic
"confidence" score to the actual empirical probability that the predicted
range covers the real outcome (range_hit).

Diagnosed earlier this session: raw confidence buckets (50-99%) showed a
flat ~25-37% real hit rate -- the displayed number doesn't track accuracy
at all. Same fix pattern as the wicket calibrator: don't invent precision
that isn't there, but do correct the number to mean what it claims to mean.

Uses the same tuning sample (seed 123, 20 IPL matches) as the original
wicket calibrator, so it's consistent with -- and distinct from -- every
reporting sample used this session. Saves models/confidence_calibrator.pkl.

Fixed 2026-08-22: was training on `prediction.confidence`, which is the
ALREADY-CALIBRATED value once models/confidence_calibrator.pkl exists on
disk (PredictionEngine.predict() applies the existing calibrator before
returning it). Since that artifact now exists and is git-tracked, this
script would silently train against its own prior output on any rerun --
a feedback loop, not real recalibration. Now uses
`prediction.metadata["raw_confidence"]`, the genuine pre-calibration
heuristic score (see app/ml/prediction_engine.py).
"""

import json
import random
from pathlib import Path

import joblib
from sklearn.isotonic import IsotonicRegression

from app.data_ingestion.json_loader import JsonLoader
from app.data_ingestion.match_parser import MatchParser
from app.live.toi_reader import ToiDelivery
from app.ml.historical_feature_store import match_exclude_key
from app.ml.prediction_engine import PredictionEngine
from app.models.match_context import MatchContext
from app.services.match_replay import MatchReplay

_NON_DISMISSAL_KINDS = {"retired hurt", "obstructing the field"}


def _to_toi_delivery(over_number: int, ball_index: int, delivery) -> ToiDelivery:
    """Convert a Cricsheet-parsed Delivery into the ToiDelivery shape
    WicketContract22FeatureComputer expects (deliveries=[]) -- mirrors the
    conversion pipeline.py/run_live_pipeline_replay.py already do."""
    wicket_kind = next(
        (
            w.get("kind")
            for w in delivery.wickets
            if w.get("kind") not in _NON_DISMISSAL_KINDS
        ),
        None,
    )
    return ToiDelivery(
        match_id="", innings=1, over=over_number, ball=ball_index,
        bowler=delivery.bowler, striker=delivery.batter,
        total_runs=delivery.runs.get("total", 0),
        batter_runs=delivery.runs.get("batter", 0),
        extras=delivery.extras, wicket_kind=wicket_kind,
        feed_total=0, feed_wickets=0, timestamp_ms=0, commentary="",
    )

N_MATCHES = 200  # raised from 20 (2026-08-22) -- validated via
# experiment_confidence_calibrator_larger_sample.py that 20 matches (764
# overs) was thin enough to leave real uncertainty about whether the
# near-constant calibration was a sample-size artifact; 200 matches
# (~7,663 overs) confirmed it is not (see docs/CTO_HANDOVER_2026-07-24.md).
#
# Fixed 2026-08-22 (later same day): the sample used to be drawn from
# ALL IPL matches regardless of date. Checked and found 171/200 (85.5%)
# landed in the run_range_v3/wicket models' own <=2023 TRAINING period --
# those models have already seen these exact matches, so raw_confidence
# on them is inflated by in-sample performance, biasing the calibrator
# toward expecting better accuracy than the models actually generalize
# to. Now draws only from the genuine holdout period (>=2025-01-01,
# IPL+T20I) those models were never fit or calibrated against at all --
# the same standard every other holdout number this session uses.
random.seed(123)
HOLDOUT_CUTOFF = "2025-01-01"
match_dir_candidates = [
    path
    for scope in ("ipl", "t20i")
    for path in (Path("../data/raw/cricsheet") / scope).glob("*.json")
]


def _is_holdout_match(path: Path) -> bool:
    try:
        dates = json.loads(path.read_text(encoding="utf-8"))["info"]["dates"]
    except (KeyError, IndexError, ValueError):
        return False
    return bool(dates) and str(dates[0]) >= HOLDOUT_CUTOFF


holdout_candidates = sorted(p for p in match_dir_candidates if _is_holdout_match(p))
sample = random.sample(holdout_candidates, N_MATCHES)

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
        # WicketContract22FeatureComputer needs the innings' deliveries so
        # far for its partnership/new-batter and bowler-spell features
        # (found missing from this exact script the same day pipeline.py's
        # equivalent gap was found and fixed -- see docs/candidate_ipl_wicket_v7_2_spell_features.md).
        # Without this, those features silently default to "brand new
        # innings, nobody has bowled/batted yet" on every single call,
        # which would bias the raw_confidence values this script fits
        # against relative to how the model actually behaves live.
        accumulated_deliveries: list[ToiDelivery] = []
        non_empty_overs = [over for over in innings.overs if over.deliveries]
        for over, frame in zip(non_empty_overs, MatchReplay().frames(innings)):
            context = MatchContext(
                team1=match.info.teams[0],
                team2=match.info.teams[1],
                venue=match.info.venue or "Unknown",
                format=match.info.match_type,
                live=frame.state,
                metadata={
                    "exclude_match_key": exclude_key,
                    "deliveries": list(accumulated_deliveries),
                },
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
            accumulated_deliveries.extend(
                _to_toi_delivery(over.over_number + 1, ball_index, delivery)
                for ball_index, delivery in enumerate(over.deliveries, start=1)
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
