"""Grid-search PredictionEngine's momentum-blend weight on a tuning sample
(seed 123 -- distinct from the seed-99/seed-42 samples already reported to
the user) so the choice isn't fit to the numbers we've been quoting."""

import random
from pathlib import Path

from app.data_ingestion.json_loader import JsonLoader
from app.data_ingestion.match_parser import MatchParser
from app.ml.historical_feature_store import match_exclude_key
from app.ml.model_repository import ModelRepository
from app.ml.prediction_engine import PredictionEngine
from app.models.match_context import MatchContext
from app.services.match_replay import MatchReplay

N_MATCHES = 20
random.seed(123)
match_dir = Path("../data/raw/cricsheet/ipl")
sample = random.sample(sorted(match_dir.glob("*.json")), N_MATCHES)

parsed_matches = []
for path in sample:
    try:
        match = MatchParser().parse(JsonLoader().load(str(path)))
    except Exception:
        continue
    parsed_matches.append((path, match, match_exclude_key(match)))

repository = ModelRepository()

best = None
for weight in (0.0, 0.1, 0.2, 0.3, 0.5):
    for centering in (-0.5, -0.3, -0.1, 0.0, 1.0):
        run_errors = []
        for path, match, exclude_key in parsed_matches:
            for innings in match.innings:
                engine = PredictionEngine(
                    repository=repository,
                    momentum_blend_weight=weight,
                    run_centering_correction=centering,
                )
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
                    run_errors.append(
                        abs(prediction.predicted_runs - frame.actual_runs)
                    )
                    engine.update_actuals(
                        actual_runs=frame.actual_runs,
                        actual_wickets=frame.actual_wickets,
                        bowler_name=frame.state.bowler,
                    )
        mae = sum(run_errors) / len(run_errors)
        print(
            f"blend_weight={weight:.1f} centering={centering:+.1f}  "
            f"Run MAE={mae:.3f}  (n={len(run_errors)})"
        )
        if best is None or mae < best[0]:
            best = (mae, weight, centering)

print(f"\nBEST: MAE={best[0]:.3f} at blend_weight={best[1]:.1f}, centering={best[2]:+.1f}")
