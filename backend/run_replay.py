from app.data_ingestion.json_loader import JsonLoader
from app.data_ingestion.match_parser import MatchParser
from app.ml.historical_feature_store import match_exclude_key
from app.ml.prediction_engine import PredictionEngine
from app.models.match_context import MatchContext
from app.services.match_replay import MatchReplay

match = MatchParser().parse(
    JsonLoader().load("../data/raw/cricsheet/t20i/1442989.json")
)

innings = match.innings[0]
exclude_key = match_exclude_key(match)

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

    prediction = engine.predict(context)

    print(
        f"{frame.state.over:2d} | "
        f"{frame.state.score_before_over:3d}/"
        f"{frame.state.wkts_down_before_over} | "
        f"Pred {prediction.predicted_runs:.2f} | "
        f"Wkt {prediction.wicket_probability:.1%} | "
        f"Actual {frame.actual_runs}/{frame.actual_wickets}"
    )

    engine.update_actuals(
        actual_runs=frame.actual_runs,
        actual_wickets=frame.actual_wickets,
        bowler_name=frame.state.bowler,
    )
