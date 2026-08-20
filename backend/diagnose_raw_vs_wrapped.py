"""Compare the raw ML model's calibration against the wrapped PredictionEngine
output, to see whether accuracy problems live in the model or the heuristics
layered on top of it (blending, bias correction, wicket multiplier, bracket)."""

import random
from pathlib import Path

import pandas as pd

from app.data_ingestion.json_loader import JsonLoader
from app.data_ingestion.match_parser import MatchParser
from app.ml.feature_builder import FeatureBuilder
from app.ml.historical_feature_store import match_exclude_key
from app.ml.model_repository import ModelRepository
from app.ml.prediction_engine import PredictionEngine
from app.models.match_context import MatchContext
from app.services.match_replay import MatchReplay

N_MATCHES = 15
random.seed(99)
match_dir = Path("../data/raw/cricsheet/ipl")
sample = random.sample(sorted(match_dir.glob("*.json")), N_MATCHES)

repository = ModelRepository()
builder = FeatureBuilder()
feature_order = repository.get_feature_columns()
cat_cols = repository.get_categorical_columns()

raw_run_errors = []
raw_wkt_brier = []
wrapped_run_errors = []
wrapped_wkt_brier = []

for path in sample:
    try:
        match = MatchParser().parse(JsonLoader().load(str(path)))
    except Exception:
        continue

    exclude_key = match_exclude_key(match)
    for innings in match.innings:
        engine = PredictionEngine(repository=repository, feature_builder=builder)
        for frame in MatchReplay().frames(innings):
            context = MatchContext(
                team1=match.info.teams[0],
                team2=match.info.teams[1],
                venue=match.info.venue or "Unknown",
                format=match.info.match_type,
                live=frame.state,
                metadata={"exclude_match_key": exclude_key},
            )

            # --- RAW: straight from the ML model, no heuristics ---
            features = builder.build(context)
            ordered = {name: features[name] for name in feature_order}
            df = pd.DataFrame([ordered])
            for col in cat_cols:
                if col in df.columns:
                    df[col] = df[col].astype("category")
            raw_runs = float(repository.get_runs_model().predict(df)[0])
            raw_wkt = float(repository.get_wicket_model().predict_proba(df)[0][1])

            had_wicket = frame.actual_wickets > 0
            raw_run_errors.append(abs(raw_runs - frame.actual_runs))
            raw_wkt_brier.append((raw_wkt - had_wicket) ** 2)

            # --- WRAPPED: full PredictionEngine (blending, bias, multiplier) ---
            try:
                prediction = engine.predict(context)
            except Exception:
                continue
            wrapped_run_errors.append(abs(prediction.predicted_runs - frame.actual_runs))
            wrapped_wkt_brier.append((prediction.wicket_probability - had_wicket) ** 2)

            engine.update_actuals(
                actual_runs=frame.actual_runs,
                actual_wickets=frame.actual_wickets,
                bowler_name=frame.state.bowler,
            )

n = len(raw_run_errors)
print(f"Overs analyzed: {n}\n")
print(f"{'':20} | {'Run MAE':>10} | {'Wicket Brier':>13}")
print(
    f"{'RAW model':20} | {sum(raw_run_errors)/n:10.3f} | "
    f"{sum(raw_wkt_brier)/n:13.4f}"
)
print(
    f"{'WRAPPED engine':20} | {sum(wrapped_run_errors)/len(wrapped_run_errors):10.3f} | "
    f"{sum(wrapped_wkt_brier)/len(wrapped_wkt_brier):13.4f}"
)
