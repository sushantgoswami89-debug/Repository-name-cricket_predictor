"""Diagnose the run-range momentum-lag finding (docs/live_field_test_2026-09-15.md)
against a real match with two consecutive massive overs, through the
ACTUAL current live PredictionEngine -- not a re-derivation.

Usage:
    .venv/bin/python3 backend/diagnose_momentum_lag.py [path/to/match.json] [innings_number]
"""

import sys

from app.data_ingestion.json_loader import JsonLoader
from app.data_ingestion.match_parser import MatchParser
from app.ml.historical_feature_store import match_exclude_key
from app.ml.prediction_engine import PredictionEngine
from app.models.match_context import MatchContext
from app.services.match_replay import MatchReplay

MATCH_PATH = sys.argv[1] if len(sys.argv) > 1 else "../data/raw/cricsheet/t20i/1528275.json"
INNINGS = int(sys.argv[2]) if len(sys.argv) > 2 else 1

match = MatchParser().parse(JsonLoader().load(MATCH_PATH))
exclude_key = match_exclude_key(match)
innings = match.innings[INNINGS - 1]

print(f"{match.info.teams[0]} vs {match.info.teams[1]} -- innings {INNINGS}: {innings.team} batting")
print(f"{'Ov':>3} | {'Actual':>7} | {'Predicted':>9} | {'recent_runs_per_ball':>20} | {'recent_legal_balls':>18}")

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
        f"{frame.state.over:3d} | {frame.actual_runs:7d} | "
        f"{prediction.predicted_runs:9.2f} | "
        f"{frame.state.recent_runs_per_ball:20.2f} | "
        f"{frame.state.recent_legal_balls:18d}"
    )
    engine.update_actuals(
        actual_runs=frame.actual_runs,
        actual_wickets=frame.actual_wickets,
        bowler_name=frame.state.bowler,
    )
