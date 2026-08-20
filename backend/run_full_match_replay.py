"""Replay a full match over-by-over: predict before each over, score after it.

Usage:
    .venv/bin/python3 backend/run_full_match_replay.py [path/to/match.json]

Defaults to the same sample match backend/run_replay.py uses.
"""

import sys

from app.data_ingestion.json_loader import JsonLoader
from app.data_ingestion.match_parser import MatchParser
from app.ml.historical_feature_store import match_exclude_key
from app.ml.prediction_engine import PredictionEngine
from app.models.match_context import MatchContext
from app.services.match_replay import MatchReplay

MATCH_PATH = sys.argv[1] if len(sys.argv) > 1 else "../data/raw/cricsheet/t20i/1442989.json"

match = MatchParser().parse(JsonLoader().load(MATCH_PATH))
exclude_key = match_exclude_key(match)

print("=" * 72)
print(f"{match.info.teams[0]} vs {match.info.teams[1]} — {match.info.venue or 'Unknown venue'}")
print("=" * 72)

overall_run_errors = []
overall_wicket_correct = []

for innings_number, innings in enumerate(match.innings, start=1):
    print(f"\n--- Innings {innings_number}: {innings.team} batting ---")
    print(f"{'Ov':>3} | {'Score':>8} | {'Predicted':>9} | {'Wkt%':>5} | {'Actual':>8} | {'RunErr':>6} | {'RunMAE':>6} | {'WktAcc':>6}")

    engine = PredictionEngine()
    run_errors = []
    wicket_correct = []

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

        actual_runs = frame.actual_runs
        actual_wicket = frame.actual_wickets > 0
        predicted_wicket = prediction.wicket_probability >= 0.5

        run_error = prediction.predicted_runs - actual_runs
        run_errors.append(abs(run_error))
        wicket_correct.append(predicted_wicket == actual_wicket)

        running_mae = sum(run_errors) / len(run_errors)
        running_wkt_acc = sum(wicket_correct) / len(wicket_correct)

        print(
            f"{frame.state.over:3d} | "
            f"{frame.state.score_before_over:3d}/{frame.state.wkts_down_before_over:<4d} | "
            f"{prediction.predicted_runs:9.2f} | "
            f"{prediction.wicket_probability*100:4.0f}% | "
            f"{frame.actual_runs:3d}/{frame.actual_wickets:<4d} | "
            f"{run_error:+6.1f} | "
            f"{running_mae:6.2f} | "
            f"{running_wkt_acc*100:5.0f}%"
        )

        engine.update_actuals(
            actual_runs=frame.actual_runs,
            actual_wickets=frame.actual_wickets,
            bowler_name=frame.state.bowler,
        )

    if run_errors:
        print(
            f"\nInnings {innings_number} summary: "
            f"Run MAE = {sum(run_errors)/len(run_errors):.2f} | "
            f"Wicket accuracy = {sum(wicket_correct)/len(wicket_correct)*100:.0f}% "
            f"over {len(run_errors)} overs"
        )
        overall_run_errors.extend(run_errors)
        overall_wicket_correct.extend(wicket_correct)

print("\n" + "=" * 72)
if overall_run_errors:
    print(
        f"MATCH TOTAL: Run MAE = {sum(overall_run_errors)/len(overall_run_errors):.2f} | "
        f"Wicket accuracy = {sum(overall_wicket_correct)/len(overall_wicket_correct)*100:.0f}% "
        f"over {len(overall_run_errors)} overs"
    )
print("=" * 72)
