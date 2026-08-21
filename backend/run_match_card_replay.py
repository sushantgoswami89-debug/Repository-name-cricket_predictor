"""Replay a full match and print a commentary-booth style card before each over.

Each card shows the prediction for the upcoming over, plus a review of how
the previous over's prediction actually did once its result is known.

Usage:
    .venv/bin/python3 backend/run_match_card_replay.py [path/to/match.json] [innings_number]

If innings_number is omitted, every innings in the match is replayed in order
(regular innings plus any super over).
"""

import sys

from app.data_ingestion.json_loader import JsonLoader
from app.data_ingestion.match_parser import MatchParser
from app.ml.historical_feature_store import match_exclude_key
from app.ml.prediction_engine import PredictionEngine
from app.models.match_context import MatchContext
from app.services.match_replay import MatchReplay

MATCH_PATH = sys.argv[1] if len(sys.argv) > 1 else "../data/raw/cricsheet/ipl/1136561.json"
INNINGS_NUMBER = int(sys.argv[2]) if len(sys.argv) > 2 else None


def confidence_bar(confidence: float) -> str:
    filled = max(0, min(10, round(confidence * 10)))
    return "█" * filled + "░" * (10 - filled)


def confidence_label(confidence: float) -> str:
    if confidence >= 0.80:
        return "HIGH"
    if confidence >= 0.60:
        return "MEDIUM"
    return "LOW"


def star_rating(abs_error: float) -> tuple[str, str]:
    if abs_error <= 1.0:
        return "★★★★★", "EXCELLENT"
    if abs_error <= 2.0:
        return "★★★★☆", "GOOD"
    if abs_error <= 3.5:
        return "★★★☆☆", "OK"
    if abs_error <= 5.0:
        return "★★☆☆☆", "FAIR"
    return "★☆☆☆☆", "POOR"


def in_range(actual: int, expected_range: str) -> bool:
    low, high = (int(x) for x in expected_range.split("-"))
    return low <= actual <= high


def replay_innings(match, innings_number, innings, match_errors, match_hits):
    print()
    print("=" * 60)
    print(f"{match.info.teams[0]} vs {match.info.teams[1]}")
    print(f"{match.info.venue or 'Unknown venue'}")
    print(f"Innings {innings_number}: {innings.team} batting")
    print("=" * 60)

    engine = PredictionEngine()
    exclude_key = match_exclude_key(match)
    prev_prediction = None
    prev_over_actual = None
    innings_errors = []
    innings_hits = []

    for frame in MatchReplay().frames(innings):
        state = frame.state
        context = MatchContext(
            team1=match.info.teams[0],
            team2=match.info.teams[1],
            venue=match.info.venue or "Unknown",
            format=match.info.match_type,
            live=state,
            metadata={"exclude_match_key": exclude_key},
        )
        prediction = engine.predict(context)

        verified_through = f"{state.over - 1}.6" if state.over > 1 else "start of innings"

        print()
        print(f"────────── OVER {state.over} ──────────")
        print(f"Score: {state.score_before_over}/{state.wkts_down_before_over}")
        print(f"Runs: {prediction.expected_range}  (point {prediction.predicted_runs:.1f})")
        print(f"Wicket probability: {prediction.wicket_probability*100:.1f}%")
        print(
            f"Confidence: {confidence_bar(prediction.confidence)} "
            f"{prediction.confidence*100:.0f}% ({confidence_label(prediction.confidence)})"
        )
        print(f"State verified through: {verified_through}")

        if prev_prediction is not None:
            prev_runs, prev_range = prev_prediction
            error = prev_runs - prev_over_actual
            stars, label = star_rating(abs(error))
            hit = in_range(prev_over_actual, prev_range)
            print(f"Previous over review: {stars} {label}")
            print(
                f"  Actual: {prev_over_actual} | Error: {error:+.1f} | "
                f"Range: {'✅ hit' if hit else '❌ missed'}"
            )
            innings_errors.append(abs(error))
            innings_hits.append(hit)
        else:
            print("Previous over review: (first over — nothing to review yet)")

        prev_prediction = (prediction.predicted_runs, prediction.expected_range)
        prev_over_actual = frame.actual_runs

        engine.update_actuals(
            actual_runs=frame.actual_runs,
            actual_wickets=frame.actual_wickets,
            bowler_name=state.bowler,
        )

    if prev_prediction is not None:
        prev_runs, prev_range = prev_prediction
        error = prev_runs - prev_over_actual
        stars, label = star_rating(abs(error))
        hit = in_range(prev_over_actual, prev_range)
        print()
        print("────────── END OF INNINGS ──────────")
        print(f"Final over review: {stars} {label}")
        print(
            f"  Actual: {prev_over_actual} | Error: {error:+.1f} | "
            f"Range: {'✅ hit' if hit else '❌ missed'}"
        )
        innings_errors.append(abs(error))
        innings_hits.append(hit)

    if innings_errors:
        mae = sum(innings_errors) / len(innings_errors)
        hit_rate = sum(innings_hits) / len(innings_hits) * 100
        print(
            f"\nInnings {innings_number} summary: Run MAE = {mae:.2f} | "
            f"Range hit rate = {hit_rate:.0f}% over {len(innings_errors)} overs"
        )
        match_errors.extend(innings_errors)
        match_hits.extend(innings_hits)


match = MatchParser().parse(JsonLoader().load(MATCH_PATH))

match_errors: list[float] = []
match_hits: list[bool] = []

if INNINGS_NUMBER is not None:
    innings_list = [(INNINGS_NUMBER, match.innings[INNINGS_NUMBER - 1])]
else:
    innings_list = list(enumerate(match.innings, start=1))

for innings_number, innings in innings_list:
    replay_innings(match, innings_number, innings, match_errors, match_hits)

if len(innings_list) > 1 and match_errors:
    print()
    print("=" * 60)
    print(
        f"MATCH TOTAL: Run MAE = {sum(match_errors)/len(match_errors):.2f} | "
        f"Range hit rate = {sum(match_hits)/len(match_hits)*100:.0f}% "
        f"over {len(match_errors)} overs"
    )
    print("=" * 60)
