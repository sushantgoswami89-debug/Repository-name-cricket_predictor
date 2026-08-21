"""Break down run and wicket prediction accuracy by match phase
(powerplay/middle/death) to see which the model handles better."""

import random
from pathlib import Path

from app.data_ingestion.json_loader import JsonLoader
from app.data_ingestion.match_parser import MatchParser
from app.ml.historical_feature_store import match_exclude_key
from app.ml.prediction_engine import PredictionEngine
from app.models.match_context import MatchContext
from app.services.match_replay import MatchReplay

N_MATCHES = 30
random.seed(7)
match_dir = Path("../data/raw/cricsheet/ipl")
sample = random.sample(sorted(match_dir.glob("*.json")), N_MATCHES)

PHASES = ("powerplay", "middle", "death")


def phase_of(over):
    if over <= 6:
        return "powerplay"
    if over <= 15:
        return "middle"
    return "death"


stats = {
    p: {
        "run_errors": [],
        "range_hits": [],
        "wkt_brier": [],
        "wkt_correct": [],
        "had_wicket": [],
    }
    for p in PHASES
}

matches_ok = 0
for path in sample:
    try:
        match = MatchParser().parse(JsonLoader().load(str(path)))
    except Exception:
        continue
    exclude_key = match_exclude_key(match)
    matches_ok += 1
    for innings in match.innings:
        engine = PredictionEngine()
        for frame in MatchReplay().frames(innings):
            phase = phase_of(frame.state.over)
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

            had_wicket = frame.actual_wickets > 0
            predicted_wicket = prediction.wicket_probability >= 0.5
            low, high = (int(x) for x in prediction.expected_range.split("-"))

            s = stats[phase]
            s["run_errors"].append(abs(prediction.predicted_runs - frame.actual_runs))
            s["range_hits"].append(low <= frame.actual_runs <= high)
            s["wkt_brier"].append((prediction.wicket_probability - had_wicket) ** 2)
            s["wkt_correct"].append(predicted_wicket == had_wicket)
            s["had_wicket"].append(had_wicket)

            engine.update_actuals(
                actual_runs=frame.actual_runs,
                actual_wickets=frame.actual_wickets,
                bowler_name=frame.state.bowler,
            )

print(f"Matches: {matches_ok}\n")
print(
    f"{'Phase':10} | {'Overs':>6} | {'RunMAE':>7} | {'RangeHit':>9} | "
    f"{'WktBrier':>9} | {'NaiveBrier':>10} | {'WktAcc':>7} | {'WktBaseRate':>11}"
)
for p in PHASES:
    s = stats[p]
    n = len(s["run_errors"])
    if n == 0:
        continue
    run_mae = sum(s["run_errors"]) / n
    range_hit = sum(s["range_hits"]) / n * 100
    wkt_brier = sum(s["wkt_brier"]) / n
    wkt_acc = sum(s["wkt_correct"]) / n * 100
    base_rate = sum(s["had_wicket"]) / n
    naive_brier = base_rate * (1 - base_rate)
    print(
        f"{p:10} | {n:6d} | {run_mae:7.3f} | {range_hit:8.1f}% | "
        f"{wkt_brier:9.4f} | {naive_brier:10.4f} | {wkt_acc:6.1f}% | {base_rate*100:9.1f}%"
    )

print()
print("(NaiveBrier = score from always predicting that phase's own wicket rate;")
print(" WktBrier below NaiveBrier = model beats naive guessing in that phase)")
