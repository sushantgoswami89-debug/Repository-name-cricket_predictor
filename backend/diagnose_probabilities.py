"""Diagnostics for run-range and wicket-probability quality across matches."""

import random
from pathlib import Path

from app.data_ingestion.json_loader import JsonLoader
from app.data_ingestion.match_parser import MatchParser
from app.ml.historical_feature_store import match_exclude_key
from app.ml.prediction_engine import PredictionEngine
from app.models.match_context import MatchContext
from app.services.match_replay import MatchReplay

N_MATCHES = 15
random.seed(99)
match_dir = Path("../data/raw/cricsheet/ipl")
sample = random.sample(sorted(match_dir.glob("*.json")), N_MATCHES)

wicket_overs = 0
total_overs = 0
brier_sum = 0.0
confidence_buckets = {}  # bucket -> [hits, total]
max_multiplier_per_match = []
range_widths = []

for path in sample:
    try:
        match = MatchParser().parse(JsonLoader().load(str(path)))
    except Exception:
        continue

    exclude_key = match_exclude_key(match)
    for innings in match.innings:
        engine = PredictionEngine()
        match_max_mult = 1.0
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

            total_overs += 1
            had_wicket = frame.actual_wickets > 0
            wicket_overs += had_wicket
            brier_sum += (prediction.wicket_probability - had_wicket) ** 2

            low, high = (int(x) for x in prediction.expected_range.split("-"))
            range_widths.append(high - low)
            hit = low <= frame.actual_runs <= high
            bucket = int(prediction.confidence * 10) * 10  # 0,10,...,90
            b = confidence_buckets.setdefault(bucket, [0, 0])
            b[0] += hit
            b[1] += 1

            engine.update_actuals(
                actual_runs=frame.actual_runs,
                actual_wickets=frame.actual_wickets,
                bowler_name=frame.state.bowler,
            )
            match_max_mult = max(match_max_mult, engine._wicket_multiplier)
        max_multiplier_per_match.append(match_max_mult)

print(f"Matches sampled: {N_MATCHES} | Overs analyzed: {total_overs}\n")

base_rate = wicket_overs / total_overs
naive_accuracy = max(base_rate, 1 - base_rate)
print("--- Wicket probability ---")
print(f"Actual wicket-in-over base rate : {base_rate*100:.1f}%")
print(f"Naive 'always predict majority' accuracy: {naive_accuracy*100:.1f}%")
print(f"Brier score (this engine)        : {brier_sum/total_overs:.4f}")
print(
    f"Wicket multiplier range across matches: "
    f"min {min(max_multiplier_per_match):.2f}, max {max(max_multiplier_per_match):.2f}, "
    f"mean-of-max {sum(max_multiplier_per_match)/len(max_multiplier_per_match):.2f}"
)

print("\n--- Run range width ---")
print(f"Mean predicted range width: {sum(range_widths)/len(range_widths):.2f} runs")
print(f"Min/Max width seen: {min(range_widths)} / {max(range_widths)}")

print("\n--- Confidence calibration (does higher displayed confidence -> higher range-hit rate?) ---")
print(f"{'Confidence bucket':>18} | {'Range hit rate':>14} | {'N overs':>8}")
for bucket in sorted(confidence_buckets):
    hits, n = confidence_buckets[bucket]
    print(f"{bucket:>15}-{bucket+9}% | {hits/n*100:13.1f}% | {n:8d}")
