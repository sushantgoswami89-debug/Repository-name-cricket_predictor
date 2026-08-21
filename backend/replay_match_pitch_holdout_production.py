"""Replay a random subsample of the same holdout matches (source_file) used
to validate the v3.10 candidate through the ACTUAL production
PredictionEngine, to get a genuinely comparable range-hit-rate.

Full holdout is 1704 matches / ~61K overs; at ~0.22s/over (CatBoost
per-row inference in the bowler-spell adjustment dominates cost), that's
~3.7 hours -- impractical. Subsampling to N matches, consistent with the
scale used for every other validation this session (40 matches for the
original bowler-spell check).
"""

import random
import time
from pathlib import Path

import pandas as pd

from app.data_ingestion.json_loader import JsonLoader
from app.data_ingestion.match_parser import MatchParser
from app.ml.historical_feature_store import match_exclude_key
from app.ml.prediction_engine import PredictionEngine
from app.models.match_context import MatchContext
from app.services.match_replay import MatchReplay

PROJECT_ROOT = Path(__file__).resolve().parents[1]
holdout = pd.read_csv(PROJECT_ROOT / "backend" / "match_pitch_holdout_result.csv")
all_source_files = sorted(holdout["source_file"].unique())
N_MATCHES = 60
random.seed(777)  # distinct from every other seed used this session
source_files = sorted(random.sample(all_source_files, min(N_MATCHES, len(all_source_files))))
print(f"Sampled {len(source_files)} of {len(all_source_files)} holdout matches to replay")

# Locate each source_file across the format directories the candidate covers.
search_dirs = [
    PROJECT_ROOT / "data/raw/cricsheet/t20i",
    PROJECT_ROOT / "data/raw/cricsheet/ipl",
]
path_by_file: dict[str, Path] = {}
for d in search_dirs:
    for p in d.glob("*.json"):
        path_by_file[p.name] = p

found = [f for f in source_files if f in path_by_file]
missing = len(source_files) - len(found)
print(f"Found on disk: {len(found)} | Missing: {missing}")

run_errors = []
range_hits = []
start = time.perf_counter()
matches_done = 0

for source_file in found:
    path = path_by_file[source_file]
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
            run_errors.append(abs(prediction.predicted_runs - frame.actual_runs))
            range_hits.append(hit)
            engine.update_actuals(
                actual_runs=frame.actual_runs,
                actual_wickets=frame.actual_wickets,
                bowler_name=frame.state.bowler,
            )
    matches_done += 1
    if matches_done % 20 == 0:
        elapsed = time.perf_counter() - start
        print(
            f"  {matches_done}/{len(found)} matches | {len(range_hits)} overs | "
            f"{elapsed:.0f}s elapsed | running hit rate {sum(range_hits)/len(range_hits)*100:.2f}%"
        )

candidate_subsample_hit_rate = (
    holdout[holdout["source_file"].isin(found)]["candidate_hit"].mean() * 100
)

print(f"\nTotal overs: {len(range_hits)}")
print(f"Production Run MAE: {sum(run_errors)/len(run_errors):.4f}")
print(f"Production Range hit rate: {sum(range_hits)/len(range_hits)*100:.2f}%")
print(f"\nCandidate (v3.10 boundary_match_pitch) hit rate, SAME {len(found)}-match subsample: {candidate_subsample_hit_rate:.2f}%")
print(f"Candidate hit rate, full 1704-match holdout (reference): 31.84%")
