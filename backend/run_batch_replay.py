"""Replay many matches over-by-over and report aggregate prediction accuracy.

Usage:
    .venv/bin/python3 backend/run_batch_replay.py [n_matches] [format]

format is one of: t20i, odi, ipl (default: t20i)

Fixed 2026-08-22: used to sample randomly across ALL matches regardless of
date. The live models are trained on matches through 2023-12-31 -- a
random sample drawn from the full corpus lands mostly inside that
training period (found 85.5% for a similar script, fit_confidence_calibrator.py),
so the model has already seen most of what it's being "tested" against,
and the accuracy number this prints would be inflated, not a real
estimate of how it performs on a genuinely new match. Now restricted to
the holdout period (>=2025-01-01) the models were never trained or
calibrated on, matching the standard every other accuracy number in this
project uses.
"""

import json
import random
import sys
from pathlib import Path

from app.data_ingestion.json_loader import JsonLoader
from app.data_ingestion.match_parser import MatchParser
from app.ml.historical_feature_store import match_exclude_key
from app.ml.prediction_engine import PredictionEngine
from app.models.match_context import MatchContext
from app.services.match_replay import MatchReplay

N_MATCHES = int(sys.argv[1]) if len(sys.argv) > 1 else 30
FORMAT = sys.argv[2] if len(sys.argv) > 2 else "t20i"
HOLDOUT_CUTOFF = "2025-01-01"

random.seed(42)

match_dir = Path(f"../data/raw/cricsheet/{FORMAT}")


def _is_holdout_match(path: Path) -> bool:
    try:
        dates = json.loads(path.read_text(encoding="utf-8"))["info"]["dates"]
    except (KeyError, IndexError, ValueError):
        return False
    return bool(dates) and str(dates[0]) >= HOLDOUT_CUTOFF


all_files = sorted(p for p in match_dir.glob("*.json") if _is_holdout_match(p))
if not all_files:
    raise SystemExit(
        f"No holdout-period (>={HOLDOUT_CUTOFF}) matches found for format={FORMAT!r}."
    )
sample = random.sample(all_files, min(N_MATCHES, len(all_files)))
print(f"(sampling from {len(all_files)} holdout-period matches, >={HOLDOUT_CUTOFF})")

print(f"Replaying {len(sample)} {FORMAT.upper()} matches...")

overall_run_errors = []
overall_wicket_correct = []
overall_range_hits = []
matches_ok = 0
matches_failed = 0
per_match_rows = []

for path in sample:
    try:
        match = MatchParser().parse(JsonLoader().load(str(path)))
    except Exception as exc:
        matches_failed += 1
        continue

    exclude_key = match_exclude_key(match)
    match_run_errors = []
    match_wicket_correct = []
    match_range_hits = []

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

            actual_wicket = frame.actual_wickets > 0
            predicted_wicket = prediction.wicket_probability >= 0.5
            low, high = (int(x) for x in prediction.expected_range.split("-"))
            match_run_errors.append(abs(prediction.predicted_runs - frame.actual_runs))
            match_wicket_correct.append(predicted_wicket == actual_wicket)
            match_range_hits.append(low <= frame.actual_runs <= high)

            engine.update_actuals(
                actual_runs=frame.actual_runs,
                actual_wickets=frame.actual_wickets,
                bowler_name=frame.state.bowler,
            )

    if not match_run_errors:
        matches_failed += 1
        continue

    matches_ok += 1
    overall_run_errors.extend(match_run_errors)
    overall_wicket_correct.extend(match_wicket_correct)
    overall_range_hits.extend(match_range_hits)
    per_match_rows.append(
        (
            path.stem,
            f"{match.info.teams[0]} vs {match.info.teams[1]}",
            sum(match_run_errors) / len(match_run_errors),
            sum(match_wicket_correct) / len(match_wicket_correct),
            sum(match_range_hits) / len(match_range_hits),
            len(match_run_errors),
        )
    )

print(f"\nMatches replayed OK: {matches_ok} | Failed/skipped: {matches_failed}")
print(f"{'Match':>10} | {'Teams':>35} | {'RunMAE':>7} | {'WktAcc':>6} | {'RangeHit':>8} | {'Overs':>5}")
for match_id, teams, mae, wacc, rhit, n in per_match_rows:
    print(f"{match_id:>10} | {teams:>35} | {mae:7.2f} | {wacc*100:5.0f}% | {rhit*100:7.0f}% | {n:5d}")

print("\n" + "=" * 60)
if overall_run_errors:
    print(
        f"AGGREGATE over {matches_ok} matches / {len(overall_run_errors)} overs:\n"
        f"  Run MAE         = {sum(overall_run_errors)/len(overall_run_errors):.3f}\n"
        f"  Wicket accuracy = {sum(overall_wicket_correct)/len(overall_wicket_correct)*100:.1f}%\n"
        f"  Range hit rate  = {sum(overall_range_hits)/len(overall_range_hits)*100:.1f}%"
    )
print("=" * 60)
