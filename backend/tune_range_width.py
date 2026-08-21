"""Measure real per-phase residual distribution (actual - pivot) on the
tuning sample to size the prediction bracket to real uncertainty, instead
of the hardcoded ~2.6-run window."""

import random
from pathlib import Path

from app.data_ingestion.json_loader import JsonLoader
from app.data_ingestion.match_parser import MatchParser
from app.ml.historical_feature_store import match_exclude_key
from app.ml.prediction_engine import PredictionEngine
from app.models.match_context import MatchContext
from app.services.match_replay import MatchReplay

N_MATCHES = 20
random.seed(123)
match_dir = Path("../data/raw/cricsheet/ipl")
sample = random.sample(sorted(match_dir.glob("*.json")), N_MATCHES)

residuals_by_phase = {"powerplay": [], "middle": [], "death": []}

for path in sample:
    try:
        match = MatchParser().parse(JsonLoader().load(str(path)))
    except Exception:
        continue
    exclude_key = match_exclude_key(match)
    for innings in match.innings:
        engine = PredictionEngine()
        for frame in MatchReplay().frames(innings):
            over = frame.state.over
            phase = "powerplay" if over <= 6 else ("middle" if over <= 15 else "death")
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
            residuals_by_phase[phase].append(frame.actual_runs - prediction.predicted_runs)
            engine.update_actuals(
                actual_runs=frame.actual_runs,
                actual_wickets=frame.actual_wickets,
                bowler_name=frame.state.bowler,
            )

import statistics
for phase, residuals in residuals_by_phase.items():
    residuals.sort()
    n = len(residuals)
    def pct(p):
        idx = min(n - 1, max(0, int(p * n)))
        return residuals[idx]
    print(f"{phase}: n={n}")
    for p in (0.05, 0.10, 0.15, 0.20, 0.25, 0.5, 0.75, 0.80, 0.85, 0.90, 0.95):
        print(f"  p{int(p*100):02d} = {pct(p):+.2f}")
    print(f"  mean abs residual = {sum(abs(r) for r in residuals)/n:.2f}")
