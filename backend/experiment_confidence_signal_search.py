"""Follow-up to the confidence-calibrator investigation
(docs/CTO_HANDOVER_2026-07-24.md 2026-08-22 addendum): confirmed the
hand-tuned `_dynamic_confidence` heuristic doesn't predict range-hit
accuracy (flat ~30% real hit rate across its whole output range). That
heuristic's weights (phase_penalty, rhythm_penalty, wicket_penalty, etc.)
were hand-picked, never fit against real outcomes.

This tests whether a genuinely FIT model -- not hand-tuned weights -- can
predict range_hit from pre-over match-state information, using the same
train/holdout rigor as the rest of this session's model work (chronological
split, no leakage, honest holdout evaluation, not a train-set score).

If holdout AUC is meaningfully above 0.5, that's real evidence a better
confidence signal is achievable and worth building. If it's close to 0.5,
that's a stronger, broader confirmation that these match-state features
don't predict per-over range-hit well -- not just that one heuristic's
weights were wrong.

Replays real IPL matches through the actual PredictionEngine (same
methodology as fit_confidence_calibrator.py) to get real expected_range /
range_hit pairs, since range_hit depends on the engine's own predicted
range, not something in the static training_overs.csv. Chronological
split: train <=2023, holdout >=2024 (same split convention as the
wicket/run-range model lines this session). Does not touch production.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

from app.data_ingestion.json_loader import JsonLoader
from app.data_ingestion.match_parser import MatchParser
from app.ml.historical_feature_store import match_exclude_key
from app.ml.prediction_engine import PredictionEngine
from app.models.match_context import MatchContext
from app.services.match_replay import MatchReplay

import random

match_dir = Path("../data/raw/cricsheet/ipl")
random.seed(123)
all_files = random.sample(sorted(match_dir.glob("*.json")), 500)
all_files.sort()

rows: list[dict] = []
model_provenance: dict[str, str] = {}
for i, path in enumerate(all_files):
    if i % 100 == 0:
        print(f"  ... {i}/{len(all_files)} matches", flush=True)
    try:
        raw_info = json.loads(path.read_text(encoding="utf-8"))["info"]
        match_date = raw_info["dates"][0]
    except Exception:
        continue
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
            if not model_provenance:
                model_provenance = {
                    "run_model": prediction.metadata.get("run_model", "unknown"),
                    "wicket_model": prediction.metadata.get("wicket_model", "unknown"),
                }
            low, high = (int(x) for x in prediction.expected_range.split("-"))
            hit = 1 if low <= frame.actual_runs <= high else 0
            state = frame.state
            rows.append({
                "match_date": match_date,
                "over": state.over,
                "phase": "powerplay" if state.over <= 6 else ("death" if state.over >= 16 else "middle"),
                "wickets_in_hand": state.wickets_in_hand,
                "balls_remaining": state.balls_remaining,
                "current_run_rate": state.current_run_rate,
                "is_chase": state.is_chase,
                "required_run_rate": state.required_run_rate,
                "recent_runs_per_ball": state.recent_runs_per_ball,
                "recent_dot_rate": state.recent_dot_rate,
                "recent_boundary_rate": state.recent_boundary_rate,
                "recent_wicket_rate": state.recent_wicket_rate,
                "range_width": high - low,
                "raw_confidence": prediction.metadata["raw_confidence"],
                "range_hit": hit,
            })
            engine.update_actuals(
                actual_runs=frame.actual_runs,
                actual_wickets=frame.actual_wickets,
                bowler_name=frame.state.bowler,
            )

data = pd.DataFrame(rows)
data["match_date"] = pd.to_datetime(data["match_date"])
print(f"\nTotal rows: {len(data)}, matches: {data['match_date'].nunique()} unique dates")

train = data[data["match_date"].dt.year <= 2023].copy()
holdout = data[data["match_date"].dt.year >= 2024].copy()
print(f"train: {len(train)} rows, holdout: {len(holdout)} rows")

FEATURES = [
    "over", "wickets_in_hand", "balls_remaining", "current_run_rate",
    "is_chase", "required_run_rate", "recent_runs_per_ball", "recent_dot_rate",
    "recent_boundary_rate", "recent_wicket_rate", "range_width",
]
phase_dummies_train = pd.get_dummies(train["phase"], prefix="phase")
phase_dummies_holdout = pd.get_dummies(holdout["phase"], prefix="phase").reindex(
    columns=phase_dummies_train.columns, fill_value=0
)
X_train = pd.concat([train[FEATURES].reset_index(drop=True), phase_dummies_train.reset_index(drop=True)], axis=1)
X_holdout = pd.concat([holdout[FEATURES].reset_index(drop=True), phase_dummies_holdout.reset_index(drop=True)], axis=1)
y_train = train["range_hit"].to_numpy()
y_holdout = holdout["range_hit"].to_numpy()

model = LogisticRegression(max_iter=1000, C=1.0)
model.fit(X_train, y_train)
holdout_proba = model.predict_proba(X_holdout)[:, 1]

baseline_rate = y_train.mean()
baseline_proba = np.full_like(y_holdout, baseline_rate, dtype=float)

print(f"\n=== Fitted logistic-regression confidence model, holdout evaluation ===")
print(f"train event rate: {baseline_rate:.3f}, holdout event rate: {y_holdout.mean():.3f}")
print(f"holdout AUC (fitted model): {roc_auc_score(y_holdout, holdout_proba):.4f}")
print(f"holdout AUC (existing raw_confidence heuristic): {roc_auc_score(y_holdout, holdout['raw_confidence']):.4f}")
print(f"holdout Brier (fitted model): {brier_score_loss(y_holdout, holdout_proba):.4f}")
print(f"holdout Brier (constant baseline): {brier_score_loss(y_holdout, baseline_proba):.4f}")

print("\nFitted coefficients (standardized-ish by feature scale, for direction only):")
for name, coef in sorted(zip(X_train.columns, model.coef_[0]), key=lambda x: -abs(x[1])):
    print(f"  {name:>24}: {coef:+.4f}")

output_dir = Path("../data/reports/confidence_signal_search")
output_dir.mkdir(parents=True, exist_ok=True)
data.to_csv(output_dir / "replay_data.csv", index=False)
report = {
    "generated_at_utc": pd.Timestamp.now("UTC").isoformat(),
    "model_provenance": model_provenance,
    "_staleness_note": (
        "This file is only valid evidence for the model_provenance stamped "
        "above. If PredictionEngine's run_model/wicket_model has changed "
        "since generated_at_utc, re-run this script before trusting "
        "these numbers -- do not reuse an old copy across a model swap."
    ),
    "n_rows": len(data),
    "train_rows": len(train),
    "holdout_rows": len(holdout),
    "holdout_auc_fitted_model": float(roc_auc_score(y_holdout, holdout_proba)),
    "holdout_auc_existing_heuristic": float(roc_auc_score(y_holdout, holdout["raw_confidence"])),
    "holdout_brier_fitted_model": float(brier_score_loss(y_holdout, holdout_proba)),
    "holdout_brier_constant_baseline": float(brier_score_loss(y_holdout, baseline_proba)),
    "coefficients": {name: float(coef) for name, coef in zip(X_train.columns, model.coef_[0])},
}
(output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(f"\nSaved replay data + report to {output_dir}")
