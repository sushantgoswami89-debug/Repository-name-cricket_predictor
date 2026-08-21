"""Train Candidate v3.3 sequence and player-response Sharp Range ablations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from train_sharp_range_candidate import MAX_RUN_CLASS, _best_bands, _phase_report
from train_validate_candidate_v3 import CATEGORICAL_FEATURES, V3_FEATURES, _features

KEYS = ["source_file", "match_date", "innings", "over"]


def _feature_groups(extra: pd.DataFrame) -> dict[str, list[str]]:
    ignored = set(KEYS + ["striker", "bowler"])
    generic = [
        column
        for column in extra.columns
        if column not in ignored
        and not column.startswith(
            (
                "batter_response_",
                "bowler_response_",
                "matchup_response_",
                "pressure_matchup_response_",
            )
        )
    ]
    batter = [column for column in extra if column.startswith("batter_response_")]
    bowler = [column for column in extra if column.startswith("bowler_response_")]
    matchup = [
        column
        for column in extra
        if column.startswith(("matchup_response_", "pressure_matchup_response_"))
    ]
    return {
        "generic_sequence": generic,
        "batter_response": batter,
        "bowler_response": bowler,
        "matchup_response": matchup,
        "combined": generic + batter + bowler + matchup,
    }


def _model() -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="multiclass",
        num_class=MAX_RUN_CLASS + 1,
        n_estimators=60,
        learning_rate=0.08,
        num_leaves=25,
        max_depth=6,
        min_child_samples=100,
        subsample=0.85,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1,
    )


def train(project_root: Path, output_dir: Path) -> dict[str, Any]:
    base = pd.read_csv(project_root / "data/candidates/v3/verified_training_overs.csv")
    extra = pd.read_csv(
        project_root / "data/candidates/v3.3/sequence_response_features.csv"
    )
    base["match_date"] = base["match_date"].astype(str)
    extra["match_date"] = extra["match_date"].astype(str)
    data = base.merge(extra, on=KEYS, how="inner", validate="one_to_one")
    if len(data) != len(base):
        raise ValueError("v3.3 features do not reconcile with the verified dataset.")
    data["match_date"] = pd.to_datetime(data["match_date"])
    train_data = data[data["match_date"].dt.year <= 2023].reset_index(drop=True)
    holdout = data[data["match_date"].dt.year >= 2025].reset_index(drop=True)
    actual = np.minimum(holdout["runs_in_over"].to_numpy(), MAX_RUN_CLASS)
    groups = _feature_groups(extra)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports: dict[str, Any] = {}

    for name, additions in groups.items():
        columns = V3_FEATURES + additions
        model = _model()
        model.fit(
            _features(train_data, columns),
            np.minimum(train_data["runs_in_over"], MAX_RUN_CLASS),
            categorical_feature=CATEGORICAL_FEATURES,
        )
        probability = model.predict_proba(_features(holdout, columns))
        low, high, _ = _best_bands(probability, 2)
        hit = (actual >= low) & (actual <= high)
        phase = _phase_report(holdout, actual, low, high)
        reports[name] = {
            "features_added": additions,
            "feature_count": len(columns),
            "holdout_hit_rate": float(hit.mean()),
            "improvement_over_v32_percentage_points": float(
                (hit.mean() - 0.31376664261821996) * 100
            ),
            "phase": phase,
        }
        joblib.dump(model, output_dir / f"{name}_sharp_range_model.pkl")
        joblib.dump(columns, output_dir / f"{name}_feature_cols.pkl")

    winner_name = max(reports, key=lambda name: reports[name]["holdout_hit_rate"])
    winner = reports[winner_name]
    baseline_phase = {
        "powerplay": 0.2774744366127271,
        "middle": 0.34074231628203766,
        "death": 0.31140663342040686,
    }
    gates = {
        "beats_required_32_38pct": winner["holdout_hit_rate"] >= 0.3238,
        "improves_over_v32": winner["holdout_hit_rate"] > 0.31376664261821996,
        "no_phase_regresses_over_2pct": all(
            winner["phase"][phase]["hit_rate"] >= value - 0.02
            for phase, value in baseline_phase.items()
        ),
        "complete_matches_at_least_100": holdout["source_file"].nunique() >= 100,
    }
    report = {
        "candidate_version": "v3.3_sequence_player_response",
        "model_scope": "candidate_only",
        "production_models_changed": False,
        "verified_rows": len(data),
        "holdout_rows": len(holdout),
        "holdout_matches": int(holdout["source_file"].nunique()),
        "v32_benchmark_hit_rate": 0.31376664261821996,
        "ablations": reports,
        "winner": winner_name,
        "winner_metrics": winner,
        "promotion_gates": gates,
        "integration_recommended": all(gates.values()),
    }
    (output_dir / "validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    print(
        json.dumps(
            train(root, root / "models/candidates/v3.3_sequence_player_response"),
            indent=2,
        )
    )
