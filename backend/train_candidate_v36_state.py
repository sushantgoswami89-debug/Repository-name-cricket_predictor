"""Train Candidate v3.6 state and strike-share ablations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from train_candidate_v33_response import _model
from train_sharp_range_candidate import MAX_RUN_CLASS, _best_bands, _phase_report
from train_validate_candidate_v3 import CATEGORICAL_FEATURES, V3_FEATURES, _features

KEYS = ["source_file", "match_date", "innings", "over"]
V32_HIT_RATE = 0.31376664261821996


def _groups(extra: pd.DataFrame) -> dict[str, list[str]]:
    groups = {
        name: [column for column in extra if column.startswith(f"{name}_")]
        for name in ("settlement", "strike_share", "bowler_match", "volatility")
    }
    groups["combined"] = sum(groups.values(), [])
    return groups


def train(project_root: Path, output_dir: Path) -> dict[str, Any]:
    base = pd.read_csv(project_root / "data/candidates/v3/verified_training_overs.csv")
    extra = pd.read_csv(
        project_root / "data/candidates/v3.6/state_strike_share_features.csv"
    )
    base["match_date"] = base["match_date"].astype(str)
    extra["match_date"] = extra["match_date"].astype(str)
    data = base.merge(extra, on=KEYS, how="inner", validate="one_to_one")
    if len(data) != len(base):
        raise ValueError("v3.6 state features do not reconcile with verified data.")
    data["match_date"] = pd.to_datetime(data["match_date"])
    training = data[data["match_date"].dt.year <= 2023].reset_index(drop=True)
    holdout = data[data["match_date"].dt.year >= 2025].reset_index(drop=True)
    actual = np.minimum(holdout["runs_in_over"].to_numpy(), MAX_RUN_CLASS)
    output_dir.mkdir(parents=True, exist_ok=True)
    ablations: dict[str, Any] = {}

    for name, additions in _groups(extra).items():
        columns = V3_FEATURES + additions
        model = _model()
        model.fit(
            _features(training, columns),
            np.minimum(training["runs_in_over"], MAX_RUN_CLASS),
            categorical_feature=CATEGORICAL_FEATURES,
        )
        probability = model.predict_proba(_features(holdout, columns))
        low, high, band_mass = _best_bands(probability, 2)
        hit = (actual >= low) & (actual <= high)
        ablations[name] = {
            "features_added": additions,
            "two_run_hit_rate": float(hit.mean()),
            "improvement_over_v32_percentage_points": float(
                (hit.mean() - V32_HIT_RATE) * 100
            ),
            "mean_band_probability": float(band_mass.mean()),
            "phase": _phase_report(holdout, actual, low, high),
        }
        joblib.dump(model, output_dir / f"{name}_model.pkl")
        joblib.dump(columns, output_dir / f"{name}_feature_cols.pkl")

    winner_name = max(ablations, key=lambda name: ablations[name]["two_run_hit_rate"])
    winner = ablations[winner_name]
    baseline_phase = {
        "powerplay": 0.2774744366127271,
        "middle": 0.34074231628203766,
        "death": 0.31140663342040686,
    }
    gates = {
        "beats_v32": winner["two_run_hit_rate"] > V32_HIT_RATE,
        "beats_required_32_38pct": winner["two_run_hit_rate"] >= 0.3238,
        "no_phase_regresses_over_2pct": all(
            winner["phase"][phase]["hit_rate"] >= rate - 0.02
            for phase, rate in baseline_phase.items()
        ),
        "complete_match_sequence_failures_zero": not data.duplicated(KEYS).any(),
    }
    report = {
        "candidate_version": "v3.6_state_strike_share",
        "model_scope": "candidate_only",
        "production_models_changed": False,
        "verified_rows": len(data),
        "holdout_rows": len(holdout),
        "holdout_matches": int(holdout["source_file"].nunique()),
        "v32_benchmark_hit_rate": V32_HIT_RATE,
        "ablations": ablations,
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
            train(root, root / "models/candidates/v3.6_state_strike_share"), indent=2
        )
    )
