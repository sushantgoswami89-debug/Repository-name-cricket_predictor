"""Train male-only phase-calibrated run-range candidate v3.3."""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

MAX_RUN_CLASS = 30
FEATURES = [
    "over",
    "score_before_over",
    "wkts_down_before_over",
    "phase",
    "wickets_in_hand",
    "legal_balls_bowled",
    "balls_remaining",
    "current_run_rate",
    "is_chase",
    "runs_required",
    "required_run_rate",
    "recent_legal_balls",
    "recent_runs_per_ball",
    "recent_dot_rate",
    "recent_single_rate",
    "recent_boundary_rate",
    "recent_wicket_rate",
]


def features(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame[columns].copy()
    result["phase"] = result["phase"].astype("category")
    return result


def best_bands(probabilities: np.ndarray, width: int = 2) -> tuple[np.ndarray, np.ndarray]:
    windows = np.column_stack(
        [
            probabilities[:, start : start + width + 1].sum(axis=1)
            for start in range(probabilities.shape[1] - width)
        ]
    )
    low = np.argmax(windows, axis=1)
    return low, low + width


def temperature_scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.clip(probabilities, 1e-12, 1.0)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    scaled = np.exp(logits)
    return scaled / scaled.sum(axis=1, keepdims=True)


def nll(probabilities: np.ndarray, actual: np.ndarray) -> float:
    return float(-np.log(np.clip(probabilities[np.arange(len(actual)), actual], 1e-12, 1)).mean())


def male_source_files(root: Path) -> set[str]:
    eligible: set[str] = set()
    for scope in ("ipl", "t20i"):
        for path in (root / "data/raw/cricsheet" / scope).glob("*.json"):
            raw = json.loads(path.read_text(encoding="utf-8"))
            info = raw.get("info", {})
            if (
                str(info.get("gender", "")).lower() == "male"
                and str(info.get("match_type", "")).upper() == "T20"
            ):
                eligible.add(path.name)
    return eligible


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    version = os.environ.get(
        "RUN_RANGE_VERSION", "v3.3_phase_calibrated_sharp_range"
    )
    direct_interactions = os.environ.get("DIRECT_PHASE_INTERACTIONS") == "1"
    phase_calibration = os.environ.get("PHASE_TEMPERATURE_SCALING", "1") == "1"
    output = root / "models/candidates" / version
    output.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(root / "data/candidates/v3/verified_training_overs.csv")
    data = data[data["source_file"].isin(male_source_files(root))].copy()
    data["match_date"] = pd.to_datetime(data["match_date"])
    active_features = list(FEATURES)
    if direct_interactions:
        phase_weight = data["phase"].astype(str).map(
            {"powerplay": 1.0, "middle": 2.0, "death": 3.0}
        ).fillna(0.0)
        data["phase_x_wickets"] = data["wkts_down_before_over"] * phase_weight
        data["phase_x_required_rate"] = data["required_run_rate"] * phase_weight
        data["phase_x_recent_scoring"] = data["recent_runs_per_ball"] * phase_weight
        data["phase_x_boundary_rate"] = data["recent_boundary_rate"] * phase_weight
        data["phase_x_dot_rate"] = data["recent_dot_rate"] * phase_weight
        active_features += [
            "phase_x_wickets",
            "phase_x_required_rate",
            "phase_x_recent_scoring",
            "phase_x_boundary_rate",
            "phase_x_dot_rate",
        ]
    train = data[data["match_date"] <= "2023-12-31"].reset_index(drop=True)
    calibration = data[data["match_date"].dt.year == 2024].reset_index(drop=True)
    holdout = data[data["match_date"] >= "2025-01-01"].reset_index(drop=True)

    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=MAX_RUN_CLASS + 1,
        n_estimators=75 if direct_interactions else 60,
        learning_rate=0.06 if direct_interactions else 0.08,
        num_leaves=31 if direct_interactions else 25,
        max_depth=7 if direct_interactions else 6,
        min_child_samples=80 if direct_interactions else 100,
        subsample=0.85,
        colsample_bytree=0.85 if direct_interactions else 0.9,
        reg_lambda=1.5 if direct_interactions else 1.0,
        random_state=42,
        verbose=-1,
    )
    train_actual = np.minimum(train["runs_in_over"].to_numpy(), MAX_RUN_CLASS)
    model.fit(features(train, active_features), train_actual, categorical_feature=["phase"])
    calibration_raw = model.predict_proba(features(calibration, active_features))
    holdout_raw = model.predict_proba(features(holdout, active_features))
    calibration_actual = np.minimum(calibration["runs_in_over"].to_numpy(), MAX_RUN_CLASS)
    holdout_actual = np.minimum(holdout["runs_in_over"].to_numpy(), MAX_RUN_CLASS)

    temperatures: dict[str, float] = {}
    calibration_scaled = calibration_raw.copy()
    holdout_scaled = holdout_raw.copy()
    for phase in ("powerplay", "middle", "death"):
        calibration_mask = calibration["phase"].astype(str).to_numpy() == phase
        holdout_mask = holdout["phase"].astype(str).to_numpy() == phase
        result = minimize_scalar(
            lambda value: nll(
                temperature_scale(calibration_raw[calibration_mask], value),
                calibration_actual[calibration_mask],
            ),
            bounds=(0.5, 3.0),
            method="bounded",
        )
        temperatures[phase] = float(result.x) if phase_calibration else 1.0
        calibration_scaled[calibration_mask] = temperature_scale(
            calibration_raw[calibration_mask], temperatures[phase]
        )
        holdout_scaled[holdout_mask] = temperature_scale(
            holdout_raw[holdout_mask], temperatures[phase]
        )

    calibration_low, calibration_high = best_bands(calibration_scaled)
    holdout_low, holdout_high = best_bands(holdout_scaled)
    calibration_hit = np.mean(
        (calibration_actual >= calibration_low) & (calibration_actual <= calibration_high)
    )
    holdout_hit = np.mean(
        (holdout_actual >= holdout_low) & (holdout_actual <= holdout_high)
    )
    raw_calibration_nll = nll(calibration_raw, calibration_actual)
    scaled_calibration_nll = nll(calibration_scaled, calibration_actual)
    old_hit = 0.2728246539222149
    accepted = bool(
        holdout_hit > old_hit and scaled_calibration_nll <= raw_calibration_nll
    )
    report = {
        "candidate_version": version,
        "scope": "male_only_ipl_and_t20i",
        "split": {
            "train_rows": len(train),
            "calibration_rows": len(calibration),
            "holdout_rows": len(holdout),
            "holdout_matches": int(holdout["source_file"].nunique()),
        },
        "phase_temperatures": temperatures,
        "direct_phase_interactions": direct_interactions,
        "interaction_features": active_features[len(FEATURES):],
        "calibration": {
            "raw_nll": raw_calibration_nll,
            "phase_scaled_nll": scaled_calibration_nll,
            "hit_rate": float(calibration_hit),
        },
        "holdout": {
            "old_hit_rate": old_hit,
            "new_hit_rate": float(holdout_hit),
            "relative_improvement": float((holdout_hit - old_hit) / old_hit),
        },
        "decision": "ACCEPT" if accepted else "REJECT",
    }
    joblib.dump(model, output / "sharp_range_model.pkl")
    joblib.dump(temperatures, output / "phase_temperatures.pkl")
    joblib.dump(active_features, output / "feature_cols.pkl")
    (output / "validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    predictions = holdout[
        ["source_file", "match_date", "innings", "over", "phase", "runs_in_over"]
    ].copy()
    predictions["range_low"] = holdout_low
    predictions["range_high"] = holdout_high
    predictions["hit"] = (
        (holdout_actual >= holdout_low) & (holdout_actual <= holdout_high)
    )
    predictions.to_csv(output / "holdout_predictions.csv", index=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
