"""Retrain and evaluate the leakage-safe cold-start candidate."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import brier_score_loss, mean_absolute_error

from train_ipl_catboost_phase_moe_v1 import (
    CATEGORICAL,
    MOE_FEATURES,
    SEED,
    _apply_point,
    _calibrate_fixed_range,
    _calibrate_point,
    _fixed_range,
)


VERSION = "ipl_cold_start_v4_range_router"
EXTRA_FEATURES = [
    "cold_start_role",
    "cold_start_status",
    "cold_start_aggressiveness",
    "cold_start_prior_seasons",
    "cold_start_consecutive_xi",
    "cold_start_important_prior",
    "cold_start_is_bowler",
    "current_batting_position",
]
FEATURES = MOE_FEATURES + EXTRA_FEATURES
EXTRA_CATEGORICAL = [
    "cold_start_role",
    "cold_start_status",
    "cold_start_aggressiveness",
]
CATEGORICAL_V3 = CATEGORICAL + EXTRA_CATEGORICAL


def _frame(data: pd.DataFrame, features: list[str], categorical: list[str]):
    result = data[features].copy()
    for column in categorical:
        result[column] = result[column].fillna("__UNKNOWN__").astype(str)
    return result


def _models():
    common = dict(
        iterations=350,
        depth=7,
        learning_rate=0.035,
        l2_leaf_reg=5,
        random_seed=SEED,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )
    return (
        CatBoostRegressor(loss_function="MAE", **common),
        CatBoostClassifier(loss_function="Logloss", **common),
    )


def _fit_predict(
    training: pd.DataFrame,
    calibration: pd.DataFrame,
    holdout: pd.DataFrame,
    features: list[str],
    categorical: list[str],
):
    runs_model, wicket_model = _models()
    train_x = _frame(training, features, categorical)
    runs_model.fit(train_x, training["runs_in_over"], cat_features=categorical)
    wicket_model.fit(train_x, training["wicket_in_over"], cat_features=categorical)
    calibration_raw = np.asarray(
        runs_model.predict(_frame(calibration, features, categorical)), dtype=float
    )
    corrections = _calibrate_point(calibration, calibration_raw)
    calibration_runs = _apply_point(calibration, calibration_raw, corrections)
    offsets = _calibrate_fixed_range(calibration, calibration_runs)
    holdout_x = _frame(holdout, features, categorical)
    runs = _apply_point(
        holdout,
        np.asarray(runs_model.predict(holdout_x), dtype=float),
        corrections,
    )
    wickets = np.asarray(wicket_model.predict_proba(holdout_x)[:, 1], dtype=float)
    calibration_wickets = np.asarray(
        wicket_model.predict_proba(
            _frame(calibration, features, categorical)
        )[:, 1],
        dtype=float,
    )
    return (
        runs_model,
        wicket_model,
        runs,
        wickets,
        corrections,
        offsets,
        calibration_runs,
        calibration_wickets,
    )


def _choose_cold_blend(
    calibration: pd.DataFrame,
    baseline_runs: np.ndarray,
    enhanced_runs: np.ndarray,
    baseline_wickets: np.ndarray,
    enhanced_wickets: np.ndarray,
    offsets: dict,
) -> dict:
    """Select the cold-start blend on calibration data only."""
    cold = (
        (calibration["cold_start_status"] == "TEMP")
        & (calibration["striker_prior_balls"] < 120)
    ).to_numpy()
    if not cold.any():
        return {"alpha": 0.0, "calibration_rows": 0}
    base = _segment_metrics(
        calibration.loc[cold],
        baseline_runs[cold],
        baseline_wickets[cold],
        offsets,
    )
    choices = []
    for alpha in np.linspace(0.0, 1.0, 21):
        for point_offset in np.linspace(-0.6, 0.6, 25):
            runs = (
                baseline_runs
                + alpha * (enhanced_runs - baseline_runs)
                + point_offset
            )
            wickets = baseline_wickets + alpha * (
                enhanced_wickets - baseline_wickets
            )
            metrics = _segment_metrics(
                calibration.loc[cold], runs[cold], wickets[cold], offsets
            )
            choices.append(
                {
                    "alpha": float(alpha),
                    "point_offset": float(point_offset),
                    **metrics,
                }
            )
    eligible = [
        item
        for item in choices
        if item["mae"] <= base["mae"]
        and item["fixed_three_run_coverage"]
        >= base["fixed_three_run_coverage"]
    ]
    selected = min(
        eligible or [choices[0]],
        key=lambda item: (
            item["mae"],
            -item["fixed_three_run_coverage"],
            item["alpha"],
            abs(item["point_offset"]),
        ),
    )
    return {
        "alpha": selected["alpha"],
        "point_offset": selected["point_offset"],
        "calibration_rows": int(cold.sum()),
        "baseline": base,
        "selected": selected,
        "grid": choices,
    }


def _segment_metrics(data, runs, wickets, offsets):
    actual = data["runs_in_over"].to_numpy()
    low, high = _fixed_range(data, runs, offsets)
    return {
        "rows": len(data),
        "mae": float(mean_absolute_error(actual, runs)),
        "bias": float(np.mean(runs - actual)),
        "fixed_three_run_coverage": float(
            np.mean((actual >= low) & (actual <= high))
        ),
        "wicket_brier": float(
            brier_score_loss(data["wicket_in_over"].to_numpy(), wickets)
        ),
    }


def _range_coverage(
    data: pd.DataFrame, range_centres: np.ndarray, offsets: dict
) -> float:
    actual = data["runs_in_over"].to_numpy()
    low, high = _fixed_range(data, range_centres, offsets)
    return float(np.mean((actual >= low) & (actual <= high)))


def _calibrate_cold_range_shifts(
    calibration: pd.DataFrame,
    baseline_runs: np.ndarray,
    offsets: dict,
) -> dict[str, int]:
    """Choose phase-specific integer shifts using only 2024 cold-start rows."""
    cold = (
        (calibration["cold_start_status"] == "TEMP")
        & (calibration["striker_prior_balls"] < 120)
    ).to_numpy()
    shifts: dict[str, int] = {}
    for phase in sorted(calibration["phase"].astype(str).unique()):
        mask = cold & (calibration["phase"].astype(str).to_numpy() == phase)
        if int(mask.sum()) < 20:
            shifts[phase] = 0
            continue
        segment = calibration.loc[mask]
        choices = []
        for shift in range(-2, 3):
            coverage = _range_coverage(
                segment, baseline_runs[mask] + shift, offsets
            )
            choices.append((coverage, -abs(shift), -shift, shift))
        shifts[phase] = int(max(choices)[-1])
    return shifts


def _apply_cold_range_shifts(
    data: pd.DataFrame,
    baseline_runs: np.ndarray,
    cold_mask: np.ndarray,
    shifts: dict[str, int],
) -> np.ndarray:
    result = baseline_runs.copy()
    phase_shift = (
        data["phase"].astype(str).map(shifts).fillna(0).to_numpy(dtype=float)
    )
    result[cold_mask] += phase_shift[cold_mask]
    return result


def train(
    root: Path,
    *,
    dataset_path: Path | None = None,
    version: str = VERSION,
    features: list[str] | None = None,
    categorical: list[str] | None = None,
    training_end_year: int = 2023,
    calibration_year: int = 2024,
    holdout_years: tuple[int, ...] = (2025, 2026),
) -> dict:
    selected_features = features or FEATURES
    selected_categorical = categorical or CATEGORICAL_V3
    data = pd.read_csv(
        dataset_path
        or root / "data/candidates/ipl_cold_start_v3/training_overs.csv"
    )
    data["match_date"] = pd.to_datetime(data["match_date"])
    data = data.sort_values(["match_date", "source_file", "innings", "over"])
    training = data[
        data["match_date"].dt.year <= training_end_year
    ].copy()
    calibration = data[
        data["match_date"].dt.year == calibration_year
    ].copy()
    holdout = data[
        data["match_date"].dt.year.isin(holdout_years)
    ].copy()

    baseline = _fit_predict(
        training, calibration, holdout, MOE_FEATURES, CATEGORICAL
    )
    enhanced = _fit_predict(
        training,
        calibration,
        holdout,
        selected_features,
        selected_categorical,
    )
    baseline_runs, baseline_wickets = baseline[2], baseline[3]
    enhanced_runs, enhanced_wickets = enhanced[2], enhanced[3]
    baseline_metrics = _segment_metrics(
        holdout, baseline_runs, baseline_wickets, baseline[5]
    )
    enhanced_metrics = _segment_metrics(
        holdout, enhanced_runs, enhanced_wickets, enhanced[5]
    )
    blend = _choose_cold_blend(
        calibration,
        baseline[6],
        enhanced[6],
        baseline[7],
        enhanced[7],
        baseline[5],
    )
    alpha = float(blend["alpha"])
    point_offset = float(blend.get("point_offset", 0.0))
    cold_mask = (
        (holdout["cold_start_status"] == "TEMP")
        & (holdout["striker_prior_balls"] < 120)
    ).to_numpy()
    bowler_mask = holdout["cold_start_is_bowler"].astype(bool).to_numpy()
    hybrid_runs = baseline_runs.copy()
    hybrid_wickets = baseline_wickets.copy()
    hybrid_runs[cold_mask] = baseline_runs[cold_mask] + alpha * (
        enhanced_runs[cold_mask] - baseline_runs[cold_mask]
    ) + point_offset
    hybrid_wickets[cold_mask] = baseline_wickets[cold_mask] + alpha * (
        enhanced_wickets[cold_mask] - baseline_wickets[cold_mask]
    )
    hybrid_metrics = _segment_metrics(
        holdout, hybrid_runs, hybrid_wickets, baseline[5]
    )
    cold_range_shifts = _calibrate_cold_range_shifts(
        calibration, baseline[6], baseline[5]
    )
    hybrid_range_centres = _apply_cold_range_shifts(
        holdout, baseline_runs, cold_mask, cold_range_shifts
    )
    hybrid_metrics["fixed_three_run_coverage"] = _range_coverage(
        holdout, hybrid_range_centres, baseline[5]
    )

    def subset(mask, runs, wickets, offsets, range_centres=None):
        metrics = _segment_metrics(
            holdout.loc[mask], runs[mask], wickets[mask], offsets
        ) if mask.any() else {"rows": 0}
        if mask.any() and range_centres is not None:
            metrics["fixed_three_run_coverage"] = _range_coverage(
                holdout.loc[mask], range_centres[mask], offsets
            )
        return metrics

    report = {
        "candidate_version": version,
        "candidate_only": True,
        "production_changed": False,
        "leakage_safe": True,
        "rows": {
            "training": len(training),
            "calibration": len(calibration),
            "holdout": len(holdout),
        },
        "chronology": {
            "training_through": training_end_year,
            "calibration_year": calibration_year,
            "holdout_years": list(holdout_years),
        },
        "baseline_corrected_canonical": baseline_metrics,
        "enhanced_cold_start": enhanced_metrics,
        "gated_candidate": {
            **hybrid_metrics,
            "routing": (
                "calibration-selected blend only for TEMP players with "
                "<120 prior IPL balls"
            ),
            "enhanced_weight": alpha,
            "calibrated_point_offset": point_offset,
            "cold_range_phase_shifts": cold_range_shifts,
        },
        "calibration_only_route_selection": blend,
        "overall_changes": {
            key: hybrid_metrics[key] - baseline_metrics[key]
            for key in ("mae", "bias", "fixed_three_run_coverage", "wicket_brier")
        },
        "cold_start_only": {
            "definition": "TEMP and fewer than 120 prior IPL balls",
            "baseline": subset(
                cold_mask, baseline_runs, baseline_wickets, baseline[5]
            ),
            "enhanced": subset(
                cold_mask, enhanced_runs, enhanced_wickets, enhanced[5]
            ),
        },
        "bowler_safeguard": {
            "baseline": subset(
                bowler_mask, baseline_runs, baseline_wickets, baseline[5]
            ),
            "enhanced": subset(
                bowler_mask, enhanced_runs, enhanced_wickets, enhanced[5]
            ),
        },
        "features_added": [
            item for item in selected_features if item not in MOE_FEATURES
        ],
    }
    segment_masks = {
        "powerplay": (holdout["phase"] == "powerplay").to_numpy(),
        "new_batter_under_24_prior_balls": (
            holdout["striker_prior_balls"] < 24
        ).to_numpy(),
        "wicket_pressure_prior_over": (
            holdout["recent_wicket_rate"] > 0
        ).to_numpy(),
        "low_history_venue": (
            holdout["venue_prior_innings"] < 10
        ).to_numpy(),
    }
    report["supported_segments"] = {
        name: {
            "baseline": subset(
                mask, baseline_runs, baseline_wickets, baseline[5]
            ),
            "candidate": subset(
                mask,
                hybrid_runs,
                hybrid_wickets,
                baseline[5],
                hybrid_range_centres,
            ),
        }
        for name, mask in segment_masks.items()
    }
    report["temporal_holdout"] = {
        str(year): {
            "baseline": subset(
                (holdout["match_date"].dt.year == year).to_numpy(),
                baseline_runs,
                baseline_wickets,
                baseline[5],
            ),
            "candidate": subset(
                (holdout["match_date"].dt.year == year).to_numpy(),
                hybrid_runs,
                hybrid_wickets,
                baseline[5],
                hybrid_range_centres,
            ),
        }
        for year in sorted(holdout["match_date"].dt.year.unique())
    }
    replay_matches = (
        holdout[["source_file", "match_date"]]
        .drop_duplicates()
        .sort_values(["match_date", "source_file"])
        .tail(20)
    )
    report["leakage_safe_live_parity_replays"] = []
    for replay in replay_matches.itertuples():
        replay_mask = (
            holdout["source_file"].to_numpy() == replay.source_file
        )
        report["leakage_safe_live_parity_replays"].append(
            {
                "source_file": replay.source_file,
                "match_date": str(replay.match_date.date()),
                "baseline": subset(
                    replay_mask,
                    baseline_runs,
                    baseline_wickets,
                    baseline[5],
                ),
                "candidate": subset(
                    replay_mask,
                    hybrid_runs,
                    hybrid_wickets,
                    baseline[5],
                    hybrid_range_centres,
                ),
            }
        )
    cold_base = report["cold_start_only"]["baseline"].get("mae", float("inf"))
    cold_candidate = subset(
        cold_mask,
        hybrid_runs,
        hybrid_wickets,
        baseline[5],
        hybrid_range_centres,
    )
    report["cold_start_only"]["gated_candidate"] = cold_candidate
    cold_new = cold_candidate.get("mae", float("inf"))
    gates = {
        "overall_mae_improves": (
            hybrid_metrics["mae"] < baseline_metrics["mae"]
        ),
        "overall_coverage_not_lower": (
            hybrid_metrics["fixed_three_run_coverage"]
            >= baseline_metrics["fixed_three_run_coverage"]
        ),
        "cold_start_mae_improves": cold_new < cold_base,
        "cold_start_coverage_not_lower": (
            cold_candidate["fixed_three_run_coverage"]
            >= report["cold_start_only"]["baseline"][
                "fixed_three_run_coverage"
            ]
        ),
        "wicket_brier_not_worse": (
            hybrid_metrics["wicket_brier"]
            <= baseline_metrics["wicket_brier"]
        ),
    }
    for year, result in report["temporal_holdout"].items():
        gates[f"{year}_mae_improves"] = (
            result["candidate"]["mae"] < result["baseline"]["mae"]
        )
        gates[f"{year}_coverage_not_lower"] = (
            result["candidate"]["fixed_three_run_coverage"]
            >= result["baseline"]["fixed_three_run_coverage"]
        )
    report["promotion_gates"] = gates
    report["decision"] = (
        "promote_candidate"
        if all(gates.values())
        else "reject_keep_research"
    )
    output = root / f"models/candidates/{version}"
    output.mkdir(parents=True, exist_ok=True)
    enhanced[0].save_model(output / "runs.cbm")
    enhanced[1].save_model(output / "wicket.cbm")
    baseline[0].save_model(output / "baseline_runs.cbm")
    baseline[1].save_model(output / "baseline_wicket.cbm")
    pd.DataFrame(
        {
            "source_file": holdout["source_file"],
            "match_date": holdout["match_date"].astype(str),
            "innings": holdout["innings"],
            "over": holdout["over"],
            "actual": holdout["runs_in_over"],
            "cold_start": cold_mask.astype(int),
            "baseline_prediction": baseline_runs,
            "enhanced_prediction": enhanced_runs,
            "gated_prediction": hybrid_runs,
            "gated_range_centre": hybrid_range_centres,
        }
    ).to_csv(output / "holdout_predictions.csv", index=False)
    (output / "validation_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report))
    return report


if __name__ == "__main__":
    train(Path(__file__).resolve().parents[1])
