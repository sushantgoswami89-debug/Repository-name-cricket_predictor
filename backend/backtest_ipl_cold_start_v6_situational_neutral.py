"""Evaluate a neutral, situation-led state machine for unknown IPL batters."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from train_ipl_catboost_phase_moe_v1 import CATEGORICAL, MOE_FEATURES
from train_ipl_cold_start_v3 import (
    _apply_cold_range_shifts,
    _calibrate_cold_range_shifts,
    _fit_predict,
    _range_coverage,
    _segment_metrics,
)


VERSION = "ipl_cold_start_v6_situational_neutral"
SURVIVAL_BALLS = 12
MIN_CALIBRATION_ROWS = 20


def _state(data: pd.DataFrame) -> np.ndarray:
    unknown = (
        (data["cold_start_status"] == "TEMP")
        & (data["t20_prior_balls"] == 0)
        & (data["striker_prior_balls"] < 120)
        & (data["cold_start_is_bowler"] == 0)
    ).to_numpy()
    survived = (data["striker_match_balls"] >= SURVIVAL_BALLS).to_numpy()
    tense = (
        data["chase_pressure"].isin(["medium", "high"])
        | (data["recent_wicket_rate"] > 0)
        | (data["state_regime"] == "wicket_pressure")
    ).to_numpy()
    return np.select(
        [
            unknown & survived,
            unknown & ~survived & tense,
            unknown & ~survived & ~tense,
        ],
        ["SURVIVED_HIGH_VARIANCE", "ENTRY_TENSE", "ENTRY_NEUTRAL"],
        default="BASELINE",
    )


def _learn_adjustments(
    calibration: pd.DataFrame,
    baseline_runs: np.ndarray,
    baseline_wickets: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    states = _state(calibration)
    adjustments = {}
    for state in ("ENTRY_TENSE", "ENTRY_NEUTRAL", "SURVIVED_HIGH_VARIANCE"):
        mask = states == state
        rows = int(mask.sum())
        run_residual = (
            float(
                np.mean(
                    calibration.loc[mask, "runs_in_over"].to_numpy()
                    - baseline_runs[mask]
                )
            )
            if rows >= MIN_CALIBRATION_ROWS and state == "SURVIVED_HIGH_VARIANCE"
            else 0.0
        )
        wicket_residual = (
            float(
                np.mean(calibration.loc[mask, "wicket_in_over"].to_numpy())
                - np.mean(baseline_wickets[mask])
            )
            if rows >= MIN_CALIBRATION_ROWS
            else 0.0
        )
        adjustments[state] = {
            "rows": rows,
            "run_residual": run_residual,
            "wicket_probability_residual": wicket_residual,
        }
    return adjustments


def _apply(
    data: pd.DataFrame,
    baseline_runs: np.ndarray,
    baseline_wickets: np.ndarray,
    adjustments: dict[str, dict[str, float | int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    states = _state(data)
    runs = baseline_runs.copy()
    wickets = baseline_wickets.copy()
    for state, values in adjustments.items():
        mask = states == state
        runs[mask] += float(values["run_residual"])
        wickets[mask] = np.clip(
            wickets[mask] + float(values["wicket_probability_residual"]),
            0.0,
            1.0,
        )
    return runs, wickets, states


def run(root: Path) -> dict:
    data = pd.read_csv(
        root / "data/candidates/ipl_cold_start_v5_t20_priors/training_overs.csv"
    )
    data["match_date"] = pd.to_datetime(data["match_date"])
    data = data.sort_values(["match_date", "source_file", "innings", "over"])
    training = data[data["match_date"].dt.year <= 2023].copy()
    calibration = data[data["match_date"].dt.year == 2024].copy()
    holdout = data[data["match_date"].dt.year.isin([2025, 2026])].copy()
    baseline = _fit_predict(
        training, calibration, holdout, MOE_FEATURES, CATEGORICAL
    )
    adjustments = _learn_adjustments(
        calibration, baseline[6], baseline[7]
    )
    runs, wickets, states = _apply(
        holdout, baseline[2], baseline[3], adjustments
    )
    routed = states != "BASELINE"
    shifts = _calibrate_cold_range_shifts(
        calibration, baseline[6], baseline[5]
    )
    range_centres = _apply_cold_range_shifts(
        holdout, baseline[2], routed, shifts
    )

    def metrics(mask, point, wicket, centres):
        frame = holdout.loc[mask]
        result = _segment_metrics(
            frame, point[mask], wicket[mask], baseline[5]
        )
        result["fixed_three_run_coverage"] = _range_coverage(
            frame, centres[mask], baseline[5]
        )
        return result

    all_rows = np.ones(len(holdout), dtype=bool)
    baseline_metrics = metrics(
        all_rows, baseline[2], baseline[3], baseline[2]
    )
    candidate_metrics = metrics(all_rows, runs, wickets, range_centres)
    state_report = {}
    for state in ("ENTRY_TENSE", "ENTRY_NEUTRAL", "SURVIVED_HIGH_VARIANCE"):
        mask = states == state
        state_report[state] = {
            "rows": int(mask.sum()),
            "baseline": metrics(mask, baseline[2], baseline[3], baseline[2]),
            "candidate": metrics(mask, runs, wickets, range_centres),
        }
    temporal = {}
    for year in (2025, 2026):
        mask = (holdout["match_date"].dt.year == year).to_numpy()
        temporal[str(year)] = {
            "baseline": metrics(mask, baseline[2], baseline[3], baseline[2]),
            "candidate": metrics(mask, runs, wickets, range_centres),
        }
    gates = {
        "overall_mae_improves": candidate_metrics["mae"] < baseline_metrics["mae"],
        "overall_coverage_not_lower": (
            candidate_metrics["fixed_three_run_coverage"]
            >= baseline_metrics["fixed_three_run_coverage"]
        ),
        "wicket_brier_not_worse": (
            candidate_metrics["wicket_brier"]
            <= baseline_metrics["wicket_brier"]
        ),
    }
    for year, values in temporal.items():
        gates[f"{year}_mae_not_worse"] = (
            values["candidate"]["mae"] <= values["baseline"]["mae"]
        )
        gates[f"{year}_coverage_not_lower"] = (
            values["candidate"]["fixed_three_run_coverage"]
            >= values["baseline"]["fixed_three_run_coverage"]
        )
        gates[f"{year}_wicket_brier_not_worse"] = (
            values["candidate"]["wicket_brier"]
            <= values["baseline"]["wicket_brier"]
        )
    report = {
        "candidate_version": VERSION,
        "candidate_only": True,
        "production_changed": False,
        "leakage_safe": True,
        "policy": {
            "unknown_players_have_numeric_player_priors": False,
            "entry_runs": "unchanged neutral baseline",
            "survival_threshold_legal_balls_faced": SURVIVAL_BALLS,
            "adjustments_learned_from": "2024 calibration only",
            "specialist_bowler_safeguard": True,
        },
        "calibration_adjustments": adjustments,
        "range_phase_shifts": shifts,
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "states": state_report,
        "temporal_holdout": temporal,
        "promotion_gates": gates,
        "decision": (
            "promote_candidate" if all(gates.values()) else "reject_keep_research"
        ),
    }
    output = root / f"models/candidates/{VERSION}"
    output.mkdir(parents=True, exist_ok=True)
    (output / "validation_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    pd.DataFrame(
        {
            "source_file": holdout["source_file"],
            "match_date": holdout["match_date"].astype(str),
            "innings": holdout["innings"],
            "over": holdout["over"],
            "state": states,
            "actual_runs": holdout["runs_in_over"],
            "actual_wicket": holdout["wicket_in_over"],
            "baseline_runs": baseline[2],
            "candidate_runs": runs,
            "baseline_wicket": baseline[3],
            "candidate_wicket": wickets,
            "range_centre": range_centres,
        }
    ).to_csv(output / "holdout_predictions.csv", index=False)
    print(json.dumps(report))
    return report


if __name__ == "__main__":
    run(Path(__file__).resolve().parents[1])
