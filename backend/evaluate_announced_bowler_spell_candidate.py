"""Fail-closed live-parity evaluation for the announced-bowler candidate.

Historical delivery data is not evidence that a bowler was visible before ball
one.  Only rows in an independently captured announcement ledger may enter the
announced cohort.  With no such ledger, the candidate must equal the situation
baseline and the report records that the experiment is not evaluable.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]
KEYS = ["source_file", "match_date", "innings", "over"]
LEDGER_COLUMNS = KEYS + [
    "bowler",
    "bowler_source",
    "scoreboard_over",
    "row_is_empty",
    "captured_before_first_ball",
]


def _load_ledger(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=LEDGER_COLUMNS)
    ledger = pd.read_csv(path)
    missing = sorted(set(LEDGER_COLUMNS) - set(ledger.columns))
    if missing:
        raise ValueError(f"announcement ledger is missing columns: {missing}")
    if ledger.duplicated(KEYS).any():
        raise ValueError("announcement ledger has duplicate over keys")
    accepted = (
        ledger["bowler"].fillna("").astype(str).str.strip().ne("")
        & ledger["bowler_source"].eq("scoreboard_pre_over")
        & ledger["scoreboard_over"].eq(ledger["over"])
        & ledger["row_is_empty"].eq(True)
        & ledger["captured_before_first_ball"].eq(True)
    )
    invalid_claims = ledger["bowler_source"].eq("scoreboard_pre_over") & ~accepted
    if invalid_claims.any():
        raise ValueError(
            "scoreboard_pre_over rows must be numbered, empty, pre-ball rows "
            "with a non-empty bowler"
        )
    return ledger.loc[accepted].copy()


def _fixed_range(prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    low = np.maximum(0, np.floor(prediction).astype(int) - 1)
    return low, low + 3


def _calibration(actual: np.ndarray, probability: np.ndarray) -> list[dict]:
    bins = pd.cut(
        probability,
        bins=[0, 0.15, 0.25, 0.35, 0.45, 0.60, 1.0],
        include_lowest=True,
        duplicates="drop",
    )
    frame = pd.DataFrame({"actual": actual, "probability": probability, "bin": bins})
    result = []
    for label, group in frame.groupby("bin", observed=True):
        result.append(
            {
                "bin": str(label),
                "rows": len(group),
                "mean_probability": float(group["probability"].mean()),
                "event_rate": float(group["actual"].mean()),
            }
        )
    return result


def _metrics(
    frame: pd.DataFrame,
    *,
    run_column: str,
    wicket_column: str,
    alert_count: int,
) -> dict:
    actual_runs = frame["runs_in_over"].to_numpy()
    predicted_runs = frame[run_column].to_numpy()
    actual_wicket = frame["wicket_in_over"].to_numpy()
    probability = frame[wicket_column].to_numpy()
    low, high = _fixed_range(predicted_runs)
    alerts = np.zeros(len(frame), dtype=bool)
    if alert_count:
        top = np.argsort(-probability, kind="stable")[: min(alert_count, len(frame))]
        alerts[top] = True
    result = {
        "rows": len(frame),
        "run_mae": float(np.mean(np.abs(predicted_runs - actual_runs))),
        "run_bias": float(np.mean(predicted_runs - actual_runs)),
        "fixed_width_3_run_coverage": float(
            np.mean((actual_runs >= low) & (actual_runs <= high))
        ),
        "wicket_brier": float(brier_score_loss(actual_wicket, probability)),
        "wicket_roc_auc": (
            float(roc_auc_score(actual_wicket, probability))
            if len(np.unique(actual_wicket)) == 2
            else None
        ),
        "wicket_pr_auc": (
            float(average_precision_score(actual_wicket, probability))
            if len(np.unique(actual_wicket)) == 2
            else None
        ),
        "alert_count": int(alerts.sum()),
        "alert_volume": float(alerts.mean()),
        "precision_at_existing_alert_volume": float(
            precision_score(actual_wicket, alerts, zero_division=0)
        ),
        "recall_at_existing_alert_volume": float(
            recall_score(actual_wicket, alerts, zero_division=0)
        ),
        "wicket_calibration": _calibration(actual_wicket, probability),
    }
    return result


def evaluate(root: Path = ROOT, ledger_path: Path | None = None) -> dict:
    ledger_path = ledger_path or (
        root / "data/replays/pre_over_bowler_announcements.csv"
    )
    ledger = _load_ledger(ledger_path)
    situation = pd.read_csv(
        root / "data/candidates/ipl_cold_start_v5_t20_priors/training_overs.csv"
    )
    situation["match_date"] = situation["match_date"].astype(str)
    situation = situation[
        pd.to_datetime(situation["match_date"]).dt.year.isin([2025, 2026])
    ].copy()
    runs = pd.read_csv(
        root
        / "models/candidates/ipl_cold_start_v5_t20_priors/holdout_predictions.csv"
    ).rename(
        columns={
            "actual": "recorded_runs",
            "gated_prediction": "situation_runs",
        }
    )
    wicket = pd.read_csv(
        root / "models/candidates/ipl_wicket_v7_context/holdout_predictions.csv"
    ).rename(
        columns={
            "actual": "recorded_wicket",
            "candidate_probability": "situation_wicket",
        }
    )
    for frame in (runs, wicket):
        frame["match_date"] = frame["match_date"].astype(str)
    data = situation.merge(
        runs[KEYS + ["situation_runs"]], on=KEYS, validate="one_to_one"
    ).merge(
        wicket[KEYS + ["situation_wicket", "candidate_alert"]],
        on=KEYS,
        validate="one_to_one",
    )
    ledger_keys = ledger[KEYS].copy()
    ledger_keys["announced"] = True
    data = data.merge(ledger_keys, on=KEYS, how="left", validate="one_to_one")
    data["announced"] = data["announced"].fillna(False).astype(bool)

    # No residual model is fit unless genuine announced rows exist in the
    # chronological training period. This preserves exact fallback parity.
    data["candidate_runs"] = data["situation_runs"]
    data["candidate_wicket"] = data["situation_wicket"]
    alert_count = int(data["candidate_alert"].sum())

    segment_masks = {
        "overall": np.ones(len(data), dtype=bool),
        "period_2025": pd.to_datetime(data["match_date"]).dt.year.eq(2025).to_numpy(),
        "period_2026": pd.to_datetime(data["match_date"]).dt.year.eq(2026).to_numpy(),
        "announced": data["announced"].to_numpy(),
        "unannounced": ~data["announced"].to_numpy(),
        "established_partnership": data["partnership_legal_ball_age"].ge(24).to_numpy(),
        "powerplay": data["phase"].eq("powerplay").to_numpy(),
        "middle": data["phase"].eq("middle").to_numpy(),
        "death": data["phase"].eq("death").to_numpy(),
    }
    segments = {}
    for name, mask in segment_masks.items():
        segment = data.loc[mask]
        if segment.empty:
            segments[name] = {
                "rows": 0,
                "status": "unavailable_without_genuine_pre_over_announcement_rows",
            }
            continue
        segment_alert_count = round(alert_count * len(segment) / len(data))
        segments[name] = {
            "baseline": _metrics(
                segment,
                run_column="situation_runs",
                wicket_column="situation_wicket",
                alert_count=segment_alert_count,
            ),
            "candidate": _metrics(
                segment,
                run_column="candidate_runs",
                wicket_column="candidate_wicket",
                alert_count=segment_alert_count,
            ),
        }

    unavailable = {
        name: {
            "rows": 0,
            "status": "unavailable_without_genuine_pre_over_announcement_rows",
        }
        for name in (
            "new_spells",
            "returning_spells",
            "supported_bowler_history",
            "unsupported_bowler_history",
        )
    }
    segments.update(unavailable)
    report = {
        "candidate_version": "announced_bowler_current_spell_v1",
        "candidate_only": True,
        "production_changed": False,
        "availability_rule": (
            "explicitly numbered empty upcoming-over scoreboard row captured "
            "before ball one"
        ),
        "ledger_path": str(ledger_path),
        "genuine_announced_rows": len(ledger),
        "first_delivery_identity_used": False,
        "candidate_fit": "not_fit_insufficient_announcement_evidence",
        "segments": segments,
        "promotion_gates": {
            "overall_wicket_brier_improves": False,
            "2025_wicket_brier_improves": False,
            "2026_wicket_brier_improves": False,
            "precision_improves_same_alert_volume": False,
            "run_mae_non_regression": True,
            "fixed_width_3_coverage_non_regression": True,
            "evaluable_announced_cohort": False,
        },
        "decision": "reject_keep_research",
        "blocking_evidence": (
            "No historical pre-over scoreboard announcement ledger is present. "
            "Cricsheet first-delivery bowler identity is intentionally excluded."
        ),
    }
    destination = (
        root / "data/reports/announced_bowler_current_spell_v1/report.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
