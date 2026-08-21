"""Replay every unseen complete match through Candidate v3 without promotion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, mean_absolute_error


def run_complete_match_test(project_root: Path, candidate_dir: Path) -> dict[str, Any]:
    dataset_file = project_root / "data/candidates/v3/verified_training_overs.csv"
    data = pd.read_csv(dataset_file)
    data["match_date"] = pd.to_datetime(data["match_date"])
    holdout = data[data["match_date"].dt.year >= 2025].copy()
    features: list[str] = joblib.load(candidate_dir / "feature_cols.pkl")
    categorical: list[str] = joblib.load(candidate_dir / "cat_cols.pkl")
    runs_model = joblib.load(candidate_dir / "runs_model.pkl")
    wicket_model = joblib.load(candidate_dir / "wkt_model.pkl")

    x = holdout[features].copy()
    for column in categorical:
        x[column] = x[column].astype("category")
    holdout["predicted_runs"] = np.clip(runs_model.predict(x), 0, None)
    holdout["wicket_probability"] = wicket_model.predict_proba(x)[:, 1]

    failures: list[dict[str, Any]] = []
    match_rows: list[dict[str, Any]] = []
    for source_file, match in holdout.groupby("source_file", sort=True):
        sequence_ok = True
        for innings_number, innings in match.groupby("innings", sort=True):
            innings = innings.sort_values("over")
            overs = innings["over"].astype(int).tolist()
            if any(b <= a for a, b in zip(overs, overs[1:], strict=False)):
                sequence_ok = False
                failures.append(
                    {
                        "source_file": source_file,
                        "innings": int(innings_number),
                        "error": "Over sequence is not strictly increasing.",
                    }
                )
            scores = innings["score_before_over"].astype(int).tolist()
            actuals = innings["runs_in_over"].astype(int).tolist()
            for index in range(1, len(scores)):
                if scores[index] != scores[index - 1] + actuals[index - 1]:
                    sequence_ok = False
                    failures.append(
                        {
                            "source_file": source_file,
                            "innings": int(innings_number),
                            "over": overs[index],
                            "error": (
                                "Pre-over score does not reconcile with prior over."
                            ),
                        }
                    )
        match_rows.append(
            {
                "source_file": source_file,
                "rows": len(match),
                "innings": int(match["innings"].nunique()),
                "sequence_verified": sequence_ok,
                "runs_mae": float(
                    mean_absolute_error(match["runs_in_over"], match["predicted_runs"])
                ),
                "wicket_brier": float(
                    brier_score_loss(
                        match["wicket_in_over"], match["wicket_probability"]
                    )
                ),
            }
        )

    per_match = pd.DataFrame(match_rows)
    report: dict[str, Any] = {
        "candidate_version": "v3_phase_impact_time_split",
        "scope": "all_complete_unseen_matches_2025_plus",
        "matches_tested": int(len(per_match)),
        "innings_tested": int(
            holdout[["source_file", "innings"]].drop_duplicates().shape[0]
        ),
        "overs_predicted": int(len(holdout)),
        "sequence_failures": len(failures),
        "runs_mae": float(
            mean_absolute_error(holdout["runs_in_over"], holdout["predicted_runs"])
        ),
        "wicket_brier": float(
            brier_score_loss(holdout["wicket_in_over"], holdout["wicket_probability"])
        ),
        "match_runs_mae_p95": float(per_match["runs_mae"].quantile(0.95)),
        "match_wicket_brier_p95": float(per_match["wicket_brier"].quantile(0.95)),
        "failures": failures,
    }
    report["gates"] = {
        "at_least_100_complete_matches": report["matches_tested"] >= 100,
        "zero_sequence_failures": report["sequence_failures"] == 0,
        "all_predictions_finite": bool(
            np.isfinite(
                holdout[["predicted_runs", "wicket_probability"]].to_numpy()
            ).all()
        ),
        "probabilities_bounded": bool(
            holdout["wicket_probability"].between(0, 1).all()
        ),
    }
    output_dir = project_root / "data/reports/candidate_v3"
    output_dir.mkdir(parents=True, exist_ok=True)
    per_match.to_csv(output_dir / "complete_match_metrics.csv", index=False)
    (output_dir / "complete_match_summary.json").write_text(
        json.dumps(report, indent=2)
    )

    validation_file = candidate_dir / "validation_report.json"
    validation = json.loads(validation_file.read_text())
    validation["complete_match_test"] = report
    all_gates = list(validation["promotion_gates"].values()) + list(
        report["gates"].values()
    )
    validation["promotion_recommended"] = all(all_gates)
    validation["production_models_changed"] = False
    validation_file.write_text(json.dumps(validation, indent=2))
    return report


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    print(
        json.dumps(
            run_complete_match_test(
                root, root / "models/candidates/v3_phase_impact_time_split"
            ),
            indent=2,
        )
    )
