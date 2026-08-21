"""Train ranked-country Candidate v3 variants and lock the best safe release."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.ml.calibrated_model import CalibratedBinaryClassifier
from train_validate_candidate_v3 import (
    CATEGORICAL_FEATURES,
    V3_FEATURES,
    _features,
    _fit,
    _intervals,
    _metrics,
)

RANKING_DATE = "2026-06-20"
TOP_12 = (
    "India",
    "England",
    "Australia",
    "New Zealand",
    "South Africa",
    "Pakistan",
    "West Indies",
    "Bangladesh",
    "Sri Lanka",
    "Afghanistan",
    "Zimbabwe",
    "Ireland",
)
TOP_10 = TOP_12[:10]


def _source_teams(project_root: Path) -> dict[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}
    for path in (project_root / "data/raw/cricsheet/t20i").glob("*.json"):
        raw = json.loads(path.read_text(encoding="utf-8"))
        result[path.name] = tuple(raw.get("info", {}).get("teams", ()))
    return result


def _scope_dataset(
    full: pd.DataFrame,
    teams_by_file: dict[str, tuple[str, ...]],
    allowed: tuple[str, ...],
) -> pd.DataFrame:
    allowed_set = set(allowed)
    eligible_files = {
        source_file
        for source_file, teams in teams_by_file.items()
        if len(teams) == 2 and set(teams) <= allowed_set
    }
    return full[full["source_file"].isin(eligible_files)].copy()


def _save_models(
    output_dir: Path,
    runs: Any,
    raw_wickets: Any,
    calibrator: Any,
    profiles: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(runs, output_dir / "runs_model.pkl")
    joblib.dump(
        CalibratedBinaryClassifier(raw_wickets, calibrator),
        output_dir / "wkt_model.pkl",
    )
    joblib.dump(raw_wickets, output_dir / "wicket_model_raw_candidate.pkl")
    joblib.dump(calibrator, output_dir / "wicket_calibrator_candidate.pkl")
    joblib.dump(V3_FEATURES, output_dir / "feature_cols.pkl")
    joblib.dump(CATEGORICAL_FEATURES, output_dir / "cat_cols.pkl")
    (output_dir / "interval_profiles.json").write_text(
        json.dumps(profiles, indent=2), encoding="utf-8"
    )


def _train_scope(
    data: pd.DataFrame,
    scope: str,
    allowed: tuple[str, ...],
    output_dir: Path,
) -> dict[str, Any]:
    train = data[data["match_date"].dt.year <= 2023]
    calibration = data[data["match_date"].dt.year == 2024]
    holdout = data[data["match_date"].dt.year >= 2025]
    if min(len(train), len(calibration), len(holdout)) == 0:
        raise ValueError(f"{scope} has an empty chronological split.")

    runs, wickets, calibrator = _fit(train, calibration, V3_FEATURES)
    metrics, holdout_runs, holdout_wickets = _metrics(
        holdout, runs, wickets, calibrator, V3_FEATURES
    )
    calibration_runs = np.clip(
        runs.predict(_features(calibration, V3_FEATURES)), 0, None
    )
    profiles, intervals = _intervals(
        calibration, calibration_runs, holdout, holdout_runs
    )
    holdout = holdout.copy()
    holdout["predicted_runs"] = holdout_runs
    holdout["wicket_probability"] = holdout_wickets

    phase_metrics: dict[str, Any] = {}
    for phase in ("powerplay", "middle", "death"):
        segment = holdout[holdout["phase"] == phase]
        values, _, _ = _metrics(segment, runs, wickets, calibrator, V3_FEATURES)
        phase_metrics[phase] = values

    failures: list[dict[str, Any]] = []
    for source_file, match in holdout.groupby("source_file"):
        for innings_number, innings in match.groupby("innings"):
            innings = innings.sort_values("over")
            scores = innings["score_before_over"].astype(int).tolist()
            actuals = innings["runs_in_over"].astype(int).tolist()
            for index in range(1, len(scores)):
                if scores[index] != scores[index - 1] + actuals[index - 1]:
                    failures.append(
                        {
                            "source_file": source_file,
                            "innings": int(innings_number),
                            "over": int(innings.iloc[index]["over"]),
                        }
                    )

    _save_models(output_dir, runs, wickets, calibrator, profiles)
    report = {
        "candidate_version": f"v3_phase_impact_{scope}",
        "ranking_snapshot": RANKING_DATE,
        "allowed_teams": list(allowed),
        "matches": int(data["source_file"].nunique()),
        "rows": len(data),
        "split": {
            "train_through_2023": len(train),
            "calibration_2024": len(calibration),
            "holdout_2025_plus": len(holdout),
            "holdout_matches": int(holdout["source_file"].nunique()),
        },
        "candidate_metrics": metrics,
        "phase_metrics": phase_metrics,
        "prediction_interval_90": intervals,
        "complete_match_test": {
            "matches_tested": int(holdout["source_file"].nunique()),
            "innings_tested": int(
                holdout[["source_file", "innings"]].drop_duplicates().shape[0]
            ),
            "overs_predicted": len(holdout),
            "sequence_failures": len(failures),
            "failures": failures,
        },
        "production_models_changed": False,
    }
    (output_dir / "validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _lock_release(
    project_root: Path,
    winner_dir: Path,
    winner: dict[str, Any],
    selection: dict[str, Any],
) -> Path:
    release_dir = project_root / "models/locked/cricketbaba_candidate_v3"
    if release_dir.exists():
        raise FileExistsError(
            f"Locked release already exists and will not be overwritten: {release_dir}"
        )
    shutil.copytree(winner_dir, release_dir)
    manifest = {
        "release": "cricketbaba_candidate_v3",
        "locked_at": "2026-07-22",
        "selected_candidate": winner["candidate_version"],
        "ranking_snapshot": winner.get("ranking_snapshot"),
        "production_promoted": False,
        "selection": selection,
        "files": {
            path.name: _sha256(path)
            for path in sorted(release_dir.iterdir())
            if path.is_file()
        },
    }
    (release_dir / "LOCKED_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    for path in release_dir.iterdir():
        if path.is_file():
            path.chmod(0o444)
    return release_dir


def train_compare_and_lock(project_root: Path) -> dict[str, Any]:
    source = project_root / "data/candidates/v3/verified_training_overs.csv"
    full = pd.read_csv(source)
    full["match_date"] = pd.to_datetime(full["match_date"])
    teams_by_file = _source_teams(project_root)
    candidate_root = project_root / "models/candidates"

    reports: dict[str, dict[str, Any]] = {}
    directories: dict[str, Path] = {}
    for name, allowed in (("ranked_top10", TOP_10), ("ranked_top12", TOP_12)):
        scoped = _scope_dataset(full, teams_by_file, allowed)
        output_dir = candidate_root / f"v3_phase_impact_{name}"
        reports[name] = _train_scope(scoped, name, allowed, output_dir)
        directories[name] = output_dir

    unrestricted_dir = candidate_root / "v3_phase_impact_time_split"
    unrestricted = json.loads(
        (unrestricted_dir / "validation_report.json").read_text(encoding="utf-8")
    )
    options = {
        "unrestricted": unrestricted,
        **reports,
    }
    eligible: list[str] = []
    for name, report in reports.items():
        metrics = report["candidate_metrics"]
        complete = report["complete_match_test"]
        if (
            metrics["runs_mae"] < unrestricted["candidate_metrics"]["runs_mae"]
            and metrics["wicket_brier"]
            < unrestricted["candidate_metrics"]["wicket_brier"]
            and report["prediction_interval_90"]["coverage"] >= 0.90
            and complete["sequence_failures"] == 0
            and complete["matches_tested"] >= 100
        ):
            eligible.append(name)

    if eligible:
        winner_name = min(
            eligible,
            key=lambda name: (
                reports[name]["candidate_metrics"]["runs_mae"],
                reports[name]["candidate_metrics"]["wicket_brier"],
            ),
        )
        winner = reports[winner_name]
        winner_dir = directories[winner_name]
        reason = "Ranked scope beat unrestricted Candidate v3 on both primary metrics."
    else:
        winner_name = "unrestricted"
        winner = unrestricted
        winner_dir = unrestricted_dir
        reason = (
            "No ranked scope passed both improvement gates; retained safer benchmark."
        )

    comparison = {}
    for name, report in options.items():
        rows = report.get("rows")
        if rows is None:
            rows = report["dataset"]["training_rows"]
        comparison[name] = {
            "rows": rows,
            "holdout_rows": report["split"]["holdout_2025_plus"],
            "runs_mae": report["candidate_metrics"]["runs_mae"],
            "wicket_brier": report["candidate_metrics"]["wicket_brier"],
        }
    selection = {
        "winner": winner_name,
        "reason": reason,
        "eligible_ranked_candidates": eligible,
        "comparison": comparison,
    }
    release_dir = _lock_release(project_root, winner_dir, winner, selection)
    result = {
        **selection,
        "locked_release": str(release_dir),
        "production_models_changed": False,
    }
    report_dir = project_root / "data/reports/candidate_v3"
    (report_dir / "ranked_scope_selection.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    print(json.dumps(train_compare_and_lock(root), indent=2))
