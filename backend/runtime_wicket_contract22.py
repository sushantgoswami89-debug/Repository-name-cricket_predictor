"""Callable runtime for contract22_wicket_rigorous -- the best-validated
wicket-probability candidate found this session (Brier skill score 3.54%
vs. the currently-live models/wkt_model.pkl's own naive estimate of
1.95%; see docs/candidate_ipl_wicket_v7_2_spell_features.md's 2026-08-22
follow-up). Mirrors runtime_run_range_v3.py's pattern: artifact integrity
verification, feature-contract validation, Platt-calibrated probability
output. Sources its features from WicketContract22FeatureComputer
(app/ml/wicket_contract22_features.py) instead of requiring the caller to
assemble a feature frame by hand.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from app.ml.model_repository import ModelRepository
from app.ml.wicket_contract22_features import WicketContract22FeatureComputer

REQUIRED_ARTIFACTS = (
    "wicket_model.pkl",
    "wicket_calibrator.pkl",
    "feature_cols.pkl",
    "categorical_cols.pkl",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _phase(over: int) -> str:
    return "powerplay" if over <= 6 else ("death" if over >= 16 else "middle")


class WicketRuntimeContract22:
    """Load contract22 artifacts + live snapshots, return calibrated wicket probability."""

    def __init__(
        self,
        artifact_dir: Path | str,
        *,
        project_root: Path | None = None,
        verify_integrity: bool = True,
        feature_computer: WicketContract22FeatureComputer | None = None,
    ) -> None:
        self.artifact_dir = Path(artifact_dir)
        if not self.artifact_dir.is_dir():
            raise FileNotFoundError(f"Artifact directory not found: {self.artifact_dir}")

        missing = [name for name in REQUIRED_ARTIFACTS if not (self.artifact_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing contract22 wicket artifacts: {', '.join(missing)}")
        if verify_integrity:
            self._verify_integrity()

        repository = ModelRepository(self.artifact_dir)
        self._model: Any = repository.load_artifact("wicket_model.pkl")
        self._calibrator: Any = repository.load_artifact("wicket_calibrator.pkl")
        self._features: list[str] = repository.load_artifact("feature_cols.pkl")
        self._categorical: list[str] = repository.load_artifact("categorical_cols.pkl")
        self._computer = feature_computer or WicketContract22FeatureComputer(project_root)

    def _verify_integrity(self) -> None:
        manifest_path = self.artifact_dir / "ARTIFACT_MANIFEST.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Integrity manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = manifest.get("artifacts", {})
        for name in REQUIRED_ARTIFACTS:
            if name not in expected:
                raise ValueError(f"Integrity manifest has no digest for {name}.")
            actual = _sha256(self.artifact_dir / name)
            if actual != expected[name]:
                raise ValueError(f"Artifact integrity check failed for {name}.")

    def predict_wicket_probability(
        self,
        *,
        registry: Mapping[str, str],
        striker_name: str,
        bowler_name: str,
        venue_name: str,
        over: int,
        non_striker_name: str = "",
        deliveries: Any = (),
        score_before_over: int,
        wkts_down_before_over: int,
        wickets_in_hand: int,
        legal_balls_bowled: int,
        balls_remaining: int,
        current_run_rate: float,
        is_chase: bool,
        runs_required: int,
        required_run_rate: float,
        recent_legal_balls: int,
        recent_runs_per_ball: float,
        recent_dot_rate: float,
        recent_single_rate: float,
        recent_boundary_rate: float,
        recent_wicket_rate: float,
    ) -> float:
        phase = _phase(over)
        enriched = self._computer.compute(
            registry=registry, striker_name=striker_name, non_striker_name=non_striker_name,
            bowler_name=bowler_name, venue_name=venue_name, phase=phase, deliveries=deliveries,
        )
        row = {
            "over": over, "score_before_over": score_before_over,
            "wkts_down_before_over": wkts_down_before_over, "phase": phase,
            "wickets_in_hand": wickets_in_hand, "legal_balls_bowled": legal_balls_bowled,
            "balls_remaining": balls_remaining, "current_run_rate": current_run_rate,
            "is_chase": int(is_chase), "runs_required": runs_required,
            "required_run_rate": required_run_rate, "recent_legal_balls": recent_legal_balls,
            "recent_runs_per_ball": recent_runs_per_ball, "recent_dot_rate": recent_dot_rate,
            "recent_single_rate": recent_single_rate, "recent_boundary_rate": recent_boundary_rate,
            "recent_wicket_rate": recent_wicket_rate,
            **enriched,
        }
        frame = pd.DataFrame([row])
        missing = [c for c in self._features if c not in frame.columns]
        if missing:
            raise ValueError(f"Missing contract22 wicket features: {', '.join(missing)}")
        model_input = frame[self._features].copy()
        for column in self._categorical:
            model_input[column] = model_input[column].fillna("__UNKNOWN__").astype("category")

        raw = float(self._model.predict_proba(model_input)[0][1])
        clipped = min(max(raw, 1e-6), 1 - 1e-6)
        logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
        calibrated = float(self._calibrator.predict_proba(logit)[0][1])
        return calibrated
