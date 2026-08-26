"""Callable runtime for the wicket GBM+NN ensemble (2026-08-25) -- see
docs/finding_wicket_nn_gbm_ensemble_real_win.md for the validated result
(beats the live GBM alone on every cut, real 2025+ holdout).

Mirrors runtime_wicket_contract22.py's feature-computation pattern
exactly (same WicketContract22FeatureComputer, same row shape) so the
GBM here is the identical model/features as the live wicket runtime --
only the NN+stacker on top are new.

**Shadow only, not yet served** (2026-08-25): PredictionEngine calls this
alongside the live GBM-only wicket_probability, logs both via
app/ml/wicket_ensemble_shadow_log.py, and reports a summary once 5 real
matches have accumulated -- same "don't route yet, observe on real
matches first" plan as the win-probability Monte Carlo work.

**Process isolation for the NN**: torch and lightgbm cannot coexist in
one process on this machine (confirmed: hangs/dies regardless of import
order or KMP_DUPLICATE_LIB_OK). The NN prediction runs in its own
subprocess (predict_wicket_nn_subprocess.py) so a torch problem can never
hang or crash the live GBM-serving process -- this trades a few seconds
of latency for that safety, acceptable since wicket predictions happen
once per over (~60-90s cadence), not per-ball.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from app.ml.model_repository import ModelRepository
from app.ml.wicket_contract22_features import WicketContract22FeatureComputer

REQUIRED_ARTIFACTS = (
    "gbm_model.pkl", "feature_cols.pkl", "categorical_cols.pkl",
    "nn_model.pt", "nn_metadata.json",
    "gbm_calibrator.pkl", "nn_calibrator.pkl", "stacker.pkl",
)
NN_SUBPROCESS_SCRIPT = Path(__file__).resolve().parents[2] / "predict_wicket_nn_subprocess.py"
NN_SUBPROCESS_TIMEOUT_SECONDS = 10


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _phase(over: int) -> str:
    return "powerplay" if over <= 6 else ("death" if over >= 16 else "middle")


def _logit(p: float) -> float:
    clipped = min(max(p, 1e-6), 1 - 1e-6)
    return float(np.log(clipped / (1 - clipped)))


class WicketEnsembleRuntime:
    """Loads the GBM (in-process, safe) + the NN's preprocessing metadata
    (the NN model itself only ever runs in the subprocess) + both Platt
    calibrators + the stacker. Returns (gbm_probability, nn_probability_or_None,
    ensemble_probability) so callers can log all three for shadow comparison."""

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
            raise FileNotFoundError(f"Missing wicket ensemble artifacts: {', '.join(missing)}")
        if verify_integrity:
            self._verify_integrity()

        repository = ModelRepository(self.artifact_dir)
        self._gbm: Any = repository.load_artifact("gbm_model.pkl")
        self._gbm_calibrator: Any = repository.load_artifact("gbm_calibrator.pkl")
        self._nn_calibrator: Any = repository.load_artifact("nn_calibrator.pkl")
        self._stacker: Any = repository.load_artifact("stacker.pkl")
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

    def _nn_probability(self, row: dict[str, Any]) -> float | None:
        try:
            result = subprocess.run(
                [sys.executable, str(NN_SUBPROCESS_SCRIPT)],
                input=json.dumps(row), capture_output=True, text=True,
                timeout=NN_SUBPROCESS_TIMEOUT_SECONDS,
            )
            if result.returncode != 0:
                return None
            return float(json.loads(result.stdout)["probability"])
        except Exception:
            # Subprocess timeout, crash, bad output -- never let the NN
            # shadow take down the caller. Ensemble falls back to GBM alone.
            return None

    def predict(
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
        batting_team_players: Any = (),
        bowling_team_players: Any = (),
    ) -> dict[str, float | None]:
        phase = _phase(over)
        enriched = self._computer.compute(
            registry=registry, striker_name=striker_name, non_striker_name=non_striker_name,
            bowler_name=bowler_name, venue_name=venue_name, phase=phase, deliveries=deliveries,
            batting_team_players=batting_team_players, bowling_team_players=bowling_team_players,
        )
        row: dict[str, Any] = {
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
            raise ValueError(f"Missing wicket ensemble features: {', '.join(missing)}")
        model_input = frame[self._features].copy()
        for column in self._categorical:
            model_input[column] = model_input[column].fillna("__UNKNOWN__").astype("category")

        gbm_raw = float(self._gbm.predict_proba(model_input)[0][1])
        gbm_probability = float(self._gbm_calibrator.predict_proba(np.array([[_logit(gbm_raw)]]))[0][1])

        nn_raw = self._nn_probability({k: (row[k].item() if hasattr(row[k], "item") else row[k]) for k in row})
        nn_probability = None
        ensemble_probability = gbm_probability
        if nn_raw is not None:
            nn_probability = float(self._nn_calibrator.predict_proba(np.array([[_logit(nn_raw)]]))[0][1])
            ensemble_probability = float(self._stacker.predict_proba(
                np.array([[_logit(gbm_probability), _logit(nn_probability)]])
            )[0][1])

        return {
            "gbm_probability": round(gbm_probability, 4),
            "nn_probability": round(nn_probability, 4) if nn_probability is not None else None,
            "ensemble_probability": round(ensemble_probability, 4),
        }
