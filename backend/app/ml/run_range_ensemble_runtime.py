"""Callable runtime for the run-range GBM+NN ensemble (2026-08-26) -- see
docs/finding_run_range_nn_gbm_ensemble_real_win.md for the validated
result (beats the live run_range_v11_partnership_rate GBM alone on all
three cuts: blended +0.18pp, IPL +0.18pp, T20I +0.20pp, real 2025+
holdout).

Mirrors runtime_run_range_v3.py's feature-computation pattern exactly
(same RunRangeV3FeatureComputer, same row shape, same width-aware
best-band selection) so the GBM here is the identical model/features as
the live run-range runtime -- only the NN + blend on top are new.

**Shadow only, not yet served** (2026-08-26): PredictionEngine calls this
alongside the live GBM-only run-range prediction, logs both via
app/ml/run_range_ensemble_shadow_log.py, and reports a summary once 5
real matches have accumulated -- same "don't route yet, observe on real
matches first" plan as the win-probability Monte Carlo and wicket
ensemble work.

**Process isolation for the NN**: torch and lightgbm cannot coexist in
one process on this machine (confirmed: hangs/dies regardless of import
order or KMP_DUPLICATE_LIB_OK). The NN prediction runs in its own
subprocess (predict_run_range_nn_subprocess.py) so a torch problem can
never hang or crash the live GBM-serving process -- if the subprocess
fails for any reason, the ensemble output degrades to the GBM alone.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from app.ml.model_repository import ModelRepository
from app.ml.run_range_v3_features import RunRangeV3FeatureComputer
from app.ml.sharp_range_predictor import select_inclusive_bands

REQUIRED_ARTIFACTS = (
    "gbm_model.pkl", "feature_cols.pkl", "categorical_cols.pkl",
    "nn_model.pt", "nn_metadata.json",
    "phase_temperatures.pkl", "blend_config.json",
)
NN_SUBPROCESS_SCRIPT = Path(__file__).resolve().parents[2] / "predict_run_range_nn_subprocess.py"
NN_SUBPROCESS_TIMEOUT_SECONDS = 10


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RunRangeEnsembleRuntime:
    """Loads the GBM (in-process, safe) + the NN's preprocessing metadata
    (the NN model itself only ever runs in the subprocess) + the blend
    weight + per-phase temperatures fit on the blended distribution.
    Returns bands for both the GBM alone and the ensemble so callers can
    log both for shadow comparison against the live-served prediction."""

    def __init__(
        self,
        artifact_dir: Path | str,
        *,
        project_root: Path | None = None,
        verify_integrity: bool = True,
        feature_computer: RunRangeV3FeatureComputer | None = None,
    ) -> None:
        self.artifact_dir = Path(artifact_dir)
        if not self.artifact_dir.is_dir():
            raise FileNotFoundError(f"Artifact directory not found: {self.artifact_dir}")

        missing = [name for name in REQUIRED_ARTIFACTS if not (self.artifact_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing run-range ensemble artifacts: {', '.join(missing)}")
        if verify_integrity:
            self._verify_integrity()

        repository = ModelRepository(self.artifact_dir)
        self._gbm: Any = repository.load_artifact("gbm_model.pkl")
        self._features: list[str] = repository.load_artifact("feature_cols.pkl")
        self._categorical: list[str] = repository.load_artifact("categorical_cols.pkl")
        self._temperatures: dict[str, float] = repository.load_artifact("phase_temperatures.pkl")
        if set(self._temperatures) != {"powerplay", "middle", "death"}:
            raise ValueError("Invalid run-range ensemble phase-temperature artifact.")
        blend_config = json.loads((self.artifact_dir / "blend_config.json").read_text(encoding="utf-8"))
        self._alpha: float = float(blend_config["alpha_gbm_weight"])
        self._computer = feature_computer or RunRangeV3FeatureComputer(project_root)

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

    @staticmethod
    def _temperature_scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
        logits = np.log(np.clip(probabilities, 1e-12, 1.0)) / temperature
        logits -= logits.max(axis=1, keepdims=True)
        scaled = np.exp(logits)
        return scaled / scaled.sum(axis=1, keepdims=True)

    def _nn_probabilities(self, row: dict[str, Any]) -> np.ndarray | None:
        try:
            result = subprocess.run(
                [sys.executable, str(NN_SUBPROCESS_SCRIPT)],
                input=json.dumps(row), capture_output=True, text=True,
                timeout=NN_SUBPROCESS_TIMEOUT_SECONDS,
            )
            if result.returncode != 0:
                return None
            return np.asarray(json.loads(result.stdout)["probabilities"], dtype=float).reshape(1, -1)
        except Exception:
            # Subprocess timeout, crash, bad output -- never let the NN
            # shadow take down the caller. Ensemble falls back to GBM alone.
            return None

    def predict_next_over(
        self,
        *,
        deliveries: Sequence[Any],
        registry: Mapping[str, str],
        striker_name: str,
        non_striker_name: str,
        venue_name: str,
        batting_team: str,
        over: int,
        bowler_name: str = "",
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
        width: int = 2,
        competition: str = "t20i",
    ) -> dict[str, Any]:
        phase = "powerplay" if over <= 6 else ("death" if over >= 16 else "middle")
        enriched = self._computer.compute(
            deliveries=deliveries, registry=registry, striker_name=striker_name,
            non_striker_name=non_striker_name, venue_name=venue_name,
            batting_team=batting_team, phase=phase, over=over,
            wickets_down=wkts_down_before_over, current_rate=current_run_rate,
            required_rate=required_run_rate, is_chase=bool(is_chase),
            bowler_name=bowler_name, competition=competition,
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
            raise ValueError(f"Missing run-range ensemble features: {', '.join(missing)}")
        model_input = frame[self._features].copy()
        for column in self._categorical:
            model_input[column] = model_input[column].fillna("__UNKNOWN__").astype("category")

        gbm_probabilities = np.asarray(self._gbm.predict_proba(model_input), dtype=float)
        gbm_low, gbm_band_probability = select_inclusive_bands(gbm_probabilities, width)

        nn_row = {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}
        nn_probabilities = self._nn_probabilities(nn_row)
        ensemble_probabilities = gbm_probabilities
        if nn_probabilities is not None and nn_probabilities.shape == gbm_probabilities.shape:
            ensemble_probabilities = self._alpha * gbm_probabilities + (1 - self._alpha) * nn_probabilities
        ensemble_probabilities = self._temperature_scale(ensemble_probabilities, self._temperatures[phase])
        ensemble_low, ensemble_band_probability = select_inclusive_bands(ensemble_probabilities, width)

        return {
            "phase": phase,
            "gbm_low": int(gbm_low[0]), "gbm_high": int(gbm_low[0] + width),
            "gbm_band_probability": float(gbm_band_probability[0]),
            "ensemble_low": int(ensemble_low[0]), "ensemble_high": int(ensemble_low[0] + width),
            "ensemble_band_probability": float(ensemble_band_probability[0]),
            "nn_available": nn_probabilities is not None,
        }
