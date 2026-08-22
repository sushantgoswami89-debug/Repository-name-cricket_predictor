"""Callable runtime for run_range_enriched_v3_batting_style -- the best
validated run-range candidate (28.49% holdout hit rate, beats v3.3's
28.10% and the v3.2 baseline's 27.28%; see
docs/candidate_run_range_enriched_v2.md). Mirrors runtime_v33.py's
pattern (artifact integrity verification, feature-contract validation,
phase-specific temperature calibration, calibrated inclusive-band output)
but sources its enriched features from RunRangeV3FeatureComputer
(app/ml/run_range_v3_features.py) instead of the caller supplying a flat
feature frame directly, since this candidate's features depend on
per-player/per-venue snapshots and in-match state the caller shouldn't
have to assemble by hand.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from app.ml.model_repository import ModelRepository
from app.ml.run_range_v3_features import RunRangeV3FeatureComputer
from app.ml.sharp_range_predictor import select_inclusive_bands

REQUIRED_ARTIFACTS = (
    "sharp_range_model.pkl",
    "phase_temperatures.pkl",
    "feature_cols.pkl",
    "categorical_cols.pkl",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RunRangeRuntimeV3:
    """Load v3 artifacts + live snapshots, return calibrated inclusive run ranges."""

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
            raise FileNotFoundError(f"Missing v3 artifacts: {', '.join(missing)}")
        if verify_integrity:
            self._verify_integrity()

        repository = ModelRepository(self.artifact_dir)
        self._model: Any = repository.load_artifact("sharp_range_model.pkl")
        self._temperatures: dict[str, float] = repository.load_artifact("phase_temperatures.pkl")
        self._features: list[str] = repository.load_artifact("feature_cols.pkl")
        self._categorical: list[str] = repository.load_artifact("categorical_cols.pkl")
        if set(self._temperatures) != {"powerplay", "middle", "death"}:
            raise ValueError("Invalid v3 phase-temperature artifact.")
        self._computer = feature_computer or RunRangeV3FeatureComputer(project_root)

    def _verify_integrity(self) -> None:
        manifest_path = self.artifact_dir / "ARTIFACT_MANIFEST.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Integrity manifest not found: {manifest_path}. "
                "Generate one (sha256 per REQUIRED_ARTIFACTS file) before "
                "trusting this artifact directory, or pass verify_integrity=False "
                "explicitly for research/replay use."
            )
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
        """Compute the enriched feature row live, then return a calibrated
        inclusive band the same way runtime_v33.py does for its own model.
        `competition` is "ipl" or "t20i" -- see
        RunRangeV3FeatureComputer.compute's docstring.
        """
        phase = "powerplay" if over <= 6 else ("death" if over >= 16 else "middle")
        enriched = self._computer.compute(
            deliveries=deliveries, registry=registry, striker_name=striker_name,
            non_striker_name=non_striker_name, venue_name=venue_name,
            batting_team=batting_team, phase=phase, over=over,
            wickets_down=wkts_down_before_over, current_rate=current_run_rate,
            required_rate=required_run_rate, is_chase=bool(is_chase),
            bowler_name=bowler_name, competition=competition,
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
            raise ValueError(f"Missing v3 features: {', '.join(missing)}")
        model_input = frame[self._features].copy()
        for column in self._categorical:
            model_input[column] = model_input[column].fillna("__UNKNOWN__").astype("category")

        probabilities = np.asarray(self._model.predict_proba(model_input), dtype=float)
        temperature = self._temperatures[phase]
        probabilities = self._temperature_scale(probabilities, temperature)

        low, band_probability = select_inclusive_bands(probabilities, width)
        return {
            "phase": phase,
            f"sharp_{width}_low": int(low[0]),
            f"sharp_{width}_high": int(low[0] + width),
            f"sharp_{width}_prob": float(band_probability[0]),
            "enriched_features": enriched,
        }
