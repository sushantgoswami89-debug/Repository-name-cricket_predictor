"""Callable runtime for the phase-calibrated Sharp Range v3.3 model."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.ml.model_repository import ModelRepository
from app.ml.sharp_range_predictor import select_inclusive_bands

REQUIRED_ARTIFACTS = (
    "sharp_range_model.pkl",
    "phase_temperatures.pkl",
    "feature_cols.pkl",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SharpRangeRuntimeV33:
    """Load v3.3 artifacts and return calibrated inclusive run ranges."""

    def __init__(
        self, artifact_dir: Path | str, *, verify_integrity: bool = True
    ) -> None:
        self.artifact_dir = Path(artifact_dir)
        if not self.artifact_dir.is_dir():
            raise FileNotFoundError(
                f"Artifact directory not found: {self.artifact_dir}"
            )

        missing = [
            name
            for name in REQUIRED_ARTIFACTS
            if not (self.artifact_dir / name).is_file()
        ]
        if missing:
            raise FileNotFoundError(f"Missing v3.3 artifacts: {', '.join(missing)}")
        if verify_integrity:
            self._verify_integrity()

        repository = ModelRepository(self.artifact_dir)
        self._model: Any = repository.load_artifact("sharp_range_model.pkl")
        self._temperatures: dict[str, float] = repository.load_artifact(
            "phase_temperatures.pkl"
        )
        self._features: list[str] = repository.load_artifact("feature_cols.pkl")
        if set(self._temperatures) != {"powerplay", "middle", "death"}:
            raise ValueError("Invalid v3.3 phase-temperature artifact.")
        if getattr(self._model, "n_features_in_", None) != len(self._features):
            raise ValueError("Model and feature manifest do not agree.")

    def _verify_integrity(self) -> None:
        manifest_path = self.artifact_dir / "ARTIFACT_MANIFEST.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Integrity manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("model") != "v3.3_phase_calibrated_sharp_range":
            raise ValueError("Integrity manifest is not for Sharp Range v3.3.")
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

    def predict_frame(self, frame: pd.DataFrame, *, width: int = 2) -> pd.DataFrame:
        """Append the most probable inclusive band of ``width + 1`` outcomes."""
        if frame.empty:
            raise ValueError("Prediction frame must contain at least one row.")
        if not isinstance(width, int) or width < 0:
            raise ValueError("width must be a non-negative integer.")
        missing = [column for column in self._features if column not in frame.columns]
        if missing:
            raise ValueError(f"Missing v3.3 features: {', '.join(missing)}")

        phases = frame["phase"].astype(str)
        unknown = sorted(set(phases) - set(self._temperatures))
        if unknown:
            raise ValueError(f"Unknown innings phase: {', '.join(unknown)}")

        features = frame[self._features].copy()
        features["phase"] = features["phase"].astype("category")
        probabilities = np.asarray(self._model.predict_proba(features), dtype=float)
        for phase, temperature in self._temperatures.items():
            mask = phases.to_numpy() == phase
            probabilities[mask] = self._temperature_scale(
                probabilities[mask], temperature
            )

        low, band_probability = select_inclusive_bands(probabilities, width)
        result = frame.copy()
        result[f"sharp_{width}_low"] = low
        result[f"sharp_{width}_high"] = low + width
        result[f"sharp_{width}_prob"] = band_probability
        return result
