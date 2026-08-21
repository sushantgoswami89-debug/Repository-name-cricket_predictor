"""
Model Repository.

Loads and provides access to all machine learning assets used by
the CricketBaba prediction engine.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import sklearn

logger = logging.getLogger(__name__)


class ModelRepository:
    """
    Repository responsible for loading and serving ML models.

    Models are loaded only once and then cached in memory.
    """

    PRODUCTION_SKLEARN_VERSION = "1.8.0"

    def __init__(
        self,
        model_dir: Path | None = None,
        expected_sklearn_version: str | None = None,
    ) -> None:
        """
        Initialize the repository.
        """

        project_root = Path(__file__).resolve().parents[3]
        model_dir = model_dir or project_root / "models"
        if (
            expected_sklearn_version is None
            and model_dir == project_root / "models"
        ):
            expected_sklearn_version = self.PRODUCTION_SKLEARN_VERSION
        if (
            expected_sklearn_version is not None
            and sklearn.__version__ != expected_sklearn_version
        ):
            raise RuntimeError(
                "Incompatible scikit-learn runtime: production artifacts require "
                f"{expected_sklearn_version}, found {sklearn.__version__}."
            )
        self.model_dir = model_dir
        self._artifacts: dict[str, Any] = {}

        self._runs_model_path = model_dir / "runs_model.pkl"
        self._wicket_model_path = model_dir / "wkt_model.pkl"
        self._feature_cols_path = model_dir / "feature_cols.pkl"
        self._cat_cols_path = model_dir / "cat_cols.pkl"

        self._wicket_calibrator_path = model_dir / "wkt_calibrator.pkl"
        self._confidence_calibrator_path = model_dir / "confidence_calibrator.pkl"

        self._runs_model: Any | None = None
        self._wicket_model: Any | None = None
        self._wicket_calibrator: Any | None = None
        self._confidence_calibrator: Any | None = None
        self._confidence_calibrator_loaded = False
        self._wicket_calibrator_loaded = False
        self._feature_columns: list[str] | None = None
        self._categorical_columns: list[str] | None = None

    def load_artifact(self, name: str) -> Any:
        """Load one named joblib artifact and cache it for this repository."""
        if not name or Path(name).name != name:
            raise ValueError("Artifact name must be a single file name.")
        if name not in self._artifacts:
            self._artifacts[name] = self._load_pickle(self.model_dir / name)
        return self._artifacts[name]

    def _load_pickle(self, path: Path) -> Any:
        """
        Load a pickle file after verifying it exists.
        """

        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        logger.info("Loading %s", path.name)

        return joblib.load(path)

    def get_runs_model(self) -> Any:
        """
        Return the cached runs prediction model.
        """

        if self._runs_model is None:
            self._runs_model = self.load_artifact(self._runs_model_path.name)

        return self._runs_model

    def get_wicket_model(self) -> Any:
        """
        Return the cached wicket prediction model.
        """

        if self._wicket_model is None:
            self._wicket_model = self.load_artifact(self._wicket_model_path.name)

        return self._wicket_model

    def get_wicket_calibrator(self) -> Any | None:
        """
        Return the fitted isotonic-regression calibrator for wicket
        probabilities, or ``None`` if this model directory has none (e.g.
        an older locked candidate that predates calibration).
        """

        if not self._wicket_calibrator_loaded:
            if self._wicket_calibrator_path.exists():
                self._wicket_calibrator = self.load_artifact(
                    self._wicket_calibrator_path.name
                )
            self._wicket_calibrator_loaded = True

        return self._wicket_calibrator

    def get_confidence_calibrator(self) -> Any | None:
        """
        Return the fitted isotonic-regression calibrator mapping the
        engine's heuristic confidence score to the actual empirical
        probability the predicted range covers the real outcome, or
        ``None`` if this model directory has none.
        """

        if not self._confidence_calibrator_loaded:
            if self._confidence_calibrator_path.exists():
                self._confidence_calibrator = self.load_artifact(
                    self._confidence_calibrator_path.name
                )
            self._confidence_calibrator_loaded = True

        return self._confidence_calibrator

    def get_feature_columns(self) -> list[str]:
        """
        Return the feature column names.
        """

        if self._feature_columns is None:
            self._feature_columns = self.load_artifact(self._feature_cols_path.name)

        return self._feature_columns

    def get_categorical_columns(self) -> list[str]:
        """
        Return the categorical feature names.
        """

        if self._categorical_columns is None:
            self._categorical_columns = self.load_artifact(self._cat_cols_path.name)

        return self._categorical_columns
