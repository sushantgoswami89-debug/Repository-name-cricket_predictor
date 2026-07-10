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

logger = logging.getLogger(__name__)


class ModelRepository:
    """
    Repository responsible for loading and serving ML models.

    Models are loaded only once and then cached in memory.
    """

    def __init__(self) -> None:
        """
        Initialize the repository.
        """

        project_root = Path(__file__).resolve().parents[3]
        model_dir = project_root / "models"

        self._runs_model_path = model_dir / "runs_model.pkl"
        self._wicket_model_path = model_dir / "wkt_model.pkl"
        self._feature_cols_path = model_dir / "feature_cols.pkl"
        self._cat_cols_path = model_dir / "cat_cols.pkl"

        self._runs_model: Any | None = None
        self._wicket_model: Any | None = None
        self._feature_columns: list[str] | None = None
        self._categorical_columns: list[str] | None = None

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
            self._runs_model = self._load_pickle(self._runs_model_path)

        return self._runs_model

    def get_wicket_model(self) -> Any:
        """
        Return the cached wicket prediction model.
        """

        if self._wicket_model is None:
            self._wicket_model = self._load_pickle(self._wicket_model_path)

        return self._wicket_model

    def get_feature_columns(self) -> list[str]:
        """
        Return the feature column names.
        """

        if self._feature_columns is None:
            self._feature_columns = self._load_pickle(self._feature_cols_path)

        return self._feature_columns

    def get_categorical_columns(self) -> list[str]:
        """
        Return the categorical feature names.
        """

        if self._categorical_columns is None:
            self._categorical_columns = self._load_pickle(self._cat_cols_path)

        return self._categorical_columns
