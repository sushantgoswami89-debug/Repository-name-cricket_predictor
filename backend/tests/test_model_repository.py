"""
Tests for ModelRepository.
"""

import warnings

import joblib
import pytest
import sklearn
from sklearn.exceptions import InconsistentVersionWarning

from app.ml.model_repository import ModelRepository


def test_model_repository_loads_models() -> None:
    """
    Ensure all ML assets can be loaded successfully.
    """

    repository = ModelRepository()

    runs_model = repository.get_runs_model()
    wicket_model = repository.get_wicket_model()
    feature_columns = repository.get_feature_columns()
    categorical_columns = repository.get_categorical_columns()

    assert runs_model is not None
    assert wicket_model is not None

    assert isinstance(feature_columns, list)
    assert isinstance(categorical_columns, list)

    assert len(feature_columns) == 22
    assert len(categorical_columns) == 6


def test_model_repository_caches_models() -> None:
    """
    Ensure models are cached after the first load.
    """

    repository = ModelRepository()

    runs_model_1 = repository.get_runs_model()
    runs_model_2 = repository.get_runs_model()

    wicket_model_1 = repository.get_wicket_model()
    wicket_model_2 = repository.get_wicket_model()

    assert runs_model_1 is runs_model_2
    assert wicket_model_1 is wicket_model_2


def test_generic_artifact_loader_caches_and_rejects_paths(
    tmp_path, monkeypatch
) -> None:
    joblib.dump({"value": 1}, tmp_path / "custom.pkl")
    repository = ModelRepository(tmp_path)
    calls = 0
    original = repository._load_pickle

    def counted(path):
        nonlocal calls
        calls += 1
        return original(path)

    monkeypatch.setattr(repository, "_load_pickle", counted)
    assert repository.load_artifact("custom.pkl") is repository.load_artifact(
        "custom.pkl"
    )
    assert calls == 1
    with pytest.raises(ValueError, match="single file name"):
        repository.load_artifact("../custom.pkl")


def test_production_repository_rejects_incompatible_sklearn(
    monkeypatch,
) -> None:
    monkeypatch.setattr(sklearn, "__version__", "0.0")

    with pytest.raises(RuntimeError, match="require 1.8.0"):
        ModelRepository()


def test_production_artifacts_load_without_version_warning() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        repository = ModelRepository()
        repository.get_runs_model()
        repository.get_wicket_model()

    assert not any(
        isinstance(warning.message, InconsistentVersionWarning)
        for warning in caught
    )
