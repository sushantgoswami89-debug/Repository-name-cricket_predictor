"""
Tests for ModelRepository.
"""

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
