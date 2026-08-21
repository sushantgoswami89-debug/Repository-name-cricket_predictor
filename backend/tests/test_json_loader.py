"""
Tests for JsonLoader.
"""

from pathlib import Path

from app.data_ingestion.json_loader import JsonLoader


def test_json_loader() -> None:
    """
    Ensure JsonLoader loads a real Cricsheet match.
    """

    project_root = Path(__file__).resolve().parents[2]

    sample_directory = project_root / "data" / "raw" / "cricsheet" / "ipl"

    sample_file = next(sample_directory.glob("*.json"))

    loader = JsonLoader()

    data = loader.load(sample_file)

    assert "info" in data
    assert "innings" in data
