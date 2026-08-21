from app.ml.current_match_pitch_dataset import _add_rates


def test_add_rates_handles_unobserved_pitch() -> None:
    row = {}
    profile = {"balls": 0, "runs": 0, "boundaries": 0, "dots": 0, "wickets": 0}
    _add_rates(row, "pitch", profile)
    assert row["pitch_samples"] == 0
    assert row["pitch_runs_rate"] == 0.0


def test_add_rates_uses_completed_balls_only() -> None:
    row = {}
    profile = {"balls": 2, "runs": 5, "boundaries": 1, "dots": 1, "wickets": 0}
    _add_rates(row, "pitch", profile)
    assert row["pitch_runs_rate"] == 2.5
    assert row["pitch_boundaries_rate"] == 0.5
