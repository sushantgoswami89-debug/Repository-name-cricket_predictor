from __future__ import annotations

import numpy as np

from app.ml.sharp_range_predictor import SharpRange, select_inclusive_bands


def test_sharp_range_is_always_two_runs_wide() -> None:
    result = SharpRange(low=8, high=10)
    assert result.display == "8-10"
    assert result.to_dict() == {
        "low": 8,
        "high": 10,
        "display": "8-10",
        "width": 2,
    }


def test_sharp_range_marks_open_ended_tail() -> None:
    assert SharpRange(low=28, high=30).display == "28-30+"


def test_shared_band_selection_matches_inclusive_window_definition() -> None:
    probabilities = np.array(
        [[0.05, 0.10, 0.15, 0.40, 0.20, 0.10]]
    )

    low, mass = select_inclusive_bands(probabilities, width=2)

    assert low.tolist() == [2]
    assert mass.tolist() == [0.75]
