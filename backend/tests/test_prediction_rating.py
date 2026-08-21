from __future__ import annotations

import pytest

from app.services.prediction_rating import rate_prediction


@pytest.mark.parametrize(
    ("actual", "stars", "label"),
    [
        (8, 5, "EXCELLENT"),
        (11, 5, "EXCELLENT"),
        (7, 4, "GOOD"),
        (12, 4, "GOOD"),
        (6, 3, "FAIR"),
        (13, 3, "FAIR"),
        (5, 2, "BAD"),
        (14, 2, "BAD"),
        (4, 2, "BAD"),
        (15, 2, "BAD"),
        (3, 1, "POOR"),
        (16, 1, "POOR"),
    ],
)
def test_locked_five_star_rating_boundaries(
    actual: int,
    stars: int,
    label: str,
) -> None:
    result = rate_prediction(
        predicted_runs=9.5,
        expected_range="8-11",
        wicket_probability=0.2,
        actual_runs=actual,
        actual_wickets=0,
    )
    assert result.stars == stars
    assert result.rating == label
    assert len(result.star_meter) == 5


def test_wicket_probability_is_scored_with_brier_loss() -> None:
    result = rate_prediction(
        predicted_runs=8.0,
        expected_range="5-11",
        wicket_probability=0.8,
        actual_runs=8,
        actual_wickets=1,
    )
    assert result.wicket_brier_score == pytest.approx(0.04)
    assert result.stars == 5


def test_invalid_range_fails_closed() -> None:
    with pytest.raises(ValueError, match="Invalid prediction range"):
        rate_prediction(
            predicted_runs=8.0,
            expected_range="unknown",
            wicket_probability=0.2,
            actual_runs=8,
            actual_wickets=0,
        )
