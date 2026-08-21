"""Five-star self-evaluation for completed CricketBaba predictions."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class PredictionRating:
    """A completed prediction compared with the verified actual over."""

    predicted_runs: float
    actual_runs: int
    absolute_run_error: float
    expected_range: str
    range_covered: bool
    wicket_probability: float
    actual_wicket: bool
    wicket_brier_score: float
    stars: int
    rating: str

    @property
    def star_meter(self) -> str:
        return "★" * self.stars + "☆" * (5 - self.stars)

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["star_meter"] = self.star_meter
        return result


def _parse_range(value: str) -> tuple[float, float]:
    try:
        low, high = value.split("-", maxsplit=1)
        result = float(low), float(high)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid prediction range: {value!r}") from error
    if result[0] > result[1]:
        raise ValueError(f"Prediction range is reversed: {value!r}")
    return result


def rate_prediction(
    *,
    predicted_runs: float,
    expected_range: str,
    wicket_probability: float,
    actual_runs: int,
    actual_wickets: int,
) -> PredictionRating:
    """Score one completed prediction using the locked five-star rubric."""

    if predicted_runs < 0 or actual_runs < 0 or actual_wickets < 0:
        raise ValueError("Prediction and actual values cannot be negative.")
    if not 0 <= wicket_probability <= 1:
        raise ValueError("Wicket probability must be between zero and one.")
    low, high = _parse_range(expected_range)
    absolute_error = abs(predicted_runs - actual_runs)
    range_covered = low <= actual_runs <= high
    actual_wicket = actual_wickets > 0
    brier = (wicket_probability - float(actual_wicket)) ** 2

    # The review label measures the miss from the nearest range boundary.
    # Wicket quality remains available separately through the Brier score.
    if range_covered:
        range_miss = 0.0
    elif actual_runs < low:
        range_miss = low - actual_runs
    else:
        range_miss = actual_runs - high

    if range_miss == 0:
        stars, label = 5, "EXCELLENT"
    elif range_miss <= 1:
        stars, label = 4, "GOOD"
    elif range_miss <= 2:
        stars, label = 3, "FAIR"
    elif range_miss <= 4:
        stars, label = 2, "BAD"
    else:
        stars, label = 1, "POOR"

    return PredictionRating(
        predicted_runs=predicted_runs,
        actual_runs=actual_runs,
        absolute_run_error=round(absolute_error, 2),
        expected_range=expected_range,
        range_covered=range_covered,
        wicket_probability=wicket_probability,
        actual_wicket=actual_wicket,
        wicket_brier_score=round(brier, 4),
        stars=stars,
        rating=label,
    )
