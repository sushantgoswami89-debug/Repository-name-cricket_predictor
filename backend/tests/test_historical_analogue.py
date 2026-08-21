from __future__ import annotations

import pandas as pd
import pytest

from app.ml.historical_analogue import ANALOGUE_FEATURES, HistoricalAnalogueEngine


def _frame(rows: int = 40) -> pd.DataFrame:
    data = {
        column: [float(index % 5) for index in range(rows)]
        for column in ANALOGUE_FEATURES
    }
    data.update(
        {
            "phase": ["middle"] * rows,
            "is_chase": [1] * rows,
            "runs_in_over": [index % 12 for index in range(rows)],
            "wicket_in_over": [int(index % 4 == 0) for index in range(rows)],
        }
    )
    return pd.DataFrame(data)


def test_analogue_query_returns_bounded_distributions() -> None:
    frame = _frame()
    result = HistoricalAnalogueEngine(neighbors=20).fit(frame).query(frame.iloc[:3])
    assert len(result.expected_runs) == 3
    assert (result.lower_runs <= result.upper_runs).all()
    assert ((result.wicket_probability >= 0) & (result.wicket_probability <= 1)).all()
    assert (result.effective_sample_size > 0).all()
    assert ((result.similarity > 0) & (result.similarity <= 1)).all()
    assert result.run_bucket_probabilities.sum(axis=1) == pytest.approx(1.0)


def test_analogue_query_requires_fit() -> None:
    with pytest.raises(RuntimeError, match="fitted first"):
        HistoricalAnalogueEngine().query(_frame().iloc[:1])
