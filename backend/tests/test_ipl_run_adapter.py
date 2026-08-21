"""Inference contract tests for the candidate IPL runs adapter."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.ml.ipl_run_adapter import PhaseAdjustedRegressor


class _RunsModel:
    def predict(self, features):
        return np.full(len(features), 6.0)


def test_phase_adjustment_is_row_specific_and_non_negative() -> None:
    adapter = PhaseAdjustedRegressor(
        _RunsModel(),
        {"powerplay": 2.0, "middle": 1.0, "death": -10.0},
    )
    features = pd.DataFrame(
        {"phase": ["powerplay", "middle", "death", "unknown"]}
    )

    assert adapter.predict(features).tolist() == [8.0, 7.0, 0.0, 6.0]


def test_venue_regime_adjustment_precedes_phase_fallback() -> None:
    adapter = PhaseAdjustedRegressor(
        _RunsModel(),
        {"powerplay": 1.0, "powerplay|high": 2.0},
        correction_columns=("phase", "venue_scoring_regime"),
    )
    features = pd.DataFrame(
        {
            "phase": ["powerplay", "powerplay"],
            "venue_scoring_regime": ["high", "low"],
        }
    )

    assert adapter.predict(features).tolist() == [8.0, 7.0]
