from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from runtime_v33 import SharpRangeRuntimeV33

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "models/candidates/v3.3_phase_calibrated_sharp_range"


def test_runtime_loads_verified_artifacts_and_predicts_inclusive_band() -> None:
    runtime = SharpRangeRuntimeV33(ARTIFACTS, verify_integrity=True)
    sample = pd.read_csv(
        ROOT / "data/candidates/v3/verified_training_overs.csv", nrows=10
    )

    result = runtime.predict_frame(sample, width=2)

    assert (result["sharp_2_high"] - result["sharp_2_low"] == 2).all()
    assert result["sharp_2_prob"].between(0, 1).all()


def test_runtime_rejects_missing_features() -> None:
    runtime = SharpRangeRuntimeV33(ARTIFACTS, verify_integrity=True)

    with pytest.raises(ValueError, match="Missing v3.3 features"):
        runtime.predict_frame(pd.DataFrame({"phase": ["powerplay"]}), width=2)
