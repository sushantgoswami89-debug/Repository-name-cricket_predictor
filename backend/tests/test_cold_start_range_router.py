import numpy as np
import pandas as pd

from train_ipl_cold_start_v3 import (
    _apply_cold_range_shifts,
    _calibrate_cold_range_shifts,
)


def test_range_shift_is_learned_only_for_supported_cold_phase():
    calibration = pd.DataFrame(
        {
            "phase": ["powerplay"] * 20 + ["death"] * 5,
            "cold_start_status": ["TEMP"] * 25,
            "striker_prior_balls": [0] * 25,
            "runs_in_over": [8] * 20 + [4] * 5,
        }
    )
    predictions = np.array([6.2] * 20 + [4.2] * 5)
    shifts = _calibrate_cold_range_shifts(
        calibration, predictions, {"powerplay": -1, "death": -1}
    )
    assert shifts["powerplay"] == 0
    assert shifts["death"] == 0


def test_range_shift_never_changes_non_cold_rows():
    data = pd.DataFrame({"phase": ["powerplay", "powerplay"]})
    predictions = np.array([6.2, 6.2])
    routed = _apply_cold_range_shifts(
        data, predictions, np.array([True, False]), {"powerplay": 1}
    )
    assert routed.tolist() == [7.2, 6.2]
