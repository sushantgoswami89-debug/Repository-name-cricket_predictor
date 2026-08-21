from app.ml.batter_entry_phase_dataset import _smoothed_rates


def test_smoothed_rates_use_matching_entry_phase_prior() -> None:
    player = {
        "samples": 1,
        "runs": 4,
        "boundary": 1,
        "single": 0,
        "wicket": 0,
        "dot": 0,
    }
    prior = {
        "samples": 2,
        "runs": 2,
        "boundary": 0,
        "single": 2,
        "wicket": 0,
        "dot": 0,
    }
    rates = _smoothed_rates(player, prior, strength=1.0)
    assert rates["runs_rate"] == 2.5
    assert rates["boundary_rate"] == 0.5
    assert rates["single_rate"] == 0.5
