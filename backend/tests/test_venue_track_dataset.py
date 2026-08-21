from app.ml.venue_track_dataset import _track_type


def test_track_type_requires_reliable_sample() -> None:
    venue = {"balls": 119, "runs": 200, "boundaries": 30, "dots": 40, "wickets": 4}
    global_profile = {
        "balls": 1000,
        "runs": 1300,
        "boundaries": 150,
        "dots": 400,
        "wickets": 50,
    }
    assert _track_type(venue, global_profile) == "unknown"


def test_track_type_detects_high_scoring_history() -> None:
    venue = {"balls": 200, "runs": 400, "boundaries": 50, "dots": 50, "wickets": 8}
    global_profile = {
        "balls": 1000,
        "runs": 1300,
        "boundaries": 150,
        "dots": 400,
        "wickets": 50,
    }
    assert _track_type(venue, global_profile) == "high_scoring"
