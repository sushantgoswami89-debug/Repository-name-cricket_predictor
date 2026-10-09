"""Pre-match win model: point-in-time features, orientation-free output, briefing text."""

import pytest

from app.live.prematch_analysis import _bat_first_team, prematch_text
from app.live.prematch_model import MODEL_PATH, build_dataset, predict

pytestmark = pytest.mark.skipif(not MODEL_PATH.exists(), reason="saved pre-match model not present")


def test_dataset_is_chronological_and_first_match_has_no_history():
    data = build_dataset()
    assert [r["date"] for r in data] == sorted(r["date"] for r in data)
    first = data[0]
    assert first["x"][:4] == [0.0, 0.0, 0.0, 0.0]  # no Elo gap, form, or H2H yet


def test_prediction_does_not_depend_on_which_team_is_listed_first():
    ab = predict("India", "West Indies", "Ekana Cricket Stadium, Lucknow", "West Indies")
    ba = predict("West Indies", "India", "Ekana Cricket Stadium, Lucknow", "West Indies")
    assert ab["p_a"] == pytest.approx(ba["p_b"])
    assert ab["p_a"] + ab["p_b"] == pytest.approx(1.0)


def test_unknown_team_gives_no_prediction():
    assert predict("Afghanistan", "India") is None


def test_bat_first_from_toss():
    assert _bat_first_team("India", "West Indies", "India", "bowl") == "West Indies"
    assert _bat_first_team("India", "West Indies", "India", "Bat") == "India"
    assert _bat_first_team("India", "West Indies", "", "") == ""
    assert _bat_first_team("India", "West Indies", "Mars", "bat") == ""


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Briefing tests must not call Open-Meteo; individual tests patch a forecast in."""
    monkeypatch.setattr("app.live.match_weather.match_day_weather", lambda city, day=None: None)


def test_briefing_shows_win_percent_with_its_measured_accuracy():
    text = prematch_text("India", "West Indies", "Ekana Cricket Stadium, Lucknow", "India", "bowl")
    assert "Win %: India" in text and "West Indies" in text
    assert "unseen 2025+ games" in text
    assert "Bats first: West Indies" in text
    assert "toss not known" in prematch_text("India", "West Indies", "Ekana Cricket Stadium, Lucknow")


WET_EVENING = {"eve_temp": 21.0, "eve_dew_point": 18.8, "eve_humidity": 87.0, "aft_cloud": 23.0, "rain_mm": 0.0}
DRY_EVENING = {"eve_temp": 31.0, "eve_dew_point": 12.0, "eve_humidity": 30.0, "aft_cloud": 10.0, "rain_mm": 0.0}


def test_dew_index_is_a_threshold_on_the_temperature_dew_point_gap():
    from app.live.prematch_conditions import dew_index

    assert dew_index(WET_EVENING) > 0.6
    assert dew_index(DRY_EVENING) < 0.01
    assert dew_index(None) == 0.0 and dew_index({}) == 0.0


def test_dew_helps_the_chasing_side_and_needs_the_toss():
    from app.live.prematch_conditions import DEW_MODEL_PATH, predict_with_dew

    if not DEW_MODEL_PATH.exists():
        pytest.skip("dew model not saved")
    wet = predict_with_dew("India", "West Indies", "India", WET_EVENING)      # West Indies chase
    dry = predict_with_dew("India", "West Indies", "India", DRY_EVENING)
    assert wet["p_b"] > dry["p_b"]
    wet_flipped = predict_with_dew("India", "West Indies", "West Indies", WET_EVENING)  # India chase
    assert wet_flipped["p_a"] > dry["p_a"] - 0.2 and wet_flipped["dew_shift"] > 0
    assert predict_with_dew("India", "West Indies", "", WET_EVENING) is None
    assert predict_with_dew("India", "West Indies", "India", None) is None


def test_briefing_shows_conditions_line_when_a_forecast_exists(monkeypatch):
    monkeypatch.setattr("app.live.match_weather.match_day_weather", lambda city, day=None: WET_EVENING)
    text = prematch_text("India", "West Indies", "JSCA International Stadium Complex, Ranchi", "India", "bowl")
    assert "Conditions (forecast, Ranchi)" in text and "DEW LIKELY" in text


def test_live_names_resolve_to_cricsheet_players_and_unknowns_stay_unmatched():
    from app.live.prematch_xi import resolve_player

    assert resolve_player("Shai Hope", "West Indies") == "SD Hope"
    assert resolve_player("Vaibhav Sooryavanshi", "India") is None  # no history -> neutral, not guessed
    assert resolve_player("Madonna", "India") is None


def test_xi_model_needs_toss_and_nine_matched_players():
    from app.live.prematch_xi import XI_MODEL_PATH, predict_with_xi

    if not XI_MODEL_PATH.exists():
        pytest.skip("xi model not saved")
    india = ["Abhishek Sharma", "Sanju Samson", "Ishan Kishan", "Shreyas Iyer", "Tilak Varma", "Shivam Dube",
             "Naman Dhir", "Axar Patel", "Arshdeep Singh", "Kuldeep Yadav", "Ravi Bishnoi"]
    windies = ["Shai Hope", "Shimron Hetmyer", "Rovman Powell", "Sherfane Rutherford", "Romario Shepherd",
               "Matthew Forde", "Roston Chase", "Akeal Hosein", "Shamar Joseph", "Gudakesh Motie", "Keemo Paul"]
    ok = predict_with_xi("India", "West Indies", "India", None, india, windies, "ranchi")
    assert ok is not None and ok["p_a"] + ok["p_b"] == pytest.approx(1.0)
    swapped = predict_with_xi("West Indies", "India", "India", None, windies, india, "ranchi")
    assert swapped["p_a"] == pytest.approx(ok["p_b"])
    assert predict_with_xi("India", "West Indies", "", None, india, windies, "ranchi") is None
    assert predict_with_xi("India", "West Indies", "India", None, india[:5], windies, "ranchi") is None
