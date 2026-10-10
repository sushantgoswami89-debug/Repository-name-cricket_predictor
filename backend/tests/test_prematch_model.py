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
    monkeypatch.setattr("app.live.prematch_analysis.toi_pitch_note", lambda a, b: "")
    from app.live import prematch_analysis

    prematch_analysis._TTL.clear()


def test_briefing_is_compact_and_shows_win_percent_with_its_measured_accuracy():
    text = prematch_text("India", "West Indies", "Ekana Cricket Stadium, Lucknow", "India", "bowl")
    assert "Win %: India" in text and "West Indies" in text
    assert "unseen full-member games" in text and "not a forecast" in text
    assert "Form (last 5)" in text and "H2H India" in text
    assert len(text.splitlines()) <= 7  # the user asked for a short briefing
    assert "(pre-toss)" in prematch_text("India", "West Indies", "Ekana Cricket Stadium, Lucknow")


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


def test_body_stays_short_with_a_forecast(monkeypatch):
    monkeypatch.setattr("app.live.match_weather.match_day_weather", lambda city, day=None: WET_EVENING)
    text = prematch_text("India", "West Indies", "JSCA International Stadium Complex, Ranchi", "India", "bowl")
    assert "Win %: India" in text
    assert len(text.splitlines()) <= 7


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


def test_track_call_uses_toi_pitch_field_not_its_boilerplate_and_admits_gaps():
    from app.live.prematch_analysis import _track_call

    flat = _track_call("Batting friendly; Dry surface | Likely good grip for slower balls/spin as the surface wears", "ranchi", 16)
    assert flat.startswith("Pitch: flat") and "turning" not in flat  # boilerplate 'grip for slower balls' is ignored
    assert "green / seam-friendly" in _track_call("Seam friendly; Grassy surface", "ranchi", 20)
    assert "turning" in _track_call("Spin friendly; Dusty surface", "ranchi", 20)
    unknown = _track_call("", "ranchi", 20)
    assert "no reliable source" in unknown  # never invents green/turning
    assert "overcast" in _track_call("", "ranchi", 85)


def test_header_has_weather_pitch_and_advantage_lines(monkeypatch):
    from app.live.prematch_analysis import prematch_header

    monkeypatch.setattr("app.live.match_weather.match_day_weather", lambda city, day=None: WET_EVENING)
    pre = prematch_header("India", "West Indies", "JSCA International Stadium Complex, Ranchi", pitch_note="Batting friendly")
    heads = [line.split(":")[0] for line in pre if not line.startswith("  ")]
    assert heads == ["Weather", "Dew", "Pitch", "Advantage"]  # the ground record lives on the title line
    assert next(l for l in pre if l.startswith("Pitch")).startswith("Pitch: flat")
    assert pre[-1] == "Advantage: toss pending -- dew would help whoever chases"
    post = prematch_header("India", "West Indies", "JSCA International Stadium Complex, Ranchi", "West Indies", "bowl", pitch_note="")
    assert post[-1].startswith("Advantage: West Indies on conditions") and "dew helps West Indies" in post[-1]


def _day(gaps_by_hour, sunset=17.5):
    return {"hours": {h: {"temp": 20.0 + g, "dew_point": 20.0, "humidity": 80} for h, g in gaps_by_hour.items()},
            "sunset_h": sunset, "utc_offset_s": 19800}


def test_dew_outlook_weight_chance_and_onset():
    from app.live.prematch_analysis import _dew_line, dew_outlook

    # the real Ranchi forecast shape: gap shrinks from 7.7C at 15:00 to 0.4C at 23:00
    ranchi = _day({14: 8.2, 15: 7.7, 16: 6.0, 17: 4.2, 18: 3.4, 19: 2.7, 20: 1.6, 21: 1.0, 22: 0.7, 23: 0.4}
                  | {h: 9.0 for h in range(0, 14)})
    out = dew_outlook(ranchi)  # no start time given -> assumes 19:00
    assert out["weight"] == "HEAVY" and out["chance"] >= 85
    assert 18.0 <= out["onset_h"] <= 19.0  # gap crosses 3.0C between 18:00 (3.4) and 19:00 (2.7)
    assert out["onset_h"] < out["chase_h"]
    line = _dew_line(out)
    assert line.startswith("Dew: HEAVY") and "before the chase" in line and "start time assumed" in line

    dry = _day({h: 12.0 for h in range(0, 24)})
    out = dew_outlook(dry, 19.0)
    assert out["weight"] == "LITTLE" and out["chance"] <= 5
    assert _dew_line(out).startswith("Dew: LITTLE")

    medium = _day({h: 2.2 for h in range(0, 24)})
    assert dew_outlook(medium, 19.0)["weight"] == "MEDIUM"
    small = _day({h: 3.5 for h in range(0, 24)})
    assert dew_outlook(small, 19.0)["weight"] == "SMALL"
    assert dew_outlook(None) is None and dew_outlook({"hours": {19: {}}}) is None


def test_dew_line_uses_the_real_start_time_when_known():
    from app.live.prematch_analysis import _dew_line, dew_outlook

    day = _day({h: 1.0 if h >= 20 else 8.0 for h in range(0, 24)})
    known = _dew_line(dew_outlook(day, 17.0))   # 17:00 start -> chase from ~18:50, dew arrives ~19:40
    assert "start time assumed" not in known and "mid-chase" in known


def test_ground_summary_is_one_t20_stats_line():
    from app.live.prematch_analysis import venue_summary

    text = venue_summary("ranchi")
    assert "\n" not in text
    assert text.startswith("T20 Stats - 11 matches (T20I 4 + IPL 7): avg score ")
    for part in ("1st inns avg ", "avg wkts ", "/inns", "most wkts by one bowler: pace ", " / spin "):
        assert part in text
    assert "All T20" not in text
    assert venue_summary("nowhere-ville") == "T20 Stats - no T20I or IPL history here"
