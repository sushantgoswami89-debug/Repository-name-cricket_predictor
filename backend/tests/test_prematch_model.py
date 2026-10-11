"""Pre-match win model: point-in-time features, orientation-free output, briefing text."""

import re

import pytest

from app.live.prematch_analysis import _bat_first_team, prematch_text
from app.live.prematch_model import MODEL_PATH, build_dataset, predict
from app.live.match_weather import match_day_weather as REAL_MATCH_DAY_WEATHER  # before any test patches it

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


def test_estimate_block_is_win_percent_then_form():
    text = prematch_text("India", "West Indies", "Ekana Cricket Stadium, Lucknow", "India", "bowl")
    lines = text.splitlines()
    assert len(lines) == 3
    assert lines[0].startswith("Win %: India ") and "| West Indies " in lines[0]
    assert lines[1].startswith("Form (last 5): India ") and "| West Indies " in lines[1]
    import re

    assert re.fullmatch(r"Head to head: India - \d vs WI - \d in last 5 matches", lines[2]), lines[2]
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
    assert len(text.splitlines()) == 3


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


def test_short_pitch_and_dew_wording():
    from app.live.prematch_analysis import _dew_short, _pitch_short

    assert _pitch_short("Batting friendly; Dry surface | Likely good grip for slower balls/spin as the surface wears") == "Pitch: flat (Batting friendly; Dry surface)"
    assert _pitch_short("Seam friendly; Grassy surface").startswith("Pitch: green")
    assert _pitch_short("Spin friendly; Dusty surface").startswith("Pitch: turning")
    assert _pitch_short("") == "Pitch: unknown"  # never invents green / turning
    assert _dew_short({"weight": "HEAVY", "chance": 90, "onset_h": 18.67}) == "Dew: HEAVY -- 90%, sets in ~18:40"
    assert _dew_short({"weight": "LITTLE", "chance": 5, "onset_h": None}) == "Dew: LITTLE -- 5%"


def test_header_has_weather_dew_and_pitch_only_the_estimate_is_one_block(monkeypatch):
    from app.live.prematch_analysis import prematch_header

    monkeypatch.setattr("app.live.match_weather.match_day_weather", lambda city, day=None: WET_EVENING)
    lines = prematch_header("India", "West Indies", "JSCA International Stadium Complex, Ranchi", "West Indies", "bowl",
                            pitch_note="Batting friendly")
    assert [l.split(":")[0] for l in lines] == ["Weather", "Dew"]  # Dew and Pitch share one line
    assert lines[1].endswith("Pitch: flat (Batting friendly)")
    assert not any(l.startswith("Advantage") for l in lines)  # the estimate is not repeated up here


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


def test_ground_summary_matches_the_requested_format():
    import re

    from app.live.prematch_analysis import venue_summary

    text = venue_summary("ranchi")
    t20, results = text.split(" | ")
    assert re.fullmatch(
        r"T20 - \d+ 1st inns avg \d+, avg wkts - \d+, pace - \d\.\d/spin - \d\.\d", t20), t20
    assert int(re.match(r"T20 - (\d+)", t20).group(1)) > 11   # every men's T20 competition, not just T20I + IPL (11 matches)
    assert "ODI" not in t20 and "avg" not in results and "2nd" not in text
    pace, spin = (float(x) for x in re.search(r"pace - ([\d.]+)/spin - ([\d.]+)", t20).groups())
    wk = float(re.search(r"avg wkts - (\d+)", t20).group(1))
    assert pace + spin <= wk + 0.5  # bowler-credited wickets cannot exceed all wickets (rounding slack)
    assert re.fullmatch(
        r"bat 1st wins \d+% / chase wins \d+% / toss winner wins \d+%, toss winner chooses (bat|bowl) \d+%( \[small sample\])?",
        results), results
    bat, chase = map(int, re.search(r"wins (\d+)% / chase wins (\d+)%", results).groups())
    assert abs(bat + chase - 100) <= 1
    assert venue_summary("nowhere-ville") == "T20 - no history here"


def test_odi_prematch_block_and_log_are_kept_apart_from_t20(tmp_path, monkeypatch):
    from app.live import prematch_log
    from app.live.prematch_odi import odi_text, predict_base

    p = predict_base("India", "West Indies")
    assert p and abs(p["p_a"] + p["p_b"] - 1) < 1e-9 and p["p_a"] > 0.5
    monkeypatch.setattr(prematch_log, "LOG_PATH", tmp_path / "p.jsonl")
    prematch_log._LAST.clear()
    text = odi_text("India", "West Indies", {"match_id": "cricbuzz:odi1", "source": "cricbuzz", "city": "x"})
    assert text.splitlines()[0].startswith("Win %: India ") and "ODI" in text.splitlines()[0]
    assert text.splitlines()[1].startswith("Form (last 5)") and text.splitlines()[2].startswith("Head to head")
    prematch_log.record_result("cricbuzz:odi1", "India", "India won", tmp_path / "r.jsonl")
    assert prematch_log.score(tmp_path / "p.jsonl", tmp_path / "r.jsonl")["matches_logged"] == 0          # T20 report
    assert prematch_log.score(tmp_path / "p.jsonl", tmp_path / "r.jsonl", fmt="odi")["by_stage"]["pre-toss"]["right"] == 1
    assert odi_text("Afghanistan", "India").startswith("Win %: not available")


def test_weather_failure_is_retried_not_cached_and_flagged(monkeypatch):
    from app.live import match_weather, prematch_analysis

    calls = {"n": 0}

    def flaky(city, day_iso):
        calls["n"] += 1
        return None if calls["n"] < 3 else WET_EVENING

    monkeypatch.setattr(match_weather, "_match_day_weather_once", flaky)
    monkeypatch.setattr("time.sleep", lambda s: None)
    assert REAL_MATCH_DAY_WEATHER("ranchi") == WET_EVENING and calls["n"] == 3  # two failures, then success

    prematch_analysis._TTL.clear()
    monkeypatch.setattr("app.live.match_weather.match_day_weather", lambda city, day=None: None)
    text = prematch_text("India", "West Indies", "JSCA International Stadium Complex, Ranchi", "West Indies", "bowl")
    assert "dew unknown" in text.splitlines()[0]
    assert ("weather", "ranchi") not in prematch_analysis._TTL  # the failure was not cached


STAGES_OK = None


def _stages_available():
    from app.live.prematch_xi import STAGES_PATH

    return STAGES_PATH.exists()


def test_stages_keep_the_base_fixed_and_steps_add_up():
    import pytest as _pytest

    from app.live.prematch_xi import predict_stages

    if not _stages_available():
        _pytest.skip("staged model not trained")
    india = ["Abhishek Sharma", "Sanju Samson", "Ishan Kishan", "Shreyas Iyer", "Tilak Varma", "Shivam Dube",
             "Naman Dhir", "Axar Patel", "Arshdeep Singh", "Kuldeep Yadav", "Mayank Yadav"]
    windies = ["Shai Hope", "Shimron Hetmyer", "Rovman Powell", "Sherfane Rutherford", "Romario Shepherd",
               "Matthew Forde", "Roston Chase", "Akeal Hosein", "Shamar Joseph", "Gudakesh Motie", "Keemo Paul"]
    pre = predict_stages("India", "West Indies", "", None, [], [], "ranchi")
    toss = predict_stages("India", "West Indies", "India", WET_EVENING, [], [], "ranchi")
    full = predict_stages("India", "West Indies", "India", WET_EVENING, india, windies, "ranchi")
    assert (pre["stage"], toss["stage"], full["stage"]) == ("pre-toss", "after toss", "with XIs")
    assert pre["p_base"] == _pytest.approx(toss["p_base"]) == _pytest.approx(full["p_base"])  # base never moves
    assert pre["p_a"] == _pytest.approx(pre["p_base"])
    assert toss["p_a"] == _pytest.approx(toss["p_base"] + toss["dew_shift"])
    assert full["p_a"] == _pytest.approx(full["p_base"] + full["dew_shift"] + full["quality_shift"] + full["matchup_shift"])
    assert toss["dew_shift"] < 0  # India bat first, West Indies chase into dew
    flipped = predict_stages("West Indies", "India", "India", WET_EVENING, windies, india, "ranchi")
    assert flipped["p_a"] == _pytest.approx(full["p_b"])  # the order the teams are named in does not matter
    assert predict_stages("Afghanistan", "India", "", None, [], [], "dubai") is None


def test_missing_forecast_is_flagged_in_the_win_percent_line(monkeypatch):
    import pytest as _pytest

    if not _stages_available():
        _pytest.skip("staged model not trained")
    text = prematch_text("India", "West Indies", "JSCA International Stadium Complex, Ranchi", "West Indies", "bowl")
    assert "dew unknown" in text.splitlines()[0]  # the autouse fixture returns no forecast


def test_rain_risk_rises_with_rain_and_is_capped():
    from app.live.rain_risk import CAP_INTERRUPTED, CAP_NO_RESULT, rain_risk

    if not (rain_risk({"eve_rain_mm": 0.0, "eve_rain_hours": 0}) or None):
        import pytest as _pytest

        _pytest.skip("rain model not trained")
    dry = rain_risk({"eve_rain_mm": 0.0, "eve_rain_hours": 0})
    drizzle = rain_risk({"eve_rain_mm": 1.0, "eve_rain_hours": 2})
    storm = rain_risk({"eve_rain_mm": 40.0, "eve_rain_hours": 7})
    assert dry["p_interrupted"] < drizzle["p_interrupted"] < storm["p_interrupted"] <= CAP_INTERRUPTED
    assert dry["p_no_result"] < drizzle["p_no_result"] < storm["p_no_result"] <= CAP_NO_RESULT
    assert all(r["p_no_result"] <= r["p_interrupted"] for r in (dry, drizzle, storm))
    assert rain_risk(None) is None and rain_risk({"rain_mm": 1.0}) is None  # no match-window field -> no guess


def test_win_percent_gets_a_no_result_share_when_rain_threatens(monkeypatch):
    from app.live import prematch_analysis

    wet = {**WET_EVENING, "eve_rain_mm": 6.0, "eve_rain_hours": 4, "rain_mm": 9.0, "hours": {}}
    monkeypatch.setattr("app.live.match_weather.match_day_weather", lambda city, day=None: wet)
    prematch_analysis._TTL.clear()
    line = prematch_text("India", "West Indies", "JSCA International Stadium Complex, Ranchi", "West Indies", "bowl").splitlines()[0]
    import re

    parts = [int(x) for x in re.findall(r"(\d+)%", line)]
    assert "No result " in line and sum(parts[:3]) == 100  # India + West Indies + No result add to 100
    dry = {**WET_EVENING, "eve_rain_mm": 0.0, "eve_rain_hours": 0, "hours": {}}
    monkeypatch.setattr("app.live.match_weather.match_day_weather", lambda city, day=None: dry)
    prematch_analysis._TTL.clear()
    line = prematch_text("India", "West Indies", "JSCA International Stadium Complex, Ranchi", "West Indies", "bowl").splitlines()[0]
    assert "No result" not in line and sum(int(x) for x in re.findall(r"(\d+)%", line)[:2]) == 100  # dry day: plain two-way split
    assert "rain risk" in prematch_analysis._weather_line(wet)


def test_prematch_log_records_changes_only_and_scores_against_results(tmp_path):
    from app.live import prematch_log as log

    preds, results = tmp_path / "p.jsonl", tmp_path / "r.jsonl"
    base = {"source": "cricbuzz", "team_a": "India", "team_b": "West Indies", "city": "ranchi",
            "pct_nr": 0, "p_nr": 0.0}
    pre = {**base, "stage": "pre-toss", "pct_a": 70, "pct_b": 30, "p_a": 0.70, "p_b": 0.30}
    toss = {**base, "stage": "after toss", "pct_a": 64, "pct_b": 36, "p_a": 0.64, "p_b": 0.36}
    assert log.log_prediction("cricbuzz:1", pre, preds) is True
    assert log.log_prediction("cricbuzz:1", pre, preds) is False      # unchanged -> not logged again
    assert log.log_prediction("cricbuzz:1", toss, preds) is True      # new stage and numbers
    assert log.log_prediction("cricbuzz:1", toss, preds) is False
    assert len(preds.read_text().splitlines()) == 2

    assert log.record_result("cricbuzz:1", "West Indies", "West Indies won", results) is True
    assert log.record_result("cricbuzz:1", "India", "again", results) is False   # a match is recorded once
    s = log.score(preds, results)
    assert s["results"] == 1 and s["matches_logged"] == 1
    assert s["by_stage"]["pre-toss"]["right"] == 0 and s["by_stage"]["after toss"]["right"] == 0   # India favoured, WI won
    assert s["by_stage"]["pre-toss"]["brier"] == pytest.approx(0.70**2)         # (0.70 - 0)^2
    assert s["by_stage"]["after toss"]["brier"] == pytest.approx(0.64**2)

    log.log_prediction("cricbuzz:2", {**pre, "team_a": "A", "team_b": "B", "p_a": 0.6, "p_b": 0.4, "pct_a": 60, "pct_b": 40}, preds)
    log.record_result("cricbuzz:2", None, "no result", results)                  # washed out: counted apart, not scored
    assert log.score(preds, results, teams=None)["no_result"] == 1
    # only matches between two ICC top-10 sides are scored by default
    assert log.score(preds, results)["no_result"] == 0 and log.score(preds, results)["matches_logged"] == 1


def test_briefing_logs_the_prediction_when_given_a_match_id(tmp_path, monkeypatch):
    from app.live import prematch_log

    monkeypatch.setattr(prematch_log, "LOG_PATH", tmp_path / "p.jsonl")
    prematch_log._LAST.clear()
    prematch_text("India", "West Indies", "JSCA International Stadium Complex, Ranchi", "West Indies", "bowl",
                  log_context={"match_id": "cricbuzz:9", "source": "cricbuzz"})
    import json

    rec = json.loads((tmp_path / "p.jsonl").read_text().splitlines()[0])
    assert rec["match_id"] == "cricbuzz:9" and rec["stage"] == "after toss" and rec["team_a"] == "India"
    assert rec["pct_a"] + rec["pct_b"] + rec["pct_nr"] == 100
    assert prematch_text("India", "West Indies", "JSCA International Stadium Complex, Ranchi", "West Indies", "bowl") is not None
    assert len((tmp_path / "p.jsonl").read_text().splitlines()) == 1  # no match_id -> nothing logged


def test_team_without_history_gets_an_explicit_no_win_percent_line():
    text = prematch_text("Afghanistan", "Zimbabwe", "Sharjah Cricket Stadium, Sharjah")
    assert text == "Win %: not available -- no match history for Afghanistan in our data"


def test_both_directions_of_the_batting_bowling_matchup_are_shown_with_full_xis():
    import re

    india = ["Abhishek Sharma", "Sanju Samson", "Ishan Kishan", "Shreyas Iyer", "Tilak Varma", "Shivam Dube",
             "Naman Dhir", "Axar Patel", "Arshdeep Singh", "Kuldeep Yadav", "Mayank Yadav"]
    windies = ["Shai Hope", "Shimron Hetmyer", "Rovman Powell", "Sherfane Rutherford", "Romario Shepherd",
               "Matthew Forde", "Roston Chase", "Akeal Hosein", "Shamar Joseph", "Gudakesh Motie", "Keemo Paul"]
    text = prematch_text("India", "West Indies", "JSCA International Stadium Complex, Ranchi", "West Indies", "bowl",
                         {"India": india, "West Indies": windies})
    lines = text.splitlines()
    pattern = r"(India|WI) bat v (India|WI) bowl: [+-]\d+ runs vs avg \| (India|WI) attack pace \d+% / spin \d+%.*"
    matchup = [l for l in lines if " bat v " in l]
    assert len(matchup) == 2 and all(re.fullmatch(pattern, l) for l in matchup), matchup
    assert {l.split(" bat v ")[0] for l in matchup} == {"India", "WI"}  # one line per direction
    no_xi = prematch_text("India", "West Indies", "JSCA International Stadium Complex, Ranchi", "West Indies", "bowl")
    assert " bat v " not in no_xi  # nothing to show before the line-ups are known


def test_dew_line_states_the_win_percent_points_for_the_chasing_side(monkeypatch):
    from app.live import prematch_analysis

    gaps = {str(h): 1.0 for h in range(12, 24)}
    hours = {h: {"temp": 21.0, "dew_point": 20.0, "humidity": 95} for h in range(0, 24)}
    wet = {**WET_EVENING, "eve_temp": 21.0, "eve_dew_point": 20.0, "hours": hours, "sunset_h": 17.5, "utc_offset_s": 19800,
           "gap_h": gaps, "eve_rain_mm": 0.0, "eve_rain_hours": 0}
    monkeypatch.setattr("app.live.match_weather.match_day_weather", lambda city, day=None: wet)
    prematch_analysis._TTL.clear()
    lines = prematch_analysis.prematch_header("India", "West Indies", "JSCA International Stadium Complex, Ranchi",
                                              "West Indies", "bowl", pitch_note="Batting friendly")
    assert re.search(r"Dew: HEAVY -- \d+%, sets in ~\d\d:\d\d \(WI \+\d+ pts\), Pitch: flat", lines[1]), lines[1]
    pre = prematch_analysis.prematch_header("India", "West Indies", "JSCA International Stadium Complex, Ranchi", pitch_note="")
    assert "pts" not in pre[1]  # before the toss nobody is the chaser yet


def test_batter_fact_is_a_100_percent_record_shown_only_when_the_ask_is_large():
    import json

    import app.live.prematch_xi as xi
    from app.live.batter_settled import ASK_MIN_RUNS, MIN_INNINGS, PATHS, all_records, batter_records, settled_line

    for fmt in ("t20", "odi"):
        records = all_records(fmt)
        assert records, fmt
        rows = json.loads(PATHS[fmt].read_text(encoding="utf-8"))
        for name, claims in records.items():
            for target, (balls_needed, innings, base) in claims.items():     # every claim is literally true in its format
                picked = [r[3] for r in rows if r[1] == name and r[2] >= balls_needed]
                assert len(picked) == innings >= MIN_INNINGS and min(picked) >= target and base <= 0.70, (fmt, name, target)
    assert set(all_records("t20")) != set(all_records("odi"))                 # formats are kept apart
    assert 100 in batter_records("Fakhar Zaman", "odi")
    original = xi.resolve_player
    xi.resolve_player = lambda full, team: full                               # live name -> Cricsheet name: identity here
    try:
        name, claims = next(iter(all_records("t20").items()))
        target = max(claims)
        n_balls = claims[target][0]
        runs_now = max(0, target - ASK_MIN_RUNS - 5)                          # a big ask: 35 runs still to get
        line = settled_line(name, "X", n_balls, runs_now)
        assert line == (f"🎯 FACT-O-METER (T20) · {name} {runs_now}*({n_balls}) \n"
                        f"   Every single time he has faced {n_balls}+ balls, he has gone on to make "
                        f"{ {50: 'a FIFTY', 100: 'a HUNDRED'}.get(target, f'{target}+ runs') }  ▰▰▰▰▰▰▰▰▰▰  " + line.rsplit("  ", 1)[1])
        assert line.rstrip().endswith("🔥")
        assert settled_line(name, "X", n_balls - 1, runs_now) == ""           # not yet at the ball count
        assert settled_line(name, "X", n_balls, target - ASK_MIN_RUNS + 1) == ""   # ask too small now (29 to go)
        assert settled_line("SA Yadav", "X", 41, 31) == ""                    # 50 after 41 balls: nothing big left to ask
    finally:
        xi.resolve_player = original
    assert settled_line("Nobody Unknown", "India", 60, 20) == ""


def test_bowler_records_are_literally_true_in_the_data_and_line_respects_the_trigger():
    import json

    import app.live.prematch_xi as xi
    from app.live.bowler_records import ASK_MIN_WICKETS, MIN_INNINGS, PATHS, all_records, bowler_line, overs_to_balls

    assert overs_to_balls("3.2") == 20 and overs_to_balls(4) == 24 and overs_to_balls("x") is None
    original = xi.resolve_player
    xi.resolve_player = lambda full, team: full
    try:
        for fmt in ("t20", "odi"):
            records = all_records(fmt)
            assert records and all(n >= MIN_INNINGS and base <= 0.70 for _, _, n, base in records.values()), fmt
            rows = json.loads(PATHS[fmt].read_text(encoding="utf-8"))
            for name, (k, balls_needed, innings, _base) in records.items():
                picked = [r[4] for r in rows if r[1] == name and r[2] >= balls_needed]
                assert len(picked) == innings and min(picked) >= k, (fmt, name)
                if k - 0 < ASK_MIN_WICKETS:   # "a wicket" is a small ask: never shown, even at 0 wickets
                    assert bowler_line(name, "X", balls_needed, 0, fmt) == ""
        big = next(((n, r, f) for f in ("t20", "odi") for n, r in all_records(f).items() if r[0] >= 2), None)
        if big:                                                                # a 2+ wicket record: a big ask, shown
            name, (k, balls_needed, innings, base), fmt = big
            line = bowler_line(name, "X", balls_needed, 0, fmt)
            assert line.startswith(f"🎯 FACT-O-METER ({fmt.upper()}) · {name} ") and "▰▰▰▰▰▰▰▰▰▰  🔥" in line and "every one of his innings" in line
            assert bowler_line(name, "X", balls_needed - 1, 0, fmt) == ""      # has not bowled that many balls yet
            assert bowler_line(name, "X", balls_needed, 1, fmt) == ""          # one wicket in: only one more needed
    finally:
        xi.resolve_player = original
    assert set(all_records("t20")) != set(all_records("odi"))
    assert bowler_line("Nobody Unknown", "India", 24, 0) == ""


def test_venue_record_is_per_format_and_matches_the_data():
    from app.live.prematch_analysis import VENUE_RECORD_MIN, _venue_event_table, venue_record, venue_summary

    for fmt in ("t20", "odi"):
        _, per_city = _venue_event_table(fmt)
        shown = {c: venue_record(c, fmt) for c in per_city if venue_record(c, fmt)}
        for city, text in shown.items():                                  # every shown record is literally true
            results = per_city[city][text.split(" has won")[0]]
            assert len(results) >= VENUE_RECORD_MIN and all(results), (fmt, city)
    assert "8 of 8" in venue_record("faisalabad", "odi") and venue_record("faisalabad", "t20") == ""
    assert venue_summary("faisalabad", "odi").endswith("🎯 Fact-o-Meter (ODI): the team chasing has won 8 of 8 here ▰▰▰▰▰▰▰▰▰▰  🔥")
    assert "Fact-o-Meter" not in venue_summary("faisalabad", "t20")
    assert venue_record("ranchi", "t20") == "" and venue_record("nowhere-ville") == ""   # Ranchi toss-winner-bowls: 6 of 11


def test_odi_watcher_briefs_once_updates_at_toss_and_records_the_result(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from app.live import prematch_log
    from app.live.odi_watch import OdiBriefingWatcher, is_men_international_odi, winner_from_status

    monkeypatch.setattr(prematch_log, "LOG_PATH", tmp_path / "p.jsonl")
    monkeypatch.setattr(prematch_log, "RESULTS_PATH", tmp_path / "r.jsonl")
    prematch_log._LAST.clear()

    def match(state, status="", **kw):
        base = dict(match_id="9", label="India vs Australia", match_format="ODI", match_type="International", state=state,
                    status=status, venue="Wankhede Stadium, Mumbai", url="http://x", match_desc="1st ODI", series="Test series")
        return SimpleNamespace(**{**base, **kw})

    assert is_men_international_odi(match("Preview"))
    assert not is_men_international_odi(match("Preview", label="India Women vs Australia Women"))
    assert not is_men_international_odi(match("Preview", label="India A vs Australia A"))
    assert not is_men_international_odi(match("Preview", label="Nepal vs Canada"))                    # not top 10
    assert is_men_international_odi(match("Preview", label="Nepal vs Canada"), only_top10=False)
    assert not is_men_international_odi(match("Preview", match_format="T20"))
    assert winner_from_status("India won by 5 wkts", ["India", "Australia"]) == "India"
    assert winner_from_status("Match tied", ["India", "Australia"]) is None

    listing = [match("Preview")]
    toss = {"value": ("", "")}
    watcher = OdiBriefingWatcher(discovery=SimpleNamespace(matches=lambda: listing), toss_fetcher=lambda m: toss["value"])

    def run():
        watcher.cycle()
        out = []
        while not watcher._results.empty():
            out.append(watcher._results.get())
        return out

    first = run()
    assert len(first) == 1 and first[0][0] == "briefing" and "Win %: India" in first[0][2] and "ODI" in first[0][2]
    assert run() == []                                                  # nothing changed: nothing re-sent
    toss["value"] = ("Australia", "bowl")
    second = run()
    assert len(second) == 1 and "Toss is in" in second[0][2] and "Australia won" in second[0][2]
    assert run() == []                                                  # toss known: the watcher stops updating
    listing[0] = match("Complete", "Australia won by 3 wkts")
    done = run()
    assert done and "result recorded" in done[0][2]
    assert prematch_log.score(tmp_path / "p.jsonl", tmp_path / "r.jsonl", fmt="odi")["by_stage"]["pre-toss"]["n"] == 1
    assert run() == []                                                  # recorded once
    # joined mid-match: no stale "pre-match" briefing
    late = OdiBriefingWatcher(discovery=SimpleNamespace(matches=lambda: [match("In Progress", match_id="10")]), toss_fetcher=lambda m: ("", ""))
    late.cycle()
    assert late._results.empty()


def test_result_facts_are_literally_true_in_the_data_and_show_only_on_the_way():
    import json

    from app.live.result_facts import APPROACH, MIN_MATCHES, PATHS, all_claims, result_line

    for fmt in ("t20", "odi"):
        claims = all_claims(fmt)
        assert claims, fmt
        rows = json.loads(PATHS[fmt].read_text(encoding="utf-8"))
        for name, items in claims.items():
            for c in items:                                                  # every record is literally true in its format
                if c["kind"] == "bat":
                    hits = [r[9] for r in rows if r[1] == name and r[3] and r[4] >= c["x"] and c["inn"] in (0, r[3])]
                else:
                    hits = [r[9] for r in rows if r[1] == name and r[6] and r[8] >= 6 and r[7] >= c["x"] and c["inn"] in (0, r[6])]
                assert len(hits) == c["n"] >= MIN_MATCHES and set(hits) == {1 if c["outcome"] == "WON" else 0}, (fmt, name, c)
                assert c["fluke"] <= 0.01
    assert set(all_claims("t20")) != set(all_claims("odi"))
    stoinis = all_claims("odi")["MP Stoinis"][0]
    assert stoinis["kind"] == "bat" and stoinis["outcome"] == "LOST" and stoinis["x"] == 50
    line = result_line("Marcus Stoinis", "MP Stoinis", "Australia", "bat", 41, 38, 2, "odi")
    assert line == ("🎯 FACT-O-METER (ODI) · Marcus Stoinis 41*(38) \n"
                    "   Australia have LOST all 7 matches in which he scored 50+  ▰▰▰▰▰▰▰▰▰▰  🔥")
    assert result_line("Marcus Stoinis", "MP Stoinis", "Australia", "bat", int(APPROACH * 50) - 1, 20, 2, "odi") == ""   # not on the way yet
    assert result_line("Marcus Stoinis", "MP Stoinis", "Australia", "bat", 50, 45, 2, "odi") == ""                       # already there
    assert result_line("Virat Kohli", "V Kohli", "India", "bat", 90, 80, 2, "odi") == ""                                 # Kohli: no 100% record
    assert result_line("Nobody", None, "X", "bat", 40, 30, 1) == ""
