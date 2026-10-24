"""The over outlook: mean runs, chance of a big / quiet over and of a wicket, in the live message (user 2026-10-24)."""

from app.live.pipeline import VerifiedLivePredictionPipeline
from app.ml.over_outlook import DEFAULT_DIR, OverOutlook
from app.models.live_match_state import LiveMatchState


def _output(outlook):
    prediction = {"expected_range": "5-7", "expected_runs": 6.0, "wicket_probability": 0.3, "confidence_level": "MEDIUM",
                  "confidence_percent": 29, "metadata": {"expected_runs_mean": 8.9}}
    return {"over": 9, "score_before_over": 112, "wickets_before_over": 1, "prediction": prediction, "outlook": outlook}


def test_telegram_message_is_plain_and_states_every_fact():
    output = _output({"mean_runs": 10.4, "p_big": 0.54, "p_quiet": 0.13, "p_wicket": 0.26, "big_runs": 10, "quiet_runs": 4})
    output.update(batting_team="India", win_probability={"status": "applied", "win_probability": 0.696, "batting_team": "India",
                                                          "bowling_team": "New Zealand"},
                  previous_over_evaluation={"actual_runs": 10, "expected_range": "9-11", "range_covered": True})
    text = VerifiedLivePredictionPipeline._telegram_text(output)
    assert text == (
        "🏏 Over 9 · India 112/1\n\n"
        "Next over: about 10 runs expected\n"
        "• Big over (10 or more runs): 54% chance\n"
        "• Quiet over (4 or fewer runs): 13% chance\n"
        "• Wicket: 26% chance of at least one\n\n"
        "Who wins: India 70% | New Zealand 30%\n\n"
        "Last over: 10 runs. We said 9 to 11. ✅"
    )
    for jargon in ("Candidate", "point", "Confidence", "Error", "★", "Win Probability"):
        assert jargon not in text


def test_plain_message_without_the_outlook_and_the_technical_text_stays_detailed():
    plain = VerifiedLivePredictionPipeline._telegram_text(_output(None))
    assert "Next over: most likely 5 to 7 runs" in plain and "• Wicket: 30% chance of at least one" in plain and "Big over" not in plain
    technical = VerifiedLivePredictionPipeline._technical_text(_output(
        {"mean_runs": 10.4, "p_big": 0.54, "p_quiet": 0.13, "p_wicket": 0.26, "big_runs": 10, "quiet_runs": 4}))
    assert "Candidate v3 — Over 9 Prediction" in technical and "Confidence:" in technical and "Most likely range: 5-7 (point 6.0)" in technical
    assert "Wicket chance: 26% (at least one wicket in the over)" in technical
    old = VerifiedLivePredictionPipeline._technical_text(_output(None))
    assert "Wicket risk (phase baseline): 30.0%" in old and "Runs: 5-7 (point 6.0) | expected 8.9" in old


def test_outlook_models_load_and_respond_to_the_situation():
    outlook = OverOutlook.try_load(DEFAULT_DIR)
    assert outlook is not None
    early = outlook.predict(LiveMatchState(over=1, wickets_in_hand=10, balls_remaining=120))
    late = outlook.predict(LiveMatchState(over=19, score_before_over=170, wkts_down_before_over=3, wickets_in_hand=7, legal_balls_bowled=108,
                                          balls_remaining=12, current_run_rate=9.4, recent_legal_balls=12, recent_runs_per_ball=1.8))
    assert early and late
    assert late["mean_runs"] > early["mean_runs"] and late["p_big"] > early["p_big"] and late["p_quiet"] < early["p_quiet"]
    assert all(0.0 <= late[k] <= 1.0 for k in ("p_big", "p_quiet", "p_wicket"))
    assert outlook.predict(LiveMatchState(over=30, balls_remaining=120)) is None                 # not a 20-over innings over
    assert OverOutlook.try_load(DEFAULT_DIR.parent / "missing") is None
