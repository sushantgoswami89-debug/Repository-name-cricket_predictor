"""The over outlook: mean runs, chance of a big / quiet over and of a wicket, in the live message (user 2026-10-24)."""

from app.live.pipeline import VerifiedLivePredictionPipeline
from app.ml.over_outlook import DEFAULT_DIR, OverOutlook
from app.models.live_match_state import LiveMatchState


def _output(outlook):
    prediction = {"expected_range": "5-7", "expected_runs": 6.0, "wicket_probability": 0.3, "confidence_level": "MEDIUM",
                  "confidence_percent": 29, "metadata": {"expected_runs_mean": 8.9}}
    return {"over": 9, "score_before_over": 112, "wickets_before_over": 1, "prediction": prediction, "outlook": outlook}


def test_message_shows_mean_big_quiet_and_wicket_chance_and_keeps_the_range_line():
    text = VerifiedLivePredictionPipeline._telegram_text(_output(
        {"mean_runs": 10.4, "p_big": 0.54, "p_quiet": 0.13, "p_wicket": 0.26, "big_runs": 10, "quiet_runs": 4}))
    assert "Most likely range: 5-7 (point 6.0)" in text and "expected 8.9" not in text
    assert "Expected: about 10.4 runs" in text
    assert "Big over (10+ runs): 54% | Quiet over (4 or fewer): 13%" in text
    assert "Wicket chance: 26% (at least one wicket in the over)" in text
    assert "phase baseline" not in text


def test_message_falls_back_to_the_old_wicket_line_without_an_outlook():
    text = VerifiedLivePredictionPipeline._telegram_text(_output(None))
    assert "Wicket risk (phase baseline): 30.0%" in text and "Big over" not in text and "Runs: 5-7 (point 6.0) | expected 8.9" in text


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
