"""2026-08-25: GBM+NN wicket ensemble shadow -- see
docs/finding_wicket_nn_gbm_ensemble_real_win.md. Covers the shadow log's
core behavior and PredictionEngine's fault isolation (a broken/missing
ensemble runtime must never affect the served GBM wicket_probability)."""

from pathlib import Path

from app.ml.wicket_ensemble_shadow_log import WicketEnsembleShadowLog
from app.models.live_match_state import LiveMatchState
from app.models.match_context import MatchContext


def _context() -> MatchContext:
    return MatchContext(
        match_id="test-match", team1="Team Batting", team2="Team Bowling",
        team1_players=["Batter One"], team2_players=["Bowler One"],
        venue="Test Venue", format="IPL", competition="IPL",
        live=LiveMatchState(
            over=10, striker="Batter One", bowler="Bowler One",
            score_before_over=100, wkts_down_before_over=3, wickets_in_hand=7,
            legal_balls_bowled=54, balls_remaining=66, current_run_rate=8.0,
            is_chase=0, runs_required=0, required_run_rate=0.0,
            recent_legal_balls=12, recent_runs_per_ball=1.2, recent_dot_rate=0.3,
            recent_single_rate=0.4, recent_boundary_rate=0.1, recent_wicket_rate=0.05,
        ),
    )


def test_shadow_log_record_and_summarize(tmp_path: Path):
    log = WicketEnsembleShadowLog(
        log_path=tmp_path / "log.jsonl", outcomes_path=tmp_path / "outcomes.jsonl",
        marker_path=tmp_path / "marker.json", min_matches_for_summary=1,
    )
    for over, wicket, in ((1, False), (2, True), (3, False), (4, False), (5, True),
                          (6, False), (7, False), (8, False), (9, True), (10, False)):
        log.record_prediction(match_id="m1", innings=1, over=over,
                               gbm_probability=0.3, nn_probability=0.25, ensemble_probability=0.28)
        log.record_outcome(match_id="m1", innings=1, over=over, wicket_fell=wicket)

    assert log.matches_with_outcome_count() == 1
    summary = log.summarize()
    assert summary["rows"] == 10
    assert summary["matches"] == 1


def test_shadow_log_outcome_idempotent(tmp_path: Path):
    log = WicketEnsembleShadowLog(
        log_path=tmp_path / "log.jsonl", outcomes_path=tmp_path / "outcomes.jsonl",
        marker_path=tmp_path / "marker.json",
    )
    log.record_outcome(match_id="m1", innings=1, over=5, wicket_fell=True)
    log.record_outcome(match_id="m1", innings=1, over=5, wicket_fell=False)  # duplicate key, must not overwrite
    rows = log._load_jsonl(log.outcomes_path)
    assert len(rows) == 1
    assert rows[0]["wicket_fell"] is True


def test_shadow_log_reminder_fires_once(tmp_path: Path):
    log = WicketEnsembleShadowLog(
        log_path=tmp_path / "log.jsonl", outcomes_path=tmp_path / "outcomes.jsonl",
        marker_path=tmp_path / "marker.json", min_matches_for_summary=1,
    )
    assert not log.reminder_already_sent()
    log.record_outcome(match_id="m1", innings=1, over=1, wicket_fell=False)
    assert log.matches_with_outcome_count() >= log.min_matches_for_summary
    log.mark_reminder_sent(match_count=1)
    assert log.reminder_already_sent()


def test_prediction_engine_missing_ensemble_runtime_still_serves_gbm():
    """If the ensemble artifacts/subprocess aren't available, PredictionEngine
    must still construct and predict exactly like before this feature
    existed -- never a hard dependency."""
    from app.ml.prediction_engine import PredictionEngine

    engine = PredictionEngine(wicket_ensemble_runtime=None)
    # Force the "artifacts missing" path by pointing at a runtime that
    # failed to construct, same as __init__'s own try/except would leave it.
    engine._wicket_ensemble_runtime = None
    result = engine.predict(_context())

    assert result.wicket_probability is not None
    assert result.metadata["wicket_ensemble_shadow"] is None


def test_prediction_engine_broken_ensemble_runtime_does_not_affect_served_result():
    """A broken ensemble runtime (predict() raises) must never affect the
    served GBM wicket_probability -- same fault-isolation standard as
    every other shadow feature in this project."""
    from app.ml.prediction_engine import PredictionEngine

    class _BrokenEnsembleRuntime:
        def predict(self, **kwargs):
            raise RuntimeError("simulated ensemble failure")

    engine = PredictionEngine(wicket_ensemble_runtime=_BrokenEnsembleRuntime())
    result = engine.predict(_context())

    assert result.wicket_probability is not None
    assert 0.0 <= result.wicket_probability <= 1.0
    assert result.metadata["wicket_ensemble_shadow"] is None
