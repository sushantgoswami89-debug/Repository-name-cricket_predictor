"""Unit tests for run_live_auto.py's Cricbuzz-primary/TOI-backup dedup
logic. Built 2026-09-18 after finding TOI's discovery has no way to be
retroactively audited (every TOI results/archive URL returns an empty
SPA shell) -- the fix is architectural (never rely on a human noticing a
gap), so the dedup behavior itself needs real test coverage, not just a
live smoke test against whatever happens to be live right now.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import run_live_auto
from app.live.cricbuzz_discovery import CricbuzzMatchSummary
from app.live.toi_discovery import ToiMatchSummary


def _cricbuzz_match(match_id="152742", label="India vs Afghanistan", **overrides):
    values = dict(
        match_id=match_id,
        url=f"https://www.cricbuzz.com/live-cricket-scores/{match_id}/x",
        label=label,
        series="",
        match_desc="1st T20I",
        match_format="T20",
        match_type="International",
        state="In Progress",
        status="",
        venue="",
        is_live=True,
        is_upcoming=False,
    )
    values.update(overrides)
    return CricbuzzMatchSummary(**values)


def _toi_match(match_id="toi-1", label="India vs Afghanistan", **overrides):
    values = dict(
        match_id=match_id,
        url="https://timesofindia.indiatimes.com/x",
        label=label,
        match_format="T20I",
        series="India tour of Afghanistan",
        venue="",
        start_time="",
        status="Live",
        is_live=True,
        is_upcoming=False,
        weather="",
        temperature="",
        humidity="",
        wind="",
        rain="",
        pitch="",
        likely_track="",
        dew_outlook="",
        team_type="International",
    )
    values.update(overrides)
    return ToiMatchSummary(**values)


class FakeCricbuzzDiscovery:
    def __init__(self, matches=()):
        self.matches = matches

    def live_matches(self):
        return self.matches


class FakeToiDiscovery:
    def __init__(self, matches=()):
        self.matches = matches

    def live_summaries(self):
        return self.matches


@pytest.fixture(autouse=True)
def _no_real_briefings(monkeypatch):
    # scan() would otherwise make real HTTP calls for pre-match briefings.
    monkeypatch.setattr(run_live_auto, "cricbuzz_briefing_text", lambda *a, **k: None)
    monkeypatch.setattr(run_live_auto, "toi_briefing_text", lambda *a, **k: None)


def _state(tmp_path: Path) -> run_live_auto.LiveAutoState:
    return run_live_auto.LiveAutoState(project_root=tmp_path, publisher=None)


def test_cricbuzz_match_is_attached_as_primary(tmp_path):
    state = _state(tmp_path)
    cricbuzz = FakeCricbuzzDiscovery((_cricbuzz_match(),))
    toi = FakeToiDiscovery(())

    state.scan(cricbuzz, toi)

    assert "152742" in state.active_cricbuzz
    assert state.covered_pairs == {frozenset({"india", "afghanistan"}): "cricbuzz"}


def test_toi_only_attaches_when_cricbuzz_does_not_have_the_same_teams(tmp_path):
    state = _state(tmp_path)
    cricbuzz = FakeCricbuzzDiscovery(())
    toi = FakeToiDiscovery((_toi_match(),))

    state.scan(cricbuzz, toi)

    assert "toi-1" in state.active_toi
    assert state.covered_pairs == {frozenset({"india", "afghanistan"}): "toi"}


def test_toi_backup_is_skipped_when_cricbuzz_already_covers_the_same_match(tmp_path):
    # The real point of this whole design: when both sources see the same
    # match, Cricbuzz (primary) wins and TOI never double-attaches it.
    state = _state(tmp_path)
    cricbuzz = FakeCricbuzzDiscovery((_cricbuzz_match(label="India vs Afghanistan"),))
    toi = FakeToiDiscovery((_toi_match(label="Afghanistan vs India"),))  # order-independent

    state.scan(cricbuzz, toi)

    assert "152742" in state.active_cricbuzz
    assert state.active_toi == {}
    assert state.covered_pairs == {frozenset({"india", "afghanistan"}): "cricbuzz"}


def test_toi_backup_stays_attached_once_claimed_even_if_cricbuzz_later_sees_it(tmp_path):
    # No mid-match source-switching: once TOI has genuinely claimed a
    # match no one else was covering, a later scan where Cricbuzz also
    # starts listing it must not double-attach or steal it back.
    state = _state(tmp_path)
    toi_match = _toi_match()
    state.scan(FakeCricbuzzDiscovery(()), FakeToiDiscovery((toi_match,)))
    assert "toi-1" in state.active_toi

    cricbuzz_match = _cricbuzz_match(label="India vs Afghanistan")
    state.scan(FakeCricbuzzDiscovery((cricbuzz_match,)), FakeToiDiscovery((toi_match,)))

    assert "toi-1" in state.active_toi
    assert state.active_cricbuzz == {}


def test_different_matches_from_both_sources_both_get_attached(tmp_path):
    state = _state(tmp_path)
    cricbuzz = FakeCricbuzzDiscovery((_cricbuzz_match(label="Australia vs Zimbabwe"),))
    toi = FakeToiDiscovery((_toi_match(label="India vs Afghanistan"),))

    state.scan(cricbuzz, toi)

    assert "152742" in state.active_cricbuzz
    assert "toi-1" in state.active_toi


def test_detaches_cricbuzz_match_and_frees_the_pair_when_no_longer_live(tmp_path):
    state = _state(tmp_path)
    match = _cricbuzz_match()
    state.scan(FakeCricbuzzDiscovery((match,)), FakeToiDiscovery(()))
    assert "152742" in state.active_cricbuzz

    state.scan(FakeCricbuzzDiscovery(()), FakeToiDiscovery(()))

    assert state.active_cricbuzz == {}
    assert state.covered_pairs == {}


def test_freed_pair_can_then_be_attached_via_toi_backup(tmp_path):
    # After Cricbuzz stops seeing a match (detached, pair freed), a
    # later scan where only TOI still sees it must be able to pick it up.
    state = _state(tmp_path)
    cricbuzz_match = _cricbuzz_match(label="India vs Afghanistan")
    state.scan(FakeCricbuzzDiscovery((cricbuzz_match,)), FakeToiDiscovery(()))
    assert "152742" in state.active_cricbuzz

    toi_match = _toi_match(label="India vs Afghanistan")
    state.scan(FakeCricbuzzDiscovery(()), FakeToiDiscovery((toi_match,)))

    assert state.active_cricbuzz == {}
    assert "toi-1" in state.active_toi
    assert state.covered_pairs == {frozenset({"india", "afghanistan"}): "toi"}


def test_unresolved_team_pair_label_never_blocks_attachment(tmp_path):
    # Both discovery modules fall back to the bare match id as the label
    # when team names can't be resolved -- _team_pair() must return None
    # for that shape rather than crash or accidentally collide two
    # different unresolved matches.
    state = _state(tmp_path)
    cricbuzz = FakeCricbuzzDiscovery((_cricbuzz_match(match_id="1", label="1"),))
    toi = FakeToiDiscovery((_toi_match(match_id="toi-2", label="toi-2"),))

    state.scan(cricbuzz, toi)

    assert "1" in state.active_cricbuzz
    assert "toi-2" in state.active_toi
    assert state.covered_pairs == {}


class _RecordingPublisher:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []

    def publish(self, key, text):
        self.sent.append((key, text))
        return True


class _AlwaysFailingToiReader:
    def fetch(self, url):
        from app.live.toi_reader import ToiFeedError

        raise ToiFeedError("simulated: feed never reconciles.")


def test_toi_backup_stuck_alert_fires_once_and_resets(tmp_path, monkeypatch):
    from app.live.pipeline import VerifiedLivePredictionPipeline

    publisher = _RecordingPublisher()
    state = run_live_auto.LiveAutoState(
        project_root=tmp_path, publisher=publisher, stuck_alert_seconds=60.0
    )
    state.active_toi["toi-1"] = (
        "https://timesofindia.indiatimes.com/x",
        _AlwaysFailingToiReader(),
        VerifiedLivePredictionPipeline(publisher=None),
    )

    clock = [1000.0]
    monkeypatch.setattr(run_live_auto.time, "time", lambda: clock[0])

    state.poll()
    assert publisher.sent == []

    clock[0] += 61.0
    state.poll()
    assert len(publisher.sent) == 1
    key, text = publisher.sent[0]
    assert key.startswith("stuck_alert:toi:toi-1:")
    assert "erroring" in text

    clock[0] += 5.0
    state.poll()  # still erroring -- must not repeat
    assert len(publisher.sent) == 1
