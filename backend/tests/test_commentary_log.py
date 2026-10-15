"""Unit tests for the per-delivery commentary log, user-requested
2026-09-18 for detailed post-match commentary review -- see
app/live/commentary_log.py's module docstring for why this exists.
"""

from __future__ import annotations

import json

from app.live.commentary_log import append_commentary_log
from app.live.live_snapshot import LiveDelivery, LiveSnapshot


def _delivery(over: int, ball: int, **overrides) -> LiveDelivery:
    values = dict(
        match_id="match",
        innings=1,
        over=over,
        ball=ball,
        bowler="Ravi Bishnoi",
        striker="Rashid Khan",
        total_runs=1,
        batter_runs=1,
        extras={},
        wicket_kind=None,
        feed_total=1,
        feed_wickets=0,
        timestamp_ms=0,
        commentary="Ravi Bishnoi to Rashid Khan, 1 run",
    )
    values.update(overrides)
    return LiveDelivery(**values)


def _snapshot(deliveries: tuple[LiveDelivery, ...]) -> LiveSnapshot:
    return LiveSnapshot(
        match_id="match",
        match_format="T20",
        batting_team="India",
        bowling_team="Afghanistan",
        innings=1,
        score=1,
        wickets=0,
        overs="0.1",
        is_live=True,
        deliveries=deliveries,
    )


def test_appends_one_json_line_per_new_delivery(tmp_path):
    path = tmp_path / "commentary.jsonl"
    logged: set[tuple[int, int, int]] = set()
    snapshot = _snapshot((_delivery(1, 1), _delivery(1, 2)))

    append_commentary_log(path, "match", snapshot, logged)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["over"] == 1
    assert first["ball"] == 1
    assert first["bowler"] == "Ravi Bishnoi"
    assert first["striker"] == "Rashid Khan"
    assert first["commentary"] == "Ravi Bishnoi to Rashid Khan, 1 run"


def test_does_not_duplicate_already_logged_deliveries(tmp_path):
    path = tmp_path / "commentary.jsonl"
    logged: set[tuple[int, int, int]] = set()
    first_poll = _snapshot((_delivery(1, 1),))
    second_poll = _snapshot((_delivery(1, 1), _delivery(1, 2)))

    append_commentary_log(path, "match", first_poll, logged)
    append_commentary_log(path, "match", second_poll, logged)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2  # not 3 -- ball 1.1 only logged once


def test_wicket_and_extras_are_captured(tmp_path):
    path = tmp_path / "commentary.jsonl"
    logged: set[tuple[int, int, int]] = set()
    delivery = _delivery(
        1, 1,
        total_runs=2,
        batter_runs=0,
        extras={"legbyes": 2},
        wicket_kind="bowled",
        commentary="leg byes, 2 runs",
    )
    snapshot = _snapshot((delivery,))

    append_commentary_log(path, "match", snapshot, logged)

    logged_row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert logged_row["extras"] == {"legbyes": 2}
    assert logged_row["wicket_kind"] == "bowled"


def test_shot_quality_and_near_wicket_are_derived_and_logged(tmp_path):
    path = tmp_path / "commentary.jsonl"
    logged: set[tuple[int, int, int]] = set()
    delivery = _delivery(
        1, 1,
        total_runs=4,
        batter_runs=4,
        commentary="chops it off the inside edge into the on-side for four",
    )
    snapshot = _snapshot((delivery,))

    append_commentary_log(path, "match", snapshot, logged)

    logged_row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert logged_row["shot_quality"] == "edge"
    assert logged_row["near_wicket"] is None


def test_no_new_deliveries_does_not_create_a_file(tmp_path):
    path = tmp_path / "commentary.jsonl"
    logged: set[tuple[int, int, int]] = set()
    snapshot = _snapshot(())

    append_commentary_log(path, "match", snapshot, logged)

    assert not path.exists()


def test_restart_with_fresh_keys_does_not_relog_existing_balls(tmp_path):
    path = tmp_path / "commentary.jsonl"
    snapshot = _snapshot((_delivery(1, 1), _delivery(1, 2)))
    append_commentary_log(path, "match", snapshot, set())

    # Simulated driver restart: new process, empty in-memory key set.
    append_commentary_log(path, "match", snapshot, set())

    assert len(path.read_text(encoding="utf-8").splitlines()) == 2
