"""How accurate is our ball-by-ball RECORD? Compare it with what Cricbuzz itself says about each finished over.

Cricbuzz posts an over separator after every over (overRuns, the over's ball tokens, the team score after it). Those are the
authority: this replays the pages the driver saved (data/raw/cricbuzz-pages/<match id>) through a fresh session, then for each
over where we hold every ball compares our runs / balls / wickets with the separator's. Nothing is written to the real logs.

    cd backend && ../.venv/bin/python audit_recording.py 170697 [more ids]
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path

from app.live.cricbuzz_reader import _extract_balanced, _extract_flight_chunks
from replay_saved_pages import _PagePlayer, build_session, load_pages


def separators(match_id: str, pages: list[tuple[int, str]]) -> dict[tuple[int, int], dict]:
    """(innings, over number) -> the LAST separator Cricbuzz sent for that over (it may be re-posted with corrections)."""
    found: dict[tuple[int, int], dict] = {}
    marker = '{"commType":"over-separator"'
    for _stamp, html in pages:
        for chunk in _extract_flight_chunks(html):
            for hit in re.finditer(re.escape(marker), chunk):
                raw = _extract_balanced(chunk, hit.start())
                try:
                    obj = json.loads(raw)
                except (TypeError, ValueError):
                    continue
                sep = obj.get("overSeparator")
                if isinstance(sep, dict) and sep.get("overNumber") is not None and obj.get("inningsId"):
                    found[(int(obj["inningsId"]), int(sep["overNumber"]))] = sep
    return found


def audit(match_id: str) -> None:
    pages = load_pages(match_id)
    truth = separators(match_id, pages)
    player = _PagePlayer(pages)
    import io
    from contextlib import redirect_stdout

    with tempfile.TemporaryDirectory() as folder:
        session = build_session(match_id, Path(folder), player)
        with redirect_stdout(io.StringIO()):
            for _ in pages:
                player.advance()
                try:
                    session.poll_once()
                except Exception:
                    pass
        ours = {}
        for (mid, innings), known in session.reader._all_deliveries.items():
            for delivery in known.values():
                ours.setdefault((innings, delivery.over), []).append(delivery)
    checked = bad = skipped = 0
    print(f"=== {match_id}: {len(pages)} pages, {len(truth)} over separators seen")
    for (innings, over), sep in sorted(truth.items()):
        balls = ours.get((innings, over), [])
        legal = sum(1 for b in balls if b.is_legal)
        runs = sum(b.total_runs for b in balls)
        wickets = sum(1 for b in balls if b.wicket_kind)
        if legal < 6:
            skipped += 1   # we never held the whole over (attached during it): nothing to compare
            continue
        checked += 1
        want_runs = int(sep.get("overRuns", -1))
        want_tokens = [t for t in str(sep.get("overSummary", "")).split() if t]
        want_wickets = sum(1 for t in want_tokens if re.fullmatch(r"W\d*", t.upper()))  # "Wd" is a wide, not a wicket
        problems = []
        if runs != want_runs:
            problems.append(f"runs ours {runs} vs Cricbuzz {want_runs}")
        if len(balls) != len(want_tokens):
            problems.append(f"deliveries ours {len(balls)} vs Cricbuzz {len(want_tokens)}")
        if wickets != want_wickets:
            problems.append(f"wickets ours {wickets} vs Cricbuzz {want_wickets}")
        team_score = str((sep.get("batTeamObj") or {}).get("teamScore", ""))
        match_total = re.fullmatch(r"(\d+)-(\d+)", team_score)
        if match_total and balls:
            last = max(balls, key=lambda b: (b.ball, b.timestamp_ms))
            if (last.feed_total, last.feed_wickets) != (int(match_total.group(1)), int(match_total.group(2))):
                problems.append(f"total after the over ours {last.feed_total}/{last.feed_wickets} vs Cricbuzz {team_score}")
        if problems:
            bad += 1
            detail = " ".join(f"{b.over - 1}.{b.ball}:{b.total_runs}{'W' if b.wicket_kind else ''}" for b in sorted(balls, key=lambda b: (b.ball)))
            print(f"  inn {innings} over {over}: " + "; ".join(problems) + f"   [tokens {' '.join(want_tokens)}]  ours: {detail}")
    print(f"  checked {checked} complete overs: {checked - bad} exact, {bad} different; {skipped} not comparable (attached mid-over)")


if __name__ == "__main__":
    for match in sys.argv[1:] or ["170697"]:
        audit(match)
