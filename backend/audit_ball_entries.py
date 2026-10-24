"""Ball-by-ball audit: is every delivery we recorded the delivery Cricbuzz says was bowled?

Cricbuzz's over separator carries one token per delivery, in order (runs, W, Wd, N1, B4, L1 ...). This replays the pages the driver saved
as one continuous session and compares OUR deliveries with those tokens POSITION BY POSITION for every over we hold completely
(runs, legal or not, wicket), and checks the names on every ball (bowler = the separator's bowler, striker and bowler present).

--raw switches the over-separator repair OFF, so the numbers measure the ball-by-ball entries themselves (what the live record would be
without Cricbuzz's own correction); without it the separator rebuilds mismatching overs, so a clean result there shows the repair works.

    cd backend && ../.venv/bin/python audit_ball_entries.py [--raw] 151598 170697 ...
"""

from __future__ import annotations

import io
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

from app.live import cricbuzz_reader
from app.live.over_summary import parse_over_summary
from audit_recording import separators
from replay_saved_pages import _PagePlayer, build_session, load_pages


def audit(match_id: str, raw: bool) -> tuple[int, int, int, int]:
    pages = load_pages(match_id)
    truth = separators(match_id, pages)
    player = _PagePlayer(pages)
    with tempfile.TemporaryDirectory() as folder:
        session = build_session(match_id, Path(folder), player)
        with redirect_stdout(io.StringIO()):
            for _ in pages:
                player.advance()
                try:
                    session.poll_once()
                except Exception:
                    pass
        ours: dict[tuple[int, int], list] = {}
        for (_mid, innings), known in session.reader._all_deliveries.items():
            for delivery in known.values():
                ours.setdefault((innings, delivery.over), []).append(delivery)
    overs = exact = ball_diffs = name_issues = 0
    print(f"=== {match_id} ({'raw ball entries' if raw else 'after the separator repair'})")
    for (innings, over), sep in sorted(truth.items()):
        tokens = parse_over_summary(sep.get("overSummary", ""))
        balls = sorted(ours.get((innings, over), []), key=lambda d: d.ball)
        if tokens is None or sum(1 for b in balls if b.is_legal) < 6:
            continue   # unreadable summary, or an over we did not hold from its first ball
        overs += 1
        problems = []
        if len(balls) != len(tokens):
            problems.append(f"{len(balls)} deliveries vs {len(tokens)}")
        else:
            for position, (ball, token) in enumerate(zip(balls, tokens), start=1):
                if (ball.total_runs, ball.is_legal, ball.wicket_kind is not None) != (token.total_runs, token.is_legal, token.wicket):
                    problems.append(f"ball {position}: ours {ball.total_runs}{'' if ball.is_legal else ' (extra)'}"
                                    f"{'W' if ball.wicket_kind else ''} vs {sep.get('overSummary','').split()[position - 1]}")
                    ball_diffs += 1
        bowler = (sep.get("bowlerObj") or {}).get("playerName", "")
        for ball in balls:
            if not ball.bowler or not ball.striker:
                name_issues += 1
                problems.append(f"{ball.over - 1}.{ball.ball} has no {'bowler' if not ball.bowler else 'striker'} name")
            elif bowler and ball.bowler != bowler:
                name_issues += 1
                problems.append(f"{ball.over - 1}.{ball.ball} bowler {ball.bowler} vs {bowler}")
        if problems:
            print(f"  inn {innings} over {over} [{sep.get('overSummary','').strip()}]: " + "; ".join(problems[:4]))
        else:
            exact += 1
    print(f"  {overs} complete overs: {exact} exact ball for ball and name for name; {ball_diffs} ball differences, {name_issues} name issues")
    return overs, exact, ball_diffs, name_issues


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    raw = "--raw" in sys.argv
    if raw:
        cricbuzz_reader._attach_over_separators = lambda chunks, objects: None
    totals = [0, 0, 0, 0]
    for match in args or ["151598"]:
        for index, value in enumerate(audit(match, raw)):
            totals[index] += value
    print(f"ALL: {totals[0]} overs, {totals[1]} exact, {totals[2]} ball differences, {totals[3]} name issues")
