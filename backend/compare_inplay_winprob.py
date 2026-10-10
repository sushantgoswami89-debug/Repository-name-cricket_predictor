"""Compare our in-play win probability with Cricbuzz's own, over by over.

Ours: data/live/match_winner_ensemble_shadow_log.jsonl (GBM+NN ensemble, probability that the BATTING
team wins, logged each over). Cricbuzz's: the `cb_win` percentages in data/live/board-<match>.jsonl
(logged since 2026-10-06; since 2026-10-10 each line also names the batting team). The outcome of a
finished match is read from the Cricbuzz match header via data/live/prematch-results.jsonl or, for the
older matches, the shadow outcomes file. Only matches where all three exist are scored.

Usage: python compare_inplay_winprob.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "data" / "live"


def _lines(path: Path) -> list[dict]:
    try:
        return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    except OSError:
        return []


def main() -> None:
    ours: dict[tuple[str, int, int], float] = {}
    for r in _lines(LIVE / "match_winner_ensemble_shadow_log.jsonl"):
        ours[(str(r["match_id"]), int(r["innings"]), int(r["over"]))] = r["ensemble_probability"]  # last write per over wins
    outcomes = {str(r["match_id"]): r["batting_team_won"] for r in _lines(LIVE / "match_winner_ensemble_shadow_outcomes.jsonl")}
    results = {r["match_id"].split(":")[-1]: r for r in _lines(LIVE / "prematch-results.jsonl")}
    rows = []
    for board_file in sorted(LIVE.glob("board-*.jsonl")):
        match_id = board_file.stem.split("-", 1)[1]
        seen = {}
        for line in _lines(board_file):
            win = (line.get("board") or {}).get("cb_win")
            batting = (line.get("ours") or {}).get("batting_team")
            if not win or not batting or line.get("innings") != 2:
                continue
            over = int(float(win["as_of_over"] or 0)) + 1  # Cricbuzz "as of 14.3" = state during over 15 => our over-15 forecast
            seen[over] = (win.get(batting), line)
        winner = (results.get(match_id) or {}).get("winner")
        for over, (cb_pct, line) in seen.items():
            p_ours = ours.get((match_id, 2, over))
            if p_ours is None or cb_pct is None or winner is None:
                continue
            batting = line["ours"]["batting_team"]
            rows.append((match_id, over, p_ours, cb_pct / 100.0, float(winner == batting)))
    for match_id, batting_won in outcomes.items():
        pass  # older matches have no named batting team in the board log, so they cannot be paired by name
    print(f"paired overs (ours vs Cricbuzz, chase innings, outcome known): {len(rows)}")
    if not rows:
        print("nothing to compare yet: the board log now records the batting team, so pairs accumulate from the next match.")
        return
    by_match = defaultdict(list)
    for r in rows:
        by_match[r[0]].append(r)
    for match_id, rs in by_match.items():
        print(f"match {match_id}: " + ", ".join(f"over {o}: ours {p:.0%} vs Cricbuzz {c:.0%}" for _, o, p, c, _ in rs))
    brier = lambda idx: sum((r[idx] - r[4]) ** 2 for r in rows) / len(rows)
    print(f"Brier ours {brier(2):.4f} | Cricbuzz {brier(3):.4f} (lower is better; n={len(rows)} over-forecasts from {len(by_match)} match(es))")


if __name__ == "__main__":
    main()
