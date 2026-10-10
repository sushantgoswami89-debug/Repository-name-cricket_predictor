"""Compare our in-play win probability with Cricbuzz's own, over by over, on finished matches.

Cricbuzz: the win-probability curve from its graphs page (cb-win-curve-<match>.json, made by
capture_cricbuzz_match.py -- one fetch gives the whole match, no live logging needed).
Ours: the GBM+NN ensemble in data/live/match_winner_ensemble_shadow_log.jsonl (probability that the BATTING team
wins, logged for the coming over). Our over k forecast is made after k-1 overs, so it is paired with Cricbuzz's
point after k-1 overs. Outcome: the match winner from Cricbuzz's scorecard header.
Caveat: some of our logged forecasts came from runs with a wrongly restarted state (see notes 2026-10-06); the
over-by-over list shows them so they can be spotted.

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
        ours[(str(r["match_id"]), int(r["innings"]), int(r["over"]))] = r["ensemble_probability"]  # last write per over
    pairs = []
    for curve_file in sorted(LIVE.glob("cb-win-curve-*.json")):
        curve = json.loads(curve_file.read_text(encoding="utf-8"))
        match_id, winner = curve["match_id"], curve.get("winner")
        if not winner:
            continue
        for innings_id, series in curve["points"].items():
            batting = curve["innings"].get(innings_id)
            by_over = {p["over"]: p for p in series}
            for (m, inn, over), p_ours in ours.items():
                if m != match_id or str(inn) != innings_id or (over - 1) not in by_over:
                    continue
                cb = by_over[over - 1].get(batting)
                if cb is None:
                    continue
                pairs.append((match_id, inn, over, batting, p_ours, cb / 100.0, float(winner == batting)))
    print(f"paired over-forecasts: {len(pairs)} from {len({p[0] for p in pairs})} match(es)")
    if not pairs:
        print("nothing to compare yet")
        return
    by_match = defaultdict(list)
    for p in pairs:
        by_match[(p[0], p[1])].append(p)
    for (match_id, inn), rows in sorted(by_match.items()):
        print(f"match {match_id} innings {inn} ({rows[0][3]} batting, {'won' if rows[0][6] else 'lost'}):")
        print("   " + " ".join(f"o{r[2]}:{r[4]:.0%}/{r[5]:.0%}" for r in sorted(rows, key=lambda r: r[2])) + "   (ours/Cricbuzz)")
    brier = lambda i: sum((r[i] - r[6]) ** 2 for r in pairs) / len(pairs)
    print(f"\nBrier over all pairs: ours {brier(4):.4f} | Cricbuzz {brier(5):.4f} (lower is better; n={len(pairs)}).")
    late = [r for r in pairs if r[2] >= 8]
    if late:
        b = lambda i: sum((r[i] - r[6]) ** 2 for r in late) / len(late)
        print(f"overs 8+ only: ours {b(4):.4f} | Cricbuzz {b(5):.4f} (n={len(late)})")


if __name__ == "__main__":
    main()
