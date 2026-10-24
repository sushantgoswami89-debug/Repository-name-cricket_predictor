"""Chase Fact-o-Meter cards from the historical record (models/situation_facts_v1.json, build_situation_facts.py).

Two cards (user 2026-10-24, after India chased 226 against New Zealand: "these facts should have come under Fact-o-Meter"):
  * target card, once the target is known: how often a first innings that big has been defended in men's T20 (since 2019);
  * checkpoint cards after 6, 10 and 15 overs of a chase: how chasing sides with the same wickets down and a similar required rate fared.
Every number is a real count from the file; a card needs at least MIN_MATCHES matches behind it.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from app.live.fact_o_meter import card

PATH = Path(__file__).resolve().parents[3] / "models" / "situation_facts_v1.json"
MIN_MATCHES = 40
RRR_BAND = 1.0          # runs per over either side of the current required rate
CHECKPOINTS = (6, 10, 15)


@lru_cache(maxsize=1)
def _facts() -> dict:
    try:
        return json.loads(PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _fill(rate: float) -> int:
    """Bar length: how far from a toss-up (50%) the record is."""
    return max(1, min(10, round(abs(rate - 0.5) * 20)))


def target_card(first_innings_total: int, batting_team: str, chasing_team: str) -> str | None:
    """"Teams that made 225+ ... won 92% of 153" for the biggest round threshold (steps of 5) at or below the first innings total."""
    totals = _facts().get("totals") or []
    threshold = (first_innings_total // 5) * 5
    while threshold >= 140:
        rows = [won for total, won in totals if total >= threshold]
        if len(rows) >= MIN_MATCHES:
            break
        threshold -= 5
    else:
        return None
    chase_wins = sum(rows)
    defended = len(rows) - chase_wins
    rate = defended / len(rows)
    if abs(rate - 0.5) < 0.15:       # a toss-up is not a fact worth a card
        return None
    side = (f"sides batting first with {threshold}+ won {defended} of {len(rows)} ({rate * 100:.0f}%)" if rate > 0.5
            else f"chasing sides facing {threshold}+ won {chase_wins} of {len(rows)} ({(1 - rate) * 100:.0f}%)")
    return card("T20", f"{chasing_team} chase {first_innings_total + 1}",
                f"In men's T20 since 2019 {side}", len(rows), _fill(rate))


def checkpoint_card(overs_done: int, score: int, wickets: int, target: int, chasing_team: str) -> str | None:
    """After 6/10/15 overs: chasing sides with the same wickets down (or one either side if that is too few) and a required rate within
    RRR_BAND of the current one."""
    if overs_done not in CHECKPOINTS or target <= 0:
        return None
    rows = (_facts().get("chase") or {}).get(str(overs_done)) or []
    balls_left = (20 - overs_done) * 6
    need = max(0, target - score)
    if need == 0 or balls_left <= 0:
        return None
    rrr = need * 6 / balls_left
    for spread in (0, 1):
        sample = [won for r, w, won in rows if abs(w - wickets) <= spread and abs(r - rrr) <= RRR_BAND]
        if len(sample) >= MIN_MATCHES:
            break
    else:
        return None
    wins = sum(sample)
    rate = wins / len(sample)
    wicket_text = f"{wickets} down" if spread == 0 else f"{max(0, wickets - 1)}-{wickets + 1} down"
    return card("T20", f"{chasing_team} {score}/{wickets} after {overs_done}, need {need} off {balls_left} (RRR {rrr:.1f})",
                f"Chasing sides {wicket_text} needing {max(0.0, rrr - RRR_BAND):.1f}-{rrr + RRR_BAND:.1f} an over went on to win {wins} of "
                f"{len(sample)} ({rate * 100:.0f}%)", len(sample), _fill(rate))
