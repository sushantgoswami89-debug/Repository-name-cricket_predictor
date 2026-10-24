"""Chase Fact-o-Meter cards from the historical record (models/situation_facts_v1.json, build_situation_facts.py).

One card (user 2026-10-24, after India chased 226 against New Zealand: "these facts should have come under Fact-o-Meter"): the target card,
sent ONCE before the chase starts, one line: how often a first innings that big has been defended in men's T20 (since 2019). The user
rejected a card at 6/10/15 overs of the chase ("we are pushing it very high"), so there is none. Every number is a real count from the
file; a card needs at least MIN_MATCHES matches behind it.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from app.live.fact_o_meter import card

PATH = Path(__file__).resolve().parents[3] / "models" / "situation_facts_v1.json"
MIN_MATCHES = 40
MIN_TOTAL = 190          # first-innings totals below this get no card


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
    """"A first innings of 225+ has been chased down only 7% of the time (12 of 177 ...)" for the biggest round threshold (steps of 5) at or
    below the first innings total; None unless the total is big and usually defended."""
    totals = _facts().get("totals") or []
    if first_innings_total < MIN_TOTAL:       # "150+" pools every bigger total too: only a big total makes the line mean something
        return None
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
    if rate < 0.65:                  # only a total that is usually defended makes a card; a toss-up or a chaser's game is not "a fact"
        return None
    return card("T20", f"{chasing_team} chase {first_innings_total + 1}",
                f"A first innings of {threshold}+ has been chased down only {chase_wins / len(rows) * 100:.0f}% of the time "
                f"({chase_wins} of {len(rows)} men's T20 matches since 2019)", len(rows), _fill(rate))
