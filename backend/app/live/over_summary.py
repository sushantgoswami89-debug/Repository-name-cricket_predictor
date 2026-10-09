"""Cricbuzz's own per-over result string ("1 6 4 N1 1 1 4"), parsed and used to
rebuild an over when the ball-by-ball entries disagree with it.

Why: Cricbuzz rewrites ball entries after posting them (prose changes, a
re-bowled ball reuses a ball number, a review replaces an earlier OUT with a
not-out entry), so rebuilding an over only from the commentary stream left a
phantom wicket and wrong runs in two overs of the 2026-10-06 chase. The
over summary is posted once the over ends and carries one token per delivery.

Token meanings, verified against the summed `overRuns` of archived pages
(matches 151565 and 173123, 2026-10-06/-07):
    "4"      four off the bat (any digit = runs off the bat, legal ball)
    "W"      wicket, no runs
    "N"      no-ball, no bat runs      -> 1 run (the penalty)
    "N1"     no-ball + 1 off the bat   -> 2 runs (digits are BAT runs)
    "Wd"     wide                      -> 1 run
    "Wd3"    wide totalling 3          -> 3 runs (digits are the TOTAL)
    "L1"     leg-bye(s)                -> 1 run (digits are the total)
Any other token ("B2", "W1", "Lb...") is NOT verified: the summary is then
treated as unreadable and nothing is rebuilt from it. Extend only with a real
example whose token sum matches `overRuns`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_DIGITS = re.compile(r"^\d+$")
_NOBALL = re.compile(r"^N(\d*)$")
_WIDE = re.compile(r"^Wd(\d*)$")
_LEGBYE = re.compile(r"^L(\d+)$")


@dataclass(frozen=True, slots=True)
class OverToken:
    total_runs: int
    batter_runs: int
    extras: dict[str, int]
    wicket: bool

    @property
    def is_legal(self) -> bool:
        return "wides" not in self.extras and "noballs" not in self.extras


def parse_over_summary(summary: str) -> list[OverToken] | None:
    """One token per delivery, or None when any token is not a verified format
    or the string is empty."""
    parts = str(summary or "").split()
    if not parts:
        return None
    tokens: list[OverToken] = []
    for part in parts:
        if _DIGITS.match(part):
            runs = int(part)
            tokens.append(OverToken(runs, runs, {}, False))
        elif part == "W":
            tokens.append(OverToken(0, 0, {}, True))
        elif (match := _NOBALL.match(part)):
            bat = int(match.group(1) or 0)
            tokens.append(OverToken(bat + 1, bat, {"noballs": 1}, False))
        elif (match := _WIDE.match(part)):
            total = int(match.group(1)) if match.group(1) else 1
            tokens.append(OverToken(total, 0, {"wides": total}, False))
        elif (match := _LEGBYE.match(part)):
            total = int(match.group(1))
            tokens.append(OverToken(total, 0, {"legbyes": total}, False))
        else:
            return None
    return tokens


def tokens_agree_with_total(tokens: list[OverToken], over_runs: object) -> bool:
    """Guard: only trust a parse whose runs add up to Cricbuzz's own overRuns."""
    try:
        return sum(t.total_runs for t in tokens) == int(over_runs)
    except (TypeError, ValueError):
        return False
