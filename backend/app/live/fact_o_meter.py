"""Fact-o-Meter: the name and the look of the 100%-record facts (batter, bowler, venue).

One place for the name, the meter and the wording style so it is easy to change. Every number in a line is a real
count from the data; only the phrasing is playful. The meter bar is always full (these are 100% records); the
flames grow with the sample (1 flame under 20 matches/innings, 2 under 40, 3 from 40). A fact is only a fact worth
showing when it is a big step up on the average player (MAX_BASELINE) and, for batters and bowlers, a big ask is
still ahead (see batter_settled / bowler_records).
"""

NAME = "Fact-o-Meter"
BAR = "▰" * 10
MAX_BASELINE = 0.70  # the average player does it at most 70% of the time (gap >= 30 points)


def label(fmt: str) -> str:
    """"Fact-o-Meter (T20)" / "Fact-o-Meter (ODI)" -- every fact says which format it is from."""
    return f"{NAME} ({fmt.upper()})"


def meter(n: int, fill: int = 10) -> str:
    """"▰▰▰▰▰▰▰▰▰▰  🔥🔥" -- bar and flames by sample size (the count itself is in the fact where it matters).
    A 100% record is a full bar; an average-gap fact fills only part of it (see ratio_fill)."""
    fill = max(0, min(10, fill))
    return "▰" * fill + "▱" * (10 - fill) + "  " + "🔥" * (1 if n < 20 else 2 if n < 40 else 3)


def ratio_fill(ratio: float) -> int:
    """Bar length for "N times his usual level": 1.5x -> 3 of 10, 2x -> 5, 3x -> 7, 5x -> 8."""
    return max(3, min(9, round(10 * (1 - 1 / max(ratio, 1.0)))))  # never a full bar: that is for 100% records


def card(fmt: str, head: str, fact: str, n: int, fill: int = 10) -> str:
    """Two fun lines for the console: title + who, then the fact with the meter."""
    return f"🎯 {NAME.upper()} ({fmt.upper()}) · {head} \n   {fact}  {meter(n, fill)}"
