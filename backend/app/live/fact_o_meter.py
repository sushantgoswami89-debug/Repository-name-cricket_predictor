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


def meter(n: int) -> str:
    """"▰▰▰▰▰▰▰▰▰▰  🔥🔥" -- full bar, flames by sample size (the count itself is in the fact where it matters)."""
    return f"{BAR}  " + "🔥" * (1 if n < 20 else 2 if n < 40 else 3)


def card(fmt: str, head: str, fact: str, n: int) -> str:
    """Two fun lines for the console: title + who, then the fact with the meter."""
    return f"🎯 {NAME.upper()} ({fmt.upper()}) · {head} \n   {fact}  {meter(n)}"
