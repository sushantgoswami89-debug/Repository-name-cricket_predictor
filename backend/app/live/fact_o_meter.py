"""Fact-o-Meter: the name and the look of the 100%-record facts (batter, bowler, venue).

One place for the name, the meter and the wording style so it is easy to change. Every number in a line is a real
count from the data; only the phrasing is playful. The meter bar is always full (these are 100% records); the
flames grow with the sample (1 flame under 20 matches/innings, 2 under 40, 3 from 40). A fact is only shown when
the gap to the average player is large (MAX_BASELINE).
"""

NAME = "Fact-o-Meter"
BAR = "▰" * 10


def label(fmt: str) -> str:
    """"Fact-o-Meter (T20)" / "Fact-o-Meter (ODI)" -- every fact says which format it is from."""
    return f"{NAME} ({fmt.upper()})"


def meter(n: int) -> str:
    """"▰▰▰▰▰▰▰▰▰▰ 26/26 🔥🔥" -- full bar, the count, flames by sample size."""
    return f"{BAR} {n}/{n} " + "🔥" * (1 if n < 20 else 2 if n < 40 else 3)


MAX_BASELINE = 0.70  # a fact must be a big step up: the average player does it at most 70% of the time (gap >= 30 points)


def versus(baseline: float, who: str) -> str:
    """"(avg batter 43%)" -- what the typical player manages, so the size of the gap is visible."""
    return f"(avg {who} {baseline:.0%})"


def card(fmt: str, head: str, fact: str, n: int, baseline: float, who: str) -> str:
    """Two fun lines for the console: title + who, then the fact with the meter and the average to beat."""
    return f"🎯 {NAME.upper()} ({fmt.upper()}) · {head}\n   {fact}  {meter(n)} {versus(baseline, who)}"
