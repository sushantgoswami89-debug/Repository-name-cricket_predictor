"""Fact-o-Meter: the name and the look of the 100%-record facts (batter, bowler, venue).

One place for the name, the meter and the wording style so it is easy to change. Every number in a line is a real
count from the data; only the phrasing is playful. The meter bar is always full (these are 100% records); the
flames grow with the sample (1 flame under 20 matches/innings, 2 under 40, 3 from 40).
"""

NAME = "Fact-o-Meter"
BAR = "▰" * 10


def label(fmt: str) -> str:
    """"Fact-o-Meter (T20)" / "Fact-o-Meter (ODI)" -- every fact says which format it is from."""
    return f"{NAME} ({fmt.upper()})"


def meter(n: int) -> str:
    """"▰▰▰▰▰▰▰▰▰▰ 26/26 🔥🔥" -- full bar, the count, flames by sample size."""
    return f"{BAR} {n}/{n} " + "🔥" * (1 if n < 20 else 2 if n < 40 else 3)


def card(fmt: str, head: str, fact: str, n: int) -> str:
    """Two fun lines for the console: title + who, then the fact with the meter."""
    return f"🎯 {NAME.upper()} ({fmt.upper()}) · {head}\n   {fact}  {meter(n)}"
