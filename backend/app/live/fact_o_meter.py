"""Fact-o-Meter: the name for the 100%-record facts (batter, bowler, venue). One place for the name so it is easy to change."""

NAME = "Fact-o-Meter"


def label(fmt: str) -> str:
    """"Fact-o-Meter (T20)" / "Fact-o-Meter (ODI)" -- every fact says which format it is from."""
    return f"{NAME} ({fmt.upper()})"
