"""CLI for app/live/prematch_analysis.py -- prints the same pre-match history
block the automatic briefing sends.

Usage:
    python3 prematch_analysis.py "India" "West Indies" --venue "Ekana Cricket Stadium, Lucknow"
"""

from __future__ import annotations

import argparse

from app.live.prematch_analysis import prematch_text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("team_a")
    parser.add_argument("team_b")
    parser.add_argument("--venue", default="", help='"<stadium>, <city>"')
    args = parser.parse_args()
    print(prematch_text(args.team_a, args.team_b, args.venue) or "No analysis available (unknown team or missing data).")


if __name__ == "__main__":
    main()
