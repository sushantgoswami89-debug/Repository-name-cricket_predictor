"""Print the ODI pre-match briefing for ODI fixtures Cricbuzz lists today (no Telegram, nothing published).

Usage: python show_prematch_odi.py [--all]   (default: only fixtures between two ICC top-10 sides)
"""

import sys

from app.live.cricbuzz_discovery import CricbuzzMatchDiscovery
from app.live.prematch_log import ICC_TOP_10
from app.live.prematch_odi import current_state, odi_briefing


def main() -> None:
    show_all = "--all" in sys.argv
    for match in CricbuzzMatchDiscovery().matches():
        if match.match_format.upper() != "ODI" or "women" in match.label.lower() or " A " in f" {match.label} ":
            continue
        teams = [t.strip() for t in match.label.split(" vs ")]
        national = set(current_state().ratings) | ICC_TOP_10  # skips state / franchise sides
        if match.match_type.lower() == "domestic" or not all(t in national for t in teams):
            continue
        if not show_all and not all(t in ICC_TOP_10 for t in teams):
            continue
        print(f"[{match.state}] {match.match_desc} -- {match.series}")
        print(odi_briefing(match.label, match.venue))
        print()


if __name__ == "__main__":
    main()
