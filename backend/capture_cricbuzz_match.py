"""Capture every Cricbuzz page of one or more FINISHED matches (works days later) and their win curve.

Usage: python capture_cricbuzz_match.py 151576 151565 ...
"""

import sys

from app.live.cricbuzz_capture import capture_match

for match_id in sys.argv[1:]:
    info = capture_match(match_id)
    print(f"{match_id}: pages {len(info['saved'])}/8, winner {info['winner']}, win-curve points {info['curve_points']}")
