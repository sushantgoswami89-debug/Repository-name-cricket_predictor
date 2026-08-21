"""Normalize free-text player style metadata into a small, cricket-meaningful
set of categories, and provide a name -> style lookup.

Source data: data/external/cricsheet_player_styles.csv (see
data/external/README.md for provenance). This is the one place both
offline training-data augmentation and online serving (historical_feature_store)
go through, so the two stay consistent with each other.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from app.ml.ipl_identities import identity_key

DEFAULT_STYLE_CSV_PATH = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "external"
    / "cricsheet_player_styles.csv"
)

UNKNOWN = "unknown"


def normalize_batting_style(raw: Any) -> str:
    if not isinstance(raw, str) or not raw.strip():
        return UNKNOWN
    text = raw.lower()
    if "bat" not in text:
        return UNKNOWN
    if "right" in text:
        return "right_hand"
    if "left" in text:
        return "left_hand"
    return UNKNOWN


def normalize_bowling_style(raw: Any) -> str:
    if not isinstance(raw, str) or not raw.strip():
        return UNKNOWN
    # Multi-style entries ("Right arm Medium, Right arm Offbreak") -- use
    # the first-listed (primary) style.
    text = raw.split(",")[0].strip().lower()
    if not text:
        return UNKNOWN

    is_left = "left" in text
    if "googly" in text or "legbreak" in text or "leg break" in text:
        return "wrist_spin"
    if "offbreak" in text or "off break" in text:
        return "left_arm_orthodox" if is_left else "right_arm_offbreak"
    if "orthodox" in text:
        return "left_arm_orthodox"
    if "wrist" in text:
        return "left_arm_wristspin" if is_left else "wrist_spin"
    if "slow" in text and "arm" not in text:
        return UNKNOWN
    # Everything else with an identifiable arm is pace/medium-pace.
    if "left" in text or "right" in text:
        return "left_arm_pace" if is_left else "right_arm_pace"
    return UNKNOWN


class PlayerStyleRegistry:
    """Name -> (batting_style, bowling_style, playing_role) lookup."""

    def __init__(self, csv_path: Path | str | None = None) -> None:
        self._csv_path = Path(csv_path) if csv_path else DEFAULT_STYLE_CSV_PATH
        self._by_key: dict[str, dict[str, str]] | None = None

    def _ensure_loaded(self) -> None:
        if self._by_key is not None:
            return
        self._by_key = {}
        if not self._csv_path.exists():
            return
        df = pd.read_csv(self._csv_path)
        df = df.drop_duplicates(subset="unique_name", keep="first")
        for row in df.itertuples(index=False):
            name = getattr(row, "unique_name", None)
            if not isinstance(name, str) or not name.strip():
                continue
            key = identity_key(name)
            self._by_key[key] = {
                "batting_style": normalize_batting_style(
                    getattr(row, "batting_style", None)
                ),
                "bowling_style": normalize_bowling_style(
                    getattr(row, "bowling_style", None)
                ),
            }

    def lookup(self, name: str) -> dict[str, str]:
        self._ensure_loaded()
        assert self._by_key is not None
        return self._by_key.get(
            identity_key(name), {"batting_style": UNKNOWN, "bowling_style": UNKNOWN}
        )
