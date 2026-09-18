"""Batter/bowler handedness matchup -- checked 2026-09-18 that this is
genuinely new information, not something a GBM could already reconstruct
from existing columns: the live `bowler_type` feature both wicket and
run-range models use is reduced all the way down to just "pace"/"spin"
(`WicketContract22FeatureComputer._classify_bowler_type`,
`RunRangeV3FeatureComputer`'s equivalent) -- bowling ARM (left/right) is
discarded entirely, even though the raw source data
(`data/external/cricsheet_player_styles.csv`) has it, and even though
`app/ml/player_style_registry.py`'s `normalize_bowling_style` already
parses it correctly (left_arm_pace/right_arm_pace/right_arm_offbreak/
left_arm_orthodox/wrist_spin/left_arm_wristspin) -- that function is
defined but used nowhere in the entire codebase. Same-handed vs
opposite-handed is a real, distinct cricket dynamic (natural angle into
or away from the batter, especially pronounced for spin), and no
existing feature lets the model see it at all.

Leakage-safe the same way as every other per-over feature here: the row
for over N uses the actual upcoming bowler and the striker on strike as
of over N's own first ball -- exactly the same information
BASE_FEATURES' `bowler_*`/`striker_*` features already assume is known
at prediction time (this project's live serving already requires
`bowler_name` be passed in when known, degrading gracefully to
"__UNKNOWN__" otherwise -- same convention followed here).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from app.ml.ipl_identities import canonical_player_id
from app.ml.player_style_registry import normalize_batting_style, normalize_bowling_style

UNKNOWN = "__UNKNOWN__"


def _load_style_lookups(root: Path) -> tuple[dict[str, str], dict[str, str]]:
    styles_path = root / "data/external/cricsheet_player_styles.csv"
    batting: dict[str, str] = {}
    bowling: dict[str, str] = {}
    if not styles_path.is_file():
        return batting, bowling
    styles = pd.read_csv(styles_path)
    for cid, bat_style, bowl_style in zip(
        styles["cricsheet_id"], styles["batting_style"], styles["bowling_style"]
    ):
        if not isinstance(cid, str):
            continue
        player_id = f"player:{cid}"
        batting[player_id] = normalize_batting_style(bat_style)
        bowling[player_id] = normalize_bowling_style(bowl_style)
    return batting, bowling


def _bowler_arm(normalized_bowling_style: str) -> str:
    if normalized_bowling_style.startswith("left_arm"):
        return "left"
    if normalized_bowling_style in {"right_arm_pace", "right_arm_offbreak"}:
        return "right"
    if normalized_bowling_style == "wrist_spin":
        # normalize_bowling_style's "wrist_spin" bucket covers legbreak/
        # googly bowlers, who are overwhelmingly right-arm in the real
        # data this project trains on (a left-arm wrist-spinner is
        # already its own distinct "left_arm_wristspin" bucket) --
        # treated as right-arm here rather than unknown, consistent with
        # that asymmetry rather than discarding real information.
        return "right"
    return "unknown"


def build_handedness_matchup_dataset(
    project_root: Path, scopes: tuple[str, ...] = ("ipl", "t20i")
) -> pd.DataFrame:
    batting_style, bowling_style = _load_style_lookups(project_root)

    paths: list[Path] = []
    for scope in scopes:
        paths.extend((project_root / "data/raw/cricsheet" / scope).glob("*.json"))
    paths.sort(key=lambda p: p.name)

    rows: list[dict[str, Any]] = []
    for path in paths:
        raw = json.loads(path.read_text(encoding="utf-8"))
        registry = raw["info"].get("registry", {}).get("people", {})
        regular_innings = [
            item for item in raw.get("innings", []) if not bool(item.get("super_over", False))
        ]
        for innings_number, innings in enumerate(regular_innings, start=1):
            for source_over in innings.get("overs", []):
                deliveries = source_over.get("deliveries", [])
                if not deliveries:
                    continue
                over_number = int(source_over["over"]) + 1
                first = deliveries[0]
                striker_id = canonical_player_id(str(first.get("batter") or ""), registry)
                bowler_id = canonical_player_id(str(first.get("bowler") or ""), registry)
                bat_style = batting_style.get(striker_id, UNKNOWN)
                bowl_style = bowling_style.get(bowler_id, UNKNOWN)
                arm = _bowler_arm(bowl_style) if bowl_style != UNKNOWN else "unknown"
                if bat_style == UNKNOWN or arm == "unknown":
                    handedness_match = UNKNOWN
                elif (bat_style == "right_hand" and arm == "right") or (
                    bat_style == "left_hand" and arm == "left"
                ):
                    handedness_match = "same"
                else:
                    handedness_match = "opposite"
                rows.append(
                    {
                        "source_file": path.name,
                        "innings": innings_number,
                        "over": over_number,
                        "bowler_arm": arm,
                        "handedness_match": handedness_match,
                    }
                )

    frame = pd.DataFrame(rows)
    return frame.sort_values(["source_file", "innings", "over"]).reset_index(drop=True)
