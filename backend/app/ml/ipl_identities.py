"""Reviewed canonical identities for IPL datasets and live-name resolution."""

from __future__ import annotations

import re
from collections.abc import Mapping

UNKNOWN_PLAYER = "player:unknown"
UNKNOWN_TEAM = "team:unknown"

# Brand changes retain a stable franchise identity. Defunct franchises remain
# distinct and are not merged with replacement teams.
TEAM_ALIASES = {
    "chennai super kings": "team:csk",
    "delhi capitals": "team:dc",
    "delhi daredevils": "team:dc",
    "gujarat titans": "team:gt",
    "kolkata knight riders": "team:kkr",
    "lucknow super giants": "team:lsg",
    "mumbai indians": "team:mi",
    "punjab kings": "team:pbks",
    "kings xi punjab": "team:pbks",
    "rajasthan royals": "team:rr",
    "royal challengers bangalore": "team:rcb",
    "royal challengers bengaluru": "team:rcb",
    "sunrisers hyderabad": "team:srh",
    "deccan chargers": "team:deccan_chargers",
    "gujarat lions": "team:gujarat_lions",
    "kochi tuskers kerala": "team:kochi_tuskers",
    "pune warriors": "team:pune_warriors",
    "rising pune supergiant": "team:rising_pune",
    "rising pune supergiants": "team:rising_pune",
}

CURRENT_FRANCHISES = frozenset(
    {
        "team:csk",
        "team:dc",
        "team:gt",
        "team:kkr",
        "team:lsg",
        "team:mi",
        "team:pbks",
        "team:rr",
        "team:rcb",
        "team:srh",
    }
)

# Official IPL 2026 primary venue list plus explicitly announced secondary
# home grounds. This is season-scoped; it must not be applied to earlier years.
HOME_VENUES_2026 = {
    "team:csk": frozenset({"chennai_chepauk"}),
    "team:dc": frozenset({"delhi_arun_jaitley"}),
    "team:gt": frozenset({"ahmedabad_narendra_modi"}),
    "team:kkr": frozenset({"kolkata_eden_gardens"}),
    "team:lsg": frozenset({"lucknow_ekana"}),
    "team:mi": frozenset({"mumbai_wankhede"}),
    "team:pbks": frozenset(
        {"new_chandigarh_mullanpur", "dharamsala_hpca"}
    ),
    "team:rr": frozenset({"jaipur_sawai_mansingh", "guwahati_barsapara"}),
    "team:rcb": frozenset(
        {"bengaluru_chinnaswamy", "raipur_shaheed_veer_narayan"}
    ),
    "team:srh": frozenset({"hyderabad_rajiv_gandhi"}),
}


def identity_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).strip()


def canonical_team_id(name: str) -> str:
    """Return a reviewed franchise identity, preserving unknown teams."""
    key = identity_key(name)
    return TEAM_ALIASES.get(key, f"team:unresolved:{key}" if key else UNKNOWN_TEAM)


def canonical_player_id(name: str, registry: Mapping[str, str]) -> str:
    """Prefer Cricsheet's person UUID; use a deterministic unresolved fallback."""
    player_uuid = str(registry.get(name, "")).strip()
    if player_uuid:
        return f"player:{player_uuid}"
    key = identity_key(name)
    return f"player:unresolved:{key}" if key else UNKNOWN_PLAYER


def first_last_key(full_name: str) -> str:
    """Reduce a full legal name to its first and last token, normalized --
    e.g. "Patrick James Cummins" -> "patrick cummins". Bridges the gap
    between Cricsheet's own player identity (often initials, e.g.
    "PJ Cummins") and how live feeds like TOI actually refer to players
    (their common first name + surname, e.g. "Pat Cummins"). Doesn't
    resolve nickname variants (Pat/Patrick, Steve/Steven) -- a known,
    smaller remaining gap, not attempted here."""
    tokens = str(full_name).split()
    if len(tokens) < 2:
        return identity_key(full_name)
    return identity_key(f"{tokens[0]} {tokens[-1]}")


def build_full_name_alias_index(styles_csv_path) -> dict[str, str]:
    """Build {first_last_key(full_name): "player:<cricsheet_id>"} from
    data/external/cricsheet_player_styles.csv's full_name column -- a
    fallback resolution tier for names that don't match Cricsheet's own
    (often abbreviated) unique_name convention. Best-effort: on a
    first_last_key collision between two different players, keeps
    whichever is seen first (same graceful-degradation philosophy as the
    rest of this identity-resolution stack; this is a fallback tier, not
    an authoritative one). Returns {} if the file is missing."""
    from pathlib import Path

    path = Path(styles_csv_path)
    if not path.is_file():
        return {}
    import pandas as pd

    styles = pd.read_csv(path)
    index: dict[str, str] = {}
    for cricsheet_id, full_name in zip(styles["cricsheet_id"], styles["full_name"]):
        if not isinstance(cricsheet_id, str) or not isinstance(full_name, str):
            continue
        key = first_last_key(full_name)
        if key and key not in index:
            index[key] = f"player:{cricsheet_id}"
    return index
