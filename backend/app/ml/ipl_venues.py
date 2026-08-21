"""Canonical IPL venue identities and current team home assignments."""

from __future__ import annotations

import re

_ALIASES = {
    "eden gardens": "kolkata_eden_gardens",
    "wankhede stadium": "mumbai_wankhede",
    "m chinnaswamy stadium": "bengaluru_chinnaswamy",
    "mchinnaswamy stadium": "bengaluru_chinnaswamy",
    "ma chidambaram stadium": "chennai_chepauk",
    "feroz shah kotla": "delhi_arun_jaitley",
    "arun jaitley stadium": "delhi_arun_jaitley",
    "rajiv gandhi international stadium": "hyderabad_rajiv_gandhi",
    "sawai mansingh stadium": "jaipur_sawai_mansingh",
    "narendra modi stadium": "ahmedabad_narendra_modi",
    "sardar patel stadium": "ahmedabad_narendra_modi",
    "bharat ratna shri atal bihari vajpayee ekana cricket stadium": "lucknow_ekana",
    "maharaja yadavindra singh international cricket stadium": (
        "new_chandigarh_mullanpur"
    ),
    "punjab cricket association stadium": "mohali_is_bindra",
    "punjab cricket association is bindra stadium": "mohali_is_bindra",
    "himachal pradesh cricket association stadium": "dharamsala_hpca",
    "dr y s rajasekhara reddy aca vdca cricket stadium": "visakhapatnam_acavdca",
    "barsapara cricket stadium": "guwahati_barsapara",
    "maharashtra cricket association stadium": "pune_mca",
    "subrata roy sahara stadium": "pune_mca",
    "dr dy patil sports academy": "mumbai_dy_patil",
    "brabourne stadium": "mumbai_brabourne",
    "sheikh zayed stadium": "abu_dhabi_zayed",
    "zayed cricket stadium": "abu_dhabi_zayed",
    "dubai international cricket stadium": "dubai_international",
    "sharjah cricket stadium": "sharjah_cricket",
    "new wanderers stadium": "johannesburg_wanderers",
    "shaheed veer narayan singh international stadium": "raipur_shaheed_veer_narayan",
}


TEAM_HOME_VENUES = {
    "Chennai Super Kings": frozenset({"chennai_chepauk"}),
    "Delhi Capitals": frozenset(
        {"delhi_arun_jaitley", "visakhapatnam_acavdca"}
    ),
    "Gujarat Titans": frozenset({"ahmedabad_narendra_modi"}),
    "Kolkata Knight Riders": frozenset({"kolkata_eden_gardens"}),
    "Lucknow Super Giants": frozenset({"lucknow_ekana"}),
    "Mumbai Indians": frozenset({"mumbai_wankhede"}),
    "Punjab Kings": frozenset(
        {"new_chandigarh_mullanpur", "dharamsala_hpca"}
    ),
    "Rajasthan Royals": frozenset(
        {"jaipur_sawai_mansingh", "guwahati_barsapara"}
    ),
    "Royal Challengers Bengaluru": frozenset({"bengaluru_chinnaswamy"}),
    "Royal Challengers Bangalore": frozenset({"bengaluru_chinnaswamy"}),
    "Sunrisers Hyderabad": frozenset({"hyderabad_rajiv_gandhi"}),
}


def _key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def normalize_ipl_venue(value: str) -> str:
    """Return a stable stadium identity for known IPL venue aliases."""
    key = _key(value)
    for alias, canonical in _ALIASES.items():
        if key == alias or key.startswith(f"{alias} "):
            return canonical
    return re.sub(r"\s+", "_", key) or "unknown"


def team_venue_context(team: str, venue: str) -> str:
    """Classify a team as home or away at a normalized IPL venue."""
    homes = TEAM_HOME_VENUES.get(team)
    if not homes:
        return "unknown"
    return "home" if normalize_ipl_venue(venue) in homes else "away"
