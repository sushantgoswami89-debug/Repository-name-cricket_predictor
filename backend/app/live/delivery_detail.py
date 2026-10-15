"""How each ball was bowled and played, read from the commentary prose.

Built 2026-10-06 (user: "learn how deliveries are being bowled"). The scoring
words at the start of a line ("1 run", "FOUR", "wide") are read elsewhere; this
reads the PROSE after them for pace, length, line, variation, shot and where it
went. Best effort: every field is None when the text does not say. It is a
logging/learning layer -- nothing here feeds a model until it has been tested
against outcomes on a large archive.

Vocabulary comes from real lines seen in live matches (India v West Indies
2026-10-06 and earlier archived T20Is); extend it only with a real example and
a test.
"""

from __future__ import annotations

import re

_PACE = re.compile(r"\b(\d{2,3})\s?kph\b", re.IGNORECASE)

# Order matters: more specific phrases first.
_LENGTH = (
    ("yorker", re.compile(r"\byorker\b", re.I)),
    ("full_toss", re.compile(r"\bfull[- ]?toss\b", re.I)),
    ("half_volley", re.compile(r"\bhalf[- ]?volley\b", re.I)),
    ("bouncer", re.compile(r"\bbouncer\b|\bbeamer\b", re.I)),
    ("back_of_a_length", re.compile(r"\bback of (?:a )?length\b|\bhard length\b", re.I)),
    ("short", re.compile(r"\bshort of (?:a )?length\b|\bshort\b(?! of the)|\bsits up\b", re.I)),
    ("good_length", re.compile(r"\bgood length\b|\bon (?:a )?length\b|\bhits the length\b|\bkeeps it on length\b", re.I)),
    ("full", re.compile(r"\bfullish\b|\bfull\b|\bpitched up\b|\bover-?pitched\b", re.I)),
)

_LINE = (
    ("wide_outside_off", re.compile(r"\bwell outside off\b|\bwide outside off\b|\bwide of off\b", re.I)),
    ("outside_off", re.compile(r"\boutside off\b|\bouside off\b|\bfourth stump\b|\bfifth stump\b", re.I)),
    ("down_leg", re.compile(r"\bdown (?:the )?leg\b|\boutside leg\b", re.I)),
    ("on_the_pads", re.compile(r"\bon the pads\b|\bat the pads\b|\bfired in on the pads\b", re.I)),
    ("middle_and_leg", re.compile(r"\bmiddle and leg\b|\bon middle and leg\b", re.I)),
    ("off_and_middle", re.compile(r"\boff and middle\b", re.I)),
    ("middle", re.compile(r"\bon middle\b|\bmiddle stump\b|\bon the stumps\b", re.I)),
    ("off", re.compile(r"\bon off\b|\boff stump\b|\bon the off\b", re.I)),
    ("leg", re.compile(r"\bon leg\b|\bleg stump\b", re.I)),
)

_VARIATION = (
    ("slower_ball", re.compile(r"\bslower[- ]?ball\b|\bslower one\b|\bbowls it slower\b|\bslower\b", re.I)),
    ("off_cutter", re.compile(r"\boff[- ]?cutter\b", re.I)),
    ("leg_cutter", re.compile(r"\bleg[- ]?cutter\b", re.I)),
    ("knuckle_ball", re.compile(r"\bknuckle[- ]?ball\b", re.I)),
    ("googly", re.compile(r"\bgoogly\b|\bwrong'?un\b", re.I)),
    ("carrom_ball", re.compile(r"\bcarrom\b", re.I)),
    ("arm_ball", re.compile(r"\barm ball\b|\bstraighter one\b|\bskids on\b", re.I)),
    ("flighted", re.compile(r"\btossed up\b|\bflighted\b|\bloopy\b", re.I)),
    ("flat_quick", re.compile(r"\bflat\b|\bquicker\b|\bfired in\b", re.I)),
)

_SHOT = (
    ("reverse_sweep", re.compile(r"\breverse[- ]?sweep|\breverse[- ]?scoop|\bswitch[- ]?hit\b", re.I)),
    ("scoop", re.compile(r"\bscoop", re.I)),
    ("sweep", re.compile(r"\bsweep", re.I)),
    ("pull", re.compile(r"\bpull(?:s|ed)?\b|\bhook(?:s|ed)?\b", re.I)),
    ("cut", re.compile(r"\bcut(?:s)?\b|\bslash(?:es|ed)?\b|\bupper[- ]?cut\b", re.I)),
    ("drive", re.compile(r"\b(?:off-|on-|cover |straight |lofted )?driv(?:e|es|en|ing)\b", re.I)),
    ("flick", re.compile(r"\bflick(?:s|ed)?\b|\bwhip(?:s|ped)?\b|\bglance[sd]?\b|\bclip(?:s|ped)?\b|\btuck(?:s|ed)?\b", re.I)),
    ("slog", re.compile(r"\bheave[sd]?\b|\bslog(?:s|ged)?\b|\bclear(?:s|ed) his front leg\b|\blaunch(?:es|ed)?\b|\bclobber", re.I)),
    ("punch", re.compile(r"\bpunch(?:es|ed)?\b|\bjam(?:s|med)? it\b|\bdug out\b|\bdig(?:s)? it out\b", re.I)),
    ("defend", re.compile(r"\bdefend(?:s|ed)?\b|\bblock(?:s|ed)?\b|\bsteers? it to\b|\bpushes? it\b|\bpokes?\b", re.I)),
    ("leave", re.compile(r"\bleaves?\b(?! the)|\bleft alone\b|\bshoulders arms\b", re.I)),
)

_AREA = re.compile(
    r"\b((?:deep |short |fine |long |extra |backward |forward |square |wide |silly )*"
    r"(?:mid-?wicket|mid-?on|mid-?off|extra cover|cover|point|third man|square leg|fine leg|long-?on|long-?off|"
    r"long leg|sweeper|slip|gully|bowler|backward point|short fine|cow corner|leg slip|midwicket))\b",
    re.IGNORECASE,
)


def _first(table, text: str) -> str | None:
    for label, pattern in table:
        if pattern.search(text):
            return label
    return None


def extract_delivery_detail(commentary: str) -> dict[str, object]:
    """Structured description of one delivery from its commentary text."""
    text = re.sub(r"<[^>]+>", "", commentary or "")
    pace = _PACE.search(text)
    area = _AREA.search(text)
    return {
        "pace_kph": int(pace.group(1)) if pace else None,
        "length": _first(_LENGTH, text),
        "line": _first(_LINE, text),
        "variation": _first(_VARIATION, text),
        "shot": _first(_SHOT, text),
        "area": area.group(1).lower() if area else None,
    }


def format_delivery_detail(detail: dict[str, object]) -> str:
    """Compact one-line form, only the fields that are known."""
    parts = []
    if detail.get("pace_kph"):
        parts.append(f"{detail['pace_kph']}kph")
    for key in ("length", "line", "variation"):
        if detail.get(key):
            parts.append(str(detail[key]).replace("_", " "))
    shot = detail.get("shot")
    area = detail.get("area")
    if shot or area:
        parts.append(
            f"{str(shot).replace('_', ' ') if shot else 'shot'}"
            + (f" -> {area}" if area else "")
        )
    return ", ".join(parts)
