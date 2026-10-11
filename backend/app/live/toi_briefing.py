"""Pre-match briefing and Fact-o-Meter card text for a TOI match (the backup source in run_live_auto.py).

Moved here 2026-10-11 from run_toi_auto.py when that old TOI-only driver was retired (docs/outdated_drivers.md).
"""

from __future__ import annotations

from app.live.prematch_analysis import (
    confirmed_xi, fact_o_meter_message, prematch_header, prematch_text, toss_line, venue_summary, xi_line, xi_names,
)
from app.live.toi_reader import ToiLiveReader


def briefing_text(reader: ToiLiveReader, match_id: str, label: str) -> str | None:
    """One-time pre-match summary: toss, pitch, weather, playing XIs.

    Same raw TOI feed and extraction the standalone match_briefing.py
    already used for this -- reused here so the automated Telegram flow
    carries it too instead of requiring a human to run that script
    separately. Returns None on any feed error so a briefing failure
    never blocks the per-over prediction flow that follows it.
    """
    try:
        raw = reader._get_json(ToiLiveReader.RAW_URL.format(match_id=match_id))
    except Exception:
        return None
    team_players = ToiLiveReader._team_players(raw)
    toss_won_by, toss_decision = ToiLiveReader._toss(raw)
    pitch_type, pitch_surface = ToiLiveReader._pitch(raw)
    weather_condition, humidity, temperature, wind = ToiLiveReader._weather(raw)

    team_names = list(team_players)
    city = ToiLiveReader._venue_name(raw).rsplit(",", 1)[-1].strip()
    lines = [f"🏏 {label}" + (f" -- {city} | {venue_summary(city.lower())}" if city else "")]
    if len(team_names) == 2:
        toi_note = " | ".join(x for x in (pitch_type, pitch_surface) if x and "unknown" not in x.lower())
        try:
            lines.extend(
                prematch_header(
                    team_names[0], team_names[1], ToiLiveReader._venue_name(raw), toss_won_by, toss_decision,
                    confirmed_xi(team_players), pitch_note=toi_note,
                )
            )
        except Exception as error:  # never block the briefing, but do not hide the bug
            print(f"[briefing] header unavailable: {error!r}")
    lines.append(
        toss_line(toss_won_by, toss_decision, team_names[0] if len(team_names) == 2 else "", team_names[1] if len(team_names) == 2 else "")
    )
    if len(team_names) == 2:
        analysis = prematch_text(
            team_names[0], team_names[1], ToiLiveReader._venue_name(raw), toss_won_by, toss_decision,
            confirmed_xi(team_players), log_context={"match_id": f"toi:{match_id}", "source": "toi"},
        )
        if analysis:
            lines.append("")
            lines.append(analysis)
    # The whole line-up block (status line, then the names once both XIs are full) goes last.
    lines.append("")
    lines.append(xi_line(team_players))
    lines.extend(xi_names(team_players))
    return "\n".join(lines)


def fact_text(reader: ToiLiveReader, match_id: str, label: str) -> str | None:
    """The Fact-o-Meter card message for a TOI match, kept apart from the briefing (None if the feed cannot be read)."""
    try:
        raw = reader._get_json(ToiLiveReader.RAW_URL.format(match_id=match_id))
    except Exception:
        return None
    team_players = ToiLiveReader._team_players(raw)
    toss_won_by, toss_decision = ToiLiveReader._toss(raw)
    team_names = list(team_players)
    if len(team_names) != 2:
        return ""
    return fact_o_meter_message(
        team_names[0], team_names[1], ToiLiveReader._venue_name(raw), toss_won_by, toss_decision, confirmed_xi(team_players)
    )
