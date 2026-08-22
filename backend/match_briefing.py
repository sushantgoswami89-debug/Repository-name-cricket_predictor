"""Pre-match / live commentary briefing: playing-XI batting depth (tail
exposure), toss result, pitch rating, and live weather -- straight from
the same raw TOI feed the live pipeline already polls for match state.

None of this is wired into the prediction models. It's a plain terminal
reference for the commentator: given a TOI match-center URL, print who's
in the XI and where the tail starts, what the toss/pitch/weather look
like. Works before the match starts too (only needs the raw match JSON,
not ball-by-ball data), and can be re-run any time during the match --
TOI updates the same raw payload as the lineup gets confirmed near toss.

Usage:
    /usr/bin/python3 match_briefing.py "<full TOI match-center URL>"
"""

from __future__ import annotations

import argparse

from app.live.toi_reader import ToiFeedError, ToiLiveReader, ToiPlayer


def _role_group(players: tuple[ToiPlayer, ...]) -> str:
    if not players:
        return "  (playing XI not yet available -- try again closer to toss)"
    lines = []
    tail_started = False
    for player in players:
        marker = ""
        if player.role == "Bowler" and not tail_started:
            tail_started = True
            marker = "  <- tail starts here"
        confirm = "" if player.confirm_xi else "  (provisional)"
        lines.append(
            f"  {player.position:>2}. {player.name:<24} {player.role}{confirm}{marker}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("match_url", help="Full TOI match-center URL")
    args = parser.parse_args()

    reader = ToiLiveReader()
    match_id = reader.match_id_from_url(args.match_url)
    try:
        raw = reader._get_json(reader.RAW_URL.format(match_id=match_id))
    except Exception as exc:  # network error, bad match id, etc.
        raise ToiFeedError(f"Could not fetch match data: {exc}") from exc

    team_players = reader._team_players(raw)
    toss_won_by, toss_decision = reader._toss(raw)
    pitch_type, pitch_surface = reader._pitch(raw)
    weather_condition, humidity, temperature, wind = reader._weather(raw)

    print(f"=== Match briefing ({match_id}) ===\n")

    if toss_won_by:
        print(f"Toss: {toss_won_by} won, chose to {toss_decision or 'unknown'}")
    else:
        print("Toss: not yet available")

    if pitch_type or pitch_surface:
        print(f"Pitch: {pitch_type or 'unknown'} / {pitch_surface or 'unknown'} surface")
    else:
        print("Pitch: not yet available")

    if weather_condition or humidity is not None:
        parts = [weather_condition or "unknown"]
        if temperature is not None:
            parts.append(f"{temperature:.0f}C")
        if humidity is not None:
            parts.append(f"{humidity:.0f}% humidity")
        if wind is not None:
            parts.append(f"wind {wind:.1f} m/s")
        print("Weather: " + ", ".join(parts))
    else:
        print("Weather: not yet available")

    print()
    if not team_players:
        print("Playing XIs not yet available -- try again closer to toss.")
        return

    for team_name, players in team_players.items():
        print(f"{team_name} -- batting depth:")
        print(_role_group(players))
        print()


if __name__ == "__main__":
    main()
