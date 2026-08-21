"""Build auditable IPL cold-start evidence for batters debuting from 2024."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from app.ml.ipl_identities import canonical_player_id, canonical_team_id


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/cricsheet/ipl"
OUT = ROOT / "data/reports/ipl_cold_start_v1"

# Reviewed role corrections for cases where IPL-only batting position is
# misleading. Prefer official IPL/ICC/national-board profiles.
REVIEWED_ROLES = {
    "Ashutosh Sharma": "BATTER",
    "SB Dubey": "BATTER",
    "MD Choudhary": "WICKETKEEPER_BATTER",
    "Himmat Singh": "BATTER",
    "MJ Suthar": "BOWLER",
    "PR Veer": "ALL_ROUNDER",
    "Sumit Kumar": "ALL_ROUNDER",
    "GF Linde": "BOWLER",
    "Krish Bhagat": "BOWLER",
    "DA Payne": "BOWLER",
    "Gulbadin Naib": "ALL_ROUNDER",
    "KA Maharaj": "BOWLER",
    "L Wood": "BOWLER",
    "LB Williams": "BOWLER",
    "PVSN Raju": "BOWLER",
    "SS Mishra": "BOWLER",
    "SZ Mulani": "ALL_ROUNDER",
    "T Dahiya": "WICKETKEEPER_BATTER",
    "MS Bhandage": "ALL_ROUNDER",
    "Musheer Khan": "ALL_ROUNDER",
    "PWA Mulder": "ALL_ROUNDER",
    "Shivam Singh": "BATTER",
    "RD Rickelton": "WICKETKEEPER_BATTER",
    "JP Inglis": "WICKETKEEPER_BATTER",
    "D Ferreira": "WICKETKEEPER_BATTER",
    "S Arora | Salil Arora": "WICKETKEEPER_BATTER",
    "Kartik Sharma": "WICKETKEEPER_BATTER",
    "Urvil Patel": "WICKETKEEPER_BATTER",
    "SD Hope": "WICKETKEEPER_BATTER",
    "Kumar Kushagra": "WICKETKEEPER_BATTER",
    "R Minz": "WICKETKEEPER_BATTER",
    "LG Pretorius": "WICKETKEEPER_BATTER",
    "BKG Mendis": "WICKETKEEPER_BATTER",
    "BR Sharath": "WICKETKEEPER_BATTER",
}


def _legal(delivery: dict) -> bool:
    extras = delivery.get("extras", {})
    return not extras.get("wides") and not extras.get("noballs")


def main() -> None:
    matches = []
    names: dict[str, set[str]] = defaultdict(set)
    for path in RAW.glob("*.json"):
        raw = json.loads(path.read_text(encoding="utf-8"))
        date = str(raw["info"]["dates"][0])
        season = int(str(raw["info"]["season"])[:4])
        registry = raw["info"].get("registry", {}).get("people", {})
        players = {
            canonical_team_id(team): [
                canonical_player_id(name, registry) for name in roster
            ]
            for team, roster in raw["info"].get("players", {}).items()
        }
        for name, player_uuid in registry.items():
            names[f"player:{player_uuid}"].add(name)
        matches.append((date, path.name, season, raw, registry, players))
    matches.sort(key=lambda row: (row[0], row[1]))

    first_season: dict[str, int] = {}
    seasons: dict[str, set[int]] = defaultdict(set)
    starts: dict[str, int] = defaultdict(int)
    current_streak: dict[tuple[str, str], int] = defaultdict(int)
    maximum_streak: dict[str, int] = defaultdict(int)
    batting_innings: dict[str, list[dict]] = defaultdict(list)
    batting_positions: dict[str, list[int]] = defaultdict(list)
    bowling_balls: dict[str, int] = defaultdict(int)
    last_team_match: dict[str, set[str]] = {}

    for date, source_file, season, raw, registry, players in matches:
        participating_teams = set(players)
        for team in participating_teams:
            prior = last_team_match.get(team, set())
            current = set(players[team])
            for player_id in prior - current:
                current_streak[(team, player_id)] = 0
            for player_id in current:
                starts[player_id] += 1
                seasons[player_id].add(season)
                first_season[player_id] = min(
                    season, first_season.get(player_id, season)
                )
                current_streak[(team, player_id)] += 1
                maximum_streak[player_id] = max(
                    maximum_streak[player_id], current_streak[(team, player_id)]
                )
            last_team_match[team] = current

        for innings in raw.get("innings", []):
            if innings.get("super_over"):
                continue
            totals: dict[str, dict[str, int]] = defaultdict(
                lambda: {"runs": 0, "balls": 0}
            )
            batting_order: list[str] = []
            team = canonical_team_id(str(innings.get("team", "")))
            for over in innings.get("overs", []):
                for delivery in over.get("deliveries", []):
                    player_id = canonical_player_id(
                        str(delivery["batter"]), registry
                    )
                    if player_id not in batting_order:
                        batting_order.append(player_id)
                    non_striker_id = canonical_player_id(
                        str(delivery["non_striker"]), registry
                    )
                    if non_striker_id not in batting_order:
                        batting_order.append(non_striker_id)
                    bowler_id = canonical_player_id(
                        str(delivery["bowler"]), registry
                    )
                    if _legal(delivery):
                        bowling_balls[bowler_id] += 1
                    totals[player_id]["runs"] += int(
                        delivery.get("runs", {}).get("batter", 0)
                    )
                    if _legal(delivery):
                        totals[player_id]["balls"] += 1
            for player_id, total in totals.items():
                balls = total["balls"]
                runs = total["runs"]
                batting_innings[player_id].append(
                    {
                        "date": date,
                        "source_file": source_file,
                        "team": team,
                        "runs": runs,
                        "balls": balls,
                        "strike_rate": round(100 * runs / balls, 2) if balls else 0,
                    }
                )
            for position, player_id in enumerate(batting_order, start=1):
                batting_positions[player_id].append(position)

    rows = []
    innings_rows = []
    for player_id, debut in first_season.items():
        if debut < 2024 or player_id not in batting_innings:
            continue
        first_five = batting_innings[player_id][:5]
        important = [
            item
            for item in first_five
            if item["runs"] >= 50
            or (item["runs"] >= 30 and item["strike_rate"] >= 175)
        ]
        total_runs = sum(item["runs"] for item in batting_innings[player_id])
        total_balls = sum(item["balls"] for item in batting_innings[player_id])
        positions = batting_positions[player_id]
        average_position = sum(positions) / len(positions)
        balls_bowled = bowling_balls[player_id]
        if average_position <= 4.5 and balls_bowled < 60:
            inferred_role = "TOP_ORDER_BATTER"
        elif average_position <= 6.5 and balls_bowled < 120:
            inferred_role = "MIDDLE_ORDER_BATTER"
        elif balls_bowled >= 120 and total_runs >= 150:
            inferred_role = "ALL_ROUNDER"
        elif balls_bowled >= 60 and average_position > 6.5:
            inferred_role = "BOWLER"
        else:
            inferred_role = "AMBIGUOUS"
        eligible_reason = []
        if len(seasons[player_id]) >= 2:
            eligible_reason.append("2+ IPL seasons")
        if maximum_streak[player_id] >= 5:
            eligible_reason.append("5+ consecutive team matches")
        if important:
            eligible_reason.append("important innings in first 5")
        player_name = " | ".join(sorted(names[player_id]))
        reviewed_role = REVIEWED_ROLES.get(player_name)
        final_role = reviewed_role or inferred_role
        rows.append(
            {
                "player_id": player_id,
                "player_name": player_name,
                "ipl_debut_season": debut,
                "ipl_seasons": len(seasons[player_id]),
                "playing_xi_matches": starts[player_id],
                "max_consecutive_starts": maximum_streak[player_id],
                "batting_innings": len(batting_innings[player_id]),
                "average_batting_position": round(average_position, 2),
                "balls_bowled": balls_bowled,
                "inferred_role": inferred_role,
                "final_role": final_role,
                "role_review": "WEB_REVIEWED" if reviewed_role else "IPL_INFERRED",
                "ipl_runs": total_runs,
                "ipl_strike_rate": (
                    round(100 * total_runs / total_balls, 2) if total_balls else 0
                ),
                "important_first_five": "Yes" if important else "No",
                "promotion_evidence": "; ".join(eligible_reason),
                "system_status": "ELIGIBLE_REVIEW" if eligible_reason else "TEMP",
            }
        )
        for number, item in enumerate(first_five, start=1):
            innings_rows.append(
                {
                    "player_id": player_id,
                    "player_name": " | ".join(sorted(names[player_id])),
                    "innings_number": number,
                    **item,
                    "important": (
                        "Yes"
                        if item["runs"] >= 50
                        or (item["runs"] >= 30 and item["strike_rate"] >= 175)
                        else "No"
                    ),
                }
            )
    rows.sort(
        key=lambda row: (
            row["system_status"] != "ELIGIBLE_REVIEW",
            -row["playing_xi_matches"],
            row["player_name"],
        )
    )
    excluded_bowlers = [row for row in rows if row["final_role"] == "BOWLER"]
    batting_rows = [row for row in rows if row["final_role"] != "BOWLER"]
    OUT.mkdir(parents=True, exist_ok=True)
    for filename, data in (
        ("temporary_profiles.csv", batting_rows),
        ("excluded_bowlers.csv", excluded_bowlers),
        ("first_five_innings.csv", innings_rows),
    ):
        with (OUT / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    summary = {
        "new_players_who_batted": len(rows),
        "cold_start_batting_profiles": len(batting_rows),
        "excluded_bowlers": len(excluded_bowlers),
        "unresolved_roles": sum(
            row["final_role"] == "AMBIGUOUS" for row in batting_rows
        ),
        "eligible_for_review": sum(
            row["system_status"] == "ELIGIBLE_REVIEW" for row in batting_rows
        ),
        "temporary": sum(
            row["system_status"] == "TEMP" for row in batting_rows
        ),
        "rules": {
            "season_threshold": 2,
            "consecutive_start_threshold": 5,
            "important_innings": "runs >= 50 OR (runs >= 30 AND strike rate >= 175)",
            "promotion_control": "Eligibility requires manual source verification before main registry promotion.",
        },
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (OUT / "registry.json").write_text(
        json.dumps(
            {
                "summary": summary,
                "profiles": batting_rows,
                "excluded_bowlers": excluded_bowlers,
                "first_five": innings_rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    certification = {
        "certified": summary["unresolved_roles"] == 0,
        "checks": {
            "unresolved_roles": summary["unresolved_roles"],
            "bowlers_in_batting_profiles": sum(
                row["final_role"] == "BOWLER" for row in batting_rows
            ),
            "non_bowlers_in_exclusion_file": sum(
                row["final_role"] != "BOWLER" for row in excluded_bowlers
            ),
            "duplicate_player_ids": len(batting_rows)
            - len({row["player_id"] for row in batting_rows}),
            "missing_batting_positions": sum(
                not row["average_batting_position"] for row in batting_rows
            ),
        },
    }
    certification["certified"] = certification["certified"] and not any(
        certification["checks"].values()
    )
    (OUT / "certification.json").write_text(
        json.dumps(certification, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
