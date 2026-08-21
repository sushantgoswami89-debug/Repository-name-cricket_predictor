"""Generate a versioned canonical identity audit without modifying raw files."""

from __future__ import annotations

import json
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from app.ml.ipl_identities import (
    CURRENT_FRANCHISES,
    HOME_VENUES_2026,
    canonical_player_id,
    canonical_team_id,
)
from app.ml.ipl_venues import normalize_ipl_venue


def audit(root: Path) -> dict[str, Any]:
    venue_aliases: dict[str, Counter[str]] = defaultdict(Counter)
    team_aliases: dict[str, Counter[str]] = defaultdict(Counter)
    player_names: dict[str, Counter[str]] = defaultdict(Counter)
    player_teams: dict[str, set[str]] = defaultdict(set)
    season_player_teams: dict[tuple[str, str], set[str]] = defaultdict(set)
    unresolved_players: Counter[str] = Counter()
    seasons: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: {"teams": set(), "venues": set(), "players": set()}
    )
    matches = deliveries = 0

    paths = sorted((root / "data/raw/cricsheet/ipl").glob("*.json"))
    for path in paths:
        raw = json.loads(path.read_text(encoding="utf-8"))
        info = raw["info"]
        date = str(info["dates"][0])
        season = date[:4]
        registry = info.get("registry", {}).get("people", {})
        venue_source = str(info.get("venue") or info.get("city") or "unknown")
        venue_id = normalize_ipl_venue(venue_source)
        venue_aliases[venue_id][venue_source] += 1
        seasons[season]["venues"].add(venue_id)
        match_teams = {
            canonical_team_id(str(name)): str(name)
            for name in info.get("teams", [])
        }
        for team_id, source_name in match_teams.items():
            team_aliases[team_id][source_name] += 1
            seasons[season]["teams"].add(team_id)
        for source_team, roster in info.get("players", {}).items():
            team_id = canonical_team_id(str(source_team))
            for source_name in roster:
                player_id = canonical_player_id(str(source_name), registry)
                player_names[player_id][str(source_name)] += 0
                player_teams[player_id].add(team_id)
                season_player_teams[(season, player_id)].add(team_id)
                seasons[season]["players"].add(player_id)

        for innings in raw.get("innings", []):
            if innings.get("super_over"):
                continue
            for over in innings.get("overs", []):
                for delivery in over.get("deliveries", []):
                    deliveries += 1
                    for field in ("batter", "non_striker", "bowler"):
                        source_name = str(delivery.get(field, "")).strip()
                        player_id = canonical_player_id(source_name, registry)
                        player_names[player_id][source_name] += 1
                        seasons[season]["players"].add(player_id)
                        if ":unresolved:" in player_id:
                            unresolved_players[source_name] += 1
        matches += 1

    multi_name_players = {
        player_id: dict(names)
        for player_id, names in player_names.items()
        if len(names) > 1
    }
    return {
        "audit_version": "ipl_canonical_identities_v1",
        "raw_files_modified": False,
        "matches": matches,
        "deliveries": deliveries,
        "current_franchise_count": len(CURRENT_FRANCHISES),
        "canonical_team_count_all_history": len(team_aliases),
        "canonical_venue_count_all_history": len(venue_aliases),
        "canonical_player_count": len(player_names),
        "unresolved_player_names": dict(unresolved_players),
        "unresolved_player_count": len(unresolved_players),
        "multi_name_player_ids": multi_name_players,
        "team_aliases": {
            key: dict(value) for key, value in sorted(team_aliases.items())
        },
        "venue_aliases": {
            key: dict(value) for key, value in sorted(venue_aliases.items())
        },
        "player_registry": {
            player_id: {
                "names": dict(player_names[player_id]),
                "teams": sorted(player_teams[player_id]),
            }
            for player_id in sorted(player_names)
        },
        "seasons": {
            season: {
                key: sorted(values)
                for key, values in season_data.items()
            }
            for season, season_data in sorted(seasons.items())
        },
        "season_player_teams": {
            f"{season}|{player_id}": sorted(teams)
            for (season, player_id), teams in sorted(season_player_teams.items())
        },
    }


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    output = (
        project_root
        / "data/reports/ipl_canonical_identities_v1/audit.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    report = audit(project_root)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (output.parent / "players.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["player_id", "names", "teams"])
        for player_id, item in report["player_registry"].items():
            writer.writerow(
                [
                    player_id,
                    " | ".join(item["names"]),
                    " | ".join(item["teams"]),
                ]
            )
    with (output.parent / "teams.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["team_id", "source_names", "current"])
        for team_id, aliases in report["team_aliases"].items():
            writer.writerow(
                [
                    team_id,
                    " | ".join(aliases),
                    int(team_id in CURRENT_FRANCHISES),
                ]
            )
    with (output.parent / "venues.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["venue_id", "source_names", "matches"])
        for venue_id, aliases in report["venue_aliases"].items():
            writer.writerow(
                [venue_id, " | ".join(aliases), sum(aliases.values())]
            )
    with (output.parent / "season_rosters.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["season", "player_id", "teams"])
        for season, season_data in report["seasons"].items():
            for player_id in season_data["players"]:
                teams = report["season_player_teams"][
                    f"{season}|{player_id}"
                ]
                writer.writerow([season, player_id, " | ".join(teams)])
    with (output.parent / "home_venues_2026.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["team_id", "venue_id", "season"])
        for team_id, venues in sorted(HOME_VENUES_2026.items()):
            for venue_id in sorted(venues):
                writer.writerow([team_id, venue_id, 2026])
    print(
        f"matches={report['matches']} players={report['canonical_player_count']} "
        f"unresolved={report['unresolved_player_count']} "
        f"venues={report['canonical_venue_count_all_history']}"
    )
