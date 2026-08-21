"""Build leakage-safe cold-start features on top of canonical IPL v2."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from app.ml.cold_start_profiles import aggressiveness_band
from app.ml.ipl_identities import canonical_player_id, canonical_team_id
from build_ipl_cold_start_registry import REVIEWED_ROLES


ROOT = Path(__file__).resolve().parents[1]
KEYS = ["source_file", "match_date", "innings", "over"]


def _profile_role(name: str, prior_balls: int) -> str:
    if name in REVIEWED_ROLES:
        return REVIEWED_ROLES[name]
    return "ESTABLISHED" if prior_balls else "UNKNOWN_NEW_PLAYER"


def build() -> pd.DataFrame:
    base = pd.read_csv(
        ROOT / "data/candidates/ipl_canonical_v2/training_overs.csv"
    )
    base["match_date"] = base["match_date"].astype(str)
    name_by_id: dict[str, set[str]] = defaultdict(set)
    raw_matches = []
    for path in (ROOT / "data/raw/cricsheet/ipl").glob("*.json"):
        raw = json.loads(path.read_text(encoding="utf-8"))
        registry = raw["info"].get("registry", {}).get("people", {})
        for name, player_uuid in registry.items():
            name_by_id[f"player:{player_uuid}"].add(name)
        raw_matches.append((str(raw["info"]["dates"][0]), path.name, raw))
    raw_matches.sort(key=lambda item: (item[0], item[1]))

    prior_seasons: dict[str, set[int]] = defaultdict(set)
    current_streak: dict[tuple[str, str], int] = defaultdict(int)
    prior_team_xi: dict[str, set[str]] = {}
    prior_important: set[str] = set()
    prior_batting_innings: dict[str, int] = defaultdict(int)
    match_features: list[dict] = []
    position_rows: list[dict] = []

    for date, source_file, raw in raw_matches:
        season = int(str(raw["info"]["season"])[:4])
        registry = raw["info"].get("registry", {}).get("people", {})
        rosters = {
            canonical_team_id(team): {
                canonical_player_id(name, registry) for name in players
            }
            for team, players in raw["info"].get("players", {}).items()
        }
        before: dict[str, tuple[int, int, bool]] = {}
        for team, current in rosters.items():
            previous = prior_team_xi.get(team, set())
            for player_id in previous - current:
                current_streak[(team, player_id)] = 0
            for player_id in current:
                before[player_id] = (
                    len(prior_seasons[player_id]),
                    current_streak[(team, player_id)],
                    player_id in prior_important,
                )

        match = base[base["source_file"] == source_file]
        for _, row in match.iterrows():
            player_id = str(row["striker"])
            names = " | ".join(sorted(name_by_id[player_id]))
            seasons, streak, important = before.get(player_id, (0, 0, False))
            prior_balls = int(row["striker_prior_balls"])
            prior_sr = (
                float(row["striker_prior_runs_per_ball"]) * 100
                if prior_balls
                else None
            )
            role = _profile_role(names, prior_balls)
            is_bowler = role == "BOWLER"
            eligible = seasons >= 2 or streak >= 5 or important
            match_features.append(
                {
                    **{key: row[key] for key in KEYS},
                    "striker": player_id,
                    "cold_start_role": role,
                    "cold_start_status": (
                        "BOWLER_SAFEGUARD"
                        if is_bowler
                        else "MAIN"
                        if eligible
                        else "TEMP"
                    ),
                    "cold_start_aggressiveness": (
                        "BOWLER_NEUTRAL"
                        if is_bowler
                        else aggressiveness_band(prior_sr)
                    ),
                    "cold_start_prior_seasons": seasons,
                    "cold_start_consecutive_xi": streak,
                    "cold_start_important_prior": int(important),
                    "cold_start_is_bowler": int(is_bowler),
                }
            )

        new_important: set[str] = set()
        for innings_no, innings in enumerate(
            [x for x in raw.get("innings", []) if not x.get("super_over")],
            start=1,
        ):
            order: list[str] = []
            totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
            for over in innings.get("overs", []):
                for delivery in over.get("deliveries", []):
                    batter = canonical_player_id(str(delivery["batter"]), registry)
                    non_striker = canonical_player_id(
                        str(delivery["non_striker"]), registry
                    )
                    for player_id in (batter, non_striker):
                        if player_id not in order:
                            order.append(player_id)
                    totals[batter][0] += int(
                        delivery.get("runs", {}).get("batter", 0)
                    )
                    extras = delivery.get("extras", {})
                    if not extras.get("wides") and not extras.get("noballs"):
                        totals[batter][1] += 1
            for position, player_id in enumerate(order, start=1):
                position_rows.append(
                    {
                        "source_file": source_file,
                        "innings": innings_no,
                        "player_id": player_id,
                        "current_batting_position": position,
                    }
                )
            for player_id, (runs, balls) in totals.items():
                if (
                    prior_batting_innings[player_id] < 5
                    and (
                        runs >= 50
                        or (
                            runs >= 30
                            and balls
                            and 100 * runs / balls >= 175
                        )
                    )
                ):
                    new_important.add(player_id)
                prior_batting_innings[player_id] += 1
        prior_important.update(new_important)
        for team, current in rosters.items():
            for player_id in current:
                current_streak[(team, player_id)] += 1
                prior_seasons[player_id].add(season)
            prior_team_xi[team] = current

    features = pd.DataFrame(match_features)
    positions = pd.DataFrame(position_rows)
    features = features.merge(
        positions,
        left_on=["source_file", "innings", "striker"],
        right_on=["source_file", "innings", "player_id"],
        how="left",
        validate="many_to_one",
    ).drop(columns=["player_id", "striker"])
    output = base.merge(features, on=KEYS, validate="one_to_one")
    if output["current_batting_position"].isna().any():
        missing = output.loc[
            output["current_batting_position"].isna(),
            ["source_file", "innings", "over", "striker_x", "striker_y"],
        ].head()
        raise ValueError(
            f"Missing current batting positions:\n{missing.to_string(index=False)}"
        )
    out = ROOT / "data/candidates/ipl_cold_start_v3"
    out.mkdir(parents=True, exist_ok=True)
    output.to_csv(out / "training_overs.csv", index=False)
    verification = {
        "rows": len(output),
        "duplicate_keys": int(output.duplicated(KEYS).sum()),
        "missing_positions": int(output["current_batting_position"].isna().sum()),
        "bowler_safeguard_rows": int(output["cold_start_is_bowler"].sum()),
        "temporary_rows": int((output["cold_start_status"] == "TEMP").sum()),
        "future_information_used": False,
    }
    (out / "verification.json").write_text(
        json.dumps(verification, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(verification))
    return output


if __name__ == "__main__":
    build()
