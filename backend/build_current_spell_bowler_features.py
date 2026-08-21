"""Build chronological current-spell features for the offline bowler candidate.

The next bowler comes from Cricsheet's first delivery and is therefore an
offline development covariate only. Live use remains gated by separately
verified pre-over TOI availability.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from app.ml.ipl_identities import canonical_player_id


ROOT = Path(__file__).resolve().parents[1]
KEYS = ["source_file", "match_date", "innings", "over"]


def _legal(delivery: dict) -> bool:
    extras = delivery.get("extras", {})
    return not extras.get("wides") and not extras.get("noballs")


def _bowler_runs(delivery: dict) -> int:
    extras = delivery.get("extras", {})
    return int(delivery["runs"]["total"]) - int(extras.get("byes", 0)) - int(
        extras.get("legbyes", 0)
    )


def _bowler_wickets(delivery: dict) -> int:
    excluded = {
        "run out",
        "retired hurt",
        "retired out",
        "obstructing the field",
    }
    return sum(
        str(wicket.get("kind", "")).lower() not in excluded
        for wicket in delivery.get("wickets", [])
    )


def _phase(over: int) -> str:
    if over <= 6:
        return "powerplay"
    if over <= 15:
        return "middle"
    return "death"


def _blank() -> dict[str, int]:
    return {
        "balls": 0,
        "runs": 0,
        "wickets": 0,
        "dots": 0,
        "boundaries": 0,
    }


def _rates(prefix: str, state: dict[str, int]) -> dict[str, float | int]:
    balls = state["balls"]
    wickets = state["wickets"]
    return {
        f"{prefix}_balls": balls,
        f"{prefix}_runs_conceded": state["runs"],
        f"{prefix}_wickets": wickets,
        f"{prefix}_dot_rate": state["dots"] / balls if balls else 0.0,
        f"{prefix}_boundary_concession_rate": (
            state["boundaries"] / balls if balls else 0.0
        ),
        f"{prefix}_strike_rate": balls / wickets if wickets else 0.0,
        f"{prefix}_economy": 6.0 * state["runs"] / balls if balls else 0.0,
    }


def build(root: Path = ROOT) -> pd.DataFrame:
    base = pd.read_csv(
        root / "data/candidates/ipl_cold_start_v5_t20_priors/training_overs.csv"
    )
    base["match_date"] = base["match_date"].astype(str)
    raw_matches = []
    for path in (root / "data/raw/cricsheet/ipl").glob("*.json"):
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw_matches.append((str(raw["info"]["dates"][0]), path.name, raw))
    raw_matches.sort(key=lambda item: (item[0], item[1]))

    career: dict[tuple[str, str], dict[str, int]] = defaultdict(_blank)
    h2h: dict[tuple[str, str], dict[str, int]] = defaultdict(_blank)
    rows: list[dict] = []
    for date, source_file, raw in raw_matches:
        registry = raw["info"].get("registry", {}).get("people", {})
        match_career: dict[tuple[str, str], dict[str, int]] = defaultdict(_blank)
        match_h2h: dict[tuple[str, str], dict[str, int]] = defaultdict(_blank)
        for innings_number, innings in enumerate(
            [item for item in raw.get("innings", []) if not item.get("super_over")],
            start=1,
        ):
            match_state: dict[str, dict[str, int]] = defaultdict(_blank)
            spell_state: dict[str, dict[str, int]] = defaultdict(
                lambda: {
                    "last_over": -1,
                    "spell_number": 0,
                    "spell_balls": 0,
                    "consecutive": 0,
                }
            )
            for source_over in innings.get("overs", []):
                deliveries = source_over.get("deliveries", [])
                if not deliveries:
                    continue
                over = int(source_over["over"]) + 1
                phase = _phase(over)
                bowler = canonical_player_id(str(deliveries[0]["bowler"]), registry)
                batter = canonical_player_id(str(deliveries[0]["batter"]), registry)
                current = match_state[bowler]
                spell = spell_state[bowler]
                gap = over - spell["last_over"] - 1 if spell["last_over"] >= 0 else -1
                new_spell = spell["last_over"] < 0 or gap >= 2
                historical = career[(bowler, phase)]
                matchup = h2h[(batter, bowler)]
                row = {
                    "source_file": source_file,
                    "match_date": date,
                    "innings": innings_number,
                    "over": over,
                    "offline_next_bowler": bowler,
                    "offline_batter": batter,
                    **_rates("bowler_match", current),
                    "overs_since_previous": gap,
                    "consecutive_overs": (
                        spell["consecutive"] + 1 if gap == 0 else 1
                    ),
                    "spell_number": spell["spell_number"] + int(new_spell),
                    "current_spell_balls": 0 if new_spell else spell["spell_balls"],
                    "spell_state": (
                        "new_spell"
                        if new_spell
                        else "returning_spell"
                    ),
                    **_rates("bowler_phase_history", historical),
                    "bowler_history_supported": historical["balls"] >= 120,
                    "h2h_balls": matchup["balls"],
                    "h2h_runs": matchup["runs"],
                    "h2h_wickets": matchup["wickets"],
                    "h2h_dot_rate": (
                        matchup["dots"] / matchup["balls"]
                        if matchup["balls"] >= 24
                        else 0.0
                    ),
                    "h2h_boundary_rate": (
                        matchup["boundaries"] / matchup["balls"]
                        if matchup["balls"] >= 24
                        else 0.0
                    ),
                    "h2h_supported": matchup["balls"] >= 24,
                }
                rows.append(row)

                over_balls = over_runs = over_wickets = over_dots = over_boundaries = 0
                for delivery in deliveries:
                    delivery_bowler = canonical_player_id(
                        str(delivery["bowler"]), registry
                    )
                    delivery_batter = canonical_player_id(
                        str(delivery["batter"]), registry
                    )
                    legal = int(_legal(delivery))
                    runs = _bowler_runs(delivery)
                    wickets = _bowler_wickets(delivery)
                    dot = int(legal and runs == 0)
                    boundary = int(legal and int(delivery["runs"]["batter"]) >= 4)
                    state = match_state[delivery_bowler]
                    for target in (
                        state,
                        match_career[(delivery_bowler, phase)],
                        match_h2h[(delivery_batter, delivery_bowler)],
                    ):
                        target["balls"] += legal
                        target["runs"] += runs
                        target["wickets"] += wickets
                        target["dots"] += dot
                        target["boundaries"] += boundary
                    if delivery_bowler == bowler:
                        over_balls += legal
                        over_runs += runs
                        over_wickets += wickets
                        over_dots += dot
                        over_boundaries += boundary
                if new_spell:
                    spell["spell_number"] += 1
                    spell["spell_balls"] = 0
                spell["consecutive"] = (
                    spell["consecutive"] + 1 if gap == 0 else 1
                )
                spell["last_over"] = over
                spell["spell_balls"] += over_balls
                row.update(
                    {
                        "bowler_runs_in_over": over_runs,
                        "bowler_wickets_in_over": over_wickets,
                        "bowler_legal_balls_in_over": over_balls,
                        "bowler_dots_in_over": over_dots,
                        "bowler_boundaries_in_over": over_boundaries,
                    }
                )
        # Historical priors are updated only after every row in the match has
        # been emitted, preventing same-match information entering "history".
        for key, update in match_career.items():
            for field, value in update.items():
                career[key][field] += value
        for key, update in match_h2h.items():
            for field, value in update.items():
                h2h[key][field] += value

    features = pd.DataFrame(rows)
    output = base.merge(features, on=KEYS, validate="one_to_one")
    destination = root / "data/candidates/announced_bowler_current_spell_v2"
    destination.mkdir(parents=True, exist_ok=True)
    output.to_csv(destination / "training_overs.csv", index=False)
    verification = {
        "rows": len(output),
        "duplicate_keys": int(output.duplicated(KEYS).sum()),
        "chronological_history": True,
        "historical_updates_after_match": True,
        "offline_first_delivery_bowler": True,
        "live_availability_verified": False,
        "candidate_only": True,
    }
    (destination / "verification.json").write_text(
        json.dumps(verification, indent=2) + "\n", encoding="utf-8"
    )
    return output


if __name__ == "__main__":
    print(json.dumps({"rows": len(build())}))
