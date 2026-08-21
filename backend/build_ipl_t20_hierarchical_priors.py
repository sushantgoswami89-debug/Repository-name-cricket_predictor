"""Build strictly pre-match T20I batting priors for IPL cold-start players."""

from __future__ import annotations

import json
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path

import pandas as pd

from app.ml.ipl_identities import canonical_player_id


ROOT = Path(__file__).resolve().parents[1]
KEYS = ["source_file", "match_date", "innings", "over"]
SHRINKAGE_BALLS = 60.0


def _legal(delivery: dict) -> bool:
    extras = delivery.get("extras", {})
    return not extras.get("wides") and not extras.get("noballs")


def _match_rows(path: Path) -> tuple[str, dict[str, dict[str, float]]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    date = str(raw["info"]["dates"][0])
    registry = raw["info"].get("registry", {}).get("people", {})
    result: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "runs": 0.0,
            "balls": 0.0,
            "dots": 0.0,
            "boundaries": 0.0,
            "dismissals": 0.0,
            "position_sum": 0.0,
            "innings": 0.0,
        }
    )
    for innings in raw.get("innings", []):
        if innings.get("super_over"):
            continue
        order: list[str] = []
        appeared: set[str] = set()
        for over in innings.get("overs", []):
            for delivery in over.get("deliveries", []):
                batter = canonical_player_id(str(delivery["batter"]), registry)
                non_striker = canonical_player_id(
                    str(delivery["non_striker"]), registry
                )
                for player_id in (batter, non_striker):
                    if player_id not in order:
                        order.append(player_id)
                appeared.add(batter)
                runs = int(delivery.get("runs", {}).get("batter", 0))
                result[batter]["runs"] += runs
                if _legal(delivery):
                    result[batter]["balls"] += 1
                    result[batter]["dots"] += int(runs == 0)
                result[batter]["boundaries"] += int(runs in (4, 6))
                for wicket in delivery.get("wickets", []):
                    dismissed = canonical_player_id(
                        str(wicket.get("player_out", "")), registry
                    )
                    kind = str(wicket.get("kind", ""))
                    if kind not in {"retired hurt", "obstructing the field"}:
                        result[dismissed]["dismissals"] += 1
        for position, player_id in enumerate(order, start=1):
            result[player_id]["position_sum"] += position
            result[player_id]["innings"] += 1
    return date, dict(result)


def _snapshots() -> tuple[
    dict[str, list[tuple[str, dict[str, float]]]],
    list[tuple[str, dict[str, float]]],
]:
    events = []
    for path in (ROOT / "data/raw/cricsheet/t20i").glob("*.json"):
        events.append((*_match_rows(path), path.name))
    events.sort(key=lambda item: (item[0], item[2]))
    player_totals: dict[str, dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    global_totals: dict[str, float] = defaultdict(float)
    players: dict[str, list[tuple[str, dict[str, float]]]] = defaultdict(list)
    global_history: list[tuple[str, dict[str, float]]] = []
    for date, rows, _ in events:
        for player_id, values in rows.items():
            for key, value in values.items():
                player_totals[player_id][key] += value
                global_totals[key] += value
            players[player_id].append((date, dict(player_totals[player_id])))
        global_history.append((date, dict(global_totals)))
    return players, global_history


def _before(
    history: list[tuple[str, dict[str, float]]], date: str
) -> dict[str, float]:
    position = bisect_left([item[0] for item in history], date) - 1
    return history[position][1] if position >= 0 else {}


def build() -> pd.DataFrame:
    base = pd.read_csv(
        ROOT / "data/candidates/ipl_cold_start_v3/training_overs.csv"
    )
    base["match_date"] = base["match_date"].astype(str)
    players, global_history = _snapshots()
    rows = []
    matches = base[["source_file", "match_date"]].drop_duplicates()
    for match in matches.itertuples(index=False):
        global_prior = _before(global_history, match.match_date)
        global_balls = float(global_prior.get("balls", 0))
        global_rpb = (
            float(global_prior.get("runs", 0)) / global_balls
            if global_balls
            else 1.2
        )
        segment = base[base["source_file"] == match.source_file]
        for row in segment.itertuples(index=False):
            player_id = str(row.striker)
            prior = _before(players.get(player_id, []), match.match_date)
            balls = float(prior.get("balls", 0))
            weight = balls / (balls + SHRINKAGE_BALLS)
            raw_rpb = (
                float(prior.get("runs", 0)) / balls if balls else global_rpb
            )
            shrunk_rpb = weight * raw_rpb + (1 - weight) * global_rpb
            rows.append(
                {
                    **{
                        key: getattr(row, key)
                        for key in KEYS
                    },
                    "t20_prior_balls": int(balls),
                    "t20_prior_weight": weight,
                    "t20_prior_runs_per_ball": shrunk_rpb,
                    "t20_prior_vs_global_rpb": shrunk_rpb - global_rpb,
                    "t20_prior_dot_rate": (
                        float(prior.get("dots", 0)) / balls
                        if balls
                        else 0.0
                    ),
                    "t20_prior_boundary_rate": (
                        float(prior.get("boundaries", 0)) / balls
                        if balls
                        else 0.0
                    ),
                    "t20_prior_batting_position": (
                        float(prior.get("position_sum", 0))
                        / float(prior.get("innings", 0))
                        if prior.get("innings", 0)
                        else 0.0
                    ),
                    "t20_prior_source": (
                        "T20I" if balls else "NEUTRAL_SHRINKAGE"
                    ),
                }
            )
    features = pd.DataFrame(rows)
    output = base.merge(features, on=KEYS, validate="one_to_one")
    destination = ROOT / "data/candidates/ipl_cold_start_v5_t20_priors"
    destination.mkdir(parents=True, exist_ok=True)
    output.to_csv(destination / "training_overs.csv", index=False)
    audit = {
        "rows": len(output),
        "duplicate_keys": int(output.duplicated(KEYS).sum()),
        "t20i_supported_rows": int(
            (output["t20_prior_source"] == "T20I").sum()
        ),
        "neutral_shrinkage_rows": int(
            (output["t20_prior_source"] == "NEUTRAL_SHRINKAGE").sum()
        ),
        "shrinkage_balls": SHRINKAGE_BALLS,
        "future_information_used": False,
        "ranji_used": False,
        "smat_used": False,
    }
    (destination / "verification.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit))
    return output


if __name__ == "__main__":
    build()
