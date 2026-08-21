"""Build chronological bowler-strike features for an announced-bowler diagnostic."""

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


def _bowler_wickets(delivery: dict) -> int:
    excluded = {"run out", "retired hurt", "obstructing the field"}
    return sum(
        str(wicket.get("kind", "")) not in excluded
        for wicket in delivery.get("wickets", [])
    )


def build(root: Path = ROOT) -> pd.DataFrame:
    base = pd.read_csv(
        root
        / "data/candidates/ipl_cold_start_v5_t20_priors/training_overs.csv"
    )
    base["match_date"] = base["match_date"].astype(str)
    paths = []
    for path in (root / "data/raw/cricsheet/ipl").glob("*.json"):
        raw = json.loads(path.read_text(encoding="utf-8"))
        paths.append((str(raw["info"]["dates"][0]), path.name, path, raw))
    paths.sort(key=lambda item: (item[0], item[1]))
    career: dict[str, dict[str, int]] = defaultdict(
        lambda: {"balls": 0, "wickets": 0}
    )
    global_balls = 0
    global_wickets = 0
    rows = []
    for date, source_file, _, raw in paths:
        registry = raw["info"].get("registry", {}).get("people", {})
        match_updates: dict[str, dict[str, int]] = defaultdict(
            lambda: {"balls": 0, "wickets": 0}
        )
        for innings_number, innings in enumerate(
            [item for item in raw.get("innings", []) if not item.get("super_over")],
            start=1,
        ):
            match_state: dict[str, dict[str, int]] = defaultdict(
                lambda: {"balls": 0, "wickets": 0, "balls_since_wicket": 0}
            )
            for source_over in innings.get("overs", []):
                deliveries = source_over.get("deliveries", [])
                if not deliveries:
                    continue
                bowler = canonical_player_id(
                    str(deliveries[0]["bowler"]), registry
                )
                prior = career[bowler]
                state = match_state[bowler]
                global_sr = (
                    global_balls / global_wickets if global_wickets else 24.0
                )
                raw_sr = (
                    prior["balls"] / prior["wickets"]
                    if prior["wickets"]
                    else global_sr
                )
                weight = prior["balls"] / (prior["balls"] + 120.0)
                shrunk_sr = weight * raw_sr + (1 - weight) * global_sr
                feature_row = {
                        "source_file": source_file,
                        "match_date": date,
                        "innings": innings_number,
                        "over": int(source_over["over"]) + 1,
                        "announced_bowler": bowler,
                        "bowler_prior_balls": prior["balls"],
                        "bowler_prior_wickets": prior["wickets"],
                        "bowler_prior_strike_rate": shrunk_sr,
                        "bowler_prior_weight": weight,
                        "bowler_match_balls": state["balls"],
                        "bowler_match_wickets": state["wickets"],
                        "bowler_balls_since_wicket": state["balls_since_wicket"],
                        "bowler_overdue_ratio": (
                            state["balls_since_wicket"] / max(1.0, shrunk_sr)
                        ),
                        "bowler_wickets_in_over": 0,
                    }
                rows.append(feature_row)
                for delivery in deliveries:
                    delivery_bowler = canonical_player_id(
                        str(delivery["bowler"]), registry
                    )
                    legal = int(_legal(delivery))
                    wickets = _bowler_wickets(delivery)
                    feature_row["bowler_wickets_in_over"] += wickets
                    current = match_state[delivery_bowler]
                    current["balls"] += legal
                    current["wickets"] += wickets
                    current["balls_since_wicket"] = (
                        0
                        if wickets
                        else current["balls_since_wicket"] + legal
                    )
                    match_updates[delivery_bowler]["balls"] += legal
                    match_updates[delivery_bowler]["wickets"] += wickets
        for bowler, update in match_updates.items():
            career[bowler]["balls"] += update["balls"]
            career[bowler]["wickets"] += update["wickets"]
            global_balls += update["balls"]
            global_wickets += update["wickets"]
    features = pd.DataFrame(rows)
    output = base.merge(features, on=KEYS, validate="one_to_one")
    destination = root / "data/candidates/ipl_wicket_v8_announced_bowler"
    destination.mkdir(parents=True, exist_ok=True)
    output.to_csv(destination / "training_overs.csv", index=False)
    verification = {
        "rows": len(output),
        "duplicate_keys": int(output.duplicated(KEYS).sum()),
        "missing_announced_bowler": int(output["announced_bowler"].isna().sum()),
        "future_information_used_for_candidate": True,
        "reason": (
            "Cricsheet reveals the bowler at the first delivery; use is valid "
            "only when a live feed explicitly announces the next-over bowler."
        ),
        "promotion_eligible": False,
    }
    (destination / "verification.json").write_text(
        json.dumps(verification, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(verification))
    return output


if __name__ == "__main__":
    build()
