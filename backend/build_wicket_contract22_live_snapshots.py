"""Precomputes the bowler-side lookups contract22_wicket_rigorous needs at
live-prediction time that aren't already covered by the run-range v3
snapshots: bowler career stats, batter-vs-bowler head-to-head, and
bowler-phase history. Batter career stats, venue par scores, and name
aliases are reused directly from data/live/run_range_v3_*.json (same
underlying quantities, built the same way, from the same match
population) -- no need to duplicate them.

Same chronological, leave-this-match-out aggregation logic as
build_run_range_v3_live_snapshots.py and train_contract22_rigorous.py's
build_enriched(), applied across all eligible matches to produce the
FINAL state as of the most recent available match.

Outputs:
  data/live/wicket_contract22_bowler_stats.json   -- canonical_player_id -> career totals
  data/live/wicket_contract22_h2h_stats.json      -- "striker_id|bowler_id" -> matchup totals
  data/live/wicket_contract22_bowler_phase_stats.json -- "bowler_id|phase" -> phase totals

Re-run periodically (e.g. after each completed match) to keep current --
not auto-refreshed by the runtime.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.ml.ipl_identities import canonical_player_id
from train_phase_calibrated_sharp_range_v33 import male_source_files


def _empty() -> dict[str, int]:
    return {"balls": 0, "runs": 0, "wickets": 0}


def _phase(over: int) -> str:
    return "powerplay" if over <= 6 else ("death" if over >= 16 else "middle")


def build(root: Path) -> None:
    eligible = male_source_files(root)
    paths: list[tuple[str, Path]] = []
    for scope in ("ipl", "t20i"):
        for path in (root / "data/raw/cricsheet" / scope).glob("*.json"):
            if path.name not in eligible:
                continue
            raw = json.loads(path.read_text(encoding="utf-8"))
            paths.append((str(raw["info"]["dates"][0]), path))
    paths.sort(key=lambda item: (item[0], item[1].name))
    print(f"Processing {len(paths)} eligible matches chronologically...")

    bowler_history: dict[str, dict[str, int]] = {}
    h2h_history: dict[str, dict[str, int]] = {}
    bowler_phase_history: dict[str, dict[str, int]] = {}

    for i, (match_date, path) in enumerate(paths):
        if i % 500 == 0:
            print(f"  ... {i}/{len(paths)}", flush=True)
        raw = json.loads(path.read_text(encoding="utf-8"))
        registry = raw["info"].get("registry", {}).get("people", {})
        regular = [inn for inn in raw.get("innings", []) if not inn.get("super_over")]
        match_bowler_events: list[tuple[str, dict[str, int]]] = []
        match_h2h_events: list[tuple[str, dict[str, int]]] = []
        match_bowler_phase_events: list[tuple[str, dict[str, int]]] = []

        for innings in regular:
            for over in innings.get("overs", []):
                over_number = int(over["over"]) + 1
                phase = _phase(over_number)
                for delivery in over.get("deliveries", []):
                    extras = delivery.get("extras", {}) or {}
                    is_legal = "wides" not in extras and "noballs" not in extras
                    if not is_legal:
                        continue
                    total_runs = int(delivery.get("runs", {}).get("total", 0))
                    batter = canonical_player_id(str(delivery.get("batter") or ""), registry)
                    bowler = canonical_player_id(str(delivery.get("bowler") or ""), registry)
                    is_wicket = any(
                        d.get("kind") not in {"retired hurt", "obstructing the field"}
                        for d in delivery.get("wickets", [])
                    )
                    event = {"balls": 1, "runs": total_runs, "wickets": int(is_wicket)}
                    match_bowler_events.append((bowler, event))
                    match_h2h_events.append((f"{batter}|{bowler}", event))
                    match_bowler_phase_events.append((f"{bowler}|{phase}", event))

        for key, event in match_bowler_events:
            profile = bowler_history.setdefault(key, _empty())
            for f, v in event.items():
                profile[f] += v
        for key, event in match_h2h_events:
            profile = h2h_history.setdefault(key, _empty())
            for f, v in event.items():
                profile[f] += v
        for key, event in match_bowler_phase_events:
            profile = bowler_phase_history.setdefault(key, _empty())
            for f, v in event.items():
                profile[f] += v

    output_dir = root / "data/live"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "wicket_contract22_bowler_stats.json").write_text(
        json.dumps(bowler_history, indent=2), encoding="utf-8"
    )
    (output_dir / "wicket_contract22_h2h_stats.json").write_text(
        json.dumps(h2h_history, indent=2), encoding="utf-8"
    )
    (output_dir / "wicket_contract22_bowler_phase_stats.json").write_text(
        json.dumps(bowler_phase_history, indent=2), encoding="utf-8"
    )
    print(
        f"Wrote {len(bowler_history)} bowler profiles, {len(h2h_history)} h2h pairs, "
        f"{len(bowler_phase_history)} bowler-phase profiles to {output_dir}"
    )


if __name__ == "__main__":
    build(Path(__file__).resolve().parents[1])
