"""Precompute the two lookup snapshots run_range_enriched_v3_batting_style
needs at live-prediction time: per-player career batting stats
(striker/partner "prior" features) and per-venue par scores. Both were
only ever computed as OFFLINE per-over columns during training
(`build_ipl_phase_moe_features`, `build_ipl_venue_regime_dataset`) --
there was no live-servable form of them before this.

Uses the exact same aggregation logic as those two training-time builders
(same rolling-average par score, same batter career running totals),
applied chronologically across every match (IPL + T20I, same
male_source_files() population the model trained on) to produce the FINAL
state as of the most recent available match -- i.e. what a live prediction
made "now" should see as each player's/venue's history to date.

Also builds a third snapshot solving the genuine-live-TOI player identity
gap flagged in `app/ml/run_range_v3_features.py`: TOI only supplies plain
player name strings, no Cricsheet-style registry, so replay-only
(registry-based) resolution isn't enough for real live use. Collects every
name string each canonical player is known by, across every match's own
registry (players are occasionally listed under slightly different name
strings in different matches/scorecards), normalized via `identity_key()`
-- same technique `BowlerSpellAdjuster` already uses for the equivalent
wicket-model problem, applied here to this model's broader IPL+T20I player
population instead of reusing that IPL-only lookup.

2026-08-22: also builds a per-(player, phase) batter breakdown
(`run_range_v3_batter_phase_stats.json`) for `run_range_v4_batter_phase`
-- batters had no phase split anywhere in this codebase before that
candidate (see `app/ml/player_venue_phase_dataset.py`'s docstring). Same
key format (`player_id|phase`) as the existing
`wicket_contract22_bowler_phase_stats.json`, computed from the same
chronological accumulator as the venue-agnostic career totals below, just
also keyed by phase.

Outputs:
  data/live/run_range_v3_player_stats.json        -- canonical_player_id -> career totals
  data/live/run_range_v3_batter_phase_stats.json  -- "player_id|phase" -> phase totals
  data/live/run_range_v3_venue_stats.json         -- normalized venue -> par score state
  data/live/run_range_v3_name_aliases.json        -- identity_key(name) -> canonical_player_id

Re-run this periodically (e.g. after each new match completes) to keep the
live snapshots current -- it is NOT auto-refreshed by the runtime.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.ml.candidate_v3_dataset import _is_legal, _phase
from app.ml.ipl_identities import canonical_player_id, identity_key
from app.ml.ipl_venues import normalize_ipl_venue
from train_phase_calibrated_sharp_range_v33 import male_source_files


def _empty_batter() -> dict[str, int]:
    return {"balls": 0, "runs": 0, "dots": 0, "boundaries": 0, "dismissals": 0}


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

    batter_history: dict[str, dict[str, int]] = {}
    batter_phase_history: dict[str, dict[str, int]] = {}
    venue_totals: dict[str, list[int]] = {}
    global_totals: list[int] = []
    name_aliases: dict[str, str] = {}  # identity_key(name) -> canonical_player_id

    for i, (match_date, path) in enumerate(paths):
        if i % 500 == 0:
            print(f"  ... {i}/{len(paths)}", flush=True)
        raw = json.loads(path.read_text(encoding="utf-8"))
        info = raw["info"]
        registry = info.get("registry", {}).get("people", {})
        venue = normalize_ipl_venue(str(info.get("venue") or info.get("city") or "unknown"))

        for squad in info.get("players", {}).values():
            for name in squad:
                key = identity_key(name)
                if key:
                    name_aliases[key] = canonical_player_id(str(name), registry)

        regular_innings = [i for i in raw.get("innings", []) if not bool(i.get("super_over", False))]
        match_batter_events: list[tuple[str, dict[str, int]]] = []
        match_batter_phase_events: list[tuple[str, dict[str, int]]] = []
        first_innings_total = 0

        for innings_number, innings in enumerate(regular_innings, start=1):
            innings_total = 0
            for over in innings.get("overs", []):
                phase = _phase(int(over.get("over", 0)) + 1)
                for delivery in over.get("deliveries", []):
                    batter = canonical_player_id(str(delivery.get("batter") or ""), registry)
                    batter_runs = int(delivery.get("runs", {}).get("batter", 0))
                    total_runs = int(delivery.get("runs", {}).get("total", 0))
                    innings_total += total_runs
                    legal = int(_is_legal(delivery))
                    boundary = int(batter_runs in {4, 6})
                    dot = int(total_runs == 0)
                    if legal:
                        event = {
                            "balls": 1, "runs": batter_runs, "dots": dot,
                            "boundaries": boundary, "dismissals": 0,
                        }
                        match_batter_events.append((batter, event))
                        match_batter_phase_events.append((f"{batter}|{phase}", event))
                    for dismissal in delivery.get("wickets", []):
                        if dismissal.get("kind") not in {"retired hurt", "obstructing the field"}:
                            dismissed = canonical_player_id(str(dismissal.get("player_out") or ""), registry)
                            event = {
                                "balls": 0, "runs": 0, "dots": 0,
                                "boundaries": 0, "dismissals": 1,
                            }
                            match_batter_events.append((dismissed, event))
                            match_batter_phase_events.append((f"{dismissed}|{phase}", event))
            if innings_number == 1:
                first_innings_total = innings_total

        if regular_innings:
            venue_totals.setdefault(venue, []).append(first_innings_total)
            global_totals.append(first_innings_total)

        # Update career profiles only after the whole match (leave-this-match-out style)
        for name, event in match_batter_events:
            profile = batter_history.setdefault(name, _empty_batter())
            for field, value in event.items():
                profile[field] += value
        for name, event in match_batter_phase_events:
            profile = batter_phase_history.setdefault(name, _empty_batter())
            for field, value in event.items():
                profile[field] += value

    output_dir = root / "data/live"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_range_v3_player_stats.json").write_text(
        json.dumps(batter_history, indent=2), encoding="utf-8"
    )
    (output_dir / "run_range_v3_batter_phase_stats.json").write_text(
        json.dumps(batter_phase_history, indent=2), encoding="utf-8"
    )
    venue_stats = {
        venue: {"par_score": sum(totals) / len(totals), "prior_innings": len(totals)}
        for venue, totals in venue_totals.items()
    }
    venue_stats["__global__"] = {
        "par_score": sum(global_totals) / len(global_totals) if global_totals else 170.0,
        "prior_innings": len(global_totals),
    }
    (output_dir / "run_range_v3_venue_stats.json").write_text(
        json.dumps(venue_stats, indent=2), encoding="utf-8"
    )
    (output_dir / "run_range_v3_name_aliases.json").write_text(
        json.dumps(name_aliases, indent=2), encoding="utf-8"
    )
    print(
        f"Wrote {len(batter_history)} player profiles, "
        f"{len(batter_phase_history)} player-phase profiles, {len(venue_stats)-1} "
        f"venue profiles, and {len(name_aliases)} name aliases to {output_dir}"
    )


if __name__ == "__main__":
    build(Path(__file__).resolve().parents[1])
