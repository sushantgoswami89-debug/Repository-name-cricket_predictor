"""Build a leakage-safe, ball-level batter-vs-bowler head-to-head profile
table, matching the pattern already used by bowler_phase_profiles.csv
(raw counts, not pre-smoothed rates -- BoundedBowlerAdjustment's own
evidence-weighting does the sample-size gating downstream).

Scoped to IPL (same player registry -- data/reports/ipl_canonical_identities_v1/
players.csv -- as bowler_spell_adjuster.py already uses), so player_id
resolution is consistent with what's already wired in.

Saves data/h2h_profiles.csv: player_id pair -> raw ball/run/wicket/dot/
boundary counts.
"""

from pathlib import Path

import pandas as pd

from app.data_ingestion.json_loader import JsonLoader
from app.data_ingestion.match_parser import MatchParser
from app.ml.ipl_identities import identity_key

PROJECT_ROOT = Path(__file__).resolve().parents[1]

players_df = pd.read_csv(PROJECT_ROOT / "data/reports/ipl_canonical_identities_v1/players.csv")
name_to_id: dict[str, str] = {}
for row in players_df.itertuples(index=False):
    for name in str(row.names).split(" | "):
        name_to_id[identity_key(name)] = str(row.player_id)

match_dir = PROJECT_ROOT / "data/raw/cricsheet/ipl"
files = sorted(match_dir.glob("*.json"))
print(f"Parsing {len(files)} IPL matches...")

# (batter_id, bowler_id) -> [balls, runs, wickets, dots, boundaries]
profiles: dict[tuple[str, str], list[int]] = {}
matches_ok = 0
matches_failed = 0

EXCLUDED_DISMISSALS = {"run out", "retired hurt", "retired out", "obstructing the field"}

for path in files:
    try:
        match = MatchParser().parse(JsonLoader().load(str(path)))
    except Exception:
        matches_failed += 1
        continue
    matches_ok += 1
    for innings in match.innings:
        for over in innings.overs:
            for delivery in over.deliveries:
                bat_id = name_to_id.get(identity_key(delivery.batter), "")
                bowl_id = name_to_id.get(identity_key(delivery.bowler), "")
                if not bat_id or not bowl_id:
                    continue
                key = (bat_id, bowl_id)
                stats = profiles.setdefault(key, [0, 0, 0, 0, 0])
                is_legal = "wides" not in delivery.extras
                if is_legal:
                    stats[0] += 1  # balls
                runs = delivery.runs.get("total", 0) - delivery.extras.get(
                    "byes", 0
                ) - delivery.extras.get("legbyes", 0)
                stats[1] += max(0, runs)  # runs off the bat, roughly
                wickets = sum(
                    w.get("kind") not in EXCLUDED_DISMISSALS for w in delivery.wickets
                )
                stats[2] += wickets  # wickets
                if is_legal and delivery.runs.get("total", 0) == 0:
                    stats[3] += 1  # dots
                if delivery.runs.get("batter", 0) in (4, 6):
                    stats[4] += 1  # boundaries

print(f"Matches parsed OK: {matches_ok} | Failed: {matches_failed}")
print(f"Unique (batter, bowler) pairs with known identities: {len(profiles)}")

rows = []
for (bat_id, bowl_id), (balls, runs, wickets, dots, boundaries) in profiles.items():
    rows.append(
        {
            "batter_id": bat_id,
            "bowler_id": bowl_id,
            "h2h_balls": balls,
            "h2h_runs": runs,
            "h2h_wickets": wickets,
            "h2h_dots": dots,
            "h2h_boundaries": boundaries,
        }
    )

out_df = pd.DataFrame(rows)
out_path = PROJECT_ROOT / "data" / "h2h_profiles.csv"
out_df.to_csv(out_path, index=False)
print(f"Saved {len(out_df)} pairs to {out_path}")
print(f"\nPairs with >=24 balls (h2h_supported threshold): {(out_df['h2h_balls'] >= 24).sum()}")
