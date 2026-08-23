"""Precomputes the live snapshots `match_winner_v1`'s
`MatchWinnerFeatureComputer` needs that no earlier promotion already
built: team-vs-team head-to-head history and recency-weighted venue par
score. Team composition, player recency form, phase-specific batter
recency, and partnership rate are all already served by earlier
snapshot builders this session (`build_wicket_contract22_live_snapshots.py`,
`build_recency_weighted_live_snapshots.py`); this only adds what's new
for the win-probability model specifically.

Outputs:
  data/live/team_h2h_pairs.json -- {"pair_matches": {...}, "pair_wins": {...}}
  data/live/venue_recency_par.json -- {"venues": {...}, "global": {...}}

Re-run periodically (e.g. after each completed match) to keep current --
not auto-refreshed by the runtime.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.ml.team_h2h_dataset import compute_final_team_h2h_state
from app.ml.venue_recency_dataset import compute_final_venue_recency_state


def build(root: Path) -> None:
    pair_matches, pair_wins = compute_final_team_h2h_state(root, scopes=("ipl",))
    output_dir = root / "data/live"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "team_h2h_pairs.json").write_text(
        json.dumps({"pair_matches": pair_matches, "pair_wins": pair_wins}, indent=2),
        encoding="utf-8",
    )

    venue_ewma, global_ewma = compute_final_venue_recency_state(root, scopes=("ipl", "t20i"))
    (output_dir / "venue_recency_par.json").write_text(
        json.dumps({"venues": venue_ewma, "global": global_ewma}, indent=2),
        encoding="utf-8",
    )
    print(
        f"Wrote {len(pair_matches)} team H2H pairings, "
        f"{len(venue_ewma)} venue recency profiles to {output_dir}"
    )


if __name__ == "__main__":
    build(Path(__file__).resolve().parents[1])
