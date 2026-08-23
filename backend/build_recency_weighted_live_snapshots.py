"""Precomputes the final recency-weighted (match-EWMA, decay=0.95)
batter/bowler form state, as of the most recent available match, for
live serving of `contract22_wicket_v9_recency_form` and later promotions
built on top of it.

Same idea as `build_wicket_contract22_live_snapshots.py`: this expensive
chronological pass is done offline, and the runtime feature computer
(`app/ml/wicket_contract22_features.py`) just does a cheap dict lookup
per prediction. Uses `app.ml.recency_weighted_prior_dataset.compute_final_recency_state`,
which shares its EWMA-update logic with the training-side dataset builder
(`train_contract22_wicket_v9_recency_form.py`'s `build_recency_weighted_prior_dataset`)
so live and training features are computed identically.

Outputs:
  data/live/recency_form_batter_stats.json -- canonical_player_id -> EWMA batter state
  data/live/recency_form_bowler_stats.json -- canonical_player_id -> EWMA bowler state
  data/live/recency_form_bowler_phase_stats.json -- "bowler_id|phase" -> EWMA bowler state
    (for contract22_wicket_v11_bowler_phase_recency, not promoted -- see
    docs/finding_bowler_phase_recency_mixed.md)
  data/live/recency_form_batter_phase_stats.json -- "batter_id|phase" -> EWMA batter state
    (for contract22_wicket_v15_batter_phase_recency, promoted 2026-08-23 --
    see docs/finding_batter_phase_recency_promoted.md)

Re-run periodically (e.g. after each completed match) to keep current --
not auto-refreshed by the runtime.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.ml.recency_weighted_prior_dataset import compute_final_recency_state


def build(root: Path) -> None:
    batter_form, bowler_form, bowler_phase_form, batter_phase_form = compute_final_recency_state(
        root, scopes=("ipl", "t20i")
    )
    output_dir = root / "data/live"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "recency_form_batter_stats.json").write_text(
        json.dumps(batter_form, indent=2), encoding="utf-8"
    )
    (output_dir / "recency_form_bowler_stats.json").write_text(
        json.dumps(bowler_form, indent=2), encoding="utf-8"
    )
    (output_dir / "recency_form_bowler_phase_stats.json").write_text(
        json.dumps(bowler_phase_form, indent=2), encoding="utf-8"
    )
    (output_dir / "recency_form_batter_phase_stats.json").write_text(
        json.dumps(batter_phase_form, indent=2), encoding="utf-8"
    )
    print(
        f"Wrote {len(batter_form)} batter recency profiles, "
        f"{len(bowler_form)} bowler recency profiles, "
        f"{len(bowler_phase_form)} bowler-phase recency profiles, "
        f"{len(batter_phase_form)} batter-phase recency profiles to {output_dir}"
    )


if __name__ == "__main__":
    build(Path(__file__).resolve().parents[1])
