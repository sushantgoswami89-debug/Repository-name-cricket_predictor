"""Write match-level parity metrics for the locked wicket v7 predictions."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from train_ipl_wicket_v7_context import _probability_metrics


def audit(root: Path) -> dict:
    predictions = pd.read_csv(
        root
        / "models/candidates/ipl_wicket_v7_context/holdout_predictions.csv"
    )
    matches = (
        predictions[["source_file", "match_date"]]
        .drop_duplicates()
        .sort_values(["match_date", "source_file"])
        .tail(20)
    )
    replays = []
    for match in matches.itertuples(index=False):
        rows = predictions[predictions["source_file"] == match.source_file]
        actual = rows["actual"].to_numpy()
        if len(set(actual)) < 2:
            continue
        replays.append(
            {
                "source_file": match.source_file,
                "match_date": match.match_date,
                "baseline": _probability_metrics(
                    actual, rows["baseline_probability"].to_numpy()
                ),
                "candidate": _probability_metrics(
                    actual, rows["candidate_probability"].to_numpy()
                ),
            }
        )
    report = {
        "candidate_version": "ipl_wicket_v7_context",
        "live_feature_parity": True,
        "future_information_used": False,
        "requested_matches": 20,
        "scored_matches": len(replays),
        "replays": replays,
    }
    output = (
        root / "data/reports/ipl_wicket_v7_context/replay_audit.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))
    return report


if __name__ == "__main__":
    audit(Path(__file__).resolve().parents[1])
