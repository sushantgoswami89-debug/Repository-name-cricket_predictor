"""Leakage-safe raw ball-by-ball sequence extraction for the LSTM wicket
prototype (2026-08-24). One sequence per (source_file, innings, over) key,
matching `data/candidates/v3/verified_training_overs.csv` exactly, holding
every legal/illegal delivery bowled strictly BEFORE that over starts.

Deliberately minimal per-ball features (no player identity) -- this is a
test of whether sequence order itself carries signal the flat/aggregated
GBM features miss, not a full production feature set.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from app.ml.candidate_v3_dataset import _is_legal, _is_wicket
from app.replay.evaluation_policy import KNOWN_RULE_ANOMALY_EXCLUSIONS

N_FEATURES = 9


def _ball_features(delivery: dict[str, Any], over_number: int, ball_index: int) -> list[float]:
    runs = delivery.get("runs", {})
    batter_runs = int(runs.get("batter", 0))
    total_runs = int(runs.get("total", 0))
    extras_total = total_runs - batter_runs
    legal = _is_legal(delivery)
    return [
        float(batter_runs),
        float(extras_total),
        float(total_runs),
        float(int(_is_wicket(delivery))),
        float(int(legal)),
        float(int(batter_runs in (4, 6))),
        float(int(total_runs == 0)),
        float(over_number) / 20.0,
        float(ball_index) / 6.0,
    ]


def build_wicket_sequences(project_root: Path, max_len: int = 120) -> dict[tuple[str, int, int], np.ndarray]:
    """Return {(source_file, innings, over): sequence[T, N_FEATURES]} for every
    over across ipl+t20i, T history capped at `max_len` most-recent balls.
    """

    excluded = set(KNOWN_RULE_ANOMALY_EXCLUSIONS)
    sequences: dict[tuple[str, int, int], np.ndarray] = {}

    for scope in ("t20i", "ipl"):
        for path in (project_root / "data/raw/cricsheet" / scope).glob("*.json"):
            if path.stem in excluded:
                continue
            raw = json.loads(path.read_text(encoding="utf-8"))
            for innings_number, innings in enumerate(raw.get("innings", []), start=1):
                history: list[list[float]] = []
                for source_over in innings.get("overs", []):
                    deliveries = source_over.get("deliveries", [])
                    if not deliveries:
                        continue
                    over_number = int(source_over["over"]) + 1
                    key = (path.name, innings_number, over_number)
                    if history:
                        window = history[-max_len:]
                        sequences[key] = np.asarray(window, dtype=np.float32)
                    else:
                        sequences[key] = np.zeros((0, N_FEATURES), dtype=np.float32)
                    for ball_index, delivery in enumerate(deliveries, start=1):
                        history.append(_ball_features(delivery, over_number, ball_index))

    return sequences
