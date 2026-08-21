"""Train the candidate-only hierarchical T20-prior cold-start model."""

from __future__ import annotations

from pathlib import Path

from train_ipl_cold_start_v3 import (
    CATEGORICAL_V3,
    FEATURES,
    train,
)


VERSION = "ipl_cold_start_v5_t20_priors"
T20_FEATURES = [
    "t20_prior_balls",
    "t20_prior_weight",
    "t20_prior_runs_per_ball",
    "t20_prior_vs_global_rpb",
    "t20_prior_dot_rate",
    "t20_prior_boundary_rate",
    "t20_prior_batting_position",
    "t20_prior_source",
]


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    train(
        root,
        dataset_path=(
            root
            / "data/candidates/ipl_cold_start_v5_t20_priors/training_overs.csv"
        ),
        version=VERSION,
        features=FEATURES + T20_FEATURES,
        categorical=CATEGORICAL_V3 + ["t20_prior_source"],
    )
