"""Time-safe historical match-state analogues for Candidate v3.1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler

ANALOGUE_FEATURES = [
    "over",
    "score_before_over",
    "wkts_down_before_over",
    "wickets_in_hand",
    "balls_remaining",
    "current_run_rate",
    "runs_required",
    "required_run_rate",
    "recent_legal_balls",
    "recent_runs_per_ball",
    "recent_dot_rate",
    "recent_single_rate",
    "recent_boundary_rate",
    "recent_wicket_rate",
]


@dataclass(frozen=True, slots=True)
class AnalogueBatch:
    """Historical outcome distribution for a batch of live-like states."""

    expected_runs: np.ndarray
    wicket_probability: np.ndarray
    lower_runs: np.ndarray
    upper_runs: np.ndarray
    effective_sample_size: np.ndarray
    similarity: np.ndarray
    outcome_std: np.ndarray
    neighbor_count: np.ndarray
    run_bucket_probabilities: np.ndarray


class HistoricalAnalogueEngine:
    """Find comparable prior states without exposing future match outcomes."""

    def __init__(self, neighbors: int = 384) -> None:
        if neighbors < 20:
            raise ValueError("At least 20 neighbours are required.")
        self.neighbors = neighbors
        self._scaler = StandardScaler()
        self._groups: dict[tuple[str, int], dict[str, Any]] = {}
        self._fitted = False

    def fit(self, reference: pd.DataFrame) -> HistoricalAnalogueEngine:
        """Fit state indexes using reference outcomes from prior matches."""

        missing = set(ANALOGUE_FEATURES + ["phase", "is_chase"]) - set(
            reference.columns
        )
        if missing:
            raise ValueError(
                f"Analogue reference is missing columns: {sorted(missing)}"
            )
        scaled = self._scaler.fit_transform(reference[ANALOGUE_FEATURES])
        phases = reference["phase"].astype(str).to_numpy()
        chases = reference["is_chase"].astype(int).to_numpy()
        runs = reference["runs_in_over"].to_numpy(dtype=float)
        wickets = reference["wicket_in_over"].to_numpy(dtype=float)
        for phase in ("powerplay", "middle", "death"):
            for is_chase in (0, 1):
                mask = (phases == phase) & (chases == is_chase)
                points = scaled[mask]
                if len(points) < 20:
                    continue
                self._groups[(phase, is_chase)] = {
                    "tree": cKDTree(points),
                    "runs": runs[mask],
                    "wickets": wickets[mask],
                }
        self._fitted = True
        return self

    def query(
        self,
        states: pd.DataFrame,
        lower_quantile: float = 0.05,
        upper_quantile: float = 0.95,
    ) -> AnalogueBatch:
        """Return weighted future outcomes for each supplied pre-over state."""

        if not self._fitted:
            raise RuntimeError("HistoricalAnalogueEngine must be fitted first.")
        if not 0 < lower_quantile < upper_quantile < 1:
            raise ValueError(
                "Analogue quantiles must be strictly between zero and one."
            )
        scaled = self._scaler.transform(states[ANALOGUE_FEATURES])
        phases = states["phase"].astype(str).to_numpy()
        chases = states["is_chase"].astype(int).to_numpy()
        size = len(states)
        expected = np.zeros(size)
        wicket_probability = np.zeros(size)
        lower = np.zeros(size)
        upper = np.zeros(size)
        effective = np.zeros(size)
        similarity = np.zeros(size)
        outcome_std = np.zeros(size)
        counts = np.zeros(size, dtype=int)
        bucket_probabilities = np.zeros((size, 5))

        for key, group in self._groups.items():
            positions = np.flatnonzero((phases == key[0]) & (chases == key[1]))
            if not len(positions):
                continue
            k = min(self.neighbors, len(group["runs"]))
            distances, indices = group["tree"].query(scaled[positions], k=k)
            if k == 1:
                distances = distances[:, None]
                indices = indices[:, None]
            neighbor_runs = group["runs"][indices]
            neighbor_wickets = group["wickets"][indices]
            local_scale = np.maximum(np.median(distances, axis=1), 0.10)
            weights = np.exp(-np.square(distances / local_scale[:, None]))
            weight_sum = weights.sum(axis=1)
            run_mean = (weights * neighbor_runs).sum(axis=1) / weight_sum
            wicket_mean = (weights * neighbor_wickets).sum(axis=1) / weight_sum
            variance = (weights * np.square(neighbor_runs - run_mean[:, None])).sum(
                axis=1
            ) / weight_sum

            expected[positions] = run_mean
            wicket_probability[positions] = wicket_mean
            effective[positions] = np.square(weight_sum) / np.square(weights).sum(
                axis=1
            )
            similarity[positions] = np.exp(
                -np.median(distances, axis=1) / np.sqrt(len(ANALOGUE_FEATURES))
            )
            outcome_std[positions] = np.sqrt(variance)
            counts[positions] = k
            bucket_index = np.select(
                [
                    neighbor_runs <= 4,
                    neighbor_runs <= 8,
                    neighbor_runs <= 12,
                    neighbor_runs <= 16,
                ],
                [0, 1, 2, 3],
                default=4,
            )
            for bucket in range(5):
                bucket_probabilities[positions, bucket] = (
                    weights * (bucket_index == bucket)
                ).sum(axis=1) / weight_sum
            for local_index, output_index in enumerate(positions):
                order = np.argsort(neighbor_runs[local_index])
                sorted_runs = neighbor_runs[local_index, order]
                sorted_weights = weights[local_index, order]
                cumulative = np.cumsum(sorted_weights) / weight_sum[local_index]
                lower[output_index] = sorted_runs[
                    np.searchsorted(cumulative, lower_quantile)
                ]
                upper_position = min(
                    len(sorted_runs) - 1,
                    np.searchsorted(cumulative, upper_quantile),
                )
                upper[output_index] = sorted_runs[upper_position]

        if np.any(counts == 0):
            raise ValueError("No historical analogue group exists for some states.")
        return AnalogueBatch(
            expected_runs=expected,
            wicket_probability=wicket_probability,
            lower_runs=lower,
            upper_runs=upper,
            effective_sample_size=effective,
            similarity=similarity,
            outcome_std=outcome_std,
            neighbor_count=counts,
            run_bucket_probabilities=bucket_probabilities,
        )
