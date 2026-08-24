"""Callable runtime for the IPL-chase Monte Carlo win-probability route
(2026-08-24) -- see `docs/finding_monte_carlo_win_probability_ipl_chase_win.md`.
Mirrors `runtime_match_winner.py`'s pattern: artifact integrity
verification, then a plain predict call.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from app.simulation.monte_carlo_win_probability import simulate_chase_win_probability

REQUIRED_ARTIFACTS = ("outcome_table.npy", "ipl_platt_calibrator.pkl")
DEFAULT_N_SIMS = 2000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class MonteCarloWinProbabilityRuntime:
    """Loads the outcome table + IPL Platt calibrator, returns a
    calibrated win probability for a real mid-chase state."""

    def __init__(
        self,
        artifact_dir: Path | str,
        *,
        verify_integrity: bool = True,
        n_sims: int = DEFAULT_N_SIMS,
    ) -> None:
        self.artifact_dir = Path(artifact_dir)
        if not self.artifact_dir.is_dir():
            raise FileNotFoundError(f"Artifact directory not found: {self.artifact_dir}")

        missing = [name for name in REQUIRED_ARTIFACTS if not (self.artifact_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing Monte Carlo win-probability artifacts: {', '.join(missing)}")
        if verify_integrity:
            self._verify_integrity()

        self._table_arr = np.load(self.artifact_dir / "outcome_table.npy")
        self._calibrator = joblib.load(self.artifact_dir / "ipl_platt_calibrator.pkl")
        self._n_sims = n_sims
        self._rng = np.random.default_rng(42)

    def _verify_integrity(self) -> None:
        manifest_path = self.artifact_dir / "ARTIFACT_MANIFEST.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Integrity manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = manifest.get("artifacts", {})
        for name in REQUIRED_ARTIFACTS:
            if name not in expected:
                raise ValueError(f"Integrity manifest has no digest for {name}.")
            actual = _sha256(self.artifact_dir / name)
            if actual != expected[name]:
                raise ValueError(f"Artifact integrity check failed for {name}.")

    def predict_ipl_chase_win_probability(
        self,
        *,
        score_before_over: int,
        wickets_in_hand: int,
        legal_balls_bowled: int,
        balls_remaining: int,
        runs_required: int,
    ) -> float:
        target = int(score_before_over + runs_required)
        total_legal_balls = int(legal_balls_bowled + balls_remaining)
        raw = simulate_chase_win_probability(
            self._table_arr,
            score=int(score_before_over),
            wickets_in_hand=int(wickets_in_hand),
            legal_balls_bowled=int(legal_balls_bowled),
            total_legal_balls=total_legal_balls,
            target=target,
            n_sims=self._n_sims,
            rng=self._rng,
        )
        clipped = min(max(raw, 1e-6), 1 - 1e-6)
        logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
        return float(self._calibrator.predict_proba(logit)[0][1])
