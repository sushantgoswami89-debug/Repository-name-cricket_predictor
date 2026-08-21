"""Transport-agnostic wrapper around the announced_bowler_current_spell_v3
candidate artifacts (run_adjustment.cbm, wicket_adjustment.cbm, Platt
scaler, bowler phase profiles), so PredictionEngine can apply this
validated adjustment regardless of whether the bowler identity came from
a live feed or a replayed match.

Independently validated against the actual production model (not the
candidate's own internal comparison baseline) via
validate_bowler_shadow_vs_production.py: Run MAE 3.87->3.75, Wicket
Brier 0.193->0.190, Wicket AUC 0.519->0.550 on a 40-match held-out IPL
replay sample.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.ml.announced_bowler_adjustment import (
    AdjustedPrediction,
    BoundedBowlerAdjustment,
    BowlerAnnouncement,
    CurrentSpellTracker,
)
from app.ml.ipl_identities import identity_key

# Process-wide cache of the heavy, static parts of _load() (CatBoost models,
# player/profile CSVs), keyed by resolved artifact_dir. A fresh
# BowlerSpellAdjuster() per innings/engine -- the pattern every replay
# script in this repo uses, since only the per-innings CurrentSpellTracker
# state needs to reset -- would otherwise re-read every artifact from disk
# each time. Mirrors HistoricalFeatureStore's and ModelRepository's caches.
_PROCESS_CACHE: dict[str, dict[str, Any]] = {}
_CACHED_ATTRS = (
    "_columns",
    "_categoricals",
    "_adjuster",
    "_run_model",
    "_wicket_model",
    "_wicket_platt",
    "_name_to_id",
    "_profiles",
    "_h2h_profiles",
)


class BowlerSpellAdjuster:
    """Loads the candidate artifacts once and applies the adjustment."""

    def __init__(
        self,
        artifact_dir: Path | None = None,
        project_root: Path | None = None,
    ) -> None:
        root = project_root or Path(__file__).resolve().parents[3]
        self._artifact_dir = artifact_dir or (
            root / "models/candidates/announced_bowler_current_spell_v3"
        )
        self._players_path = root / "data/reports/ipl_canonical_identities_v1/players.csv"
        self._h2h_path = root / "data/h2h_profiles.csv"
        self._loaded = False
        self._available = False
        self._tracker = CurrentSpellTracker()

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self._artifact_dir.exists() or not self._players_path.exists():
            self._available = False
            return
        cache_key = str(self._artifact_dir.resolve())
        cached = _PROCESS_CACHE.get(cache_key)
        if cached is not None:
            for attr in _CACHED_ATTRS:
                setattr(self, attr, cached[attr])
            self._available = True
            return
        try:
            from catboost import CatBoostClassifier, CatBoostRegressor

            manifest = json.loads(
                (self._artifact_dir / "runtime_manifest.json").read_text()
            )
            self._columns = manifest["adjustment_context"] + manifest[
                "adjustment_categorical"
            ]
            self._categoricals = manifest["adjustment_categorical"]
            self._adjuster = BoundedBowlerAdjustment(
                maximum_run_delta=manifest["maximum_run_delta"],
                maximum_wicket_logit_delta=manifest["maximum_wicket_logit_delta"],
            )
            self._run_model = CatBoostRegressor()
            self._run_model.load_model(str(self._artifact_dir / "run_adjustment.cbm"))
            self._wicket_model = CatBoostClassifier()
            self._wicket_model.load_model(
                str(self._artifact_dir / "wicket_adjustment.cbm")
            )
            self._wicket_platt = joblib.load(
                self._artifact_dir / "wicket_adjustment_platt.pkl"
            )
            self._name_to_id: dict[str, str] = {}
            players = pd.read_csv(self._players_path)
            for row in players.itertuples(index=False):
                for name in str(row.names).split(" | "):
                    self._name_to_id[identity_key(name)] = str(row.player_id)
            self._profiles: dict[tuple[str, str], dict[str, Any]] = {}
            profiles = pd.read_csv(self._artifact_dir / "bowler_phase_profiles.csv")
            for row in profiles.itertuples(index=False):
                self._profiles[(row.offline_next_bowler, row.phase)] = {
                    "balls": row.bowler_phase_history_balls,
                    "runs": row.bowler_phase_history_runs_conceded,
                    "wickets": row.bowler_phase_history_wickets,
                }
            self._h2h_profiles: dict[tuple[str, str], dict[str, Any]] = {}
            if self._h2h_path.exists():
                h2h = pd.read_csv(self._h2h_path)
                for row in h2h.itertuples(index=False):
                    self._h2h_profiles[(row.batter_id, row.bowler_id)] = {
                        "balls": row.h2h_balls,
                        "runs": row.h2h_runs,
                        "wickets": row.h2h_wickets,
                        "dots": row.h2h_dots,
                        "boundaries": row.h2h_boundaries,
                    }
            self._available = True
            _PROCESS_CACHE[cache_key] = {
                attr: getattr(self, attr) for attr in _CACHED_ATTRS
            }
        except Exception:
            self._available = False

    def adjust(
        self,
        *,
        bowler: str,
        batter: str,
        over: int,
        phase: str,
        wickets_in_hand: int,
        recent_wicket_rate: float,
        current_run_rate: float,
        required_run_rate: float,
        base_runs: float,
        base_wicket_probability: float,
    ) -> AdjustedPrediction | None:
        """Return an adjusted prediction, or None if unavailable/unresolvable."""

        self._load()
        if not self._available or not bowler:
            return None
        player_id = self._name_to_id.get(identity_key(bowler), "")
        if not player_id:
            return None
        batter_id = self._name_to_id.get(identity_key(batter), "") if batter else ""

        announcement = BowlerAnnouncement(
            bowler=bowler,
            source="scoreboard_pre_over",
            over=over,
            expected_over=over,
            row_is_empty=True,
            captured_before_first_ball=True,
        )
        profile = self._profiles.get((player_id, phase), {})
        h2h_profile = (
            self._h2h_profiles.get((batter_id, player_id), {}) if batter_id else {}
        )
        features = self._tracker.features(
            announcement,
            history_phase_balls=profile.get("balls", 0),
            history_phase_runs=profile.get("runs", 0),
            history_phase_wickets=profile.get("wickets", 0),
            h2h_balls=h2h_profile.get("balls", 0),
            h2h_runs=h2h_profile.get("runs", 0),
            h2h_wickets=h2h_profile.get("wickets", 0),
        )
        if features.bowler_source != "scoreboard_pre_over":
            return None
        h2h_balls = h2h_profile.get("balls", 0)
        h2h_dot_rate = h2h_profile.get("dots", 0) / h2h_balls if h2h_balls else 0.0
        h2h_boundary_rate = (
            h2h_profile.get("boundaries", 0) / h2h_balls if h2h_balls else 0.0
        )

        row = {
            "base_prediction": base_runs,
            "over": over,
            "wickets_in_hand": wickets_in_hand,
            "partnership_legal_ball_age": 0,
            "recent_wicket_rate": recent_wicket_rate,
            "current_run_rate": current_run_rate,
            "required_run_rate": required_run_rate,
            "bowler_match_balls": features.match_balls,
            "bowler_match_runs_conceded": features.match_runs_conceded,
            "bowler_match_wickets": features.match_wickets,
            "bowler_match_dot_rate": features.match_dot_rate,
            "bowler_match_boundary_concession_rate": features.match_boundary_concession_rate,
            "bowler_match_strike_rate": features.match_strike_rate,
            "bowler_match_economy": features.match_economy,
            "overs_since_previous": features.overs_since_previous,
            "consecutive_overs": features.consecutive_overs,
            "spell_number": features.spell_number,
            "current_spell_balls": features.current_spell_balls,
            "bowler_phase_history_balls": features.history_phase_balls,
            "bowler_phase_history_runs_conceded": profile.get("runs", 0),
            "bowler_phase_history_wickets": profile.get("wickets", 0),
            "bowler_phase_history_dot_rate": 0.0,
            "bowler_phase_history_boundary_concession_rate": 0.0,
            "bowler_phase_history_strike_rate": features.history_phase_strike_rate,
            "bowler_phase_history_economy": features.history_phase_economy,
            "h2h_balls": features.h2h_balls,
            "h2h_runs": features.h2h_runs,
            "h2h_wickets": features.h2h_wickets,
            "h2h_dot_rate": h2h_dot_rate,
            "h2h_boundary_rate": h2h_boundary_rate,
            "phase": phase,
            "spell_state": (
                "new_spell" if features.current_spell_balls == 0 else "returning_spell"
            ),
            "bowler_history_supported": features.history_supported,
            "h2h_supported": features.h2h_supported,
        }
        frame = pd.DataFrame([row])[self._columns]
        for column in self._categoricals:
            frame[column] = frame[column].astype(str)

        proposed_runs = float(self._run_model.predict(frame)[0])
        raw_wicket = float(self._wicket_model.predict_proba(frame)[0, 1])
        proposed_wicket = self._platt(raw_wicket)

        return self._adjuster.apply(
            base_runs=base_runs,
            base_wicket_probability=base_wicket_probability,
            features=features,
            proposed_run_delta=proposed_runs - base_runs,
            proposed_wicket_logit_delta=(
                self._logit(proposed_wicket) - self._logit(base_wicket_probability)
            ),
        )

    def record_completed_over(
        self, *, bowler: str, over: int, actual_runs: int, actual_wickets: int
    ) -> None:
        """Update leakage-safe spell state after the over's true outcome."""

        if not bowler:
            return
        self._tracker.record_completed_over(
            bowler=bowler,
            over=over,
            legal_balls=6,
            runs_conceded=actual_runs,
            wickets=actual_wickets,
            dots=0,
            boundaries=0,
        )

    def _platt(self, probability: float) -> float:
        return float(
            self._wicket_platt.predict_proba([[self._logit(probability)]])[0, 1]
        )

    @staticmethod
    def _logit(probability: float) -> float:
        probability = min(1 - 1e-6, max(1e-6, probability))
        return float(np.log(probability / (1 - probability)))
