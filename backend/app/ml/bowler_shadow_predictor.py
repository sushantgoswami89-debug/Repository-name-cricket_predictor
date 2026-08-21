"""Fail-safe shadow inference for the Cricsheet-trained bowler candidate."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor

from app.live.toi_reader import ToiDelivery
from app.ml.ipl_identities import identity_key
from app.models.match_context import MatchContext


class BowlerShadowPredictor:
    """Apply v3 only when an announced TOI bowler resolves exactly."""

    def __init__(self, project_root: Path | None = None) -> None:
        self.root = project_root or Path(__file__).resolve().parents[3]
        self.artifact_dir = (
            self.root / "models/candidates/announced_bowler_current_spell_v3"
        )
        self._loaded = False

    def predict(
        self,
        *,
        context: MatchContext,
        baseline: dict[str, object],
        announced_bowler: str,
        deliveries: list[ToiDelivery],
    ) -> dict[str, object]:
        if not announced_bowler:
            return self._unavailable("unannounced")
        try:
            self._load()
            player_id = self._name_to_id.get(identity_key(announced_bowler), "")
            if not player_id:
                return self._unavailable("bowler_identity_unresolved")
            phase = self._phase(context.live.over)
            profile = self._profiles.get((player_id, phase), {})
            current = self._current_match(
                announced_bowler, context.live.over, deliveries
            )
            row = {
                "base_prediction": float(baseline["expected_runs"]),
                "over": context.live.over,
                "wickets_in_hand": context.live.wickets_in_hand,
                "partnership_legal_ball_age": 0,
                "recent_wicket_rate": context.live.recent_wicket_rate,
                "current_run_rate": context.live.current_run_rate,
                "required_run_rate": context.live.required_run_rate,
                **current,
                **profile,
                "h2h_balls": 0,
                "h2h_runs": 0,
                "h2h_wickets": 0,
                "h2h_dot_rate": 0.0,
                "h2h_boundary_rate": 0.0,
                "phase": phase,
                "spell_state": current["spell_state"],
                "bowler_history_supported": bool(
                    profile.get("bowler_phase_history_balls", 0) >= 120
                ),
                "h2h_supported": False,
            }
            frame = pd.DataFrame([row])[self._columns]
            for column in self._categoricals:
                frame[column] = frame[column].astype(str)
            proposed_runs = float(self._run_model.predict(frame)[0])
            raw_wicket = float(self._wicket_model.predict_proba(frame)[0, 1])
            proposed_wicket = self._platt(self._wicket_platt, raw_wicket)
            base_runs = float(baseline["expected_runs"])
            base_wicket = float(baseline["wicket_probability"])
            run_delta = np.clip(
                proposed_runs - base_runs,
                -self._manifest["maximum_run_delta"],
                self._manifest["maximum_run_delta"],
            )
            wicket_delta = np.clip(
                self._logit(proposed_wicket) - self._logit(base_wicket),
                -self._manifest["maximum_wicket_logit_delta"],
                self._manifest["maximum_wicket_logit_delta"],
            )
            adjusted_runs = base_runs + self._manifest["run_alpha"] * run_delta
            adjusted_wicket = self._logistic(
                self._logit(base_wicket)
                + self._manifest["wicket_alpha"] * wicket_delta
            )
            return {
                "status": "applied",
                "candidate_version": self._manifest["candidate_version"],
                "publishing_enabled": False,
                "bowler": announced_bowler,
                "player_id": player_id,
                "phase": phase,
                "expected_runs": round(float(adjusted_runs), 4),
                "wicket_probability": round(float(adjusted_wicket), 6),
                "baseline_expected_runs": base_runs,
                "baseline_wicket_probability": base_wicket,
                "history_supported": row["bowler_history_supported"],
            }
        except Exception as exc:
            return self._unavailable(
                "shadow_inference_failed", detail=type(exc).__name__
            )

    def _load(self) -> None:
        if self._loaded:
            return
        self._manifest = json.loads(
            (self.artifact_dir / "runtime_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        self._columns = self._manifest["adjustment_context"] + self._manifest[
            "adjustment_categorical"
        ]
        self._categoricals = self._manifest["adjustment_categorical"]
        self._run_model = CatBoostRegressor()
        self._run_model.load_model(self.artifact_dir / "run_adjustment.cbm")
        self._wicket_model = CatBoostClassifier()
        self._wicket_model.load_model(self.artifact_dir / "wicket_adjustment.cbm")
        self._wicket_platt = joblib.load(
            self.artifact_dir / "wicket_adjustment_platt.pkl"
        )
        self._name_to_id = {}
        players = pd.read_csv(
            self.root / "data/reports/ipl_canonical_identities_v1/players.csv"
        )
        for item in players.itertuples(index=False):
            for name in str(item.names).split(" | "):
                self._name_to_id[identity_key(name)] = str(item.player_id)
        features = pd.read_csv(
            self.artifact_dir / "bowler_phase_profiles.csv"
        )
        self._profiles = {}
        for item in features.itertuples(index=False):
            self._profiles[(item.offline_next_bowler, item.phase)] = {
                name: getattr(item, name)
                for name in (
                    "bowler_phase_history_balls",
                    "bowler_phase_history_runs_conceded",
                    "bowler_phase_history_wickets",
                    "bowler_phase_history_dot_rate",
                    "bowler_phase_history_boundary_concession_rate",
                    "bowler_phase_history_strike_rate",
                    "bowler_phase_history_economy",
                )
            }
        self._loaded = True

    @staticmethod
    def _current_match(
        bowler: str, next_over: int, deliveries: list[ToiDelivery]
    ) -> dict[str, int | float | str]:
        key = identity_key(bowler)
        selected = [item for item in deliveries if identity_key(item.bowler) == key]
        legal = [item for item in selected if item.is_legal]
        runs = sum(
            item.total_runs
            - item.extras.get("byes", 0)
            - item.extras.get("legbyes", 0)
            for item in selected
        )
        excluded = {"run out", "retired hurt", "retired out", "obstructing the field"}
        wickets = sum(
            item.wicket_kind is not None
            and str(item.wicket_kind).lower() not in excluded
            for item in selected
        )
        dots = sum(
            item.total_runs
            - item.extras.get("byes", 0)
            - item.extras.get("legbyes", 0)
            == 0
            for item in legal
        )
        boundaries = sum(item.batter_runs in {4, 6} for item in legal)
        overs = sorted({item.over for item in selected})
        last = overs[-1] if overs else -1
        gap = next_over - last - 1 if last >= 0 else -1
        new_spell = last < 0 or gap >= 2
        spell_overs = []
        for over in reversed(overs):
            if spell_overs and spell_overs[-1] - over >= 3:
                break
            spell_overs.append(over)
        spell_balls = sum(item.is_legal and item.over in spell_overs for item in selected)
        balls = len(legal)
        return {
            "bowler_match_balls": balls,
            "bowler_match_runs_conceded": runs,
            "bowler_match_wickets": wickets,
            "bowler_match_dot_rate": dots / balls if balls else 0.0,
            "bowler_match_boundary_concession_rate": (
                boundaries / balls if balls else 0.0
            ),
            "bowler_match_strike_rate": balls / wickets if wickets else 0.0,
            "bowler_match_economy": 6 * runs / balls if balls else 0.0,
            "overs_since_previous": gap,
            "consecutive_overs": 1,
            "spell_number": 1 + int(new_spell and bool(overs)),
            "current_spell_balls": 0 if new_spell else spell_balls,
            "spell_state": "new_spell" if new_spell else "returning_spell",
        }

    @staticmethod
    def _phase(over: int) -> str:
        return "powerplay" if over <= 6 else "middle" if over <= 15 else "death"

    @staticmethod
    def _platt(model, probability: float) -> float:
        return float(model.predict_proba([[BowlerShadowPredictor._logit(probability)]])[0, 1])

    @staticmethod
    def _logit(probability: float) -> float:
        probability = min(1 - 1e-6, max(1e-6, probability))
        return float(np.log(probability / (1 - probability)))

    @staticmethod
    def _logistic(value: float) -> float:
        return float(1 / (1 + np.exp(-value)))

    @staticmethod
    def _unavailable(reason: str, detail: str = "") -> dict[str, object]:
        result: dict[str, object] = {
            "status": "not_applied",
            "candidate_version": "announced_bowler_current_spell_v3",
            "publishing_enabled": False,
            "reason": reason,
        }
        if detail:
            result["detail"] = detail
        return result
