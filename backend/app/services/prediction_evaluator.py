"""Evaluate CricketBaba predictions against replayed match outcomes."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from app.ml.feature_builder import FeatureBuilder
from app.ml.historical_feature_store import match_exclude_key
from app.ml.model_repository import ModelRepository
from app.models.match_context import MatchContext
from app.models.match_data import Match
from app.replay.evaluation_policy import (
    MatchExcludedError,
    exclusion_reason,
    match_eligibility,
)
from app.replay.match_state_verifier import MatchStateVerifier
from app.replay.replay_loader import ReplayLoader
from app.services.match_replay import MatchReplay, ReplayFrame


@dataclass(frozen=True, slots=True)
class PredictionEvaluation:
    """One predicted over and the recorded result used to check it."""

    source_file: str
    match_format: str
    innings_number: int
    is_super_over: bool
    batting_team: str
    over: int
    score_before_over: int
    wickets_before_over: int
    striker: str
    non_striker: str
    bowler: str
    predicted_runs: float
    actual_runs: int
    run_error: float
    wicket_probability: float
    predicted_wicket: bool
    actual_wicket: bool
    wicket_prediction_correct: bool
    confidence: float
    model_scope: str
    rule_exception: bool
    rule_exception_codes: str
    eligibility_scope: str

    def to_dict(self) -> dict[str, Any]:
        """Return a CSV-ready representation."""

        return asdict(self)


class PredictionEvaluator:
    """Apply the current model to replay frames and retain actual outcomes."""

    _WICKET_THRESHOLD = 0.50
    _MODEL_CONFIDENCE = 0.90

    def __init__(
        self,
        repository: ModelRepository | None = None,
        feature_builder: FeatureBuilder | None = None,
        replay: MatchReplay | None = None,
        replay_loader: ReplayLoader | None = None,
        verifier: MatchStateVerifier | None = None,
    ) -> None:
        self._repository = repository or ModelRepository()
        self._feature_builder = feature_builder or FeatureBuilder()
        self._replay = replay or MatchReplay()
        self._replay_loader = replay_loader or ReplayLoader()
        self._verifier = verifier or MatchStateVerifier()

    def evaluate_match(
        self,
        match: Match,
        source_file: str,
    ) -> list[PredictionEvaluation]:
        """Evaluate every non-empty over in a parsed match."""

        eligibility = match_eligibility(match)
        if not eligibility.eligible:
            raise MatchExcludedError(
                source_file.removesuffix(".json"),
                eligibility.reason,
                category="ineligible_competition",
            )

        reason = exclusion_reason(source_file)
        if reason is not None:
            raise MatchExcludedError(
                source_file.removesuffix(".json"),
                reason,
                category="known_rule_anomaly",
            )

        verification_frames = list(
            self._replay_loader.frames(match, match_id=source_file)
        )
        self._verifier.assert_valid(match, verification_frames, match_id=source_file)
        anomalies_by_innings_over = {
            (frame.innings_index + 1, frame.over_number): frame.rule_anomalies
            for frame in verification_frames
        }

        evaluations: list[PredictionEvaluation] = []
        for innings_number, innings in enumerate(match.innings, start=1):
            frames = list(self._replay.frames(innings))
            if not frames:
                continue

            predictions = self._predict_frames(match, frames)
            model_scope = self._model_scope(match.info.match_type)
            evaluations.extend(
                self._to_evaluation(
                    source_file=source_file,
                    match_format=match.info.match_type,
                    innings_number=innings_number,
                    is_super_over=innings.super_over,
                    batting_team=innings.team,
                    frame=frame,
                    predicted_runs=predicted_runs,
                    wicket_probability=wicket_probability,
                    model_scope=model_scope,
                    rule_anomalies=anomalies_by_innings_over.get(
                        (innings_number, frame.state.over), ()
                    ),
                    eligibility_scope=eligibility.scope,
                )
                for frame, (predicted_runs, wicket_probability) in zip(
                    frames, predictions, strict=True
                )
            )
        return evaluations

    def _predict_frames(
        self,
        match: Match,
        frames: Sequence[ReplayFrame],
    ) -> list[tuple[float, float]]:
        exclude_key = match_exclude_key(match)
        features = [
            self._feature_builder.build(
                MatchContext(
                    team1=match.info.teams[0],
                    team2=match.info.teams[1],
                    venue=match.info.venue or "",
                    format=match.info.match_type,
                    competition=match.info.event_name or "",
                    live=frame.state,
                    metadata={"exclude_match_key": exclude_key},
                )
            )
            for frame in frames
        ]
        feature_order = self._repository.get_feature_columns()
        dataframe = pd.DataFrame(features)[feature_order]
        for column in self._repository.get_categorical_columns():
            if column in dataframe.columns:
                dataframe[column] = dataframe[column].astype("category")

        runs = self._repository.get_runs_model().predict(dataframe)
        wicket_probabilities = self._repository.get_wicket_model().predict_proba(
            dataframe
        )[:, 1]
        return [
            (float(predicted_runs), float(wicket_probability))
            for predicted_runs, wicket_probability in zip(
                runs, wicket_probabilities, strict=True
            )
        ]

    def _to_evaluation(
        self,
        source_file: str,
        match_format: str,
        innings_number: int,
        is_super_over: bool,
        batting_team: str,
        frame: ReplayFrame,
        predicted_runs: float,
        wicket_probability: float,
        model_scope: str,
        rule_anomalies: Sequence[Any],
        eligibility_scope: str,
    ) -> PredictionEvaluation:
        state = frame.state
        actual_wicket = frame.actual_wickets > 0
        predicted_wicket = wicket_probability >= self._WICKET_THRESHOLD
        return PredictionEvaluation(
            source_file=source_file,
            match_format=match_format,
            innings_number=innings_number,
            is_super_over=is_super_over,
            batting_team=batting_team,
            over=state.over,
            score_before_over=state.score_before_over,
            wickets_before_over=state.wkts_down_before_over,
            striker=state.striker,
            non_striker=state.non_striker,
            bowler=state.bowler,
            predicted_runs=predicted_runs,
            actual_runs=frame.actual_runs,
            run_error=predicted_runs - frame.actual_runs,
            wicket_probability=wicket_probability,
            predicted_wicket=predicted_wicket,
            actual_wicket=actual_wicket,
            wicket_prediction_correct=predicted_wicket == actual_wicket,
            confidence=self._MODEL_CONFIDENCE,
            model_scope=model_scope,
            rule_exception=bool(rule_anomalies),
            rule_exception_codes="|".join(
                anomaly.code for anomaly in rule_anomalies
            ),
            eligibility_scope=eligibility_scope,
        )

    @staticmethod
    def _model_scope(match_format: str) -> str:
        if match_format.upper() == "T20":
            return "in_scope_t20_model"
        return "exploratory_t20_model_on_non_t20_match"
