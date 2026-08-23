"""
Prediction Engine v3.

Runs prediction is sourced from run_range_v11_partnership_rate's own
calibrated inclusive band (see docs/finding_partnership_scoring_rate.md):
run_range_v7_competition_prior plus current-partnership scoring rate
(team runs/balls since the fall of the last wicket, resets at each
dismissal -- a cleaner signal than partnership_legal_ball_age alone,
which only tracks how long a pair has been together, not how well).
28.80% blended / 25.16% IPL / 29.45% T20I holdout hit rate, beats v7 on
every split. v7 itself (28.75%/25.00%/29.43%) is
run_range_v4_batter_phase (28.60% blended) plus competition-specific
(IPL vs T20I) batter/bowler prior-stat features (see
app/ml/player_competition_dataset.py's docstring) -- see
docs/candidate_run_range_enriched_v2.md and
docs/finding_blended_holdout_masks_ipl_accuracy.md.
Wicket prediction is sourced from contract22_wicket_v16_team_composition's
calibrated probability (see docs/finding_team_composition.md):
contract22_wicket_v15_batter_phase_recency plus match-level team role
composition (batting-side all-rounder depth, bowling-side pace/spin
balance from the confirmed XI -- known before ball 1, same timing as
toss). Mostly flat when the bowler is known (redundant with that
bowler's own individual stats) but a real, consistent gain specifically
when the bowler is unknown (AUC +0.0015 blended, +0.0029 IPL, +0.0009
T20I) -- the harder, more common real live-serving case. v15 itself (see
docs/finding_batter_phase_recency_promoted.md) is
contract22_wicket_v10_partnership_rate plus phase-specific batter
recency (recent strike-rate/dismissal-rate within the same phase as the
current over -- "finishing ability" is a distinct, form-sensitive skill,
and death overs are the single most decisive IPL phase per external
analysis). Real, consistent IPL gain there (AUC 0.6104->0.6149 known,
0.6059->0.6102 unknown), T20I/blended roughly flat. v10 itself (see
docs/finding_partnership_scoring_rate.md) is
contract22_wicket_v9_recency_form plus current-partnership scoring rate
(team runs/balls since the fall of the last wicket, resets at each
dismissal). AUC improved on every split tested there too (blended/IPL/
T20I x known/unknown, all 6 up), Brier improved on every split too. v9
itself (see docs/finding_recency_weighted_form.md) is
contract22_wicket_v2_batter_state plus recency-weighted (match-EWMA,
decay=0.95) batter/bowler form added alongside the existing flat
career-average features -- every flat prior-stat feature in this codebase
weighted a player's form from 40 matches ago exactly as much as last
week's until that promotion. v2 itself (AUC 0.6081/0.6072 blended) is
contract22_wicket_rigorous plus a new-batter/partnership-age signal (see
docs/candidate_ipl_wicket_v7_2_spell_features.md's 2026-08-22 follow-ups),
validated end-to-end through the real VerifiedLivePredictionPipeline
(AUC 0.6067, Brier 0.2000 on 669 replayed overs, 20 held-out matches).
BowlerSpellAdjuster is NOT applied to either model (removed 2026-08-22):
it was tuned as an adjustment layer on top of the old, weaker models, and
a direct replay comparison on 745 real overs showed it measurably HURTS
the new wicket model (AUC 0.5945 -> 0.5768, Brier 0.2098 -> 0.2117 with
the adjustment applied) rather than helping -- not just an untested
combination this time, an empirically harmful one.
"""

from __future__ import annotations

import time
from pathlib import Path

from app.ml.engine_router import EngineFamily, EngineRouter
from app.ml.model_repository import ModelRepository
from app.ml.prediction_result import PredictionResult
from app.models.match_context import MatchContext
from runtime_run_range_v3 import RunRangeRuntimeV3
from runtime_wicket_contract22 import WicketRuntimeContract22

RUN_RANGE_V3_ARTIFACTS = (
    Path(__file__).resolve().parents[3]
    / "models/candidates/run_range_v11_partnership_rate"
)
WICKET_CONTRACT22_ARTIFACTS = (
    Path(__file__).resolve().parents[3]
    / "models/candidates/contract22_wicket_v16_team_composition"
)


class PredictionEngine:
    def __init__(
        self,
        repository: ModelRepository | None = None,
        run_range_runtime: RunRangeRuntimeV3 | None = None,
        wicket_runtime: WicketRuntimeContract22 | None = None,
    ) -> None:
        self._repository = repository or ModelRepository()
        # run_range_v7_competition_prior (28.75% blended / 25.00% IPL /
        # 29.43% T20I, beats run_range_v4_batter_phase's 28.60%/24.19%/
        # 29.22% on every split -- see docs/candidate_run_range_enriched_v2.md)
        # replaces the runs_model.pkl prediction entirely: its own
        # calibrated inclusive band becomes expected_range directly.
        self._run_range_runtime = run_range_runtime or RunRangeRuntimeV3(RUN_RANGE_V3_ARTIFACTS)
        # contract22_wicket_v2_batter_state (Brier skill score 3.58% vs the
        # old models/wkt_model.pkl's own naive estimate of 1.95% -- see
        # docs/candidate_ipl_wicket_v7_2_spell_features.md's 2026-08-22
        # follow-ups) replaces the wkt_model.pkl prediction.
        self._wicket_runtime = wicket_runtime or WicketRuntimeContract22(WICKET_CONTRACT22_ARTIFACTS)

        # --- STATE MANAGEMENT ---
        self._last_predicted_over = -1
        self._is_awaiting_actuals = False

        # --- TRACKERS ---
        self._bowler_stats = {}
        self._error_history = []

    def export_runtime_state(self) -> dict[str, object]:
        """Return the state needed for a safe restart."""
        return {
            "last_predicted_over": self._last_predicted_over,
            "is_awaiting_actuals": self._is_awaiting_actuals,
            "bowler_stats": dict(self._bowler_stats),
        }

    def reset_runtime_state(self) -> None:
        """Clear match state before rebuilding from corrected facts."""
        self._last_predicted_over = -1
        self._is_awaiting_actuals = False
        self._bowler_stats = {}
        self._error_history = []

    def restore_runtime_state(self, state: dict[str, object]) -> None:
        """Restore validated state from live persistence."""
        self._last_predicted_over = int(state.get("last_predicted_over", -1))
        self._is_awaiting_actuals = bool(state.get("is_awaiting_actuals", False))
        bowler_stats = state.get("bowler_stats", {})
        if not isinstance(bowler_stats, dict):
            raise ValueError("Persisted bowler stats must be an object.")
        self._bowler_stats = {
            str(name): int(value) for name, value in bowler_stats.items()
        }

    def _get_format_rules(self, style: str, competition: str = ""):
        specification = EngineRouter.resolve(style, competition)
        return {
            "pp": specification.powerplay_overs,
            "max_overs": 10 if specification.family is EngineFamily.ODI else 4,
            "death": specification.death_starts,
            "engine_family": specification.family.value,
        }

    def _build_analysis(
        self, context, pred_runs, wkt_prob, latency, rules
    ) -> list[str]:
        analysis: list[str] = []
        over = context.live.over
        # Wording matches the real phase-accuracy ranking measured in
        # experiment_confidence_signal_search.py against the CURRENT model
        # stack (run_range_enriched_v3_batting_style / contract22_wicket_v2)
        # -- see the phase_penalty comment in _dynamic_confidence below for
        # the same evidence, including the earlier version of this fix that
        # was validated against a stale pre-swap replay and got it backwards.
        if over <= rules["pp"]:
            analysis.append("Phase: Powerplay (High Volatility).")
        elif over >= rules["death"]:
            analysis.append("Phase: Death Overs (Extreme Momentum).")
        else:
            analysis.append("Phase: Middle Overs (Our most reliable prediction window).")

        return analysis

    @staticmethod
    def _dynamic_confidence(
        context: MatchContext,
        raw_runs: float,
        evolved_runs: float,
        range_width: int,
    ) -> tuple[float, dict[str, float | str]]:
        """Estimate per-over confidence from information available at prediction time.

        This is a bounded stability score rather than a probability that the
        prediction will be exactly correct. It deliberately falls during
        volatile phases and when current match rhythm disagrees with history.
        """

        over = context.live.over
        recent_runs = context.live.runs_last_3_overs
        recent_wickets = context.live.wickets_last_3_overs
        observed_overs = max(0, over - 1)

        maturity_bonus = min(0.10, observed_overs * 0.0125)
        # phase_penalty is fit from real range-hit outcomes, not hand-picked.
        # NOTE: an earlier version of this fix (2026-08-22) was validated
        # against data/reports/confidence_signal_search/report.json without
        # checking it was stale -- that replay predated the runs/wicket
        # model swap to run_range_enriched_v3_batting_style /
        # contract22_wicket_v2_batter_state, and had powerplay/middle
        # backwards for the CURRENT models. Re-validated against a fresh
        # 3,091-row holdout replay (2024+) run through today's actual
        # PredictionEngine: middle overs have the highest real range-hit
        # rate (25.1%), powerplay and death are both worse and roughly tied
        # (20.3%/20.6%). Values below are scaled from each phase's holdout
        # hit-rate deviation from the overall mean onto a 0.02-0.09 range;
        # this landed within noise of the original hand-picked constants
        # (holdout AUC 0.5436 vs 0.5438), which turned out to already be
        # approximately correct for the current models by coincidence.
        if over <= 6:
            phase_penalty = 0.09
            phase_name = "powerplay"
        elif over >= 16:
            phase_penalty = 0.086
            phase_name = "death"
        else:
            phase_penalty = 0.02
            phase_name = "middle"

        if recent_runs > 0:
            recent_rate = recent_runs / min(3, max(1, observed_overs))
            disagreement = abs(raw_runs - recent_rate)
            rhythm_penalty = min(0.14, disagreement * 0.018)
        else:
            disagreement = 0.0
            rhythm_penalty = 0.07 if observed_overs > 0 else 0.10

        # wicket_bonus and range_bonus were originally signed as penalties
        # (more recent wickets / wider range -> lower confidence), on the
        # assumption that both signal volatility. A holdout evaluation
        # against 3,091 real IPL overs (2024+, see
        # experiment_confidence_signal_search.py,
        # docs/CTO_HANDOVER_2026-07-24.md 2026-08-22 addendum) found both
        # signs were backwards: recent_wicket_rate and range_width were the
        # two strongest POSITIVE predictors of actual range-hit accuracy --
        # a wider range mechanically covers more outcomes, and overs
        # following a wicket tend to be more conservative/predictable
        # (new batter settling in), not less. Flipped to bonuses.
        wicket_bonus = min(0.08, max(0, recent_wickets) * 0.025)
        range_bonus = min(0.06, max(0, range_width - 2) * 0.02)
        adaptation_penalty = min(0.08, abs(evolved_runs - raw_runs) * 0.015)
        confidence = (
            0.84
            + maturity_bonus
            + wicket_bonus
            + range_bonus
            - (
                phase_penalty
                + rhythm_penalty
                + adaptation_penalty
            )
        )
        confidence = round(min(0.92, max(0.50, confidence)), 2)
        factors: dict[str, float | str] = {
            "method": "dynamic_match_stability_v2",
            "phase": phase_name,
            "maturity_bonus": round(maturity_bonus, 3),
            "phase_penalty": round(phase_penalty, 3),
            "rhythm_disagreement": round(disagreement, 3),
            "rhythm_penalty": round(rhythm_penalty, 3),
            "wicket_bonus": round(wicket_bonus, 3),
            "range_bonus": round(range_bonus, 3),
            "adaptation_penalty": round(adaptation_penalty, 3),
        }
        return confidence, factors

    def predict(self, context: MatchContext) -> PredictionResult:
        current_over = context.live.over
        style = context.format or context.live.match_style
        rules = self._get_format_rules(style, context.competition)

        # 1. GATEKEEPER
        if self._is_awaiting_actuals and current_over > self._last_predicted_over:
            raise ValueError(
                f"FEED REQUIRED: Over {self._last_predicted_over} missing."
            )

        start = time.perf_counter()

        # Which named team (team1/team2) is currently batting -- same
        # is_chase/batting_first/bowling_first resolution the run-range
        # section below already used, computed once here so the wicket
        # model's team-composition lookup (added 2026-08-23 for
        # contract22_wicket_v16_team_composition) and run-range's
        # competition-prior features share one answer instead of deriving
        # it twice.
        is_chase_for_batting_team = bool(context.live.is_chase)
        batting_team = (
            (context.bowling_first or context.team2)
            if is_chase_for_batting_team
            else (context.batting_first or context.team1)
        )
        if batting_team == context.team1:
            batting_team_players, bowling_team_players = context.team1_players, context.team2_players
        else:
            batting_team_players, bowling_team_players = context.team2_players, context.team1_players

        # 2. WICKET PROBABILITY (contract22_wicket_v2_batter_state)
        # Replaces the old models/wkt_model.pkl + phase-keyed calibrator
        # pipeline. Also retires the FeatureBuilder/HistoricalFeatureStore
        # 22-feature-contract path entirely (nothing else in predict() used
        # it once both models moved off it -- see git history for the old
        # `df`/`features = self._builder.build(context)` construction this
        # replaced).
        raw_wkt_prob = self._wicket_runtime.predict_wicket_probability(
            registry=context.metadata.get("registry", {}),
            striker_name=context.live.striker,
            non_striker_name=context.live.non_striker,
            bowler_name=context.live.bowler,
            venue_name=context.venue,
            deliveries=context.metadata.get("deliveries", []),
            over=current_over,
            score_before_over=context.live.score_before_over,
            wkts_down_before_over=context.live.wkts_down_before_over,
            wickets_in_hand=context.live.wickets_in_hand,
            legal_balls_bowled=context.live.legal_balls_bowled,
            balls_remaining=context.live.balls_remaining,
            current_run_rate=context.live.current_run_rate,
            is_chase=bool(context.live.is_chase),
            runs_required=context.live.runs_required,
            required_run_rate=context.live.required_run_rate,
            recent_legal_balls=context.live.recent_legal_balls,
            recent_runs_per_ball=context.live.recent_runs_per_ball,
            recent_dot_rate=context.live.recent_dot_rate,
            recent_single_rate=context.live.recent_single_rate,
            recent_boundary_rate=context.live.recent_boundary_rate,
            recent_wicket_rate=context.live.recent_wicket_rate,
            batting_team_players=batting_team_players,
            bowling_team_players=bowling_team_players,
        )

        # 3. RUN RANGE (run_range_v7_competition_prior)
        # Replaces the old runs_model.pkl point-estimate + momentum-blend +
        # match_bias + fixed-width-bracket pipeline entirely: this model's
        # own calibrated inclusive band (from a full probability
        # distribution over 0-30 runs) becomes expected_range directly.
        # That combination of steps was never validated together and v3's
        # 28.49% holdout hit rate assumed none of it -- see
        # docs/candidate_run_range_enriched_v2.md for the full comparison
        # against v3.3 and a fair rebuild of the legacy 22-feature contract.
        #
        # width is competition-aware (2026-08-22 finding, see
        # docs/finding_blended_holdout_masks_ipl_accuracy.md): the
        # headline "28.6%" holdout hit rate was a blended IPL+T20I number.
        # Split apart, IPL alone sits at 24.19% vs T20I's 29.22% on the
        # same fixed width-2 band -- traced to IPL's genuinely wider
        # run-scoring distribution (std 4.72 vs 4.58), not a fixable
        # training-population bug (reweighting/full separation were both
        # tested and neither moved IPL's hit rate). A cheap, real fix:
        # IPL at width=3 scores 32.69%, already beating T20I's own
        # width=2 rate -- one extra run of band width closes the gap
        # without touching the model at all.
        width = 3 if rules["engine_family"] == "ipl" else 2
        # Also feeds run_range_v7_competition_prior's competition-specific
        # player features (see app/ml/player_competition_dataset.py) --
        # the model trained on "ipl"/"t20i" scopes only, so ODI (not part
        # of that population) maps to "t20i" the same way it already
        # falls back gracefully for a player with no competition-specific
        # history at all (shrinks to the pooled rate, doesn't crash).
        competition_scope = "ipl" if rules["engine_family"] == "ipl" else "t20i"
        is_chase = is_chase_for_batting_team
        deliveries = context.metadata.get("deliveries", [])
        registry = context.metadata.get("registry", {})
        run_range = self._run_range_runtime.predict_next_over(
            deliveries=deliveries,
            registry=registry,
            striker_name=context.live.striker,
            non_striker_name=context.live.non_striker,
            venue_name=context.venue,
            batting_team=batting_team,
            over=current_over,
            bowler_name=context.live.bowler,
            score_before_over=context.live.score_before_over,
            wkts_down_before_over=context.live.wkts_down_before_over,
            wickets_in_hand=context.live.wickets_in_hand,
            legal_balls_bowled=context.live.legal_balls_bowled,
            balls_remaining=context.live.balls_remaining,
            current_run_rate=context.live.current_run_rate,
            is_chase=is_chase,
            runs_required=context.live.runs_required,
            required_run_rate=context.live.required_run_rate,
            recent_legal_balls=context.live.recent_legal_balls,
            recent_runs_per_ball=context.live.recent_runs_per_ball,
            recent_dot_rate=context.live.recent_dot_rate,
            recent_single_rate=context.live.recent_single_rate,
            recent_boundary_rate=context.live.recent_boundary_rate,
            recent_wicket_rate=context.live.recent_wicket_rate,
            width=width,
            competition=competition_scope,
        )
        low_bound = run_range[f"sharp_{width}_low"]
        high_bound = run_range[f"sharp_{width}_high"]
        evolved_runs = (low_bound + high_bound) / 2.0
        # No adaptive multiplier here: contract22_wicket_v2_batter_state's
        # own calibration (Platt-fit on the 2024 holdout year) already handles
        # this. A per-match adaptive "wicket risk dampener," tuned against
        # the old wkt_model.pkl, was found to measurably hurt this model
        # instead (see the module docstring) -- removed, not kept as an
        # untested/harmful legacy correction.
        evolved_wkt = min(1.0, max(0.0, raw_wkt_prob))

        pivot = round(evolved_runs, 1)

        confidence, confidence_factors = self._dynamic_confidence(
            context,
            evolved_runs,
            evolved_runs,
            high_bound - low_bound,
        )
        raw_confidence = confidence
        confidence_calibrator = self._repository.get_confidence_calibrator()
        if confidence_calibrator is not None:
            confidence = round(
                float(confidence_calibrator.predict([confidence])[0]), 3
            )

        # Update Internal State
        self._last_predicted_over = current_over
        self._is_awaiting_actuals = True

        latency_ms = (time.perf_counter() - start) * 1000
        analysis = self._build_analysis(
            context, evolved_runs, evolved_wkt, latency_ms, rules
        )

        return PredictionResult(
            predicted_runs=pivot,
            expected_range=f"{low_bound}-{high_bound}",
            wicket_probability=round(evolved_wkt, 3),
            confidence=confidence,
            analysis=analysis,
            metadata={
                # Derived from the actual loaded artifact directories, not
                # hardcoded -- a hardcoded pair here silently went stale
                # across all five model promotions earlier in this session
                # (found 2026-08-23 while regenerating
                # data/reports/confidence_signal_search/report.json: its
                # own provenance stamp, sourced from this exact field,
                # still said run_range_v7/contract22_wicket_v2 after v11/v16
                # had been live for hours). This self-corrects on every
                # future promotion instead of requiring a manual edit here.
                "run_model": self._run_range_runtime.artifact_dir.name,
                "wicket_model": self._wicket_runtime.artifact_dir.name,
                "sharp_band_width": width,
                "sharp_band_prob": round(run_range[f"sharp_{width}_prob"], 3),
                "display_runs": pivot,
                "confidence_type": "dynamic stability indicator",
                "raw_confidence": round(raw_confidence, 3),
                "confidence_factors": confidence_factors,
                "engine_family": rules["engine_family"],
            },
        )

    def update_actuals(self, actual_runs: int, actual_wickets: int, bowler_name: str):
        if not self._is_awaiting_actuals:
            return "No pending over."

        self._bowler_stats[bowler_name] = self._bowler_stats.get(bowler_name, 0) + 1
        self._is_awaiting_actuals = False
        return "Evolved"
