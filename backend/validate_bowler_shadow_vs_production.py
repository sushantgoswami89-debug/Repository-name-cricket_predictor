"""Independently validate the announced_bowler_current_spell_v3 shadow
candidate against the ACTUAL current production model (PredictionEngine /
models/wkt_model.pkl + phase calibrator), not the candidate's own internal
"situation" baseline it was originally validated against (a different
CatBoost model). Uses real Cricsheet IPL replay, matching exactly the
feature-computation gaps the live wiring has today (h2h and
partnership_legal_ball_age hardcoded to 0/unsupported, per
bowler_shadow_predictor.py).
"""

import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import brier_score_loss, mean_absolute_error, roc_auc_score

from app.data_ingestion.json_loader import JsonLoader
from app.data_ingestion.match_parser import MatchParser
from app.ml.announced_bowler_adjustment import (
    BoundedBowlerAdjustment,
    BowlerAnnouncement,
    CurrentSpellTracker,
)
from app.ml.historical_feature_store import match_exclude_key
from app.ml.ipl_identities import identity_key
from app.ml.prediction_engine import PredictionEngine
from app.models.match_context import MatchContext
from app.services.match_replay import MatchReplay

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "models" / "candidates" / "announced_bowler_current_spell_v3"

# --- load shadow candidate artifacts ---
manifest = __import__("json").loads((ARTIFACT_DIR / "runtime_manifest.json").read_text())
columns = manifest["adjustment_context"] + manifest["adjustment_categorical"]
categoricals = manifest["adjustment_categorical"]
run_model = CatBoostRegressor()
run_model.load_model(str(ARTIFACT_DIR / "run_adjustment.cbm"))
wicket_model = CatBoostClassifier()
wicket_model.load_model(str(ARTIFACT_DIR / "wicket_adjustment.cbm"))
wicket_platt = joblib.load(ARTIFACT_DIR / "wicket_adjustment_platt.pkl")

players_df = pd.read_csv(PROJECT_ROOT / "data/reports/ipl_canonical_identities_v1/players.csv")
name_to_id = {}
for row in players_df.itertuples(index=False):
    for name in str(row.names).split(" | "):
        name_to_id[identity_key(name)] = str(row.player_id)

profiles_df = pd.read_csv(ARTIFACT_DIR / "bowler_phase_profiles.csv")
profiles = {}
for row in profiles_df.itertuples(index=False):
    profiles[(row.offline_next_bowler, row.phase)] = {
        "history_phase_balls": row.bowler_phase_history_balls,
        "history_phase_runs": row.bowler_phase_history_runs_conceded,
        "history_phase_wickets": row.bowler_phase_history_wickets,
    }

adjuster = BoundedBowlerAdjustment(
    maximum_run_delta=manifest["maximum_run_delta"],
    maximum_wicket_logit_delta=manifest["maximum_wicket_logit_delta"],
)


def logit(p):
    p = min(1 - 1e-6, max(1e-6, p))
    return float(np.log(p / (1 - p)))


def phase_of(over):
    return "powerplay" if over <= 6 else "middle" if over <= 15 else "death"


def platt(prob):
    return float(wicket_platt.predict_proba([[logit(prob)]])[0, 1])


N_MATCHES = 40
random.seed(555)  # distinct from every other seed used this session
match_dir = PROJECT_ROOT / "data/raw/cricsheet/ipl"
sample = random.sample(sorted(match_dir.glob("*.json")), N_MATCHES)

baseline_runs_err, adjusted_runs_err = [], []
baseline_wkt_prob, adjusted_wkt_prob, actual_wkt = [], [], []
n_announced = 0
n_total = 0

for path in sample:
    try:
        match = MatchParser().parse(JsonLoader().load(str(path)))
    except Exception:
        continue
    exclude_key = match_exclude_key(match)
    for innings in match.innings:
        engine = PredictionEngine()
        tracker = CurrentSpellTracker()
        for frame in MatchReplay().frames(innings):
            n_total += 1
            over = frame.state.over
            bowler_name = frame.state.bowler
            phase = phase_of(over)

            context = MatchContext(
                team1=match.info.teams[0],
                team2=match.info.teams[1],
                venue=match.info.venue or "Unknown",
                format=match.info.match_type,
                live=frame.state,
                metadata={"exclude_match_key": exclude_key},
            )
            baseline = engine.predict(context)

            player_id = name_to_id.get(identity_key(bowler_name), "")
            adjusted_runs, adjusted_wkt = baseline.predicted_runs, baseline.wicket_probability
            if player_id:
                announcement = BowlerAnnouncement(
                    bowler=bowler_name, source="scoreboard_pre_over",
                    over=over, expected_over=over, row_is_empty=True,
                    captured_before_first_ball=True,
                )
                profile = profiles.get((player_id, phase), {})
                features = tracker.features(
                    announcement,
                    history_phase_balls=profile.get("history_phase_balls", 0),
                    history_phase_runs=profile.get("history_phase_runs", 0),
                    history_phase_wickets=profile.get("history_phase_wickets", 0),
                )
                if features.bowler_source == "scoreboard_pre_over":
                    n_announced += 1
                    row = {
                        "base_prediction": baseline.predicted_runs,
                        "over": over,
                        "wickets_in_hand": frame.state.wickets_in_hand,
                        "partnership_legal_ball_age": 0,
                        "recent_wicket_rate": frame.state.recent_wicket_rate,
                        "current_run_rate": frame.state.current_run_rate,
                        "required_run_rate": frame.state.required_run_rate,
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
                        "bowler_phase_history_runs_conceded": profile.get("history_phase_runs", 0),
                        "bowler_phase_history_wickets": profile.get("history_phase_wickets", 0),
                        "bowler_phase_history_dot_rate": 0.0,
                        "bowler_phase_history_boundary_concession_rate": 0.0,
                        "bowler_phase_history_strike_rate": features.history_phase_strike_rate,
                        "bowler_phase_history_economy": features.history_phase_economy,
                        "h2h_balls": 0, "h2h_runs": 0, "h2h_wickets": 0,
                        "h2h_dot_rate": 0.0, "h2h_boundary_rate": 0.0,
                        "phase": phase,
                        "spell_state": "new_spell" if features.current_spell_balls == 0 else "returning_spell",
                        "bowler_history_supported": features.history_supported,
                        "h2h_supported": False,
                    }
                    frame_df = pd.DataFrame([row])[columns]
                    for col in categoricals:
                        frame_df[col] = frame_df[col].astype(str)
                    proposed_runs = float(run_model.predict(frame_df)[0])
                    raw_wkt = float(wicket_model.predict_proba(frame_df)[0, 1])
                    proposed_wkt = platt(raw_wkt)
                    result = adjuster.apply(
                        base_runs=baseline.predicted_runs,
                        base_wicket_probability=baseline.wicket_probability,
                        features=features,
                        proposed_run_delta=proposed_runs - baseline.predicted_runs,
                        proposed_wicket_logit_delta=logit(proposed_wkt) - logit(baseline.wicket_probability),
                    )
                    adjusted_runs, adjusted_wkt = result.runs, result.wicket_probability

            had_wicket = frame.actual_wickets > 0
            baseline_runs_err.append(abs(baseline.predicted_runs - frame.actual_runs))
            adjusted_runs_err.append(abs(adjusted_runs - frame.actual_runs))
            baseline_wkt_prob.append(baseline.wicket_probability)
            adjusted_wkt_prob.append(adjusted_wkt)
            actual_wkt.append(1 if had_wicket else 0)

            engine.update_actuals(
                actual_runs=frame.actual_runs,
                actual_wickets=frame.actual_wickets,
                bowler_name=bowler_name,
            )
            tracker.record_completed_over(
                bowler=bowler_name, over=over,
                legal_balls=6, runs_conceded=frame.actual_runs,
                wickets=frame.actual_wickets, dots=0, boundaries=0,
            )

print(f"Overs: {n_total} | Announced (spell-tracked) overs: {n_announced} ({n_announced/n_total*100:.1f}%)\n")
print(f"{'':20} | {'Run MAE':>10} | {'Wkt Brier':>10} | {'Wkt AUC':>9}")
print(
    f"{'Baseline (prod)':20} | {mean_absolute_error(actual_wkt, actual_wkt)*0 + sum(baseline_runs_err)/len(baseline_runs_err):10.4f} | "
    f"{brier_score_loss(actual_wkt, baseline_wkt_prob):10.4f} | {roc_auc_score(actual_wkt, baseline_wkt_prob):9.4f}"
)
print(
    f"{'Shadow-adjusted':20} | {sum(adjusted_runs_err)/len(adjusted_runs_err):10.4f} | "
    f"{brier_score_loss(actual_wkt, adjusted_wkt_prob):10.4f} | {roc_auc_score(actual_wkt, adjusted_wkt_prob):9.4f}"
)
