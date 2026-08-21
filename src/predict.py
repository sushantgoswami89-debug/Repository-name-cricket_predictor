"""
Given batsman, bowler, over number, match situation, and pitch type -> predicts
next-over outcome and generates the human-readable insight a commentator can
use on air.
"""

import pandas as pd
import numpy as np
import joblib
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(SCRIPT_DIR)
from features import build_features, get_feature_columns
from generate_synthetic_data import BATSMEN, BOWLERS

MODEL_DIR = os.path.join(PROJECT_DIR, "models")
_REAL_DATA_PATH = os.path.join(PROJECT_DIR, "data", "real_overs.csv")
_SYNTHETIC_DATA_PATH = os.path.join(PROJECT_DIR, "data", "synthetic_overs.csv")
DATA_PATH = _REAL_DATA_PATH if os.path.exists(_REAL_DATA_PATH) else _SYNTHETIC_DATA_PATH


def phase_of_over(over_num):
    if over_num <= 6:
        return "powerplay"
    elif over_num <= 15:
        return "middle"
    else:
        return "death"


class NextOverPredictor:
    def __init__(self):
        self.runs_model = joblib.load(f"{MODEL_DIR}/runs_model.pkl")
        self.wkt_model = joblib.load(f"{MODEL_DIR}/wkt_model.pkl")
        self.feature_cols = joblib.load(f"{MODEL_DIR}/feature_cols.pkl")
        self.cat_cols = joblib.load(f"{MODEL_DIR}/cat_cols.pkl")
        # historical data used to compute "as of now" rolling stats
        self.history = pd.read_csv(DATA_PATH)
        data_label = (
            "REAL historical data"
            if DATA_PATH == _REAL_DATA_PATH
            else "SYNTHETIC (test) data"
        )
        print(
            f"[Loaded {data_label} from {os.path.basename(DATA_PATH)} — {len(self.history)} rows]"
        )

    def _historical_stat(self, mask, col, default):
        vals = self.history.loc[mask, col]
        return vals.mean() if len(vals) > 0 else default

    def build_live_features(
        self,
        batsman,
        bowler,
        over_num,
        score_before,
        wkts_down,
        balls_faced_by_batsman,
        pitch_type,
        venue_avg_score=170,
    ):
        global_run_mean = self.history["runs_in_over"].mean()
        global_wkt_mean = self.history["wicket_in_over"].mean()

        bat_style = BATSMEN.get(batsman, {}).get("style", "unknown")
        bat_class = BATSMEN.get(batsman, {}).get("class", "unknown")
        bowl_type = BOWLERS.get(bowler, {}).get("type", "unknown")
        bowl_quality = BOWLERS.get(bowler, {}).get("quality", "unknown")
        phase = phase_of_over(over_num)

        h = self.history
        bat_mask = h["batsman"] == batsman
        bowl_mask = h["bowler"] == bowler
        h2h_mask = bat_mask & bowl_mask
        bat_vs_type_mask = bat_mask & (h["bowler_type"] == bowl_type)
        bowl_phase_mask = bowl_mask & (h["phase"] == phase)

        row = {
            "over": over_num,
            "score_before_over": score_before,
            "wkts_down_before_over": wkts_down,
            "balls_faced_before_over": balls_faced_by_batsman,
            "venue_avg_score": venue_avg_score,
            "bat_career_overs_faced": bat_mask.sum(),
            "bat_hist_avg_runs_per_over": self._historical_stat(
                bat_mask, "runs_in_over", global_run_mean
            ),
            "bat_hist_wicket_rate": self._historical_stat(
                bat_mask, "wicket_in_over", global_wkt_mean
            ),
            "bowl_career_overs_bowled": bowl_mask.sum(),
            "bowl_hist_avg_runs_conceded": self._historical_stat(
                bowl_mask, "runs_in_over", global_run_mean
            ),
            "bowl_hist_wicket_rate": self._historical_stat(
                bowl_mask, "wicket_in_over", global_wkt_mean
            ),
            "h2h_overs": h2h_mask.sum(),
            "h2h_avg_runs": self._historical_stat(
                h2h_mask, "runs_in_over", global_run_mean
            ),
            "bat_vs_bowltype_avg_runs": self._historical_stat(
                bat_vs_type_mask, "runs_in_over", global_run_mean
            ),
            "bat_vs_bowltype_wicket_rate": self._historical_stat(
                bat_vs_type_mask, "wicket_in_over", global_wkt_mean
            ),
            "bowl_phase_avg_runs": self._historical_stat(
                bowl_phase_mask, "runs_in_over", global_run_mean
            ),
            "phase": phase,
            "batsman_style": bat_style,
            "batsman_class": bat_class,
            "bowler_type": bowl_type,
            "bowler_quality": bowl_quality,
            "pitch_type": pitch_type,
        }
        return pd.DataFrame([row])

    def predict(
        self,
        batsman,
        bowler,
        over_num,
        score_before=0,
        wkts_down=0,
        balls_faced_by_batsman=0,
        pitch_type="balanced",
        venue_avg_score=170,
    ):
        X = self.build_live_features(
            batsman,
            bowler,
            over_num,
            score_before,
            wkts_down,
            balls_faced_by_batsman,
            pitch_type,
            venue_avg_score,
        )
        for c in self.cat_cols:
            X[c] = X[c].astype("category")
        X = X[self.feature_cols]

        pred_runs = float(self.runs_model.predict(X)[0])
        pred_wkt_prob = float(self.wkt_model.predict_proba(X)[0][1])

        # rough uncertainty band using model residual scale (~ +/- MAE)
        low = max(0, round(pred_runs - 2.5))
        high = round(pred_runs + 2.5)

        bat_overs_seen = int(X.iloc[0]["bat_career_overs_faced"])
        bowl_overs_seen = int(X.iloc[0]["bowl_career_overs_bowled"])
        LOW_DATA_THRESHOLD = (
            15  # fewer than this many historical overs = shaky personalization
        )
        limited_data = (
            bat_overs_seen < LOW_DATA_THRESHOLD or bowl_overs_seen < LOW_DATA_THRESHOLD
        )

        return {
            "expected_runs": round(pred_runs, 1),
            "expected_range": f"{low}-{high}",
            "wicket_probability": round(pred_wkt_prob * 100, 1),
            "commentary_lines": self._generate_commentary(
                batsman, bowler, over_num, pred_runs, pred_wkt_prob, X.iloc[0]
            ),
            "data_confidence": {
                "limited_data": limited_data,
                "batsman_overs_in_history": bat_overs_seen,
                "bowler_overs_in_history": bowl_overs_seen,
                "note": (
                    (
                        f"Limited history for {'batsman' if bat_overs_seen < LOW_DATA_THRESHOLD else 'bowler'} "
                        f"— treat this as a rough estimate, not a personalized read."
                    )
                    if limited_data
                    else None
                ),
            },
        }

    def _generate_commentary(
        self, batsman, bowler, over_num, pred_runs, pred_wkt_prob, feat_row
    ):
        lines = []
        phase = phase_of_over(over_num)

        bowler_type_label = BOWLERS.get(bowler, {}).get("type", "this bowler")
        if (
            feat_row["bat_vs_bowltype_avg_runs"]
            > feat_row["bat_hist_avg_runs_per_over"] * 1.15
        ):
            lines.append(
                f"{batsman} has scored well against {bowler_type_label} bowling historically "
                f"(~{feat_row['bat_vs_bowltype_avg_runs']:.1f} runs/over vs type average)."
            )
        elif (
            feat_row["bat_vs_bowltype_wicket_rate"]
            > feat_row["bat_hist_wicket_rate"] * 1.2
        ):
            lines.append(
                f"{batsman} has a higher dismissal rate against {bowler_type_label} bowling "
                f"than his overall average — a wicket here wouldn't be a shock."
            )

        if feat_row["bowl_hist_wicket_rate"] > 0.35:
            lines.append(
                f"{bowler} has a strong strike rate this tournament, taking wickets often."
            )

        if feat_row["balls_faced_before_over"] < 12:
            lines.append(
                f"{batsman} is still relatively new at the crease — early dismissal risk is elevated."
            )

        if phase == "death" and pred_runs > 9:
            lines.append(
                "Death overs, and the model expects the batting side to accelerate hard here."
            )

        if pred_wkt_prob > 0.35:
            lines.append(
                f"Wicket probability this over is elevated at {pred_wkt_prob*100:.0f}%."
            )

        if not lines:
            lines.append(
                f"A fairly even contest this over — no strong historical edge either way."
            )

        return lines


if __name__ == "__main__":
    predictor = NextOverPredictor()

    # Example: Kohli facing Bumrah, over 8, 2 wickets down, settled batsman
    result = predictor.predict(
        batsman="V Kohli",
        bowler="J Bumrah",
        over_num=8,
        score_before=62,
        wkts_down=2,
        balls_faced_by_batsman=24,
        pitch_type="seaming_track",
    )
    print("=== Example 1: Kohli vs Bumrah, Over 8, Seaming Track ===")
    print(
        f"Expected runs: {result['expected_runs']} (range: {result['expected_range']})"
    )
    print(f"Wicket probability: {result['wicket_probability']}%")
    print("Commentary insights:")
    for l in result["commentary_lines"]:
        print(f"  - {l}")

    print()
    # Example: new batsman facing spin at death
    result2 = predictor.predict(
        batsman="R Parag",
        bowler="Y Chahal",
        over_num=18,
        score_before=145,
        wkts_down=5,
        balls_faced_by_batsman=6,
        pitch_type="slow_turner",
    )
    print("=== Example 2: Parag (new) vs Chahal, Over 18, Slow Turner ===")
    print(
        f"Expected runs: {result2['expected_runs']} (range: {result2['expected_range']})"
    )
    print(f"Wicket probability: {result2['wicket_probability']}%")
    print("Commentary insights:")
    for l in result2["commentary_lines"]:
        print(f"  - {l}")
