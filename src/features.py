"""
Feature engineering for the next-over prediction models.

Core idea: for every over in the dataset, compute features using ONLY prior
information (no leakage from the over we're trying to predict):
  - Batsman's historical form vs this bowler type, in this phase
  - Bowler's historical economy/wicket rate in this phase, on this pitch type
  - Match situation: score, wickets down, balls faced by current batsman
  - Pitch category

This mirrors what you'd compute live: "here's what we know walking into this
over" -> predict runs & wicket probability for it.
"""

import pandas as pd
import numpy as np

CATEGORICAL_COLS = [
    "phase",
    "batsman_style",
    "batsman_class",
    "bowler_type",
    "bowler_quality",
    "pitch_type",
]


def build_features(
    df: pd.DataFrame,
    order_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Build leakage-safe history in the supplied chronological row order."""

    df = df.sort_values(order_columns or ["match_id", "over"]).reset_index(drop=True)

    # --- Historical (expanding, shifted) stats per batsman ---
    df["bat_career_overs_faced"] = df.groupby("batsman").cumcount()
    df["bat_hist_avg_runs_per_over"] = (
        df.groupby("batsman")["runs_in_over"]
        .apply(lambda s: s.shift().expanding().mean())
        .reset_index(level=0, drop=True)
    )
    df["bat_hist_wicket_rate"] = (
        df.groupby("batsman")["wicket_in_over"]
        .apply(lambda s: s.shift().expanding().mean())
        .reset_index(level=0, drop=True)
    )

    # --- Historical stats per bowler ---
    df["bowl_career_overs_bowled"] = df.groupby("bowler").cumcount()
    df["bowl_hist_avg_runs_conceded"] = (
        df.groupby("bowler")["runs_in_over"]
        .apply(lambda s: s.shift().expanding().mean())
        .reset_index(level=0, drop=True)
    )
    df["bowl_hist_wicket_rate"] = (
        df.groupby("bowler")["wicket_in_over"]
        .apply(lambda s: s.shift().expanding().mean())
        .reset_index(level=0, drop=True)
    )

    # --- Head-to-head: batsman vs this specific bowler (sparse early on) ---
    df["h2h_overs"] = df.groupby(["batsman", "bowler"]).cumcount()
    df["h2h_avg_runs"] = (
        df.groupby(["batsman", "bowler"])["runs_in_over"]
        .apply(lambda s: s.shift().expanding().mean())
        .reset_index(level=[0, 1], drop=True)
    )

    # --- Batsman vs bowler TYPE (pace/spin) - more reliable than exact h2h ---
    df["bat_vs_bowltype_avg_runs"] = (
        df.groupby(["batsman", "bowler_type"])["runs_in_over"]
        .apply(lambda s: s.shift().expanding().mean())
        .reset_index(level=[0, 1], drop=True)
    )
    df["bat_vs_bowltype_wicket_rate"] = (
        df.groupby(["batsman", "bowler_type"])["wicket_in_over"]
        .apply(lambda s: s.shift().expanding().mean())
        .reset_index(level=[0, 1], drop=True)
    )

    # --- Bowler in this phase ---
    df["bowl_phase_avg_runs"] = (
        df.groupby(["bowler", "phase"])["runs_in_over"]
        .apply(lambda s: s.shift().expanding().mean())
        .reset_index(level=[0, 1], drop=True)
    )

    # --- Fill NaNs (first-ever appearance of a player) with global priors ---
    global_run_mean = df["runs_in_over"].mean()
    global_wkt_mean = df["wicket_in_over"].mean()

    fill_map = {
        "bat_hist_avg_runs_per_over": global_run_mean,
        "bat_hist_wicket_rate": global_wkt_mean,
        "bowl_hist_avg_runs_conceded": global_run_mean,
        "bowl_hist_wicket_rate": global_wkt_mean,
        "h2h_avg_runs": global_run_mean,
        "bat_vs_bowltype_avg_runs": global_run_mean,
        "bat_vs_bowltype_wicket_rate": global_wkt_mean,
        "bowl_phase_avg_runs": global_run_mean,
    }
    df = df.fillna(fill_map)

    return df


def get_feature_columns():
    numeric = [
        "over",
        "score_before_over",
        "wkts_down_before_over",
        "balls_faced_before_over",
        "venue_avg_score",
        "bat_career_overs_faced",
        "bat_hist_avg_runs_per_over",
        "bat_hist_wicket_rate",
        "bowl_career_overs_bowled",
        "bowl_hist_avg_runs_conceded",
        "bowl_hist_wicket_rate",
        "h2h_overs",
        "h2h_avg_runs",
        "bat_vs_bowltype_avg_runs",
        "bat_vs_bowltype_wicket_rate",
        "bowl_phase_avg_runs",
    ]
    return numeric, CATEGORICAL_COLS
