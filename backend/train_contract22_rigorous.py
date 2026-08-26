"""Resolves the "does v3 actually beat what's live" gap flagged in
docs/candidate_run_range_enriched_v2.md: models/runs_model.pkl (the
actually-live model) was trained via `train_test_split(..., random_state=42)`
on data/real_overs.csv, which has NO match_date column at all -- a genuine
chronological holdout for it has never existed and (with that data file)
can't be built.

This builds the closest faithful, rigorously-validated equivalent:
the SAME 22-feature contract docs/model_contract.md describes (venue_avg_score,
bat_career_*, bat_hist_*, bowl_career_*, bowl_hist_*, h2h_*,
bat_vs_bowltype_*, bowl_phase_avg_runs, pitch_type, batsman_style,
batsman_class, bowler_type, bowler_quality), computed from DATED data with
a real chronological split (train<=2023 / calib=2024 / holdout>=2025,
same convention as every other candidate this session).

Two things found while inspecting data/real_overs.csv before building this:
1. batsman_style, batsman_class, bowler_type, and bowler_quality are
   "unknown" for ALL 247,071 rows of the live model's own training data --
   constant placeholders contributing zero signal to the trained model.
   Not faithfully reproduced as constants here; instead this candidate
   fills batsman_style and bowler_type with REAL data
   (data/external/cricsheet_player_styles.csv) to test whether real signal
   there helps, mirroring the same finding for run_range_enriched_v3
   (striker_batting_style). batsman_class/bowler_quality left out
   entirely -- there's no available data source for a genuine quality-tier
   classification, and reproducing an always-"unknown" placeholder adds
   nothing.
2. MatchReplay (used by run_full_match_replay.py, fit_confidence_calibrator.py,
   and this session's earlier "31.1%/31.35% naive hit rate" estimates for
   the live model) populates the ACTUAL bowler for the upcoming over during
   replay, since it's replaying known history -- NOT available in genuine
   live prediction, where the next over's bowler is usually unannounced.
   This means those earlier naive estimates were inflated by bowler
   information a real live prediction wouldn't have. This candidate is
   evaluated BOTH ways: with real bowler identity (a ceiling estimate) and
   with bowler deliberately blanked (a live-realistic floor estimate).

pitch_type: not reproduced -- no available ground-surface classification
data source independent of runs already scored there (using scored-runs to
classify the pitch and then predicting runs from that classification
would be circular). venue_par_score/venue_scoring_regime (already in
run_range_enriched_v3) serves the same purpose without that risk.
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from app.ml.ipl_venue_regime_dataset import build_ipl_venue_regime_dataset
from app.ml.ipl_identities import canonical_player_id
from train_phase_calibrated_sharp_range_v33 import (
    MAX_RUN_CLASS, best_bands, male_source_files, nll, temperature_scale,
)

VERSION = "contract22_rigorous"

BASE_FEATURES = [
    "over", "score_before_over", "wkts_down_before_over", "phase",
    "wickets_in_hand", "legal_balls_bowled", "balls_remaining",
    "current_run_rate", "is_chase", "runs_required", "required_run_rate",
    "recent_legal_balls", "recent_runs_per_ball", "recent_dot_rate",
    "recent_single_rate", "recent_boundary_rate", "recent_wicket_rate",
]
BATTER_FEATURES = [
    "bat_career_overs_faced", "bat_hist_avg_runs_per_over", "bat_hist_wicket_rate",
    "striker_batting_style",
]
BOWLER_FEATURES = [
    "bowl_career_overs_bowled", "bowl_hist_avg_runs_conceded", "bowl_hist_wicket_rate",
    "bowl_phase_avg_runs", "bowler_type", "bowler_match_overs_bowled",
    "bowler_spell_over_number", "bowler_is_return_spell",
]
MATCHUP_FEATURES = ["h2h_overs", "h2h_avg_runs"]
H2H_SHRINKAGE_OVERS = 4.0
VENUE_FEATURES = ["venue_par_score", "venue_prior_innings", "venue_scoring_regime", "venue_par_source"]
FEATURES = BASE_FEATURES + BATTER_FEATURES + BOWLER_FEATURES + MATCHUP_FEATURES + VENUE_FEATURES
CATEGORICAL = ["phase", "striker_batting_style", "bowler_type", "venue_scoring_regime", "venue_par_source"]


def _empty() -> dict[str, int]:
    return {"balls": 0, "runs": 0, "wickets": 0}


def _phase(over: int) -> str:
    return "powerplay" if over <= 6 else ("death" if over >= 16 else "middle")


def build_enriched(root: Path, eligible: set[str]) -> pd.DataFrame:
    """Chronological pass building batter/bowler/h2h/phase history with
    strict leave-this-match-out semantics (profiles updated only after the
    whole match is processed), same principle as ipl_phase_moe_dataset.py.
    """
    from app.ml.player_style_registry import normalize_batting_style

    styles = pd.read_csv(root / "data/external/cricsheet_player_styles.csv")
    batting_style_lookup = {
        f"player:{cid}": normalize_batting_style(style)
        for cid, style in zip(styles["cricsheet_id"], styles["batting_style"])
        if isinstance(cid, str) and normalize_batting_style(style) != "unknown"
    }
    def _bowler_type(style: str) -> str:
        if not isinstance(style, str):
            return "__UNKNOWN__"
        lowered = style.lower()
        if "spin" in lowered or "break" in lowered or "orthodox" in lowered or "chinaman" in lowered:
            return "spin"
        if "fast" in lowered or "medium" in lowered or "pace" in lowered:
            return "pace"
        return "__UNKNOWN__"
    bowling_style_lookup = {
        f"player:{cid}": _bowler_type(style)
        for cid, style in zip(styles["cricsheet_id"], styles["bowling_style"])
        if isinstance(cid, str)
    }

    paths: list[tuple[str, Path]] = []
    for scope in ("ipl", "t20i"):
        for path in (root / "data/raw/cricsheet" / scope).glob("*.json"):
            if path.name not in eligible:
                continue
            raw = json.loads(path.read_text(encoding="utf-8"))
            paths.append((str(raw["info"]["dates"][0]), path))
    paths.sort(key=lambda item: (item[0], item[1].name))
    print(f"Building enriched contract-22 features from {len(paths)} matches...")

    batter_history: dict[str, dict[str, int]] = {}
    bowler_history: dict[str, dict[str, int]] = {}
    bowler_phase_history: dict[tuple[str, str], dict[str, int]] = {}
    h2h_history: dict[tuple[str, str], dict[str, int]] = {}
    rows: list[dict] = []

    for i, (match_date, path) in enumerate(paths):
        if i % 500 == 0:
            print(f"  ... {i}/{len(paths)}", flush=True)
        raw = json.loads(path.read_text(encoding="utf-8"))
        info = raw["info"]
        registry = info.get("registry", {}).get("people", {})
        regular = [inn for inn in raw.get("innings", []) if not inn.get("super_over")]
        match_batter_events: list[tuple[str, dict[str, int]]] = []
        match_bowler_events: list[tuple[str, dict[str, int]]] = []
        match_bowler_phase_events: list[tuple[tuple[str, str], dict[str, int]]] = []
        match_h2h_events: list[tuple[tuple[str, str], dict[str, int]]] = []

        for innings_number, innings in enumerate(regular, start=1):
            legal_balls = 0
            score = wkts = 0
            bowler_match_balls: dict[str, int] = {}
            # Spell tracking: is the upcoming over a continuation of the
            # bowler's active spell, or a return after being taken off?
            # See WicketContract22FeatureComputer._partnership_state for
            # the same logic on the live side.
            current_spell_bowler = ""
            current_spell_length = 0
            bowler_spell_count: dict[str, int] = {}
            for over in innings.get("overs", []):
                over_number = int(over["over"]) + 1
                deliveries = over.get("deliveries", [])
                if not deliveries:
                    continue
                phase = _phase(over_number)
                striker = canonical_player_id(str(deliveries[0].get("batter") or ""), registry)
                bowler = canonical_player_id(str(deliveries[0].get("bowler") or ""), registry)

                bp = batter_history.get(striker, _empty())
                bwp = bowler_history.get(bowler, _empty())
                bph = bowler_phase_history.get((bowler, phase), _empty())
                h2h = h2h_history.get((striker, bowler), _empty())
                bowler_match_overs_bowled = bowler_match_balls.get(bowler, 0) / 6.0
                if bowler == current_spell_bowler:
                    bowler_spell_over_number = current_spell_length + 1
                    bowler_is_return_spell = int(bowler_spell_count.get(bowler, 0) > 1)
                else:
                    bowler_spell_over_number = 1
                    bowler_is_return_spell = int(bowler_spell_count.get(bowler, 0) >= 1)

                over_runs = 0
                over_wkts = 0
                for delivery in deliveries:
                    total_runs = int(delivery.get("runs", {}).get("total", 0))
                    batter_runs = int(delivery.get("runs", {}).get("batter", 0))
                    over_runs += total_runs
                    is_wicket = any(
                        d.get("kind") not in {"retired hurt", "obstructing the field"}
                        for d in delivery.get("wickets", [])
                    )
                    over_wkts += int(is_wicket)
                    extras = delivery.get("extras", {}) or {}
                    is_legal = "wides" not in extras and "noballs" not in extras
                    batter_now = canonical_player_id(str(delivery.get("batter") or ""), registry)
                    bowler_now = canonical_player_id(str(delivery.get("bowler") or ""), registry)
                    if is_legal:
                        legal_balls += 1
                        match_batter_events.append((batter_now, {"balls": 1, "runs": batter_runs, "wickets": 0}))
                        match_bowler_events.append((bowler_now, {"balls": 1, "runs": total_runs, "wickets": int(is_wicket)}))
                        match_bowler_phase_events.append(((bowler_now, phase), {"balls": 1, "runs": total_runs, "wickets": int(is_wicket)}))
                        match_h2h_events.append(((batter_now, bowler_now), {"balls": 1, "runs": total_runs, "wickets": int(is_wicket)}))
                        bowler_match_balls[bowler_now] = bowler_match_balls.get(bowler_now, 0) + 1
                score += over_runs
                wkts += over_wkts

                bp_overs = bp["balls"] / 6.0
                bwp_overs = bwp["balls"] / 6.0
                bowl_hist_avg_runs_conceded = (bwp["runs"] / bwp_overs) if bwp_overs > 0 else 0.0
                h2h_overs = h2h["balls"] / 6.0
                h2h_raw_avg = (h2h["runs"] / h2h_overs) if h2h_overs > 0 else bowl_hist_avg_runs_conceded
                # Same shrinkage as WicketContract22FeatureComputer._h2h --
                # most pairs have well under an over of shared history, so a
                # raw small-sample average is mostly noise. Shrink toward
                # the bowler's own overall average, weighted by real overs
                # of head-to-head (H2H_SHRINKAGE_OVERS = pseudo-count).
                h2h_avg_runs = (
                    h2h_overs * h2h_raw_avg + H2H_SHRINKAGE_OVERS * bowl_hist_avg_runs_conceded
                ) / (h2h_overs + H2H_SHRINKAGE_OVERS)
                # User-requested (2026-08-25): this bowler's own history of
                # dismissing THIS specific batter. h2h["wickets"] was
                # already tracked (see match_h2h_events below) but never
                # surfaced as a feature -- only the runs-based h2h_avg_runs
                # was. Same shrinkage convention as h2h_avg_runs: raw H2H
                # wicket rate is mostly noise below ~4 overs of shared
                # history, so shrink toward the bowler's own overall
                # wicket rate, weighted by real shared overs.
                bowl_hist_wicket_rate = (bwp["wickets"] / bwp["balls"]) if bwp["balls"] > 0 else 0.0
                h2h_raw_wicket_rate = (h2h["wickets"] / h2h["balls"]) if h2h["balls"] > 0 else bowl_hist_wicket_rate
                h2h_wicket_rate = (
                    h2h["balls"] * h2h_raw_wicket_rate + (H2H_SHRINKAGE_OVERS * 6) * bowl_hist_wicket_rate
                ) / (h2h["balls"] + H2H_SHRINKAGE_OVERS * 6)
                rows.append({
                    "source_file": path.name, "innings": innings_number, "over": over_number,
                    "bat_career_overs_faced": bp_overs,
                    "bat_hist_avg_runs_per_over": (bp["runs"] / bp_overs) if bp_overs > 0 else 0.0,
                    "bat_hist_wicket_rate": (bp["wickets"] / bp["balls"]) if bp["balls"] > 0 else 0.0,
                    "striker_batting_style": batting_style_lookup.get(striker, "__UNKNOWN__"),
                    "bowl_career_overs_bowled": bwp_overs,
                    "bowl_hist_avg_runs_conceded": bowl_hist_avg_runs_conceded,
                    "bowl_hist_wicket_rate": (bwp["wickets"] / bwp["balls"]) if bwp["balls"] > 0 else 0.0,
                    "bowl_phase_avg_runs": (bph["runs"] / (bph["balls"] / 6.0)) if bph["balls"] > 0 else 0.0,
                    "bowler_match_overs_bowled": bowler_match_overs_bowled,
                    "bowler_spell_over_number": bowler_spell_over_number,
                    "bowler_is_return_spell": bowler_is_return_spell,
                    "bowler_type": bowling_style_lookup.get(bowler, "__UNKNOWN__"),
                    "h2h_overs": h2h_overs,
                    "h2h_avg_runs": h2h_avg_runs,
                    "h2h_wicket_rate": h2h_wicket_rate,
                    # Bowler identity for the live-realistic evaluation variant
                    "actual_bowler_id": bowler,
                })
                if bowler == current_spell_bowler:
                    current_spell_length += 1
                else:
                    current_spell_bowler = bowler
                    current_spell_length = 1
                    bowler_spell_count[bowler] = bowler_spell_count.get(bowler, 0) + 1

        for name, event in match_batter_events:
            profile = batter_history.setdefault(name, _empty())
            for f, v in event.items():
                profile[f] += v
        for name, event in match_bowler_events:
            profile = bowler_history.setdefault(name, _empty())
            for f, v in event.items():
                profile[f] += v
        for key, event in match_bowler_phase_events:
            profile = bowler_phase_history.setdefault(key, _empty())
            for f, v in event.items():
                profile[f] += v
        for key, event in match_h2h_events:
            profile = h2h_history.setdefault(key, _empty())
            for f, v in event.items():
                profile[f] += v

    return pd.DataFrame(rows)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "models/candidates" / VERSION
    output.mkdir(parents=True, exist_ok=True)

    eligible = male_source_files(root)
    base = pd.read_csv(root / "data/candidates/v3/verified_training_overs.csv")
    base = base[base["source_file"].isin(eligible)].copy()
    base["match_date"] = pd.to_datetime(base["match_date"])

    enriched = build_enriched(root, eligible)
    venue = build_ipl_venue_regime_dataset(root, scopes=("ipl", "t20i"))
    keys = ["source_file", "innings", "over"]
    data = base.merge(enriched, on=keys, how="inner", validate="one_to_one")
    data = data.merge(
        venue[keys + ["venue_par_score", "venue_prior_innings", "venue_scoring_regime", "venue_par_source"]],
        on=keys, how="inner", validate="one_to_one",
    )
    print(f"Merged rows: {len(data)} (base was {len(base)})")

    train = data[data["match_date"] <= "2023-12-31"].reset_index(drop=True)
    calibration = data[data["match_date"].dt.year == 2024].reset_index(drop=True)
    holdout = data[data["match_date"] >= "2025-01-01"].reset_index(drop=True)

    def frame(d: pd.DataFrame, bowler_known: bool) -> pd.DataFrame:
        result = d[FEATURES].copy()
        if not bowler_known:
            for col in BOWLER_FEATURES + MATCHUP_FEATURES:
                if col in CATEGORICAL:
                    result[col] = "__UNKNOWN__"
                else:
                    result[col] = 0.0
        for column in CATEGORICAL:
            result[column] = result[column].fillna("__UNKNOWN__").astype("category")
        return result

    model = lgb.LGBMClassifier(
        objective="multiclass", num_class=MAX_RUN_CLASS + 1,
        n_estimators=60, learning_rate=0.08, num_leaves=25, max_depth=6,
        min_child_samples=100, subsample=0.85, colsample_bytree=0.9,
        reg_lambda=1.0, random_state=42, verbose=-1,
    )
    train_actual = np.minimum(train["runs_in_over"].to_numpy(), MAX_RUN_CLASS)
    model.fit(frame(train, bowler_known=True), train_actual, categorical_feature=CATEGORICAL)

    def evaluate(bowler_known: bool) -> dict:
        calibration_raw = model.predict_proba(frame(calibration, bowler_known))
        holdout_raw = model.predict_proba(frame(holdout, bowler_known))
        calibration_actual = np.minimum(calibration["runs_in_over"].to_numpy(), MAX_RUN_CLASS)
        holdout_actual = np.minimum(holdout["runs_in_over"].to_numpy(), MAX_RUN_CLASS)

        temperatures: dict[str, float] = {}
        calibration_scaled = calibration_raw.copy()
        holdout_scaled = holdout_raw.copy()
        for phase in ("powerplay", "middle", "death"):
            cm = calibration["phase"].astype(str).to_numpy() == phase
            hm = holdout["phase"].astype(str).to_numpy() == phase
            result = minimize_scalar(
                lambda v: nll(temperature_scale(calibration_raw[cm], v), calibration_actual[cm]),
                bounds=(0.5, 3.0), method="bounded",
            )
            temperatures[phase] = float(result.x)
            calibration_scaled[cm] = temperature_scale(calibration_raw[cm], temperatures[phase])
            holdout_scaled[hm] = temperature_scale(holdout_raw[hm], temperatures[phase])

        _, _ = best_bands(calibration_scaled)
        holdout_low, holdout_high = best_bands(holdout_scaled)
        holdout_hit = float(np.mean((holdout_actual >= holdout_low) & (holdout_actual <= holdout_high)))
        return {"phase_temperatures": temperatures, "holdout_hit_rate": holdout_hit}

    ceiling = evaluate(bowler_known=True)
    realistic = evaluate(bowler_known=False)

    report = {
        "candidate_version": VERSION,
        "candidate_only": True,
        "production_changed": False,
        "note": (
            "Full 22-feature-contract-equivalent (docs/model_contract.md), "
            "computed from DATED IPL+T20I data with a real chronological "
            "split (train<=2023/calib=2024/holdout>=2025), unlike the live "
            "model's own random-split training on undated data. Evaluated "
            "two ways: bowler_known_ceiling (real bowler identity available, "
            "matching what MatchReplay-based estimates like this session's "
            "earlier 31.1%/31.35% numbers implicitly assumed) and "
            "bowler_unknown_realistic (bowler-dependent features blanked, "
            "matching genuine live pre-over conditions where the next "
            "over's bowler is usually unannounced)."
        ),
        "split": {"train_rows": len(train), "calibration_rows": len(calibration), "holdout_rows": len(holdout)},
        "v32_baseline_hit_rate": 0.2728246539222149,
        "v33_accepted_hit_rate": 0.28103713469567126,
        "run_range_enriched_v3_hit_rate": 0.28493293755496923,
        "bowler_known_ceiling": ceiling,
        "bowler_unknown_realistic": realistic,
    }
    (output / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
