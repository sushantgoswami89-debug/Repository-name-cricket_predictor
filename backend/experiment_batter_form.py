"""Does the batters' form in THIS innings (their runs and balls so far -- the strike rate) help the run-per-over forecast, and does the mean
(L2) rather than the median (L1) objective remove the 'flat and low' forecasts? (user 2026-10-11 and 2026-10-24: "two batters on strike rate
200 and we predict 5-7"; "we were flat in prediction").

Adds to the verified v3 over table, per over: the striker's (first ball of the over) and the non-striker's runs and balls in the innings so
far (all before the over, no leakage) and their shrunk strike rates. Fits the v3 recipe (train <= 2023, holdout 2025+) in four variants:
baseline L1 (what the live GBM is), baseline L2, + batter form L1, + batter form L2. Reports MAE, RMSE, bias, the spread of the predictions
(std) against the spread of the actual overs, and the correlation. Usage: python experiment_batter_form.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

from train_validate_candidate_v3 import V3_FEATURES, CATEGORICAL_FEATURES, _features
from train_lock_ranked_candidate_v3 import TOP_10

root = Path(__file__).resolve().parents[1]
D = root / "data/candidates/v3"
FORM_PATH = D / "batter_form_overs.csv"
PRIOR_RPB, PRIOR_BALLS = 1.25, 12.0   # shrink a batter's in-innings strike rate toward a typical T20 rate (runs per ball)


def build_form() -> pd.DataFrame:
    rows = []
    for scope in ("ipl", "t20i"):
        for path in sorted((root / "data/raw/cricsheet" / scope).glob("*.json")):
            raw = json.loads(path.read_text(encoding="utf-8"))
            # every file: the verified table (merged on below) decides which matches are used (it holds IT20 internationals too)
            for index, innings in enumerate(raw.get("innings", []), start=1):
                runs: dict[str, int] = {}
                balls: dict[str, int] = {}
                for source_over in innings.get("overs", []):
                    deliveries = source_over.get("deliveries", [])
                    if not deliveries:
                        continue
                    first = deliveries[0]
                    striker, partner = first.get("batter"), first.get("non_striker")
                    rows.append({
                        "source_file": path.name, "innings": index, "over": int(source_over["over"]) + 1,
                        "striker_runs": runs.get(striker, 0), "striker_balls": balls.get(striker, 0),
                        "partner_runs": runs.get(partner, 0), "partner_balls": balls.get(partner, 0),
                    })
                    for d in deliveries:
                        batter = d.get("batter")
                        extras = d.get("extras", {})
                        runs[batter] = runs.get(batter, 0) + int(d["runs"]["batter"])
                        if "wides" not in extras:
                            balls[batter] = balls.get(batter, 0) + 1
    return pd.DataFrame(rows)


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for who in ("striker", "partner"):
        frame[f"{who}_sr"] = (frame[f"{who}_runs"] + PRIOR_RPB * PRIOR_BALLS) / (frame[f"{who}_balls"] + PRIOR_BALLS)
    frame["pair_sr"] = (frame["striker_sr"] + frame["partner_sr"]) / 2
    frame["pair_balls"] = frame["striker_balls"] + frame["partner_balls"]
    return frame


FORM = ["striker_runs", "striker_balls", "partner_runs", "partner_balls", "striker_sr", "partner_sr", "pair_sr", "pair_balls"]


def model(objective: str) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(objective=objective, n_estimators=600, learning_rate=0.025, num_leaves=31, max_depth=7,
                             min_child_samples=80, subsample=0.85, colsample_bytree=0.9, reg_lambda=1.0, random_state=42, verbose=-1)


def main() -> None:
    if not FORM_PATH.exists():
        print("building batter form table ...", flush=True)
        build_form().to_csv(FORM_PATH, index=False)
    base = pd.read_csv(D / "verified_training_overs.csv")
    form = pd.read_csv(FORM_PATH)
    frame = add_features(base.merge(form, on=["source_file", "innings", "over"], how="inner"))
    frame["match_date"] = pd.to_datetime(frame["match_date"])
    print("rows", len(base), "with form", len(frame), flush=True)
    t20i = {p.name for p in (root / "data/raw/cricsheet/t20i").glob("*.json")}
    teams = {p.name: tuple(json.loads(p.read_text())["info"].get("teams", ())) for p in (root / "data/raw/cricsheet/t20i").glob("*.json")}
    top10 = {n for n, t in teams.items() if len(t) == 2 and set(t) <= set(TOP_10)}
    train = frame[frame["match_date"].dt.year <= 2023]
    hold = frame[frame["match_date"].dt.year >= 2025]
    subsets = {"all 2025+": hold, "T20I 2025+": hold[hold["source_file"].isin(t20i)], "T20I top-10 2025+": hold[hold["source_file"].isin(top10)]}
    variants = {
        "baseline L1 (live GBM)": (V3_FEATURES, "regression_l1"),
        "baseline L2 (mean)": (V3_FEATURES, "regression"),
        "+ batter form L1": (V3_FEATURES + FORM, "regression_l1"),
        "+ batter form L2 (mean)": (V3_FEATURES + FORM, "regression"),
    }
    results = {}
    for name, (columns, objective) in variants.items():
        m = model(objective)
        m.fit(_features(train, columns), train["runs_in_over"], categorical_feature=CATEGORICAL_FEATURES)
        results[name] = {}
        for subset, part in subsets.items():
            pred = np.clip(m.predict(_features(part, columns)), 0, None)
            actual = part["runs_in_over"].to_numpy()
            # hit rate of the 3-run band centred on the rounded prediction (what a "window" forecast around the point would score)
            centre = np.round(pred)
            band = float(np.mean((actual >= centre - 1) & (actual <= centre + 1)))
            results[name][subset] = {
                "rows": int(len(part)), "mae": round(float(np.mean(np.abs(pred - actual))), 3),
                "rmse": round(float(np.sqrt(np.mean((pred - actual) ** 2))), 3), "bias": round(float(np.mean(pred - actual)), 3),
                "pred_std": round(float(pred.std()), 3), "actual_std": round(float(actual.std()), 3),
                "corr": round(float(np.corrcoef(pred, actual)[0, 1]), 3), "band3_hit": round(band, 3),
            }
    (D / "batter_form_experiment.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    for subset in subsets:
        print(f"\n{subset}")
        print(f"{'variant':30} {'MAE':>6} {'RMSE':>6} {'bias':>6} {'pred sd':>8} {'act sd':>7} {'corr':>6} {'band3':>6}")
        for name in variants:
            r = results[name][subset]
            print(f"{name:30} {r['mae']:6.3f} {r['rmse']:6.3f} {r['bias']:6.3f} {r['pred_std']:8.3f} {r['actual_std']:7.3f} {r['corr']:6.3f} {r['band3_hit']:6.3f}")


if __name__ == "__main__":
    main()
