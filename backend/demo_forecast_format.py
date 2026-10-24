"""Demo of the proposed over forecast: the MEAN (L2) plus the chance of a big / small over and of a wicket, on the real New Zealand v India
1st T20I (2026-10-22) overs from our own commentary log. Trained on T20I+IPL overs up to 2024. Usage: python demo_forecast_format.py [match_id]"""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from train_validate_candidate_v3 import V3_FEATURES, CATEGORICAL_FEATURES, _features

root = Path(__file__).resolve().parents[1]
MATCH = sys.argv[1] if len(sys.argv) > 1 else "161190"


def match_rows(match_id: str) -> pd.DataFrame:
    balls = {}
    for line in (root / "data/live" / f"commentary-{match_id}.jsonl").read_text().splitlines():
        r = json.loads(line)
        balls[(r["innings"], r["over"], r["ball"])] = r
    rows, first_total = [], None
    for innings in (1, 2):
        keys = sorted(k for k in balls if k[0] == innings)
        score = wickets = legal = 0
        recent = []
        for over in sorted({k[1] for k in keys}):
            over_balls = [balls[k] for k in keys if k[1] == over]
            n = len(recent)
            target = first_total + 1 if innings == 2 and first_total is not None else 0
            left = max(0, 120 - legal)
            req = max(0, target - score) if target else 0
            rows.append({
                "innings": innings, "over": over, "phase": "powerplay" if over <= 6 else "middle" if over <= 15 else "death",
                "score_before_over": score, "wkts_down_before_over": wickets, "wickets_in_hand": 10 - wickets, "legal_balls_bowled": legal,
                "balls_remaining": left, "current_run_rate": score * 6 / legal if legal else 0.0, "is_chase": int(target > 0),
                "runs_required": req, "required_run_rate": req * 6 / left if target and left else 0.0, "recent_legal_balls": n,
                "recent_runs_per_ball": sum(x[0] for x in recent) / n if n else 0.0, "recent_dot_rate": sum(x[1] for x in recent) / n if n else 0.0,
                "recent_single_rate": sum(x[2] for x in recent) / n if n else 0.0, "recent_boundary_rate": sum(x[3] for x in recent) / n if n else 0.0,
                "recent_wicket_rate": sum(x[4] for x in recent) / n if n else 0.0,
                "actual": sum(b["total_runs"] for b in over_balls), "actual_wicket": int(any(b["wicket_kind"] for b in over_balls)),
            })
            for b in over_balls:
                score += b["total_runs"]
                wickets += int(bool(b["wicket_kind"]))
                ex = b["extras"] or {}
                if "wides" not in ex and "noballs" not in ex:
                    legal += 1
                    recent.append((b["total_runs"], int(b["total_runs"] == 0), int(b["total_runs"] == 1), int(b["batter_runs"] in (4, 6)), int(bool(b["wicket_kind"]))))
                    recent = recent[-12:]
        if innings == 1:
            first_total = score
    return pd.DataFrame(rows)


def main() -> None:
    base = pd.read_csv(root / "data/candidates/v3/verified_training_overs.csv")
    base["match_date"] = pd.to_datetime(base["match_date"])
    train = base[base["match_date"].dt.year <= 2024]
    X = _features(train, V3_FEATURES)
    kw = dict(n_estimators=400, learning_rate=0.03, num_leaves=31, max_depth=7, min_child_samples=80, subsample=0.85, colsample_bytree=0.9, random_state=42, verbose=-1)
    mean = lgb.LGBMRegressor(objective="regression", **kw).fit(X, train["runs_in_over"], categorical_feature=CATEGORICAL_FEATURES)
    big = lgb.LGBMClassifier(**kw).fit(X, (train["runs_in_over"] >= 10).astype(int), categorical_feature=CATEGORICAL_FEATURES)
    small = lgb.LGBMClassifier(**kw).fit(X, (train["runs_in_over"] <= 4).astype(int), categorical_feature=CATEGORICAL_FEATURES)
    wick = lgb.LGBMClassifier(**kw).fit(X, train["wicket_in_over"], categorical_feature=CATEGORICAL_FEATURES)
    rows = match_rows(MATCH)
    F = _features(rows, V3_FEATURES)
    rows["mean"], rows["p_big"], rows["p_small"], rows["p_wkt"] = mean.predict(F), big.predict_proba(F)[:, 1], small.predict_proba(F)[:, 1], wick.predict_proba(F)[:, 1]
    print(f"Match {MATCH}: {len(rows)} overs. Model trained on overs up to 2024.\n")
    for innings, over in [(1, 1), (1, 6), (1, 10), (1, 16), (1, 19), (1, 20), (2, 4), (2, 9), (2, 15), (2, 19)]:
        r = rows[(rows.innings == innings) & (rows.over == over)]
        if r.empty:
            continue
        r = r.iloc[0]
        chase = f" | need {int(r.runs_required)} off {int(r.balls_remaining)}" if r.is_chase else ""
        print(f"🏏 Over {over} Prediction  (innings {innings}: {int(r.score_before_over)}/{int(r.wkts_down_before_over)}{chase})")
        print(f"   Expected: about {r['mean']:.1f} runs")
        print(f"   Big over (10+ runs): {100*r.p_big:.0f}%   |   Quiet over (4 or fewer): {100*r.p_small:.0f}%")
        print(f"   Wicket chance: {100*r.p_wkt:.0f}%")
        print(f"   [actual: {int(r.actual)} runs{', wicket' if r.actual_wicket else ''}]\n")
    s = rows
    print(f"Whole match: mean forecast {s['mean'].mean():.1f} vs actual {s['actual'].mean():.1f} per over; predicted mean ranged {s['mean'].min():.1f} to {s['mean'].max():.1f}")
    print(f"Big-over calls: when P(10+) >= 40%: {int((s.p_big>=.4).sum())} overs, {int(((s.p_big>=.4)&(s.actual>=10)).sum())} were 10+ | when P(10+) < 25%: {int((s.p_big<.25).sum())} overs, {int(((s.p_big<.25)&(s.actual>=10)).sum())} were 10+")


if __name__ == "__main__":
    main()
