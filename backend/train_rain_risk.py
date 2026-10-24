"""Rain risk: how likely a match is interrupted (D/L) or washed out (no result), from the rain
recorded during the match window, learned from every men's T20I and IPL match in Cricsheet.

Ground truth comes from Cricsheet's own outcome records: `method` == "D/L" (a revised target) or
`result` == "no result". Features come from the Open-Meteo weather for the match city and date
(fetch_match_weather.py): rain between 17:00 and midnight local ("eve_rain_mm", hours of rain) and,
for comparison, the whole-day total. Two small logistic models:
    P(no result)              washout, nobody wins
    P(interrupted or no result) any rain effect on the game
Evaluated with a chronological split (fit on matches before 2024, score 2024+); the saved models are
fitted on all data. Usage: python train_rain_risk.py
"""

from __future__ import annotations

import json
import math

import numpy as np
from sklearn.linear_model import LogisticRegression

from app.live.prematch_model import ROOT, city_key

WEATHER = ROOT / "data" / "external" / "weather" / "city_daily.json"
OUT = ROOT / "models" / "rain_risk.json"
FEATURE_SETS = {
    "daily total only": ("rain_mm",),
    "match-window rain": ("eve_rain_mm", "eve_rain_hours"),
    "window + daily": ("eve_rain_mm", "eve_rain_hours", "rain_mm"),
}


def load_rows() -> list[dict]:
    weather = json.loads(WEATHER.read_text(encoding="utf-8"))
    rows = []
    for source in ("t20i", "ipl"):
        for path in sorted((ROOT / "data" / "raw" / "cricsheet" / source).glob("*.json")):
            info = json.loads(path.read_text(encoding="utf-8"))["info"]
            if info.get("gender") not in (None, "male") or len(info.get("teams", [])) != 2:
                continue
            day = weather.get(city_key(info), {}).get(info["dates"][0])
            if not day or "eve_rain_mm" not in day:
                continue
            outcome = info.get("outcome", {})
            no_result = outcome.get("result") == "no result"
            rows.append({
                "date": info["dates"][0], "no_result": no_result,
                "interrupted": no_result or bool(outcome.get("method")), "day": day,
            })
    rows.sort(key=lambda r: r["date"])
    return rows


def features(day: dict, names) -> list[float]:
    return [math.log1p(float(day.get(n) or 0.0)) for n in names]


def fit(rows, names, target):
    x = np.array([features(r["day"], names) for r in rows], float)
    y = np.array([r[target] for r in rows], float)
    model = LogisticRegression(C=1.0, max_iter=2000).fit(x, y)
    return model


def main() -> None:
    rows = load_rows()
    train = [r for r in rows if r["date"] < "2024-01-01"]
    test = [r for r in rows if r["date"] >= "2024-01-01"]
    print(f"matches with weather: {len(rows)} (train {len(train)}, test {len(test)})")
    for target, label in (("no_result", "no result"), ("interrupted", "interrupted (D/L or no result)")):
        y = np.array([r[target] for r in test], float)
        base = np.mean([r[target] for r in train])
        print(f"\n{label}: base rate {base:.1%} in train, {y.mean():.1%} in test; "
              f"Brier of the constant base rate {np.mean((base - y) ** 2):.4f}")
        for name, feats in FEATURE_SETS.items():
            m = fit(train, feats, target)
            p = m.predict_proba(np.array([features(r["day"], feats) for r in test], float))[:, 1]
            print(f"  {name:<20} Brier {np.mean((p - y) ** 2):.4f}")
        if target == "interrupted":
            best = FEATURE_SETS["match-window rain"]
            m = fit(train, best, target)
            p = m.predict_proba(np.array([features(r["day"], best) for r in test], float))[:, 1]
            for lo, hi in ((0, 0.03), (0.03, 0.08), (0.08, 0.15), (0.15, 0.3), (0.3, 1.01)):
                mk = (p >= lo) & (p < hi)
                if mk.sum() >= 8:
                    print(f"    said {lo:.0%}-{min(hi, 1):.0%}: n={int(mk.sum()):<4} said {p[mk].mean():.1%}  actually {y[mk].mean():.1%}")
    names = FEATURE_SETS["match-window rain"]
    saved = {"features": list(names), "trained_on": len(rows), "trained_through": rows[-1]["date"]}
    for target in ("no_result", "interrupted"):
        m = fit(rows, names, target)
        saved[target] = {"coef": [float(c) for c in m.coef_[0]], "intercept": float(m.intercept_[0])}
    OUT.write_text(json.dumps(saved, indent=2))
    print("\nsaved", OUT)


if __name__ == "__main__":
    main()
