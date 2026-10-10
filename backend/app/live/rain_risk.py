"""Rain risk for a match from the weather forecast (models/rain_risk.json, train_rain_risk.py).

P(no result) and P(interrupted: D/L or no result) from the rain forecast for the match window
(17:00-midnight local). Learned from 4,687 men's T20Is and IPL matches with Open-Meteo's recorded
rain: modest skill (interruption Brier 0.0440 vs 0.0466 for the constant base rate on 2024+), mostly
calibrated; the top risk band was overstated (said 32%, 17% happened, n=42), hence the cap below.
The model was trained on recorded rain but is applied to a forecast, so it understates how
uncertain the forecast is.
"""

from __future__ import annotations

import json
import math

from app.live.prematch_model import ROOT

MODEL_PATH = ROOT / "models" / "rain_risk.json"
CAP_INTERRUPTED = 0.25
CAP_NO_RESULT = 0.15


def rain_risk(day: dict | None) -> dict | None:
    """{"p_no_result", "p_interrupted"} for a weather summary that has the match-window rain fields,
    or None when there is no forecast or no saved model."""
    if not day or "eve_rain_mm" not in day:
        return None
    try:
        saved = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    x = [math.log1p(float(day.get(name) or 0.0)) for name in saved["features"]]

    def prob(model):
        z = model["intercept"] + sum(c * v for c, v in zip(model["coef"], x))
        return 1.0 / (1.0 + math.exp(-z))

    p_nr = min(prob(saved["no_result"]), CAP_NO_RESULT)
    p_int = min(max(prob(saved["interrupted"]), p_nr), CAP_INTERRUPTED)
    return {"p_no_result": p_nr, "p_interrupted": p_int}
