"""Shadow version of the over outlook with the batters' career strike rates (models/over_outlook_v2_career, backend/train_over_outlook_career.py).
It is never published: the pipeline logs it next to the live outlook so both can be scored on the same real overs (backend/score_career_shadow.py).
Anything missing or failing gives None."""

from __future__ import annotations

from pathlib import Path

from app.ml.career_strike_rates import CareerStrikeRates
from app.ml.over_outlook import OverOutlook

DEFAULT_DIR = Path(__file__).resolve().parents[3] / "models" / "over_outlook_v2_career"


class CareerOverOutlook(OverOutlook):
    def __init__(self, directory: Path = DEFAULT_DIR, career: CareerStrikeRates | None = None) -> None:
        super().__init__(directory)
        self._career = career if career is not None else CareerStrikeRates()

    @classmethod
    def try_load(cls, directory: Path = DEFAULT_DIR) -> "CareerOverOutlook | None":
        try:
            return cls(directory)
        except Exception:
            return None

    def predict_for_pair(self, live, striker: str | None, partner: str | None) -> dict | None:
        """The outlook for the coming over given the two batters expected at the crease; adds which name-matching tier found each batter."""
        try:
            features, tiers = self._career.features(striker, partner, live.over)
            result = self.predict(live, extra=features)
            if result is not None:
                result["tiers"] = tiers
            return result
        except Exception:
            return None
