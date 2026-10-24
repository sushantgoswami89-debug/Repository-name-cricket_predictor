"""Over outlook for the live message (user 2026-10-24): mean runs of the next over, chance of a big (10+) and a quiet (4 or fewer) over, and
of at least one wicket. Models: models/over_outlook_v1 (backend/train_over_outlook.py). Fault-isolated: anything missing or failing gives None
and the message simply omits the outlook."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd

DEFAULT_DIR = Path(__file__).resolve().parents[3] / "models" / "over_outlook_v1"


def _phase(over: int) -> str:
    return "powerplay" if over <= 6 else "middle" if over <= 15 else "death"


class OverOutlook:
    def __init__(self, directory: Path = DEFAULT_DIR) -> None:
        manifest = json.loads((directory / "OUTLOOK_MANIFEST.json").read_text(encoding="utf-8"))
        for name, digest in manifest["sha256"].items():
            if hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"over outlook artifact {name} does not match its manifest")
        self._features: list[str] = manifest["features"]
        self.big_runs: int = int(manifest["big_over_runs"])
        self.quiet_runs: int = int(manifest["quiet_over_runs"])
        self._models = {name: joblib.load(directory / f"{name}.pkl") for name in ("mean", "big", "quiet", "wicket")}

    @classmethod
    def try_load(cls, directory: Path = DEFAULT_DIR) -> "OverOutlook | None":
        try:
            return cls(directory)
        except Exception:
            return None

    def predict(self, live) -> dict | None:
        """`live` is the pipeline's LiveMatchState for the over about to be bowled. T20 only (20-over innings)."""
        try:
            if live.balls_remaining > 120 or live.over < 1 or live.over > 20:
                return None
            row = {name: getattr(live, name) for name in self._features if name != "phase"}
            row["over"] = live.over
            row["phase"] = _phase(live.over)
            frame = pd.DataFrame([row])[self._features]
            frame["phase"] = frame["phase"].astype(pd.CategoricalDtype(["death", "middle", "powerplay"]))
            return {
                "mean_runs": max(0.0, float(self._models["mean"].predict(frame)[0])),
                "p_big": float(self._models["big"].predict_proba(frame)[0, 1]),
                "p_quiet": float(self._models["quiet"].predict_proba(frame)[0, 1]),
                "p_wicket": float(self._models["wicket"].predict_proba(frame)[0, 1]),
                "big_runs": self.big_runs,
                "quiet_runs": self.quiet_runs,
            }
        except Exception:
            return None
