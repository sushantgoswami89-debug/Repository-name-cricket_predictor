"""Live shadow-comparison logging for the match_winner GBM+NN ensemble
(2026-08-26) -- mirrors app/ml/win_probability_shadow_log.py's pattern
(the real outcome is known only at MATCH END, not per-over, same as the
Monte Carlo shadow -- unlike the wicket/run-range ensembles where the
outcome is known one over later). The "5 real matches" threshold for the
review reminder counts distinct MATCHES, consistent with every other
shadow system in this project.

Files (gitignored, same data/live/* convention as every other live
snapshot): match_winner_ensemble_shadow_log.jsonl (per-over: gbm/nn/
ensemble probabilities), match_winner_ensemble_shadow_outcomes.jsonl
(per-match real outcome, idempotent per match_id),
match_winner_ensemble_shadow_reminder_sent.json (fires the review
reminder once per threshold crossing).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_LOG_PATH = Path(__file__).resolve().parents[3] / "data/live/match_winner_ensemble_shadow_log.jsonl"
DEFAULT_OUTCOMES_PATH = Path(__file__).resolve().parents[3] / "data/live/match_winner_ensemble_shadow_outcomes.jsonl"
DEFAULT_MARKER_PATH = Path(__file__).resolve().parents[3] / "data/live/match_winner_ensemble_shadow_reminder_sent.json"
MIN_MATCHES_FOR_SUMMARY = 5


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class MatchWinnerEnsembleShadowLog:
    def __init__(
        self,
        log_path: Path | None = None,
        outcomes_path: Path | None = None,
        marker_path: Path | None = None,
        min_matches_for_summary: int = MIN_MATCHES_FOR_SUMMARY,
    ) -> None:
        self.log_path = log_path or DEFAULT_LOG_PATH
        self.outcomes_path = outcomes_path or DEFAULT_OUTCOMES_PATH
        self.marker_path = marker_path or DEFAULT_MARKER_PATH
        self.min_matches_for_summary = min_matches_for_summary

    def record_prediction(
        self,
        *,
        match_id: str,
        innings: int,
        over: int,
        gbm_probability: float,
        nn_probability: float | None,
        ensemble_probability: float,
    ) -> None:
        row = {
            "match_id": match_id, "innings": innings, "over": over,
            "gbm_probability": gbm_probability, "nn_probability": nn_probability,
            "ensemble_probability": ensemble_probability, "recorded_at": _now_iso(),
        }
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")

    def _recorded_outcome_match_ids(self) -> set[str]:
        if not self.outcomes_path.exists():
            return set()
        ids: set[str] = set()
        for line in self.outcomes_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                ids.add(json.loads(line)["match_id"])
        return ids

    def record_outcome(self, *, match_id: str, batting_team_won: bool) -> None:
        if match_id in self._recorded_outcome_match_ids():
            return
        row = {"match_id": match_id, "batting_team_won": batting_team_won, "recorded_at": _now_iso()}
        self.outcomes_path.parent.mkdir(parents=True, exist_ok=True)
        with self.outcomes_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")

    def matches_with_outcome_count(self) -> int:
        return len(self._recorded_outcome_match_ids())

    def reminder_already_sent(self) -> bool:
        return self.marker_path.exists()

    def mark_reminder_sent(self, *, match_count: int) -> None:
        self.marker_path.parent.mkdir(parents=True, exist_ok=True)
        self.marker_path.write_text(
            json.dumps({"sent_at": _now_iso(), "match_count": match_count}, indent=2), encoding="utf-8")

    def _load_jsonl(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def summarize(self) -> dict[str, Any]:
        import numpy as np
        from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score

        predictions = self._load_jsonl(self.log_path)
        outcomes = {row["match_id"]: bool(row["batting_team_won"]) for row in self._load_jsonl(self.outcomes_path)}
        joined = [row for row in predictions if row["match_id"] in outcomes]
        matches = sorted({row["match_id"] for row in joined})
        if not joined:
            return {"rows": 0, "matches": 0}

        actual = np.array([int(outcomes[row["match_id"]]) for row in joined])

        def _metrics(values: list[float | None]) -> dict[str, float] | None:
            pairs = [(a, v) for a, v in zip(actual, values) if v is not None]
            if len(pairs) < 10 or len({a for a, _ in pairs}) < 2:
                return None
            a = np.array([p[0] for p in pairs])
            p = np.array([p[1] for p in pairs])
            predicted = (p >= 0.5).astype(int)
            return {"rows": len(pairs), "auc": float(roc_auc_score(a, p)),
                    "brier": float(brier_score_loss(a, p)), "accuracy": float(accuracy_score(a, predicted))}

        return {
            "rows": len(joined), "matches": len(matches),
            "gbm": _metrics([row["gbm_probability"] for row in joined]),
            "nn": _metrics([row["nn_probability"] for row in joined]),
            "ensemble": _metrics([row["ensemble_probability"] for row in joined]),
        }

    def summary_text(self) -> str:
        summary = self.summarize()
        if summary["rows"] == 0:
            return "No real live match-winner-ensemble comparisons logged yet."
        lines = [f"Match-winner ensemble: GBM vs NN vs blend, {summary['matches']} real match(es), "
                 f"{summary['rows']} over-level predictions."]
        for label, key in (("GBM (served)", "gbm"), ("NN (shadow)", "nn"), ("Ensemble", "ensemble")):
            m = summary.get(key)
            if m is None:
                lines.append(f"{label}: not enough data yet")
            else:
                lines.append(f"{label}: accuracy {m['accuracy'] * 100:.1f}%, AUC {m['auc']:.3f}, Brier {m['brier']:.3f}")
        return "\n".join(lines)
