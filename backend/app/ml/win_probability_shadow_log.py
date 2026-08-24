"""Live shadow-comparison logging for Monte Carlo vs GBM win-probability
on IPL chases (2026-08-24, user-requested).

The 2025+ holdout comparison (docs/finding_monte_carlo_win_probability_ipl_chase_win.md)
showed Monte Carlo beating the GBM on IPL chases, but the user explicitly
does NOT want that holdout result to decide production routing by itself:
"dont route now, it should check when match is live and after 5 or 10
matches it should show me summary and then remind me to pick one." So
`MatchWinnerEngine` keeps serving the GBM (unchanged from before this
whole investigation) and runs Monte Carlo alongside as a logged shadow,
purely for IPL chases. This module is the persistent record of that
comparison: one append-only JSONL row per real IPL-chase over prediction
(both models' numbers), and one row per completed chase with the real
outcome, so a later session can compute real accuracy for both models
over actual live matches -- not just simulate/assume it.

Files (gitignored, same convention as every other `data/live/*` live
snapshot in this project):
  - `data/live/win_probability_shadow_log.jsonl` -- per-over predictions
  - `data/live/win_probability_shadow_outcomes.jsonl` -- per-match real
    outcomes (idempotent: `record_outcome` skips a `match_id` it has
    already recorded, so a duplicate completion signal from the live
    feed can't double-count a match)
  - `data/live/win_probability_shadow_reminder_sent.json` -- marker so
    the "time to review" reminder fires once per threshold crossing, not
    on every subsequent match
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_LOG_PATH = Path(__file__).resolve().parents[3] / "data/live/win_probability_shadow_log.jsonl"
DEFAULT_OUTCOMES_PATH = Path(__file__).resolve().parents[3] / "data/live/win_probability_shadow_outcomes.jsonl"
DEFAULT_MARKER_PATH = Path(__file__).resolve().parents[3] / "data/live/win_probability_shadow_reminder_sent.json"

# User said "5 or 10" -- picked the lower end so the first check-in
# arrives sooner; easy to raise later if 5 real matches turns out too
# noisy a sample to act on.
MIN_MATCHES_FOR_SUMMARY = 5


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class WinProbabilityShadowLog:
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
        batting_team: str,
        bowling_team: str,
        monte_carlo_win_probability: float,
        gbm_win_probability: float,
    ) -> None:
        row = {
            "match_id": match_id,
            "innings": innings,
            "over": over,
            "batting_team": batting_team,
            "bowling_team": bowling_team,
            "monte_carlo_win_probability": monte_carlo_win_probability,
            "gbm_win_probability": gbm_win_probability,
            "recorded_at": _now_iso(),
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
        row = {
            "match_id": match_id,
            "batting_team_won": batting_team_won,
            "recorded_at": _now_iso(),
        }
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
            json.dumps({"sent_at": _now_iso(), "match_count": match_count}, indent=2),
            encoding="utf-8",
        )

    def _load_jsonl(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def summarize(self) -> dict[str, Any]:
        """Join per-over predictions to their match's real outcome and
        report real AUC/Brier/accuracy for both models over every logged
        IPL chase that has a known real result. Returns
        `{"rows": 0, ...}` gracefully if nothing's logged yet -- callers
        should check `matches_with_outcome_count()` before treating this
        as a meaningful comparison."""

        import numpy as np
        from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score

        predictions = self._load_jsonl(self.log_path)
        outcomes = {row["match_id"]: bool(row["batting_team_won"]) for row in self._load_jsonl(self.outcomes_path)}

        joined = [row for row in predictions if row["match_id"] in outcomes]
        matches = sorted({row["match_id"] for row in joined})
        if not joined:
            return {"rows": 0, "matches": 0, "matches_list": []}

        actual = np.array([int(outcomes[row["match_id"]]) for row in joined])
        mc = np.array([float(row["monte_carlo_win_probability"]) for row in joined])
        gbm = np.array([float(row["gbm_win_probability"]) for row in joined])

        def _metrics(proba: np.ndarray) -> dict[str, float]:
            predicted = (proba >= 0.5).astype(int)
            result = {"accuracy": float(accuracy_score(actual, predicted))}
            if len(np.unique(actual)) >= 2:
                result["auc"] = float(roc_auc_score(actual, proba))
                result["brier"] = float(brier_score_loss(actual, proba))
            return result

        return {
            "rows": len(joined),
            "matches": len(matches),
            "matches_list": matches,
            "monte_carlo": _metrics(mc),
            "gbm": _metrics(gbm),
        }

    def summary_text(self) -> str:
        summary = self.summarize()
        if summary["rows"] == 0:
            return "No real live IPL-chase comparisons logged yet."
        mc, gbm = summary["monte_carlo"], summary["gbm"]
        lines = [
            f"IPL chase win-probability: Monte Carlo vs GBM, {summary['matches']} real live "
            f"match(es), {summary['rows']} over-level predictions.",
            f"Monte Carlo: accuracy {mc['accuracy'] * 100:.1f}%"
            + (f", AUC {mc['auc']:.3f}, Brier {mc['brier']:.3f}" if "auc" in mc else ""),
            f"GBM (served):     accuracy {gbm['accuracy'] * 100:.1f}%"
            + (f", AUC {gbm['auc']:.3f}, Brier {gbm['brier']:.3f}" if "auc" in gbm else ""),
        ]
        return "\n".join(lines)
