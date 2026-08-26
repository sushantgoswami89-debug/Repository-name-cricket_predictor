"""Live shadow-comparison logging for the run-range GBM+NN ensemble
(2026-08-26) -- mirrors app/ml/wicket_ensemble_shadow_log.py's pattern,
adapted for run-range: the real outcome (actual runs scored) is known at
the very next over, per-over here, not per-match. The "5 real matches"
threshold for the review reminder still counts distinct MATCHES, for
consistency with the win-probability and wicket-ensemble shadow systems.

Metric here is band HIT RATE (was the actual over-runs total within the
predicted inclusive [low, high] band), matching how this project
evaluates every run-range candidate -- not AUC/Brier, which don't apply
to a multiclass band prediction.

Files (gitignored, same data/live/* convention as every other live
snapshot): run_range_ensemble_shadow_log.jsonl (per-over: GBM-alone and
ensemble bands), run_range_ensemble_shadow_outcomes.jsonl (per-over real
actual runs, idempotent per (match_id, innings, over)),
run_range_ensemble_shadow_reminder_sent.json (fires the review reminder
once per threshold crossing).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_LOG_PATH = Path(__file__).resolve().parents[3] / "data/live/run_range_ensemble_shadow_log.jsonl"
DEFAULT_OUTCOMES_PATH = Path(__file__).resolve().parents[3] / "data/live/run_range_ensemble_shadow_outcomes.jsonl"
DEFAULT_MARKER_PATH = Path(__file__).resolve().parents[3] / "data/live/run_range_ensemble_shadow_reminder_sent.json"
MIN_MATCHES_FOR_SUMMARY = 5


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunRangeEnsembleShadowLog:
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
        gbm_low: int,
        gbm_high: int,
        ensemble_low: int,
        ensemble_high: int,
        nn_available: bool,
    ) -> None:
        row = {
            "match_id": match_id, "innings": innings, "over": over,
            "gbm_low": gbm_low, "gbm_high": gbm_high,
            "ensemble_low": ensemble_low, "ensemble_high": ensemble_high,
            "nn_available": nn_available, "recorded_at": _now_iso(),
        }
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")

    def _recorded_outcome_keys(self) -> set[tuple[str, int, int]]:
        if not self.outcomes_path.exists():
            return set()
        keys: set[tuple[str, int, int]] = set()
        for line in self.outcomes_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                keys.add((row["match_id"], row["innings"], row["over"]))
        return keys

    def record_outcome(self, *, match_id: str, innings: int, over: int, actual_runs: int) -> None:
        if (match_id, innings, over) in self._recorded_outcome_keys():
            return
        row = {"match_id": match_id, "innings": innings, "over": over,
               "actual_runs": actual_runs, "recorded_at": _now_iso()}
        self.outcomes_path.parent.mkdir(parents=True, exist_ok=True)
        with self.outcomes_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")

    def matches_with_outcome_count(self) -> int:
        return len({key[0] for key in self._recorded_outcome_keys()})

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
        predictions = self._load_jsonl(self.log_path)
        outcomes = {
            (row["match_id"], row["innings"], row["over"]): int(row["actual_runs"])
            for row in self._load_jsonl(self.outcomes_path)
        }
        joined = [row for row in predictions if (row["match_id"], row["innings"], row["over"]) in outcomes]
        matches = sorted({row["match_id"] for row in joined})
        if not joined:
            return {"rows": 0, "matches": 0}

        def _hit_rate(low_key: str, high_key: str) -> dict[str, float]:
            hits = 0
            for row in joined:
                actual = outcomes[(row["match_id"], row["innings"], row["over"])]
                if row[low_key] <= actual <= row[high_key]:
                    hits += 1
            return {"rows": len(joined), "hit_rate": hits / len(joined)}

        return {
            "rows": len(joined), "matches": len(matches),
            "gbm": _hit_rate("gbm_low", "gbm_high"),
            "ensemble": _hit_rate("ensemble_low", "ensemble_high"),
        }

    def summary_text(self) -> str:
        summary = self.summarize()
        if summary["rows"] == 0:
            return "No real live run-range-ensemble comparisons logged yet."
        lines = [f"Run-range ensemble: GBM vs blend, {summary['matches']} real match(es), "
                 f"{summary['rows']} over-level predictions."]
        for label, key in (("GBM (served)", "gbm"), ("Ensemble", "ensemble")):
            m = summary.get(key)
            if m is None:
                lines.append(f"{label}: not enough data yet")
            else:
                lines.append(f"{label}: band hit rate {m['hit_rate'] * 100:.1f}%")
        return "\n".join(lines)
