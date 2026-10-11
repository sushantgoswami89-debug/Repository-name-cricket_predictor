"""A permanent record of every pre-match win % we publish, and how each match ended.

Two append-only files under data/live/:
  prematch-predictions.jsonl  one line each time the pre-match number changes (stage, probabilities,
                              the components behind it, the rain risk, the model version)
  prematch-results.jsonl      one line per finished match (winner, or "no result")
`score_prematch_log.py` joins them and reports accuracy and Brier by stage. Until 2026-10-10 the
pre-match numbers had only ever been backtested, never scored on live matches.

Logging never raises: a failure here must not block a briefing.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from app.live.prematch_model import ROOT

LOG_PATH = ROOT / "data" / "live" / "prematch-predictions.jsonl"
RESULTS_PATH = ROOT / "data" / "live" / "prematch-results.jsonl"
_LAST: dict[str, tuple] = {}


def _read(path: Path) -> list[dict]:
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, ValueError):
        return []


def log_prediction(match_id: str, record: dict, path: Path | None = None) -> bool:
    """Append a prediction unless the match's latest logged one has the same stage and numbers."""
    path = path or LOG_PATH
    try:
        signature = (record["stage"], record["pct_a"], record["pct_b"], record["pct_nr"])
        key = f"{path}:{match_id}"
        if key not in _LAST:  # first call in this process: pick up where an earlier run left off
            earlier = [r for r in _read(path) if r.get("match_id") == match_id]
            if earlier:
                last = earlier[-1]
                _LAST[key] = (last["stage"], last["pct_a"], last["pct_b"], last["pct_nr"])
        if _LAST.get(key) == signature:
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        line = {"ts_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "match_id": match_id, **record}
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(line) + "\n")
        _LAST[key] = signature
        return True
    except Exception:
        return False


def record_result(match_id: str, winner: str | None, text: str, path: Path | None = None) -> bool:
    """Append a finished match's outcome once (winner None = no result)."""
    path = path or RESULTS_PATH
    try:
        if any(r.get("match_id") == match_id for r in _read(path)):
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        line = {"ts_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "match_id": match_id, "winner": winner, "text": text}
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(line) + "\n")
        return True
    except Exception:
        return False


# ICC men's T20I top 10 (user choice 2026-10-11): only matches between two of these are scored.
# Update when the ranking changes. Names as Cricbuzz/TOI/Cricsheet spell them.
ICC_TOP_10 = frozenset({
    "India", "England", "Australia", "Pakistan", "South Africa",
    "New Zealand", "West Indies", "Sri Lanka", "Afghanistan", "Bangladesh",
})


def score(pred_path: Path | None = None, result_path: Path | None = None, teams: frozenset | None = ICC_TOP_10) -> dict:
    """Per stage: games, picks right, Brier, using the LAST prediction logged at each stage.
    Two-way probability = p_a / (p_a + p_b), i.e. conditional on the match finishing; matches
    that ended without a result are counted separately."""
    predictions = _read(pred_path or LOG_PATH)
    if teams is not None:
        predictions = [p for p in predictions if p["team_a"] in teams and p["team_b"] in teams]
    results = {r["match_id"]: r for r in _read(result_path or RESULTS_PATH)}
    by_stage: dict[str, dict] = {}
    no_result = 0
    for match_id, result in results.items():
        mine = [p for p in predictions if p["match_id"] == match_id]
        if not mine:
            continue
        if not result.get("winner"):
            no_result += 1
            continue
        latest: dict[str, dict] = {}
        for p in mine:
            latest[p["stage"]] = p  # later lines overwrite earlier ones of the same stage
        for stage, p in latest.items():
            two_way = p["p_a"] / (p["p_a"] + p["p_b"])
            outcome = 1.0 if result["winner"] == p["team_a"] else 0.0
            cell = by_stage.setdefault(stage, {"n": 0, "right": 0, "brier": 0.0, "games": []})
            cell["n"] += 1
            cell["right"] += int((two_way >= 0.5) == bool(outcome))
            cell["brier"] += (two_way - outcome) ** 2
            cell["games"].append((match_id, p["team_a"], p["team_b"], round(two_way, 3), result["winner"]))
    for cell in by_stage.values():
        cell["brier"] = cell["brier"] / cell["n"] if cell["n"] else None
    return {"by_stage": by_stage, "no_result": no_result, "matches_logged": len({p["match_id"] for p in predictions}),
            "results": len(results)}
