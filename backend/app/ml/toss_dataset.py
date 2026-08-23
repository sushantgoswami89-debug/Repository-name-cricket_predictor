"""Leakage-safe toss state, from Cricsheet's ``info.toss`` block.

CTO-agreed roadmap priority #2 (2026-08-23): of the four fields TOI's
``ToiSnapshot`` already fetches live but the pipeline drops
(``toss_won_by``/``toss_decision``/``pitch_type``/``weather_*``), toss is
the only one with a historical source (Cricsheet) and is therefore the
only one backtestable before shipping -- weather/pitch/lineup would need
to ship blind and be observed on a real live match.

Known before a ball is bowled and constant for the whole match, so this is
maximally leakage-safe. Derived without needing team-name matching: toss
``decision`` (``bat``/``field``) plus the over's own ``innings`` number is
sufficient to know whether the team currently batting is the team that won
the toss --

    batting_team_won_toss = (innings == 1 and decision == "bat")
                          or (innings == 2 and decision == "field")

-- since the toss winner either bats first (innings 1) or elects to field
first, sending themselves in to bat second (innings 2).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from app.replay.evaluation_policy import KNOWN_RULE_ANOMALY_EXCLUSIONS

TOSS_FEATURES = ["toss_decision", "batting_team_won_toss"]


def build_toss_dataset(project_root: Path, scopes: tuple[str, ...] = ("ipl", "t20i")) -> pd.DataFrame:
    excluded = set(KNOWN_RULE_ANOMALY_EXCLUSIONS)
    rows: list[dict[str, object]] = []
    for scope in scopes:
        for path in (project_root / "data/raw/cricsheet" / scope).glob("*.json"):
            if path.stem in excluded:
                continue
            raw = json.loads(path.read_text(encoding="utf-8"))
            toss = raw["info"].get("toss", {})
            decision = str(toss.get("decision", ""))
            if decision not in ("bat", "field"):
                continue
            for innings_number, innings in enumerate(raw.get("innings", []), start=1):
                if innings_number not in (1, 2):
                    # Super-over innings aren't governed by the main-match
                    # toss; the bat/field derivation below doesn't apply.
                    continue
                won_toss = (innings_number == 1 and decision == "bat") or (
                    innings_number == 2 and decision == "field"
                )
                for source_over in innings.get("overs", []):
                    if not source_over.get("deliveries"):
                        continue
                    rows.append(
                        {
                            "source_file": path.name,
                            "innings": innings_number,
                            "over": int(source_over["over"]) + 1,
                            "toss_decision": decision,
                            "batting_team_won_toss": int(won_toss),
                        }
                    )
    result = pd.DataFrame(rows)
    if result.duplicated(["source_file", "innings", "over"]).any():
        raise ValueError("Toss dataset contains duplicate over keys.")
    return result
