"""Strict IPL raw-data certification with reviewed exceptional match events."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.ml.candidate_v3_dataset import _is_legal
from app.ml.ipl_identities import canonical_player_id, canonical_team_id
from app.ml.ipl_venues import normalize_ipl_venue

# (source file, innings number, zero-based source over)
REVIEWED_EXCEPTIONS = {
    ("335994.json", 2, 10): {
        "type": "seven_legal_ball_over",
        "status": "verified_historical_event",
        "source": "ESPN scorecard match notes",
    },
    ("392198.json", 2, 10): {
        "type": "seven_legal_ball_over",
        "status": "scorecard_reconciled_historical_event",
        "source": "Cricsheet delivery record and innings scorecard",
    },
    ("419155.json", 1, 18): {
        "type": "seven_legal_ball_over",
        "status": "verified_historical_event",
        "source": "ESPN scorecard match notes",
    },
    ("1136564.json", 2, 11): {
        "type": "seven_legal_ball_over",
        "status": "verified_historical_event",
        "source": "contemporary match report",
    },
    ("501247.json", 2, 2): {
        "type": "37_run_over_with_no_ball",
        "status": "valid_scoring_event",
        "source": "Cricsheet delivery record and innings scorecard",
    },
    ("1254076.json", 1, 19): {
        "type": "37_run_over_with_no_ball",
        "status": "valid_scoring_event",
        "source": "Cricsheet delivery record and innings scorecard",
    },
}


def certify_ipl_raw_data(project_root: Path) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    observed_exceptions: dict[tuple[str, int, int], dict[str, Any]] = {}
    files = sorted((project_root / "data/raw/cricsheet/ipl").glob("*.json"))
    rows = deliveries_seen = 0

    for path in files:
        raw = json.loads(path.read_text(encoding="utf-8"))
        info = raw.get("info", {})
        registry = info.get("registry", {}).get("people", {})
        if str(info.get("match_type", "")).upper() != "T20":
            failures.append({"file": path.name, "reason": "not_t20"})
        for team in info.get("teams", []):
            if ":unresolved:" in canonical_team_id(str(team)):
                failures.append(
                    {"file": path.name, "reason": "unresolved_team", "value": team}
                )
        venue = normalize_ipl_venue(
            str(info.get("venue") or info.get("city") or "unknown")
        )
        if venue == "unknown":
            failures.append({"file": path.name, "reason": "unresolved_venue"})

        regular = [
            innings
            for innings in raw.get("innings", [])
            if not innings.get("super_over")
        ]
        if len(regular) not in {1, 2}:
            failures.append(
                {
                    "file": path.name,
                    "reason": "unexpected_regular_innings_count",
                    "value": len(regular),
                }
            )
        for innings_number, innings in enumerate(regular, start=1):
            over_numbers = [int(over["over"]) for over in innings.get("overs", [])]
            if over_numbers != sorted(set(over_numbers)):
                failures.append(
                    {"file": path.name, "reason": "invalid_over_sequence"}
                )
            for over in innings.get("overs", []):
                source_over = int(over["over"])
                key = (path.name, innings_number, source_over)
                balls = over.get("deliveries", [])
                if not balls:
                    failures.append(
                        {
                            "file": path.name,
                            "innings": innings_number,
                            "over": source_over,
                            "reason": "empty_over",
                        }
                    )
                    continue
                legal = sum(int(_is_legal(ball)) for ball in balls)
                runs = sum(
                    int(ball.get("runs", {}).get("total", 0)) for ball in balls
                )
                has_illegal = any(not _is_legal(ball) for ball in balls)
                exceptional = legal > 6 or runs > 36
                if exceptional:
                    if key not in REVIEWED_EXCEPTIONS:
                        failures.append(
                            {
                                "file": path.name,
                                "innings": innings_number,
                                "over": source_over,
                                "reason": "unreviewed_exceptional_over",
                                "legal_balls": legal,
                                "runs": runs,
                            }
                        )
                    else:
                        observed_exceptions[key] = {
                            **REVIEWED_EXCEPTIONS[key],
                            "legal_balls": legal,
                            "runs": runs,
                            "has_illegal_delivery": has_illegal,
                        }
                for ball in balls:
                    deliveries_seen += 1
                    for field in ("batter", "non_striker", "bowler"):
                        name = str(ball.get(field, "")).strip()
                        player_id = canonical_player_id(name, registry)
                        if ":unresolved:" in player_id or player_id.endswith(
                            ":unknown"
                        ):
                            failures.append(
                                {
                                    "file": path.name,
                                    "reason": "unresolved_player",
                                    "value": name,
                                }
                            )
                rows += 1

    missing_reviewed = sorted(set(REVIEWED_EXCEPTIONS) - set(observed_exceptions))
    if missing_reviewed:
        failures.append(
            {
                "reason": "reviewed_exception_missing_from_source",
                "values": [list(item) for item in missing_reviewed],
            }
        )
    return {
        "certification_version": "ipl_raw_clean_v1",
        "certified": not failures,
        "raw_files_modified": False,
        "source_files": len(files),
        "over_rows": rows,
        "deliveries": deliveries_seen,
        "failure_count": len(failures),
        "failures": failures,
        "reviewed_exception_count": len(observed_exceptions),
        "reviewed_exceptions": {
            "|".join(map(str, key)): value
            for key, value in sorted(observed_exceptions.items())
        },
        "rules": {
            "duplicate_over_keys_forbidden": True,
            "missing_player_uuid_forbidden": True,
            "unresolved_team_or_venue_forbidden": True,
            "unreviewed_more_than_six_legal_balls_forbidden": True,
            "unreviewed_more_than_36_runs_forbidden": True,
            "super_overs_separate_from_regular_innings": True,
        },
    }
