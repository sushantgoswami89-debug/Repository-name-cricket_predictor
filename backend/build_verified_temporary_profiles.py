"""Compile source-tracked temporary profiles without inventing missing stats."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from app.ml.cold_start_profiles import TemporaryProfile, aggressiveness_band


def build(root: Path) -> dict:
    source = root / "data/profiles/temporary_batters.json"
    raw = json.loads(source.read_text(encoding="utf-8"))
    profiles = [TemporaryProfile.from_dict(row) for row in raw["profiles"]]
    if len({profile.player_id for profile in profiles}) != len(profiles):
        raise ValueError("Duplicate temporary-profile player_id")
    rows = [
        {
            "player_id": profile.player_id,
            "player_name": profile.player_name,
            "role": profile.role,
            "usual_position": profile.usual_position,
            "batting_average": profile.batting_average,
            "strike_rate": profile.strike_rate,
            "aggressiveness": aggressiveness_band(profile.strike_rate),
            "experience": "; ".join(profile.experience),
            "source_url": profile.source_url,
            "source_title": profile.source_title,
            "source_published": profile.source_published.isoformat(),
            "verified_at": profile.verified_at.isoformat(),
        }
        for profile in profiles
    ]
    output = root / "data/reports/ipl_cold_start_v1"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "verified_external_profiles.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "schema_version": raw["schema_version"],
        "profiles": len(rows),
        "source_tracked": sum(bool(row["source_url"]) for row in rows),
        "missing_average_preserved": sum(
            row["batting_average"] is None for row in rows
        ),
        "missing_strike_rate_preserved": sum(
            row["strike_rate"] is None for row in rows
        ),
        "specialist_bowlers": sum(row["role"] == "BOWLER" for row in rows),
    }
    (output / "external_profile_build.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    print(json.dumps(build(project_root), indent=2))
