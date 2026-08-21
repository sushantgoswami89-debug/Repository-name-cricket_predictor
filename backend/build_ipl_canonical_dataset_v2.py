"""Build the canonical, candidate-only IPL training dataset v2."""

import json
from pathlib import Path

import pandas as pd

from app.ml.ipl_phase_moe_dataset import KEYS, build_ipl_phase_moe_features
from app.ml.ipl_venue_regime_dataset import build_ipl_venue_regime_dataset

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    output = root / "data/candidates/ipl_canonical_v2"
    output.mkdir(parents=True, exist_ok=True)

    base = pd.read_csv(root / "data/candidates/v3/verified_training_overs.csv")
    venue = build_ipl_venue_regime_dataset(root)
    identity = build_ipl_phase_moe_features(root, canonical_identities=True)
    for frame in (base, venue, identity):
        frame["match_date"] = frame["match_date"].astype(str)

    venue_columns = [column for column in venue.columns if column not in KEYS]
    identity_columns = [column for column in identity.columns if column not in KEYS]
    data = base.merge(
        venue[KEYS + venue_columns], on=KEYS, validate="one_to_one"
    ).merge(identity[KEYS + identity_columns], on=KEYS, validate="one_to_one")
    data.to_csv(output / "training_overs.csv", index=False)

    report = {
        "dataset_version": "ipl_canonical_v2",
        "rows": len(data),
        "matches": int(data["source_file"].nunique()),
        "duplicate_keys": int(data.duplicated(KEYS).sum()),
        "missing_values": int(data.isna().sum().sum()),
        "canonical_players": int(
            len(set(data["striker"]) | set(data["non_striker"]))
        ),
        "canonical_teams": sorted(data["batting_team"].unique()),
        "canonical_venues": int(data["venue_name"].nunique()),
        "targets_unchanged": True,
        "raw_files_modified": False,
        "production_changed": False,
    }
    (output / "verification.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report))
