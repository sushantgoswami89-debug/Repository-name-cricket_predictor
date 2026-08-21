"""Store the CricketData API key in a local owner-only configuration file."""

from __future__ import annotations

import json
import os
from pathlib import Path


def main() -> None:
    api_key = os.environ.get("CRICKETDATA_API_KEY", "").strip()
    if not api_key:
        raise SystemExit(
            "CRICKETDATA_API_KEY is not available in this Terminal session."
        )
    destination = Path(__file__).with_name(".cricketdata.json")
    destination.write_text(json.dumps({"api_key": api_key}), encoding="utf-8")
    destination.chmod(0o600)
    print("CricketData key saved securely.")


if __name__ == "__main__":
    main()
