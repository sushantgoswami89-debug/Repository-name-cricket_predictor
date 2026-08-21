"""Write the immutable IPL raw-data cleanliness certificate."""

import json
from pathlib import Path

from app.ml.ipl_dataset_validator import certify_ipl_raw_data

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    output = root / "data/reports/ipl_raw_clean_v1/certification.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    report = certify_ipl_raw_data(root)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"certified={report['certified']} failures={report['failure_count']} "
        f"exceptions={report['reviewed_exception_count']}"
    )
    if not report["certified"]:
        raise SystemExit(1)
