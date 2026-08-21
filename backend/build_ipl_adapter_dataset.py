"""Build shared-history IPL adapter features and targets."""

from pathlib import Path

from app.ml.ipl_adapter_dataset import build_ipl_adapter_dataset

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    output = root / "data/candidates/ipl_v3/adapter_features.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = build_ipl_adapter_dataset(root)
    frame.to_csv(output, index=False)
    print({"rows": len(frame), "output": str(output)})
