from pathlib import Path

print("=" * 50)
print("CricketBaba Replay")
print("=" * 50)

project_root = Path(__file__).resolve().parents[3]

print("Project Root :", project_root)

models_dir = project_root / "models"

print("Models Found :", models_dir.exists())

data_dir = project_root / "data"

print("Data Found   :", data_dir.exists())

print("=" * 50)
