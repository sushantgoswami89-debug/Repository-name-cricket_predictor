"""Produce data/real_overs_styled.csv: real_overs.csv with batsman_style and
bowler_type replaced by real values from the external player-style registry,
instead of the constant "unknown" every row currently has.

Does not touch data/real_overs.csv itself.
"""

from pathlib import Path

import pandas as pd

from app.ml.player_style_registry import PlayerStyleRegistry

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "data" / "real_overs.csv"
OUT_PATH = PROJECT_ROOT / "data" / "real_overs_styled.csv"

df = pd.read_csv(SRC_PATH)
registry = PlayerStyleRegistry()

print("Looking up styles for", df["batsman"].nunique(), "batsmen and", df["bowler"].nunique(), "bowlers...")

batsman_lookup = {name: registry.lookup(name)["batting_style"] for name in df["batsman"].unique()}
bowler_lookup = {name: registry.lookup(name)["bowling_style"] for name in df["bowler"].unique()}

df["batsman_style"] = df["batsman"].map(batsman_lookup)
df["bowler_type"] = df["bowler"].map(bowler_lookup)

known_bat = (df["batsman_style"] != "unknown").mean()
known_bowl = (df["bowler_type"] != "unknown").mean()
print(f"Rows with known batsman_style: {known_bat*100:.1f}%")
print(f"Rows with known bowler_type:   {known_bowl*100:.1f}%")
print(df["bowler_type"].value_counts())

df.to_csv(OUT_PATH, index=False)
print(f"Saved {OUT_PATH}")
