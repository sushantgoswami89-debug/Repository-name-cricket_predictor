"""
Converts Cricsheet JSON match files into the same over-level row schema our
model already trains on. Run this AFTER downloading and unzipping real data
from https://cricsheet.org/downloads/ (e.g. the "IPL" JSON zip).

Usage:
    python3 parse_cricsheet.py /path/to/folder/of/match/json/files

This will write real_overs.csv into the data/ folder, ready for train.py.

NOTE ON MISSING FIELDS:
Cricsheet doesn't label batting style, bowler type (pace/spin), or "batsman
class" the way our synthetic data did. Rather than guess at those, this
parser relies more heavily on each player's own historical numbers (which
is a stronger, more honest signal anyway) and derives a simple pitch/venue
category from that venue's own historical scoring level.
"""

import json
import os
import sys
import glob
import pandas as pd
import numpy as np


def phase_of_over(over_num):
    if over_num <= 6:
        return "powerplay"
    elif over_num <= 15:
        return "middle"
    else:
        return "death"


def parse_match_file(filepath, match_id):
    with open(filepath, "r") as f:
        match = json.load(f)

    info = match.get("info", {})
    venue = info.get("venue", "unknown")

    rows = []
    innings_list = match.get("innings", [])

    first_innings_total = None

    for innings_no, innings in enumerate(innings_list, start=1):
        batting_team = innings.get("team", "unknown")
        teams = info.get("teams", [])
        bowling_team = next((t for t in teams if t != batting_team), "unknown")
        score = 0
        wkts_down = 0
        balls_faced = {}  # batter -> balls faced so far in this innings
        overs_data = innings.get("overs", [])

        for over_block in overs_data:
            over_num = over_block.get("over", 0) + 1  # cricsheet is 0-indexed
            deliveries = over_block.get("deliveries", [])
            if not deliveries:
                continue

            # the batter facing the FIRST ball of the over is who we predict for
            first_ball = deliveries[0]
            batsman = first_ball.get("batter", "unknown")
            bowler = first_ball.get("bowler", "unknown")

            score_before_over = score
            wkts_before_over = wkts_down
            balls_before_over = balls_faced.get(batsman, 0)

            runs_this_over = 0
            wicket_this_over = 0
            boundaries_this_over = 0

            for ball in deliveries:
                batter = ball.get("batter", "unknown")
                runs = ball.get("runs", {})
                batter_runs = runs.get("batter", 0)
                total_runs = runs.get("total", 0)

                runs_this_over += total_runs
                score += total_runs

                if batter_runs == 4 or batter_runs == 6:
                    boundaries_this_over += 1

                # only count a "ball faced" if it's not a wide (extras.wides present)
                extras = ball.get("extras", {})
                if "wides" not in extras:
                    balls_faced[batter] = balls_faced.get(batter, 0) + 1

                if "wickets" in ball:
                    wicket_this_over = 1
                    wkts_down += 1

            balls_bowled = max(0, (over_num - 1) * 6)
            balls_remaining = max(0, 120 - balls_bowled)
            current_run_rate = (
                score_before_over / (balls_bowled / 6) if balls_bowled else 0.0
            )

            if innings_no == 2 and first_innings_total is not None:
                target = first_innings_total + 1
                overs_left = balls_remaining / 6 if balls_remaining else 0
                runs_needed = max(0, target - score_before_over)
                required_run_rate = runs_needed / overs_left if overs_left else 0.0
            else:
                target = 0
                required_run_rate = 0.0

            rows.append(
                {
                    "match_id": match_id,
                    "venue": venue,
                    "innings": innings_no,
                    "batting_team": batting_team,
                    "bowling_team": bowling_team,
                    "over": over_num,
                    "phase": phase_of_over(over_num),
                    "batsman": batsman,
                    "bowler": bowler,
                    "balls_faced_before_over": balls_before_over,
                    "score_before_over": score_before_over,
                    "wkts_down_before_over": wkts_before_over,
                    "wickets_in_hand": 10 - wkts_before_over,
                    "current_run_rate": round(current_run_rate, 3),
                    "balls_remaining": balls_remaining,
                    "target": target,
                    "required_run_rate": round(required_run_rate, 3),
                    "runs_in_over": runs_this_over,
                    "wicket_in_over": wicket_this_over,
                    "boundaries_in_over": boundaries_this_over,
                }
            )

        if innings_no == 1:
            first_innings_total = score

    return rows


def add_pitch_category(df):
    """Derive a simple pitch/venue scoring category from the venue's own
    historical average total score, since Cricsheet doesn't label pitch type."""
    venue_totals = df.groupby(["match_id", "venue"])["runs_in_over"].sum().reset_index()
    venue_avg = venue_totals.groupby("venue")["runs_in_over"].mean()

    tertiles = venue_avg.quantile([0.33, 0.66])

    def categorize(v):
        avg = venue_avg.get(v, venue_avg.median())
        if avg <= tertiles.iloc[0]:
            return "slow_turner"
        elif avg <= tertiles.iloc[1]:
            return "balanced"
        else:
            return "batting_paradise"

    df["pitch_type"] = df["venue"].apply(categorize)
    df["venue_avg_score"] = (
        df["venue"].map(venue_avg * 6.5).fillna(venue_avg.mean() * 6.5)
    )  # rough full-innings estimate
    return df


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 parse_cricsheet.py /path/to/json/folder")
        sys.exit(1)

    folder = sys.argv[1]
    json_files = glob.glob(os.path.join(folder, "*.json"))
    # skip the "people.json" / "README" style metadata files Cricsheet includes
    json_files = [
        f for f in json_files if "people" not in f.lower() and "readme" not in f.lower()
    ]

    print(f"Found {len(json_files)} match files")

    all_rows = []
    match_id = 0
    errors = 0
    for filepath in json_files:
        match_id += 1
        try:
            rows = parse_match_file(filepath, match_id)
            all_rows.extend(rows)
        except Exception as e:
            errors += 1
            continue

    print(
        f"Parsed {match_id - errors} matches successfully ({errors} skipped due to format issues)"
    )

    df = pd.DataFrame(all_rows)
    if df.empty:
        print(
            "No data parsed - check that the folder path is correct and contains Cricsheet JSON files."
        )
        sys.exit(1)

    df = add_pitch_category(df)

    # These columns don't exist for real players unless you add them yourself later.
    # Filling with a neutral placeholder so the SAME features.py / train.py code works unchanged.
    df["batsman_style"] = "unknown"
    df["batsman_class"] = "unknown"
    df["bowler_type"] = "unknown"
    df["bowler_quality"] = "unknown"

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    out_path = os.path.join(out_dir, "real_overs.csv")

    if os.path.exists(out_path):
        existing = pd.read_csv(out_path)
        # offset match_ids so they don't collide with the previous batch
        df["match_id"] = df["match_id"] + existing["match_id"].max()
        df = pd.concat([existing, df], ignore_index=True)
        print(f"Appending to existing {out_path} (previously {len(existing)} rows)")

    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} total over-level rows to {out_path}")
    print(
        f"Matches: {df.match_id.nunique()}, Unique batsmen: {df.batsman.nunique()}, Unique bowlers: {df.bowler.nunique()}"
    )


if __name__ == "__main__":
    main()
