"""
Converts Cricsheet JSON match files into the same over-level row schema our
model already trains on.
"""

import json, os, sys, glob
import pandas as pd
import numpy as np


def phase_of_over(over_num):
    if over_num <= 6:
        return "powerplay"
    elif over_num <= 15:
        return "middle"
    return "death"


def parse_match_file(filepath, match_id):
    with open(filepath, "r") as f:
        match = json.load(f)

    info = match.get("info", {})
    venue = info.get("venue", "unknown")
    teams = info.get("teams", [])

    rows = []
    innings_list = match.get("innings", [])

    for innings_no, innings in enumerate(innings_list, start=1):
        batting_team = innings.get("team", "unknown")
        bowling_team = next((t for t in teams if t != batting_team), "unknown")

        score = 0
        wkts_down = 0
        balls_faced = {}
        overs_data = innings.get("overs", [])

        for over_block in overs_data:
            over_num = over_block.get("over", 0) + 1
            deliveries = over_block.get("deliveries", [])
            if not deliveries:
                continue

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

                if batter_runs in (4, 6):
                    boundaries_this_over += 1

                extras = ball.get("extras", {})
                if "wides" not in extras:
                    balls_faced[batter] = balls_faced.get(batter, 0) + 1

                if "wickets" in ball:
                    wicket_this_over = 1
                    wkts_down += 1

            balls_bowled = (over_num - 1) * 6
            current_rr = (
                round(score_before_over / max(1, over_num - 1), 2)
                if over_num > 1
                else 0.0
            )

            rows.append(
                {
                    "match_id": match_id,
                    "innings": innings_no,
                    "batting_team": batting_team,
                    "bowling_team": bowling_team,
                    "venue": venue,
                    "over": over_num,
                    "phase": phase_of_over(over_num),
                    "batsman": batsman,
                    "bowler": bowler,
                    "balls_faced_before_over": balls_before_over,
                    "score_before_over": score_before_over,
                    "wkts_down_before_over": wkts_before_over,
                    "wickets_in_hand": 10 - wkts_before_over,
                    "balls_remaining": 120 - balls_bowled,
                    "current_run_rate": current_rr,
                    "runs_in_over": runs_this_over,
                    "wicket_in_over": wicket_this_over,
                    "boundaries_in_over": boundaries_this_over,
                }
            )
    return rows


def add_pitch_category(df):
    venue_totals = df.groupby(["match_id", "venue"])["runs_in_over"].sum().reset_index()
    venue_avg = venue_totals.groupby("venue")["runs_in_over"].mean()
    tertiles = venue_avg.quantile([0.33, 0.66])

    def categorize(v):
        avg = venue_avg.get(v, venue_avg.median())
        if avg <= tertiles.iloc[0]:
            return "slow_turner"
        elif avg <= tertiles.iloc[1]:
            return "balanced"
        return "batting_paradise"

    df["pitch_type"] = df["venue"].apply(categorize)
    df["venue_avg_score"] = (
        df["venue"].map(venue_avg * 6.5).fillna(venue_avg.mean() * 6.5)
    )
    return df


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 parse_cricsheet.py /path/to/json/folder")
        sys.exit(1)
    folder = sys.argv[1]
    files = [
        f
        for f in glob.glob(os.path.join(folder, "*.json"))
        if "people" not in f.lower() and "readme" not in f.lower()
    ]
    all_rows = []
    mid = 0
    for fp in files:
        mid += 1
        try:
            all_rows.extend(parse_match_file(fp, mid))
        except Exception:
            pass
    df = pd.DataFrame(all_rows)
    df = add_pitch_category(df)
    df["batsman_style"] = "unknown"
    df["batsman_class"] = "unknown"
    df["bowler_type"] = "unknown"
    df["bowler_quality"] = "unknown"
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    out = os.path.join(out_dir, "real_overs.csv")
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows to {out}")


if __name__ == "__main__":
    main()
