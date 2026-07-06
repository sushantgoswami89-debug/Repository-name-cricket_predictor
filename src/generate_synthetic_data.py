"""
Generates synthetic ball-by-ball T20 data structured the same way Cricsheet data
would look once parsed into a flat dataframe. This lets us build and test the
entire pipeline now. Swap this out later with a real Cricsheet parser (same
output schema) and nothing downstream needs to change.
"""
import numpy as np
import pandas as pd

np.random.seed(42)

# --- Define a pool of players with "true" underlying skill (hidden ground truth
# the simulator uses to generate outcomes realistically) ---
BATSMEN = {
    "V Kohli":      {"style": "RHB", "class": "top",  "vs_pace": 1.15, "vs_spin": 1.05},
    "R Sharma":     {"style": "RHB", "class": "top",  "vs_pace": 1.20, "vs_spin": 0.95},
    "S Yadav":      {"style": "RHB", "class": "top",  "vs_pace": 1.30, "vs_spin": 1.10},
    "KL Rahul":     {"style": "RHB", "class": "solid","vs_pace": 1.05, "vs_spin": 1.00},
    "R Pant":       {"style": "LHB", "class": "top",  "vs_pace": 1.25, "vs_spin": 1.15},
    "H Pandya":     {"style": "RHB", "class": "solid","vs_pace": 1.10, "vs_spin": 1.05},
    "D Padikkal":   {"style": "LHB", "class": "avg",  "vs_pace": 0.90, "vs_spin": 0.95},
    "S Gill":       {"style": "RHB", "class": "solid","vs_pace": 1.05, "vs_spin": 1.00},
    "N Pooran":     {"style": "LHB", "class": "top",  "vs_pace": 1.20, "vs_spin": 1.05},
    "R Parag":      {"style": "RHB", "class": "avg",  "vs_pace": 0.95, "vs_spin": 0.90},
}

BOWLERS = {
    "J Bumrah":     {"type": "pace", "quality": "elite", "economy_factor": 0.75, "wicket_factor": 1.40},
    "M Shami":      {"type": "pace", "quality": "good",  "economy_factor": 0.90, "wicket_factor": 1.15},
    "T Boult":      {"type": "pace", "quality": "good",  "economy_factor": 0.88, "wicket_factor": 1.10},
    "Y Chahal":     {"type": "spin", "quality": "good",  "economy_factor": 0.92, "wicket_factor": 1.25},
    "R Jadeja":     {"type": "spin", "quality": "good",  "economy_factor": 0.85, "wicket_factor": 1.05},
    "A Rashid":     {"type": "spin", "quality": "elite", "economy_factor": 0.80, "wicket_factor": 1.30},
    "H Patel":      {"type": "pace", "quality": "avg",   "economy_factor": 1.10, "wicket_factor": 0.90},
    "K Ahmed":      {"type": "pace", "quality": "avg",   "economy_factor": 1.15, "wicket_factor": 0.85},
}

PITCHES = {
    "batting_paradise": {"run_factor": 1.25, "wicket_factor": 0.80},
    "balanced":         {"run_factor": 1.00, "wicket_factor": 1.00},
    "slow_turner":      {"run_factor": 0.80, "wicket_factor": 1.15},
    "seaming_track":    {"run_factor": 0.85, "wicket_factor": 1.25},
}

def phase_of_over(over_num):
    if over_num <= 6:
        return "powerplay"
    elif over_num <= 15:
        return "middle"
    else:
        return "death"

def base_run_rate(phase):
    return {"powerplay": 8.2, "middle": 7.6, "death": 10.5}[phase]

def base_wicket_prob(phase):
    return {"powerplay": 0.045, "middle": 0.035, "death": 0.065}[phase]

def simulate_over(batsman, bowler, over_num, pitch, balls_faced_so_far, current_score, wkts_down):
    b = BATSMEN[batsman]
    bw = BOWLERS[bowler]
    p = PITCHES[pitch]
    phase = phase_of_over(over_num)

    skill_vs = b["vs_pace"] if bw["type"] == "pace" else b["vs_spin"]
    class_mult = {"top": 1.15, "solid": 1.0, "avg": 0.88}[b["class"]]

    exp_runs = base_run_rate(phase) * skill_vs * class_mult * bw["economy_factor"] * p["run_factor"]
    # settled-in batsman scores a bit more; new batsman (low balls faced) scores less & is more dismissal-prone
    settle_mult = 0.75 if balls_faced_so_far < 10 else (0.95 if balls_faced_so_far < 20 else 1.05)
    exp_runs *= settle_mult

    runs = max(0, int(np.random.poisson(exp_runs)))
    runs = min(runs, 36)  # cap sanity

    wicket_prob = base_wicket_prob(phase) * bw["wicket_factor"] * p["wicket_factor"] / class_mult
    wicket_prob *= (1.3 if balls_faced_so_far < 10 else 1.0)
    wicket_prob = min(wicket_prob * 6, 0.55)  # scale to per-over prob, cap

    wicket = np.random.random() < wicket_prob
    if wicket:
        runs = max(0, runs - np.random.randint(0, 4))  # fewer runs when dismissed mid-over typically

    boundary_prob = min(0.15 + (exp_runs - 7) * 0.03, 0.55)
    boundaries = np.random.binomial(1, boundary_prob) + (1 if runs >= 12 else 0)

    return runs, wicket, boundaries

def generate_matches(n_matches=400):
    rows = []
    match_id = 0
    batsmen_list = list(BATSMEN.keys())
    bowlers_list = list(BOWLERS.keys())
    pitches_list = list(PITCHES.keys())

    for m in range(n_matches):
        match_id += 1
        pitch = np.random.choice(pitches_list)
        venue_avg_score = int(np.random.normal(165, 20))

        # pick a batting lineup order and bowling rotation for this match
        order = np.random.choice(batsmen_list, size=6, replace=False)
        bowl_rotation = np.random.choice(bowlers_list, size=6, replace=False)

        wkts_down = 0
        score = 0
        balls_faced = {b: 0 for b in order}
        cur_batsman_idx = 0

        for over_num in range(1, 21):
            if wkts_down >= 6 or cur_batsman_idx >= len(order):
                break
            bowler = bowl_rotation[over_num % len(bowl_rotation)]
            batsman = order[cur_batsman_idx]

            runs, wicket, boundaries = simulate_over(
                batsman, bowler, over_num, pitch,
                balls_faced[batsman], score, wkts_down
            )

            rows.append({
                "match_id": match_id,
                "over": over_num,
                "phase": phase_of_over(over_num),
                "batsman": batsman,
                "batsman_style": BATSMEN[batsman]["style"],
                "batsman_class": BATSMEN[batsman]["class"],
                "bowler": bowler,
                "bowler_type": BOWLERS[bowler]["type"],
                "bowler_quality": BOWLERS[bowler]["quality"],
                "pitch_type": pitch,
                "venue_avg_score": venue_avg_score,
                "balls_faced_before_over": balls_faced[batsman],
                "score_before_over": score,
                "wkts_down_before_over": wkts_down,
                "runs_in_over": runs,
                "wicket_in_over": int(wicket),
                "boundaries_in_over": boundaries,
            })

            balls_faced[batsman] += 6
            score += runs
            if wicket:
                wkts_down += 1
                cur_batsman_idx += 1

    return pd.DataFrame(rows)

if __name__ == "__main__":
    import os
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    OUT_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), "data", "synthetic_overs.csv")
    df = generate_matches(n_matches=500)
    df.to_csv(OUT_PATH, index=False)
    print(f"Generated {len(df)} over-level rows across {df.match_id.nunique()} matches")
    print(df.head(10))
    print("\nRuns per over distribution:")
    print(df.runs_in_over.describe())
    print("\nWicket rate:", df.wicket_in_over.mean())
