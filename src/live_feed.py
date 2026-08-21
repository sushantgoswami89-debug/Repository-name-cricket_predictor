"""
Automated live-over predictor - EVOLVED VERSION.
Now includes the Feedback Loop for 90% accuracy evolution.
"""

import argparse
import json
import time
import os
import sys
import requests
import random

# Import the new Engine and Context models we updated
from app.ml.prediction_engine import PredictionEngine
from app.models.match_context import MatchContext, LiveContext, PitchContext

OUTPUT_FILE = "live_prediction.json"

# ---------------------------------------------------------------------------
# MOCK MODE — Now with Evolution Training
# ---------------------------------------------------------------------------
def run_mock_feed(poll_seconds=3):
    print("Running in EVOLVING MOCK mode. System will learn from errors every over.\n")
    
    # Initialize the Smart Engine
    engine = PredictionEngine()
    
    score = 0
    wkts = 0
    last_batsman = "SC Ganguly"
    last_bowler = "P Kumar"

    for over_num in range(1, 21):
        # 1. SETUP CONTEXT
        # We build a proper MatchContext so the engine knows the format and momentum
        context = MatchContext(
            match_id="mock_match_001",
            match_style="IPL", # Set format to stop confusion
            live=LiveContext(
                over=over_num,
                balls_in_over=0,
                batsman=last_batsman,
                non_striker="BB McCullum",
                bowler=last_bowler,
                wickets_down=wkts,
                total_score=score,
                current_run_rate=round(score/over_num, 2) if over_num > 1 else 0
            ),
            pitch=PitchContext(batting_rating=75, pace_assistance=50, spin_assistance=30)
        )

        # 2. GENERATE PREDICTION
        try:
            result = engine.predict(context)
            
            # Save to JSON for the dashboard
            output = {
                "over": over_num,
                "batsman": last_batsman,
                "bowler": last_bowler,
                "prediction": {
                    "expected_runs": result.predicted_runs,
                    "range": result.expected_range,
                    "wicket_probability": result.wicket_probability,
                    "analysis": result.analysis
                }
            }
            with open(OUTPUT_FILE, "w") as f:
                json.dump(output, f, indent=2)

            print(f"--- Over {over_num} Prediction: {result.expected_range} runs ---")
        
        except ValueError as e:
            print(f"GATEKEEPER BLOCK: {e}")
            break

        # 3. SIMULATE ACTUAL OVER (This is where the 'Match' happens)
        time.sleep(poll_seconds)
        actual_runs = random.randint(2, 12)
        actual_wickets = 1 if random.random() < 0.1 else 0
        
        # 4. THE EVOLUTION HOOK (Crucial!)
        # We tell the engine what happened so it can adjust its Match Bias
        report = engine.update_actuals(actual_runs, actual_wickets, last_bowler)
        
        score += actual_runs
        wkts += actual_wickets
        print(f"  [Actual: {actual_runs} runs, {actual_wickets} wkts] -> {report}\n")

# ---------------------------------------------------------------------------
# LIVE MODE — Real API Polling with Evolution
# ---------------------------------------------------------------------------
def run_live_feed(api_key, match_id, poll_seconds=30):
    print(f"Running LIVE Evolution. Polling ID: {match_id}")
    engine = PredictionEngine()
    
    last_over_seen = -1
    last_score = 0
    last_wickets = 0

    while True:
        try:
            # 1. Fetch data from API
            raw = fetch_raw_match_data(api_key, match_id)
            state = parse_api_response(raw) # This needs to return current score/over
            
            if state and state["over_num"] > last_over_seen:
                # A NEW OVER HAS COMPLETED
                if last_over_seen != -1:
                    # EVOLVE: Calculate what happened in the over that just finished
                    actual_runs = state["score_before"] - last_score
                    actual_wickets = state["wkts_down"] - last_wickets
                    
                    engine.update_actuals(actual_runs, actual_wickets, state["bowler"])
                    print(f"Engine Evolved after Over {last_over_seen}")

                # PREDICT: Now predict the NEXT over
                context = MatchContext(
                    match_id=match_id,
                    match_style="IPL", # Or detect from API
                    live=LiveContext(
                        over=state["over_num"],
                        batsman=state["batsman"],
                        bowler=state["bowler"],
                        wickets_down=state["wkts_down"],
                        total_score=state["score_before"]
                    ),
                    pitch=PitchContext(batting_rating=70, pace_assistance=50, spin_assistance=50)
                )
                
                prediction = engine.predict(context)
                last_over_seen = state["over_num"]
                last_score = state["score_before"]
                last_wickets = state["wkts_down"]
                
                print(f"Over {state['over_num']} Prediction Ready: {prediction.expected_range}")

        except Exception as e:
            print(f"Error in Live Loop: {e}")
        
        time.sleep(poll_seconds)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()

    if args.mock:
        run_mock_feed()