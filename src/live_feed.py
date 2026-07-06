"""
Automated live-over predictor.

Polls a live cricket data API on an interval, detects when a new over has
completed, builds features from the current match state, runs the prediction
models, and writes the result to a JSON file that a dashboard can read.

Two modes:
  --mock   : simulates a live match locally (no API key needed) so you can see
             the full automated loop working right now.
  --live   : calls a real API (CricketData.org / CricAPI style) using your key.
             NOTE: the parsing in `parse_api_response()` is a best-effort based
             on commonly documented CricketData.org response shapes. The exact
             field names can shift between providers/plans, so the FIRST time
             you run --live mode, it will print the raw JSON so you can confirm
             field names match, before it starts feeding the model.
"""
import argparse
import json
import time
import os
import sys
import requests
import random

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(SCRIPT_DIR)
from predict import NextOverPredictor
from generate_synthetic_data import BATSMEN, BOWLERS

OUTPUT_FILE = os.path.join(PROJECT_DIR, "live_prediction.json")

# ---------------------------------------------------------------------------
# MOCK MODE — simulates a live match progressing over by over
# ---------------------------------------------------------------------------
def run_mock_feed(poll_seconds=5):
    print("Running in MOCK mode — simulating a live match, no API key needed.\n")
    predictor = NextOverPredictor()

    batsmen_pool = list(BATSMEN.keys())
    bowlers_pool = list(BOWLERS.keys())
    pitch = "balanced"

    score = 0
    wkts = 0
    balls_faced = 0
    current_batsman = random.choice(batsmen_pool)

    for over_num in range(1, 21):
        bowler = bowlers_pool[over_num % len(bowlers_pool)]

        # simulate "fetching live state" -- in --live mode this comes from the API
        match_state = {
            "batsman": current_batsman,
            "bowler": bowler,
            "over_num": over_num,
            "score_before": score,
            "wkts_down": wkts,
            "balls_faced_by_batsman": balls_faced,
            "pitch_type": pitch,
        }

        result = predictor.predict(
            batsman=match_state["batsman"],
            bowler=match_state["bowler"],
            over_num=match_state["over_num"],
            score_before=match_state["score_before"],
            wkts_down=match_state["wkts_down"],
            balls_faced_by_batsman=match_state["balls_faced_by_batsman"],
            pitch_type=match_state["pitch_type"],
        )

        output = {
            "over": over_num,
            "batsman": current_batsman,
            "bowler": bowler,
            "score_before_over": score,
            "wickets_down": wkts,
            "prediction": result,
        }
        with open(OUTPUT_FILE, "w") as f:
            json.dump(output, f, indent=2)

        print(f"--- Over {over_num} about to be bowled: {current_batsman} facing {bowler} ---")
        print(f"  Prediction: {result['expected_runs']} runs (range {result['expected_range']}), "
              f"{result['wicket_probability']}% wicket chance")
        for line in result["commentary_lines"]:
            print(f"  > {line}")

        # simulate the over actually happening (this would come from the live API)
        runs_this_over = random.randint(2, 14)
        wicket_fell = random.random() < (result["wicket_probability"] / 100)
        score += runs_this_over
        balls_faced += 6
        print(f"  [Simulated actual result: {runs_this_over} runs"
              f"{', WICKET!' if wicket_fell else ''}]\n")

        if wicket_fell:
            wkts += 1
            balls_faced = 0
            current_batsman = random.choice([b for b in batsmen_pool if b != current_batsman])
            if wkts >= 6:
                print("Innings folds. Match simulation ending.")
                break

        time.sleep(poll_seconds)

    print(f"\nDone. Last prediction written to {OUTPUT_FILE}")


# ---------------------------------------------------------------------------
# LIVE MODE — real API polling (CricketData.org / CricAPI-style)
# ---------------------------------------------------------------------------
def fetch_raw_match_data(api_key, match_id):
    """
    CricketData.org's match_info endpoint includes live batsmen/bowlers for
    in-progress matches (match_scorecard does not - that's for completed
    scorecards only). Adjust if you're using a different provider.
    """
    url = "https://api.cricapi.com/v1/match_info"
    params = {"apikey": api_key, "id": match_id}
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def parse_api_response(raw_json):
    """
    !! ADAPT THIS FUNCTION FIRST !!
    Run once with --live --debug to print the raw JSON, confirm the actual
    field names/paths for your provider/plan, then fill these in.
    This is a best-effort placeholder structure.
    """
    try:
        data = raw_json["data"]
        current_batsman = data["batsman"][0]["batsman"]["name"]
        current_bowler = data["bowler"][0]["bowler"]["name"]
        over_num = int(float(data["score"][-1]["o"])) + 1
        score_before = data["score"][-1]["r"]
        wkts_down = data["score"][-1]["w"]
        balls_faced = data["batsman"][0].get("balls", 0)
        return {
            "batsman": current_batsman,
            "bowler": current_bowler,
            "over_num": over_num,
            "score_before": score_before,
            "wkts_down": wkts_down,
            "balls_faced_by_batsman": balls_faced,
        }
    except (KeyError, IndexError) as e:
        print(f"Could not parse response with expected schema: {e}")
        print("Raw response:", json.dumps(raw_json, indent=2)[:1000])
        return None


def run_live_feed(api_key, match_id, poll_seconds=30, debug=False, pitch_type="balanced"):
    print(f"Running in LIVE mode — polling every {poll_seconds}s. Match ID: {match_id}\n")
    predictor = NextOverPredictor()
    last_over_seen = None

    while True:
        try:
            raw = fetch_raw_match_data(api_key, match_id)
            if debug:
                print(json.dumps(raw, indent=2))
                print("\n--- Debug mode: stopping after one fetch. Copy everything above and send it back. ---")
                return

            state = parse_api_response(raw)
            if state is None:
                print("Skipping this poll — fix parse_api_response() field names, see printed JSON above.")
                time.sleep(poll_seconds)
                continue

            if state["over_num"] == last_over_seen:
                time.sleep(poll_seconds)
                continue
            last_over_seen = state["over_num"]

            # map real player names to known names, or fall back gracefully
            batsman = state["batsman"] if state["batsman"] in BATSMEN else list(BATSMEN.keys())[0]
            bowler = state["bowler"] if state["bowler"] in BOWLERS else list(BOWLERS.keys())[0]

            result = predictor.predict(
                batsman=batsman,
                bowler=bowler,
                over_num=state["over_num"],
                score_before=state["score_before"],
                wkts_down=state["wkts_down"],
                balls_faced_by_batsman=state["balls_faced_by_batsman"],
                pitch_type=pitch_type,
            )

            output = {"over": state["over_num"], **state, "prediction": result}
            with open(OUTPUT_FILE, "w") as f:
                json.dump(output, f, indent=2)

            print(f"Over {state['over_num']}: {result['expected_runs']} runs expected "
                  f"({result['expected_range']}), {result['wicket_probability']}% wicket chance")

        except requests.RequestException as e:
            print(f"API request failed: {e}")

        time.sleep(poll_seconds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", help="Run simulated match, no API key needed")
    parser.add_argument("--live", action="store_true", help="Poll a real live API")
    parser.add_argument("--api-key", default=os.environ.get("CRICKET_API_KEY"))
    parser.add_argument("--match-id", default=None)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.mock:
        run_mock_feed(poll_seconds=3)
    elif args.live:
        if not args.api_key or not args.match_id:
            print("--live mode needs --api-key and --match-id (or set CRICKET_API_KEY env var)")
        else:
            run_live_feed(args.api_key, args.match_id, args.poll_seconds, args.debug)
    else:
        print("Specify --mock (test locally now) or --live (real API, needs key + match id)")
