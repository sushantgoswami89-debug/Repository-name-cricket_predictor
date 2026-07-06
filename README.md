# Cricket Next-Over Predictor — Prototype

## What's here
- `src/generate_synthetic_data.py` — generates realistic ball-by-ball data (structured exactly like a parsed Cricsheet dataset) to build/test the pipeline
- `src/features.py` — feature engineering: rolling batsman/bowler/matchup stats using ONLY prior information (no leakage)
- `src/train.py` — trains two LightGBM models: runs-in-over (regression) and wicket-probability (classification)
- `src/predict.py` — the live prediction function + commentary-line generator
- `models/` — trained model files (`.pkl`)
- `data/synthetic_overs.csv` — the training data used

## Current results (on synthetic data)
- Runs model: MAE 2.35 runs vs a naive baseline of 3.22 — real signal, not just guessing the mean
- Wicket model: AUC 0.61 — wickets are inherently high-variance events (even pro models rarely exceed 0.65-0.70 AUC), this is a reasonable prototype baseline

## How to use it right now
```bash
python3 src/generate_synthetic_data.py   # regenerate data if needed
python3 src/train.py                     # retrain models
python3 src/predict.py                   # see example predictions
```

To predict any over, use `NextOverPredictor` from `predict.py`:
```python
from predict import NextOverPredictor
p = NextOverPredictor()
result = p.predict(
    batsman="V Kohli", bowler="J Bumrah", over_num=8,
    score_before=62, wkts_down=2, balls_faced_by_batsman=24,
    pitch_type="seaming_track"
)
print(result)
```

## Next step: swap in REAL data
This whole pipeline was built on synthetic data because my sandbox can't reach cricsheet.org directly — but you can, from your own machine:

1. Go to https://cricsheet.org/downloads/ and download match data (JSON format, "all matches" or specific league like IPL)
2. Write a parser that flattens each match's ball-by-ball JSON into the SAME row schema used in `synthetic_overs.csv`:
   `match_id, over, phase, batsman, batsman_style, batsman_class, bowler, bowler_type, bowler_quality, pitch_type, venue_avg_score, balls_faced_before_over, score_before_over, wkts_down_before_over, runs_in_over, wicket_in_over, boundaries_in_over`
3. Replace `data/synthetic_overs.csv` with your real parsed data
4. Re-run `train.py` — nothing else changes

Some fields (batsman_class, bowler_quality, pitch_type) aren't in raw Cricsheet data — you'll need to derive these yourself (e.g. batsman_class from career strike rate/average tiers, pitch_type from historical venue scoring data, which Cricsheet also provides via match results).

## For live use during a broadcast
This is currently a batch/offline pipeline. For live in-match use you'll need:
- A live ball-by-ball feed (paid API — CricAPI, Sportradar, etc.) instead of Cricsheet (which only has completed matches)
- A small always-running script that calls `predictor.predict(...)` before each over using the live match state, and pushes output to a dashboard your commentator can glance at

## Automated live feed (`src/live_feed.py`)

Test it right now with zero setup:
```bash
python3 src/live_feed.py --mock
```
This simulates a real match — before each over it prints the prediction, then shows what "actually happened" (simulated), and writes the latest prediction to `live_prediction.json`. This is the exact loop that will run automatically once connected to a real live API — no manual data entry.

### Going live with a real API
1. Sign up for an API key — recommended for you: **CricketData.org** (cheap, ball-by-ball, India-friendly pricing) or **Roanuz Cricket API** (India-focused, built for exactly broadcast/commentary use cases)
2. Run once with debug mode to see the real response shape:
   ```bash
   python3 src/live_feed.py --live --api-key YOUR_KEY --match-id MATCH_ID --debug
   ```
3. Open `parse_api_response()` in `live_feed.py` and adjust the field names to match what you see printed — every provider's JSON schema differs slightly, and I built this off publicly documented shapes, not a confirmed live response, so this step matters.
4. Once parsing is confirmed, drop `--debug` and let it run — it'll poll automatically (default every 30s, configurable), detect new overs, and keep `live_prediction.json` updated with zero manual input.

### Why not scrape ESPNcricinfo directly
Their terms prohibit automated scraping, and it's a shaky foundation for a commercial radio product — an official API (even a cheap one) is the safer, more reliable path.

## Training on REAL historical data (recommended once you're comfortable with the prototype)

### Step 1: Download real match data
Go to https://cricsheet.org/downloads/ and download a JSON zip for the competition you care about (e.g. "IPL" under T20 matches). Unzip it somewhere on your Mac — you'll get one `.json` file per match.

### Step 2: Convert it to our format
```bash
python3 src/parse_cricsheet.py /path/to/your/unzipped/folder
```
This creates `data/real_overs.csv`.

### Step 3: Retrain on the real data
```bash
python3 src/train.py --data real
```
This overwrites the model files with ones trained on actual match history. Compare the MAE/AUC printed here against the synthetic run — real data is noisier, so don't be alarmed if numbers look slightly different; what matters is it's now learning from real players and real matches.

### Honest limitation to know about
Cricsheet doesn't label things like "batting style" or "pace vs spin" — so the real-data model relies more heavily on each player's own historical numbers (which is actually a more trustworthy signal than broad categories anyway). If you want pace/spin distinction back, you'd need to build a small lookup table yourself (player name → bowling type) and merge it in — happy to help with that once you've got real data flowing.



## The Dashboard (`dashboard.html`)

A clean, auto-refreshing screen your commentator can glance at — shows the matchup, expected runs, wicket probability, and talking points. It updates itself every 5 seconds by reading `live_prediction.json`.

### How to run it
Browsers block web pages from reading local files directly (a security rule), so we need a tiny local web server — this is one command, not a real "server" to worry about:

```bash
cd ~/Downloads/cricket_predictor
/usr/bin/python3 -m http.server 8000
```
Leave that Terminal window running. Then open your browser to:
```
http://localhost:8000/dashboard.html
```

### To see it working right now (with mock data)
Open a **second** Terminal tab/window (Cmd+T) and run:
```bash
cd ~/Downloads/cricket_predictor
/usr/bin/python3 src/live_feed.py --mock
```
Keep both windows running — the dashboard tab will update automatically every few seconds as the mock match progresses.

### For real matches
Once your API quota resets and we've confirmed the real data format, you'll instead run:
```bash
/usr/bin/python3 src/live_feed.py --live --api-key YOUR_KEY --match-id MATCH_ID
```
(no `--debug` once it's working) — leave that running in one tab, and the dashboard tab updates itself automatically the whole match, zero manual input.
