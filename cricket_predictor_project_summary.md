# Cricket Next-Over Predictor — Project Summary

## What this is
An offline-capable system that predicts what's likely to happen in the next over
of a cricket match (runs, wicket probability) plus projected final total and a
heuristic win probability — built to support live radio cricket commentary.

## Who's using this
Sushant — cricket commentator at a small radio company, India-based, also has
an FMCG management background. Non-technical with command line/Python, so
needs step-by-step Terminal instructions (exact commands, not just concepts).
Uses a Mac. Default Python on his machine is `/usr/bin/python3` (not `python3`
alone — installs done via `/usr/bin/python3 -m pip install ...`).

## Current status (as of last session)
- ✅ Full pipeline built: data → features → LightGBM models → predictions → dashboard
- ✅ Trained on REAL historical data (not just synthetic test data):
  - Downloaded via Cricsheet.org: IPL + T20 Internationals (ODIs deliberately
    excluded — mixing 20-over and 50-over matches distorted the "over phase" logic)
  - ~250,000 real over-level rows, 6,767 matches, 6,185 batsmen, 5,172 bowlers
  - Runs model: MAE 3.18 (vs 3.55 baseline) — modest real improvement
  - Wicket model: AUC 0.615 — real signal, not random
- ✅ Dashboard built (dashboard.html + serve_dashboard.py) — dark, broadcast-booth
  styled, auto-refreshes every 4s, shows expected runs/range, wicket %, commentary
  insight lines, and a "limited data" warning banner for under-covered players
- ✅ Manual entry tool (manual_predict.py) — fully offline, no API/internet needed,
  menu-driven: (1) next over, (2) projected final total, (3) heuristic win probability
- ⏳ Live API automation (live_feed.py) — built but NOT yet confirmed working with
  real live match data. The CricketData.org credential is stored only in the
  ignored, owner-only local configuration (Lifetime Free tier, 100 hits/day).
  Hit daily rate limits during testing before this could be fully verified.
  Endpoint used: `https://api.cricapi.com/v1/match_info` — the parser function
  `parse_api_response()` in live_feed.py is a BEST-EFFORT guess at field names
  and has NOT been confirmed against a real successful live-match response yet.

## Known limitations (be upfront about these, don't overstate accuracy)
- Real player data has no labeled "batting style" or "bowler type" (pace/spin) —
  Cricsheet doesn't provide this. The model leans on each player's own rolling
  historical stats instead, which is reasonable but means those categorical
  features are just "unknown" placeholders in real data.
- New/rare players fall back to generic league averages (graceful, not broken,
  but flagged via the "limited_data" field in prediction output).
- Win probability is a HEURISTIC (required run rate vs wickets in hand formula),
  NOT a trained model. A real one would need match outcome data (who won) as
  training labels — not yet built.
- Projected final total assumes the SAME batsman/bowler continue for all
  remaining overs (simplification — real matches rotate both).
- Model doesn't yet weight recent form more heavily than old matches (all
  history currently weighted equally).

## File locations (in ~/Downloads/cricket_predictor on his Mac)
- `src/generate_synthetic_data.py` — synthetic data for pipeline testing
- `src/parse_cricsheet.py` — converts real Cricsheet JSON into training CSV
  (usage: `python3 src/parse_cricsheet.py /path/to/unzipped/json/folder`,
  appends to existing data/real_overs.csv rather than overwriting)
- `src/features.py` — rolling historical feature engineering (no leakage)
- `src/train.py` — trains LightGBM models (`--data real` or `--data synthetic`)
- `src/predict.py` — NextOverPredictor class: `.predict()`, `.project_total_score()`,
  `.win_probability_heuristic()`. Auto-loads real_overs.csv if it exists, else
  falls back to synthetic_overs.csv (prints which one on load).
- `src/manual_predict.py` — interactive CLI, no internet needed
- `src/live_feed.py` — live API polling (--mock for testing, --live for real API)
- `dashboard.html` + `serve_dashboard.py` — visual dashboard, run server then
  open `http://localhost:8899/dashboard.html`
- `models/` — trained model files (runs_model.pkl, wkt_model.pkl)
- `data/real_overs.csv` — the real training data (IPL + T20I combined)

## Common gotchas already solved (don't repeat these mistakes)
- His Python is `/usr/bin/python3`, and `pip3 install --break-system-packages`
  doesn't work on it — use `/usr/bin/python3 -m pip install PACKAGE` instead
  (no --break-system-packages flag on his pip version)
- lightgbm needs `libomp` on Mac — requires Homebrew (`brew install libomp`)
- Player names must match Cricsheet's exact spelling (e.g. "JJ Bumrah" not
  "J Bumrah") — use pandas to search data/real_overs.csv for exact spellings:
  `df[df['batsman'].str.contains('NAME', case=False, na=False)]['batsman'].unique()`
- He needs the exact zip-and-redownload flow when files change (Finder folder
  downloads have failed before; zipping into one file works reliably)
- When giving Terminal instructions, always give the FULL exact copy-pasteable
  command, never a description of what to type

## Next steps to pick up from
1. Retry live API connection once daily rate limit resets or he upgrades to
   paid CricketData.org plan ($5.99/mo, 2000 hits/day) — need to find a
   genuinely LIVE (in-progress) match and confirm parse_api_response() field
   names against a real response
2. Consider building a real trained win-probability model (needs match outcome
   labels from Cricsheet's info.outcome field — not yet parsed/used)
3. Consider weighting recent matches more heavily than old ones for current form
4. Optionally build a pace/spin lookup table to restore bowler-type distinction
