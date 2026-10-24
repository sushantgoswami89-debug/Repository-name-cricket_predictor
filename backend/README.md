# CricketBaba Backend

AI-powered cricket prediction backend.

## Setup

Create a virtual environment:

```bash
python3.12 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run tests:

```bash
pytest
```

Current Python Version:

- Python 3.12+

Current Module:

- CB-002 Match Context Engine

## Live matches

One driver only: `run_live_auto.py` (Cricbuzz; TOI only with `--toi-backup`). In production it runs as the
`com.cricketbaba.driver` LaunchAgent (copy of the plist in `../ops/`), which restarts it automatically:

```bash
python run_live_auto.py --shadow --facts-telegram --poll-seconds 5
```

Shadow mode logs forecasts and prints them to the per-match console (`../data/live/console-<id>.log`); only
Fact-o-Meter cards go to Telegram. Never run a second copy: two drivers on one match write the same state files.
The older drivers (`run_toi_live.py`, `run_toi_auto.py`, `run_cricbuzz_auto.py`, `run_cricbuzz_match.py`) were
deleted -- see `../docs/outdated_drivers.md`.
