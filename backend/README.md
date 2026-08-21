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

## Verified TOI live feed (Candidate v3)

Run one read-only verification against a TOI match-center URL:

```bash
python run_toi_live.py --once \
  --model-dir ../models/locked/cricketbaba_candidate_v3 \
  "TOI_MATCH_CENTER_URL"
```

Poll a live match, write verified predictions to `live_prediction.json`, and
publish one idempotent Telegram message per predicted over:

```bash
export TELEGRAM_BOT_TOKEN="..."
export TELEGRAM_CHAT_ID="..."
python run_toi_live.py --telegram \
  --model-dir ../models/locked/cricketbaba_candidate_v3 \
  "TOI_MATCH_CENTER_URL"
```

The process stops on delivery-order, score, wicket, scorecard, or silent feed
revision mismatches. It never publishes a prediction from unverified state.
