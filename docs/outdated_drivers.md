# Outdated drivers -- record

Record started 2026-10-11 (user request). Cleanup done the same night after the India v West Indies and Pakistan v Sri Lanka
matches finished. The only driver to run live is `backend/run_live_auto.py` (Cricbuzz primary, TOI backup, ODI briefings,
one console file per match). Run exactly one copy: two drivers watching the same match would both write
`data/live/cricbuzz-<id>.state.json`, the commentary / board logs and the prediction log.

## Deleted 2026-10-11

| File | Why it was outdated |
|---|---|
| `backend/run_cricbuzz_auto.py` | Cricbuzz-only auto driver; `run_live_auto.py` does the same plus TOI backup, ODI watcher, sync/resync, Fact-o-Meter, per-match consoles. |
| `backend/run_cricbuzz_match.py` | Single-URL fallback from the 2026-09-15 field test. |
| `backend/run_toi_live.py` + `tests/test_run_toi_live_stuck_alert.py` | Oldest single-match TOI driver; its stuck-feed alert now lives in `cricbuzz_session.py` / `run_live_auto.py`. |
| `backend/run_toi_auto.py` | TOI-only auto driver (it published to Telegram unless `--shadow-t20i`). Its briefing and Fact-o-Meter text builders moved to `backend/app/live/toi_briefing.py`; its shadow-prediction writer had no other user and went with it (test removed). |

## Found while cleaning: the Telegram control bot started the old driver

`backend/telegram_control.py` (LaunchAgent `com.cricketbaba.telegram-control`, always on) launched `run_toi_auto.py` -- in
LIVE mode, publishing predictions to Telegram -- whenever the chat said "match is live". It now launches
`run_live_auto.py --shadow --facts-telegram --poll-seconds 5` (shadow, as agreed), and refuses to start when a
`run_live_auto.py` is already running from a terminal. `data/live/auto-watcher.log` is that bot's watcher log (not stale
by itself), so it was kept.

## Kept (not live drivers)

- `backend/run_commentary_study.py` -- study-only commentary logger, no predictions (`tests/test_commentary_study.py`).
- Offline replay and evaluation tools: `run_batch_replay.py`, `run_full_match_replay.py`, `run_match_card_replay.py`,
  `run_live_pipeline_replay.py`, `run_replay.py`, `run_candidate_v3_complete_match_test.py`, `run_prediction_evaluation.py`.

## Re-checked 2026-10-20 (user: "delete all the outdated drivers")

Nothing left to delete: `backend/run_live_auto.py` is the only live driver; the four old ones were removed on 2026-10-11 and no
process or LaunchAgent starts them (running: `com.cricketbaba.driver` and `com.cricketbaba.telegram-control`, which launches
`run_live_auto.py`). Cleaned the leftovers: `backend/README.md` now documents `run_live_auto.py` instead of `run_toi_live.py`, and
`docs/next_live_field_test_checklist.md` is marked superseded. Code comments that mention the old names are history and were left.
