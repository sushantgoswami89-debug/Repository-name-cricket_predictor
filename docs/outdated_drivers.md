# Outdated drivers -- candidates to delete

Record kept 2026-10-11 (user request). **Nothing has been deleted yet.** The only driver to run live is
`backend/run_live_auto.py` (Cricbuzz primary, TOI backup, ODI briefings, one console file per match).
Run exactly one copy: two drivers watching the same match would both write `data/live/cricbuzz-<id>.state.json`,
the commentary / board logs and the prediction log, and would double-send anything that goes to Telegram.
Checked 2026-10-11 21:50: one `run_live_auto.py` process (plus its caffeinate) was running, nothing else.

## A. Superseded live drivers (safe to delete once the notes below are handled)

| File | Why it is outdated | What still depends on it |
|---|---|---|
| `backend/run_cricbuzz_auto.py` (Sep 18) | Cricbuzz-only auto driver; `run_live_auto.py` does the same plus TOI backup, ODI watcher, sync/resync, Fact-o-Meter, per-match consoles. It has none of the fixes made since 2026-09-18. | Nothing imports it, no test. |
| `backend/run_cricbuzz_match.py` (Sep 18) | Single-URL fallback from the 2026-09-15 field test. Same gaps as above. | Nothing imports it, no test. |
| `backend/run_toi_live.py` (Aug 22) | Oldest single-match TOI driver. | `tests/test_run_toi_live_stuck_alert.py` imports it -- delete the test with it (the stuck alert now lives in `cricbuzz_session.py` / `run_live_auto.py`). |
| `backend/run_toi_auto.py` (Oct 11) | TOI-only auto driver, replaced as a driver by `run_live_auto.py`. **Still imported** for helpers. | `run_live_auto.py` imports `_briefing_text`, `_fact_text`; `tests/test_toi_auto.py` imports `_write_shadow_prediction`. Move those three helpers into an `app/live/` module first, then delete the driver. |

## B. Study / offline tools (not live drivers; keep unless clutter matters)

- `backend/run_commentary_study.py` -- study-only commentary logger, no predictions (`tests/test_commentary_study.py`).
- Replay and evaluation tools, never run live: `run_batch_replay.py`, `run_full_match_replay.py`,
  `run_match_card_replay.py`, `run_live_pipeline_replay.py`, `run_replay.py`,
  `run_candidate_v3_complete_match_test.py`, `run_prediction_evaluation.py` (the last has a test).

## C. Stale files / processes to clean at the same time

- `data/live/auto-watcher.log` -- output of the old TOI-only watcher ("CricketBaba is watching TOI ... No match is live"), contains a
  Telegram `ConnectionError` traceback. Not written by the current driver.
- Terminal tabs from earlier runs: only the current `run_live_auto.py` tab should exist (closed 2026-10-11: c57, c58, c59, c60, c61).
- LaunchAgent `com.cricketbaba.telegram-control.plist` is the Telegram control bot, not a prediction driver -- leave it.

## Suggested order when approved

1. Move the three `run_toi_auto.py` helpers to `app/live/`, point `run_live_auto.py` and `tests/test_toi_auto.py` at them.
2. `git rm` run_cricbuzz_auto.py, run_cricbuzz_match.py, run_toi_live.py, run_toi_auto.py and their tests.
3. Full test suite, then one dry run of `run_live_auto.py --shadow` between matches.
4. Delete `data/live/auto-watcher.log`.
