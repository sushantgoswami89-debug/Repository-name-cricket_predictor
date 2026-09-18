# Checklist for the next real live match (3rd field test)

Compiled 2026-09-18, after closing all 11 items from the 09-15 punch
list and building the Cricbuzz-primary/TOI-backup unified driver. Next
real opportunity: Asian Games men's T20I, starting 2026-09-24 (confirmed
via TOI's own schedule — Afghanistan vs Japan, match 1, listed there
already). Log this field test the same way as the first two
(`docs/live_field_test_2026-09-15.md`, `docs/live_field_test_2026-09-17.md`)
when it happens.

## A. First real end-to-end test of today's new work

None of this has been run against a genuine live match yet — only
against replays, real-but-static page fetches, and unit tests with fakes.

1. **Run `run_live_auto.py` as the primary driver**, not
   `run_cricbuzz_match.py`/`run_toi_auto.py` directly — this is the
   whole point of today's build. Confirm:
   - Cricbuzz actually claims the match first (expected — it covered
     both prior real matches TOI missed once).
   - If TOI's own discovery ALSO lists the match, confirm the backup
     path correctly does NOT double-attach it (dedup by team-pair).
   - If Cricbuzz somehow misses this one too, confirm TOI backup
     attaches it automatically, with no manual intervention.
2. **Pre-match briefing** fires and is readable (toss/venue/confirmed
   XI counts) — both the auto-discovery path and, if used standalone,
   `run_cricbuzz_match.py`'s.
3. **Self-checking guards** (`MatchStateAnomaly`) — ideally never fire.
   If one does, that's a real, currently-live bug class working exactly
   as intended (catching something automatically instead of a human
   noticing wrong output) — treat any firing as a priority follow-up,
   not noise.
4. **Stuck-feed Telegram alert** — ideally never fires. If Cricbuzz's
   known feed-inconsistency problem (confirmed real on 09-17, still
   plausible) recurs badly enough to trigger it, confirm exactly one
   alert arrives and it stops repeating.
5. **Over-14-skip fix** — confirm over-boundary predictions are served
   even if a later over's first ball arrives in the same poll (watch for
   "OVER GAP DETECTED" — should not appear anymore; if it does, the fix
   has a real gap, treat as a priority bug).
6. **"Confidence" label** (renamed from "System Health") — just confirm
   it reads correctly in the live Telegram messages.
7. **Commentary log** (`app/live/commentary_log.py`, wired into every
   driver — `run_cricbuzz_match.py`, `run_cricbuzz_auto.py`,
   `run_toi_auto.py`, and both paths of `run_live_auto.py`) — confirm
   `data/live/commentary-<match_id>.jsonl` is actually being written
   during the match (one JSON line per delivery, every ball, not just
   tagged events). This is the detail-analysis tool for section B below:
   after the match, grep this file for the still-unconfirmed event types
   instead of relying on memory of what scrolled past in the console.

## A.5 Read the full commentary log as a human, not just as data

User-requested 2026-09-18: after the match, actually READ
`data/live/commentary-<match_id>.jsonl` end-to-end (Claude reading the
prose directly, not a script classifying tags) and form a genuine
narrative understanding of the match -- momentum, pressure, standout
performances, turning points, what a piece of commentary text actually
*means* beyond its structured fields. Demonstrated in this session on a
real 3-ball tail-end (Australia vs Zimbabwe, 2nd ODI, 2026-09-18):
the structured data said "wicket_kind=caught, wicket_kind=bowled, margin
84 runs"; reading the actual prose instead surfaced that the dismissed
batter was the recognized/set man playing a battling (not fluent)
innings who got out attempting to force a boundary under pressure, that
the very next ball ended the innings outright, and that the commentary's
own framing ("untouchable today," "take the game and the series") meant
this was a series-deciding rout, not a nervy finish -- none of which the
per-ball structured fields alone convey.

This is a genuine gap between parsing correctness (sections A.7/B below)
and match *understanding* -- do both, they check different things:
parsing correctness asks "did we record this ball right," a human read
of the full log asks "do we understand what actually happened in this
match, in the way a person watching it would." Worth doing for its own
sake (a real, readable match summary) and as a sanity check on whether
the model's own predictions (win-probability trajectory, run-range
hits/misses) told the same story the real match did, or missed a
dramatic swing / overstated a close finish that a human reading would
have caught immediately.

## A.6 Confirm shot-quality/near-wicket detection live, and log the correlation

`app/live/shot_quality.py` (built 2026-09-18, directly from a user
example: an edge for four scores identically to a well-timed four in
every structured field) flags `edge`/`beaten`/`well_timed` shot quality
and `dropped_catch`/`survived_close_call` near-wicket events, live, in
both the console (a `⚠️` line) and the commentary log. `edge`/`beaten`
were verified against real commentary text; `well_timed`/
`dropped_catch`/`survived_close_call` are standard broadcast vocabulary
not yet confirmed against a real live instance -- check they fire
correctly (or add a real counter-example if the wording turns out
different in practice).

More importantly: this is the start of the accumulation process
discussed with the user (same "how many matches" answer already given --
~5 real matches, the same bar the live ensemble shadows use). For every
`edge`/`beaten`/`dropped_catch`/`survived_close_call` flag that fires,
note in the field-test log whether a real wicket actually followed
within the next few balls. That is the raw material for eventually
testing, honestly, whether these prose-only signals predict outcomes
the structured features miss -- not yet enough data to conclude
anything from a single match.

## B. Event types still unconfirmed against real data — check the commentary log for these

From `docs/finding_cricbuzz_event_types_2026-09-18.md` — grep
`data/live/commentary-<match_id>.jsonl` (see A.7) for these specifically
after the match, not just the console's `[event]` lines (which only
print for balls that already tripped a *known* tag — the commentary log
covers every ball, so it can also catch a genuinely new, unanticipated
parsing case the known-tag list doesn't cover):

- `byes` (plain, not leg-byes) — same code path as leg-byes (already
  fixed), high confidence, not yet directly observed.
- Wicket kinds `lbw`, `stumped`, `hit wicket`, `caught and bowled` — none
  observed yet. If one occurs, check the printed `WICKET(kind)` tag
  matches the real dismissal.
- Also worth a final confirmation of a **multi-run** wide/no-ball/bye
  specifically (2+ runs) — the exact bug class fixed today
  (`_leading_run_count`'s comma parsing). A single-run case won't
  exercise the fix.

## C. Recurring bugs from the first two field tests — confirm actually fixed

- **Checkpoint-mismatch / stale-snapshot stalls** (real Cricbuzz feed
  inconsistency, confirmed 09-17): auto-reseed and the monotonicity
  guard should keep handling this automatically. If it happens
  repeatedly, the stuck-alert (A.4) should now surface it instead of
  requiring a human to notice.
- **Team-swap bug class** (batting/bowling team identity, found twice
  historically in two different files): the self-checking guard (A.3)
  should catch any recurrence immediately if it ever comes back a third
  time.
- **TOI discovery miss**: since this match IS already on TOI's schedule
  ahead of time (confirmed today), this is actually a good test of
  whether TOI's LIVE discovery correctly picks it up once play starts —
  a real chance to observe whether the 09-15 miss was a one-off or
  systemic. Either way it doesn't block prediction (Cricbuzz primary),
  but worth noting for the standing "why does TOI miss things"
  question, even though direct forensic root-causing is a confirmed
  dead end.

## D. Ensemble shadows — this match counts toward their review threshold

Four live shadows have been built, validated on holdout, and waiting on
real match data since 2026-08-26: win-probability Monte Carlo, wicket
GBM+NN ensemble (`wicket_v19_nn_gbm_ensemble`), run-range GBM+NN ensemble
(`run_range_v21_nn_gbm_ensemble`), match-winner GBM+NN ensemble. None has
ever been reviewed because no live match has occurred since they were
built. This match should count toward each one's 5-match review
threshold — check the shadow logs
(`app/ml/*_ensemble_shadow_log.py`-backed files) after the match and see
how close each one is to enough real observations to actually evaluate
promotion, not just holdout numbers.

## E. Team/data-quality watch

- Afghanistan vs Japan (and any other Asian Games associate-nation
  matchup): Japan almost certainly has sparse-to-nonexistent Cricsheet
  history — a real test of the cold-start path (unresolved player IDs,
  shrinkage-to-pooled-average behavior) under genuinely new conditions,
  not just a historical replay. Worth explicitly checking a Japan
  batter's live features resolve to sane defaults, not garbage.
- Confirm `EngineRouter.resolve()` accepts whatever match_format string
  the feed reports for an Asian Games T20I (should just be "T20"/"T20I",
  but worth a quick check before assuming).

## Standing reminder

Per feedback memory: validate everything against the ACTUAL current
production baseline in the moment, not memory of what these numbers were
when each feature was tested — models may have moved on by 09-24, and
the individual-feature rejections this session (momentum window,
batting position, handedness matchup — commits `9c708b5`, `f1a34b7`,
`a1e4736`) mean nothing about the live model has changed since today's
`v11`/`v16`/`v21`/`v19` state, but always re-confirm the "currently live"
model names via `app/ml/prediction_engine.py`'s `RUN_RANGE_V3_ARTIFACTS`/
`WICKET_CONTRACT22_ARTIFACTS` before assuming.
