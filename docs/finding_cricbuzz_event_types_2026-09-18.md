# Finding: Cricbuzz event-type validation, 2026-09-18 — real bug found and fixed, partial coverage remains

**Punch-list item 5** (`docs/live_field_test_2026-09-15.md`): validate the
Cricbuzz reader's untested event types (six/wide/no-ball/bye, every
wicket kind except "caught") against real archived matches. Previously
confirmed live 2026-09-17: SIX, WIDE, WICKET(run out), WICKET(bowled).

## Method, and a real dead end

Tried to validate against a real *completed* match's full ball-by-ball
history first. Confirmed this doesn't work: even a finished match's
Cricbuzz page (and its `/live-cricket-full-commentary/...` and
`/api/cricket-match/commentary/...` variants) only ever carries a small
rolling window of recent balls — the same limitation
`app/live/cricbuzz_reader.py`'s own module docstring already documents
for *live* matches. Cricbuzz has no retrievable full ball-by-ball
archive for a finished match, the same kind of dead end found earlier
this session investigating TOI's own history.

Fallback: `scan_cricbuzz_event_types.py` samples many different
completed matches' tail-end windows instead of one full match, to raise
the odds of catching each rare event type somewhere.

## Real bug found and fixed

Scanning turned up a real, previously-undetected correctness bug, not
just a presence/absence gap. A real leg-bye delivery
("`Josh Bohannon to S Northeast, leg byes, 2 runs`") parsed as 1 run, not
2. Root cause: `_leading_run_count()`'s regex expected the run count to
start immediately (`\s*(\d+)\s*runs?\b`), but Cricbuzz's real phrasing
always puts a comma before it (`, 2 runs`) — the comma broke the match on
every extra worth more than one run, silently falling through to the
hardcoded default of 1. This affects wides and no-balls the same way,
not just byes — confirmed with a real no-ball example too, where the
same code path is exercised.

**Fixed**: `_leading_run_count()` now tolerates the leading comma.
Verified against the real leg-bye example (now correctly returns 2) plus
synthetic multi-run wide/no-ball cases. 6 new regression tests in
`tests/test_cricbuzz_reader.py`, using the real observed strings where
available.

## What's now confirmed against real data (as of 2026-09-18)

- `bowled` — real example (`docs/live_field_test_2026-09-17.md` +
  re-confirmed today against a different real delivery).
- `caught` — real example, confirmed today for the first time.
- `noballs` — real example, confirmed today (both the ambiguous-count
  case, which honestly defaults to 1, and a synthetic explicit-count case
  after the fix).
- `legbyes` — real example, confirmed today, and the bug it exposed is
  now fixed.
- `wides`, `WICKET(run out)` — confirmed 2026-09-17 (unchanged).

## What's still unconfirmed against real data

- `byes` (plain, not leg-byes) — same code path as `legbyes`, now fixed
  by the same comma change, but no real "byes" example was found in
  today's sample. High confidence by construction, not directly observed.
- Wicket kinds `lbw`, `stumped`, `hit wicket`, `caught and bowled` — no
  real example found in today's sample (all four are less common events;
  the rolling-window sampling approach only sees a handful of balls per
  match, so a full scan across ~30 matches still isn't guaranteed to hit
  a rare dismissal type). Each relies on the exact same trailing
  scorecard-notation convention already confirmed correct for `bowled`
  and `caught` (`"<b>Batter kind Bowler runs(balls)</b>"`, always
  appended by Cricbuzz after the prose commentary) — high confidence,
  not a direct confirmation.

## Honest conclusion

This is a real, bounded, structural limitation, not a gap in effort: full
confirmation of every remaining event type requires either watching many
full live matches over time (as the actual live drivers already do and
will keep doing) or a full match archive Cricbuzz doesn't expose. Keep
`scan_cricbuzz_event_types.py` for opportunistic future runs (e.g.,
during the next live match, or periodically against whatever's currently
completed) — it costs nothing to re-run and each real event type it
confirms is genuine progress, same as today's.
