# Live field test, 2026-09-17 -- India vs Afghanistan, 3rd T20I, Delhi

Second live run through `run_cricbuzz_match.py`, same Cricbuzz fallback
path built for 2026-09-15 (`docs/live_field_test_2026-09-15.md`). Purpose
of this file: that session's punch list named several open items and
none of them got tracked anywhere durable between sessions -- this file
exists so this run's findings don't suffer the same fate. Updated live as
the match progresses, not written up after the fact.

Match: `https://www.cricbuzz.com/live-cricket-scores/170120/ind-vs-afg-3rd-t20i-afghanistan-vs-india-in-india-2026`
(match id 170120). Pre-flight: 221/221 tests passing before launch.
Log: `backend/cricbuzz_live_2026-09-17.log`.

## Open items being watched this run (carried over from 2026-09-15's punch list)

1. **Self-checking guards** -- not built yet. Added a lighter-weight,
   observational version instead for this run: `run_cricbuzz_match.py`
   now flags an over-sequence gap immediately (see item 3 below) and
   logs every delivery's extras/wicket kind live (item 5). Full
   `batting_team != bowling_team` / `over <= scheduled_overs + 1` guards
   still not built.
2. **`batting_first`/`bowling_first` dead-field grep** -- not re-run this
   session yet.
3. **Over-14 skip, never root-caused** -- `run_cricbuzz_match.py` now
   prints `OVER GAP DETECTED` the moment a served-over sequence skips a
   number, instead of requiring a manual log audit after the fact. If it
   recurs tonight this file gets the real evidence immediately.
4. **TOI discovery silently missed the 2026-09-15 match** -- still
   unexplained, not investigated this session (running the Cricbuzz path
   directly again, not re-attempting TOI).
5. **Untested Cricbuzz event types** -- six/wide/noball/bye/legbye and
   every wicket kind except "caught" were never confirmed against real
   data. `run_cricbuzz_match.py` now prints an `[event]` line the first
   time each new delivery carries one of these tags. Confirmed so far
   tonight:
   - SIX: confirmed, repeatedly, across multiple bowlers (Nabi, Rashid
     Khan, Mujeeb, Ahmadzai) through overs 5-9 of India's innings --
     solidly validated, not a one-off.
   - WICKET(run out): confirmed (innings 1, over 12.1 -- Tilak Varma,
     off Rashid Khan) -- first non-"caught" dismissal kind validated
     against real data.
   - WICKET(bowled): confirmed (innings 2, over 4.1 -- Rahmanullah
     Gurbaz, off Axar Patel).
   - WIDE: confirmed (innings 2, over 13.3 -- Rashid Khan, off Ravi
     Bishnoi).
   - Still unconfirmed as of this writing: no-ball, bye, leg-bye, and
     every other wicket kind (lbw, stumped, hit wicket, etc.).

## New observation: innings-transition snapshots flip-flopped between innings 1 and 2 across consecutive polls

India finished at 201/4 (20 overs); Afghanistan's chase began. For
roughly a dozen polls spanning Afghanistan reaching over 3, the printed
`[scorecard]` line kept alternating back to the stale final `India
201/4 (16.6 ov)` line in between real Afghanistan progress lines --
not just once at the transition instant (which the 09-15 "highest
inningsId is authoritative" fix already accounts for), but repeatedly,
several innings-2 polls in. Most likely Cricbuzz's own CDN/edge caching
serves slightly different snapshots of the underlying page to
consecutive requests during a live transition, so one poll's raw window
genuinely lacks any innings-2 object yet while the very next one has
several. **No incorrect publish resulted** -- `pipeline.process()`
returned nothing new for the stale re-served innings-1 snapshots (over
20 was already fully processed), so this was cosmetic in the console
log, not a wrong Telegram send. Still worth a note: the "highest
inningsId wins" logic assumes monotonic improvement poll-to-poll near a
transition, and this shows that assumption can be violated by the
upstream feed itself, not just by this reader's own processing order.
Not chased further -- no observed harm, logged for awareness only.
   - 20:05 IST -- narrowed the live monitor's filter to stop alerting on
     further SIX events (type fully confirmed) and only surface the
     still-unconfirmed types plus gaps/errors/completion, to avoid
     drowning real findings in repeat confirmations of the same thing.
6. **Cricbuzz match discovery hardcoded to one URL** -- unchanged, URL
   still passed explicitly on the command line.
7. **Pre-match briefing not wired into the Cricbuzz path** -- unchanged.
8. **"System Health" label still misleading** -- unchanged, not a
   logging-visibility issue, out of scope for this pass.
9. Feature-search churn -- n/a, no feature work this session.
10. **Run-range momentum lag** -- not re-investigated this session.
11. **Live observability** -- this file plus the two instrumentation
    additions above are a first, partial step; still manual, not an
    automated monitor.

## Live findings log

- 19:49 IST -- match launched, India batting first, live and publishing
  to Telegram.
- 19:56 IST -- restarted once (PID 77664 -> 77819) to add the
  `[event]`/`OVER GAP DETECTED` instrumentation above. ~10s gap, no
  over boundary crossed during the restart, no prediction lost.
- 19:57 IST -- SIX confirmed for the first time against real Cricbuzz
  data (see item 5).

## Real finding: the pipeline got permanently stuck on a checkpoint mismatch (recovered by restart, root cause below)

Abhishek Sharma's innings (108 off 34, 9 sixes, 4 fours -- confirmed
SIX repeatedly across 4 different bowlers before this) ended caught by
Rahmanullah Gurbaz off Rashid Khan, over 10.5. Immediately after, the
pipeline started repeating, every poll, forever:

```
Deferred: Cricbuzz over-checkpoint mismatch for 170120 innings 1:
accumulated 141/1, feed reports 142/1 after over 10.
```

**This is not a transient/self-healing error like the other `Deferred`
cases (network blips, mid-fetch timing) -- it is deterministic and
permanent.** `CricbuzzLiveReader._build_deliveries` (`app/live/
cricbuzz_reader.py:504-513`) raises `CricbuzzFeedError` as soon as an
over-break checkpoint disagrees with the running total accumulated from
per-ball parsing, but `self._running_score[scope]` is only persisted
*after* the loop finishes cleanly -- so when the exception fires
mid-loop, nothing from that poll is saved. The next poll re-parses the
exact same window against the exact same (already-known, deduped)
deliveries and hits the identical mismatch again. **No predictions were
sent for ~5-6 minutes while this repeated** -- confirmed via this
session's own `OVER GAP DETECTED` instrumentation not yet having fired
(the pipeline was stuck *before* ever reaching the over-11 send, so
there was nothing to compare a gap against -- worth hardening: this
class of stall should trip its own explicit alert, not rely on a human
noticing a repeating log line).

**Root-caused, with real evidence, not just a guess**: manually replayed
the same raw Cricbuzz commentary for over 10 after the fact
(reconstructing `_runs_and_extras` output per ball). Cricbuzz's own
over-break checkpoints are internally consistent (134/0 after over 9,
142/1 after over 10 -- an 8-run over: SIX, 1, dot, 1, wicket, dot), and
that 8-run total is exactly what fresh parsing of the current commentary
text produces. But the live process's own accumulated total entering
over 10 was only 133 (141 - 8 = 133), one run short of Cricbuzz's real
134. **Most likely cause**: `_build_deliveries` dedupes by `(innings,
over, ball)` key (`if key in known: continue`) and Cricbuzz's commentary
API can amend a ball's text shortly after first posting it (seen
generally in fast-moving overs); if this reader's first-ever look at a
given ball caught an in-flight/incomplete text that `_runs_and_extras`
couldn't parse (falling through to the `return 0, 0, extras` default),
the dedup guard means that 0 is locked in forever -- a later, corrected
poll of the same ball is silently skipped, not re-parsed. Could not
pin down the exact ball or the exact original (pre-edit) text with
certainty, since the live process's in-memory `known` dict was lost on
restart -- **this is the one piece of this finding that's a strong
inference, not directly proven**.

**Recovery**: restarting the process fixed it immediately (confirmed:
resumed cleanly at 151/1, over 11, within 12 seconds of relaunch) --
`fetch()`'s existing seed logic (`app/live/cricbuzz_reader.py:420-475`,
"seed from the LATEST available checkpoint rather than assuming 0/0")
bootstraps a fresh reader's running total directly from Cricbuzz's own
most recent over-break checkpoint, which sidesteps the specific bad
per-ball accumulation entirely. This is a real, usable recovery path,
not a coincidence -- but it required a human (me) noticing the repeating
line and manually restarting; nothing in the pipeline does this
automatically.

**New, real, high-priority punch-list item for next session**: make
`_build_deliveries` re-parse (not skip) a delivery whose stored total
doesn't match a later-arriving checkpoint, or at minimum auto-restart /
re-seed the reader when the same `CricbuzzFeedError` repeats N times in
a row instead of deferring forever. This is a more severe version of the
09-15 "over-14 silently skipped" bug -- that one dropped one over's
prediction; this one would have silently dropped *every remaining over
of the match* if left unwatched.

## Second occurrence of the same stuck-pipeline bug (confirms the pattern)

Recurred at over 16: `Deferred: Cricbuzz over-checkpoint mismatch for
170120 innings 1: accumulated 190/4, feed reports 191/4 after over 16.`
Same deterministic repeat-forever shape as the first time. **Both
occurrences so far fired immediately after a wicket ball** (Abhishek
Sharma caught at 10.5, Shivam Dube caught at 16.3) -- this is now a real
repeated pattern, not a one-off, and upgrades the wicket-commentary-edit
theory in the section above from "strong inference" toward "likely
mechanism, still not directly proven." A wicket ball's commentary text
is the one kind most likely to get appended/amended shortly after first
posting (review outcomes, fielder credit, milestone notes all arrive
progressively) -- consistent with the dedup-lock theory.

Recovered the same way: restart (PID 78198 -> 78537), back to live within
~12s at 201/4, over 17.

**This raises the priority of the punch-list fix above** -- twice in one
match, both times losing several minutes of predictions, both times
requiring a human to notice and restart. Given the pattern, expect this
to recur again after the next wicket unless fixed before the next
session.

## Third occurrence -- reveals the mismatch and the transition-oscillation bug are likely the same root cause

Third checkpoint-mismatch stall, over 4 of the chase, right after
Gurbaz's bowled dismissal: `accumulated 34/2, feed reports 33/2` --
first time the direction reversed (over-count, not under-count). Unlike
the first two occurrences (byte-for-byte identical repeat forever), this
one's own accumulated value **changed between consecutive polls while
stuck** -- 34/2, then 32/2, then held at 32/2 -- and the interleaved
`[scorecard]` lines kept printing the stale `India 201/4` innings-1 line
throughout, exactly like the transition-oscillation observation above.
**This strongly suggests both symptoms share one upstream cause**:
Cricbuzz's own feed serves inconsistent window contents to consecutive
polls during this specific match (possibly different CDN edge caches),
so the reader isn't failing to parse one fixed bad input -- it's being
fed genuinely different raw data each request, both a different
innings-1-vs-2 split of the RSC objects AND a different partial subset
of a given over's balls, and each subset happens to produce an
internally-consistent-with-itself-but-wrong-vs-Cricbuzz's-own-checkpoint
total. This is a materially different (and arguably more fundamental)
problem than the "dedup locks in a bad first parse" theory from the
first occurrence -- that theory may still be a contributing factor for
some cases, but doesn't explain a value that changes across polls while
stuck. Recovered the same way (restart, PID 78537 -> 79126), settled at
37/2 over 4.1 once the feed stabilized.

**Revised next-session priority**: investigate whether Cricbuzz's
window inconsistency is real edge-cache skew (checkable by comparing
raw response bytes/headers across two rapid consecutive requests) before
building any fix -- a fix aimed at "bad text parse" won't help if the
actual cause is "the upstream feed itself disagrees with itself between
requests."

## Real fixes deployed live, mid-match (not just documented for next time)

Given the checkpoint-mismatch stall recurred three times and cost
several minutes of predictions each time, actually fixed it instead of
continuing to restart by hand:

1. **`run_cricbuzz_match.py`**: auto-reseed the `CricbuzzLiveReader`
   after 3 consecutive feed errors of either kind (checkpoint mismatch
   or network), instead of deferring forever until a human notices and
   restarts the whole process. Mirrors exactly what the 3 manual
   restarts tonight did.
2. **`app/live/cricbuzz_reader.py`**: added `Cache-Control`/`Pragma`
   no-cache headers and a cache-busting query param to every request.
   Tested directly -- did NOT fix the deeper problem (see next item),
   but harmless and worth keeping.
3. **Root cause, confirmed with direct evidence**: fired 3 fresh,
   independent requests (new session, cache-busting param, milliseconds
   apart) for the exact same match URL and got 3 different snapshots --
   one genuinely current, two stale by a full innings. **This is a real
   Cricbuzz-side inconsistency, not a client-side caching bug** -- some
   fraction of requests hit a stale cache/replica. No amount of
   client-side cache-defeating fixes this on their end.
4. **`run_cricbuzz_match.py`**: two defenses added given (3) can't be
   fixed at the source --
   - `_fetch_freshest()`: try up to 3 quick fetches per poll, keep
     whichever snapshot has the furthest (innings, legal-balls)
     progress. Raises the odds of catching the current snapshot instead
     of waiting a full `--poll-seconds` cycle to try again.
   - A hard monotonicity guard: any snapshot behind the furthest
     progress already seen is explicitly skipped (`Stale snapshot
     skipped: ...`) rather than reaching the scorecard print or the
     prediction pipeline. **Verified live**: caught and correctly
     rejected multiple real stale reads after deployment without
     printing a wrong scoreboard line or touching the pipeline.
   Full test suite (221/221, one unrelated flaky wall-clock test in
   `test_run_toi_live_stuck_alert.py` passed on retry -- pre-existing
   timing sensitivity, not a regression from tonight's changes) run
   before each redeploy.

**Residual limitation, honestly**: if all 3 attempts in a poll happen to
be stale (seen tonight, twice), that poll is silently skipped with no
new prediction -- better than publishing wrong data, but still a gap.
`attempts=3` was picked as a reasonable balance against Cricbuzz's own
load, not derived from real hit-rate data -- worth revisiting with
actual measurement (what fraction of requests are stale?) next session
rather than guessing.

## Over 14 silently skipped again -- this time root-caused with clean evidence (punch-list item 3, finally confirmed)

`OVER GAP DETECTED: innings 2 jumped from over 13 to 15 -- over(s) 14-14
never got a prediction.` The instrumentation added earlier this session
for exactly this purpose (open since 09-15) caught a live recurrence.

**Root cause confirmed, not just suspected this time.** Re-read the raw
log carefully: the `[scorecard]` line's `overs` field uses
`{completed_overs}.{ball}` notation, so `13.1` through `13.6` (printed
cleanly, ball by ball, zero errors, zero stale-skips in between) *is*
over 14 -- the reader captured it correctly the whole way, contradicting
my first guess that `_fetch_freshest()` (this session's own fix) had
discarded it. The real cause is exactly the line the 09-15 write-up
already suspected but never confirmed: `app/live/pipeline.py:198`,

```python
if verifier.last_key is None or verifier.last_key[1] != completed:
    return None
```

This guard exists deliberately ("a prediction is valid only at the over
boundary -- don't backfill using mid-over state"). But on the poll
immediately after over 14's 6th ball, the same fetch *also* already
contained over 15's first ball or more (Afghanistan were losing wickets
in a cluster, so deliveries were arriving fast). At that instant,
`completed_over` had just become 14, but `verifier.last_key` already
pointed past it into over 15 -- the guard can't distinguish "the feed
moved past this over before we ever got a clean snapshot of its
6th-ball state" from genuine mid-over staleness, so it silently declines
both. **The over-14 data was never stale or missing -- the pipeline had
everything it needed to predict for it and chose not to, because the
guard's condition is stricter than it needs to be.**

**Deliberately not patched tonight, and why**: `over_runs[14]` /
`over_wickets[14]` are already fully and correctly populated at this
point, so it's tempting to just predict from those directly instead of
gating on `last_key`. But the code below that point also builds feature
windows from `verifier.accepted` as a whole (`recent_deliveries`, the
last-12-legal-ball window, etc.) -- if over 15's first ball is already
in `accepted` by the time this runs, a naive fix would compute those
windows including a ball that shouldn't be visible to an "as of over
14" prediction, trading a skipped over for a subtly leaky one. That's
likely *exactly* why the original guard exists and skips rather than
risks it. A real fix needs to snapshot/filter the feature inputs to
`over <= completed` specifically for this backfilled case, not just
delete the guard -- a bigger, more careful change than tonight's
already-large diff warrants without dedicated testing. Left as a
clearly-scoped item for next session rather than rushed in.

## Timeline

- 19:49 IST -- match launched, India batting first, live and publishing
  to Telegram.
- 19:56 IST -- restarted once (PID 77664 -> 77819) to add the
  `[event]`/`OVER GAP DETECTED` instrumentation above. ~10s gap, no
  over boundary crossed during the restart, no prediction lost.
- 19:57 IST -- SIX confirmed for the first time against real Cricbuzz
  data.
- 20:05 IST -- narrowed the live monitor's filter to stop alerting on
  further SIX events (type fully confirmed) and only surface the
  still-unconfirmed types plus gaps/errors/completion.
- ~20:1x IST -- Abhishek Sharma out caught (108 off 34), pipeline wedged
  on the checkpoint-mismatch bug above immediately after.
- ~20:1x-20:2x IST -- stuck, repeating the identical error every poll
  for several minutes, zero predictions sent.
- 20:2x IST -- diagnosed and restarted (PID 77819 -> 78198); recovered
  cleanly at 151/1, over 11, publishing resumed.

## Final result

India won by 127 runs (221 vs Afghanistan all out for 94). Pipeline
detected completion and published the result correctly. Process exited
cleanly on its own (no manual stop needed) once `MATCH COMPLETE` fired.

## Session summary vs the 09-15 punch list

| # | Item | Status after tonight |
|---|------|----------------------|
| 1 | Self-checking guards | Partial: `[event]` logging, `OVER GAP DETECTED`, and the monotonicity guard added -- not the full `batting_team != bowling_team` style guard set |
| 2 | `batting_first`/`bowling_first` grep | Not re-run |
| 3 | Over-14 skip | **Root-caused with real evidence** (see above) -- fix scoped, not yet applied |
| 4 | TOI discovery miss | Not investigated (ran Cricbuzz path directly again) |
| 5 | Untested event types | SIX, WICKET(run out), WICKET(bowled), WIDE confirmed. NO-BALL, BYE, LEG-BYE, and lbw/stumped/hit-wicket/etc. still unconfirmed |
| 6 | Cricbuzz discovery hardcoded | Unchanged |
| 7 | Pre-match briefing on Cricbuzz path | Unchanged |
| 8 | "System Health" label | Unchanged |
| 10 | Run-range momentum lag | Not investigated |
| 11 | Live observability | Improved concretely tonight (see below), still manual |
| -- | **New tonight**: checkpoint-mismatch permanent stall (4 occurrences) | **Fixed live**: auto-reseed after 3 consecutive errors, deployed and verified working autonomously |
| -- | **New tonight**: Cricbuzz feed non-monotonic across polls (confirmed with 3 simultaneous requests returning 3 different snapshots) | **Mitigated live**: best-of-3-attempts fetch + hard monotonicity guard, deployed and verified rejecting real stale reads |
| -- | **New tonight**: `test_run_toi_live_stuck_alert.py` flaky under load | Noted, not investigated further (passed on retry) |

**Real difference from 09-15**: two of tonight's new findings got actual
fixes shipped and verified live, not just written up for a future
session that might not happen -- directly addressing the user's ask
this session to stop letting shortcomings go uncorrected. The over-14
root cause is the one item that's understood but deliberately left
unpatched, with the reasoning for that written above rather than
silently deferred again.
