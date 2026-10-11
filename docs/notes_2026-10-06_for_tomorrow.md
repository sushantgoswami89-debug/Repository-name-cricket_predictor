# Notes from 2026-10-06 (India vs West Indies, 1st T20I, Lucknow) -- build tomorrow

Status tonight: `run_live_auto.py --shadow` running in the user's Terminal
panel (tab c1). Cricbuzz claimed the match first; briefing printed
correctly. Shadow mode, so nothing was published to Telegram. Review the
commentary log and per-over results after the match, per
`docs/next_live_field_test_checklist.md`.

## Shipped tonight (uncommitted)
- `backend/app/live/prematch_analysis.py` -- `prematch_text()`: H2H, last-5
  form, venue record from raw Cricsheet men's T20Is. Descriptive history,
  explicitly "not a model forecast".
- Wired into both briefings: `run_toi_auto.py::_briefing_text` and
  `app/live/cricbuzz_session.py::briefing_text`. Verified against this
  real match on both paths; 297 tests pass.
- `backend/prematch_analysis.py` -- CLI wrapper for the same text.
- `backend/evaluate_prematch_elo.py` -- Elo backtest (not wired in).

## Requested, NOT built: a real pre-match win predictor in the briefing
User ask: "current winning predictor, based on historic matches, pitch,
weather and other things".

What exists today: nothing model-based before the first ball. The live
match-winner model's H2H snapshot is IPL-only (113 franchise pairings, no
international teams), so for a T20I its H2H input is neutral 0.5.

Elo backtest (K chosen on 2023-24 only, holdout 2025+ untouched, every
prediction made before its own result is applied; K=96 chosen, grid 16-128):
- all men's T20Is: n=830, accuracy 0.704, Brier 0.1995 (coin 0.25)
- both full members: n=141, accuracy 0.638, Brier 0.2343, log-loss 0.6776
  (coin 0.6931) -- barely better than a coin flip
- associate involved: n=689, accuracy 0.717, Brier 0.1924
Read: the headline accuracy is carried by lopsided associate fixtures.
For a full-member matchup like India vs West Indies, plain Elo is only
slightly better than a coin flip. Do not present it as a confident
forecast.

To do tomorrow:
1. Add features beyond Elo and test each against the Elo baseline on the
   same 2025+ holdout, reporting full-member and associate splits
   separately: recent form, shrunk H2H, venue bat-first/chase bias,
   rating of the specific XI where available. Logistic regression, then
   a calibration check.
2. Pitch and weather: Cricsheet has no pitch/weather/dew history, so they
   cannot be trained on. Options: show TOI's pitch note and weather as
   display-only context, or find a historical source. Toss has no signal
   in earlier tests -- do not re-add without new evidence.
3. Only wire into the briefing if it beats the Elo baseline on the
   full-member split; label it with its measured accuracy.

## Data hole found: no Afghanistan in the local T20I data
0 of 5,524 files in `data/raw/cricsheet/t20i` involve Afghanistan (108
distinct teams listed, none Afghan). Consequences: the pre-match block is
correctly skipped for any Afghanistan fixture, and Afghanistan sits at
the 1500 default in Elo. Check whether the project's T20I training
population is missing the same matches (it would affect every "T20I"
number reported so far) and whether a re-download fixes it.

## Operational notes
- Background commands in this environment are killed after roughly
  18-30 minutes even with the longest timeout (two restarts of
  `run_live_auto.py` were killed). For match-length runs use the Terminal
  panel tab, not a background Bash command.
- Both discovery sources returned transient connection resets
  (RemoteDisconnected / ConnectionResetError) tonight; the driver deferred
  the scan and carried on both times.
- TOI pre-match pitch note for Lucknow: "Batting friendly... comparatively
  true batting track"; dew and weather unavailable.

## Match timings (India vs West Indies, 1st T20I, Ekana Stadium, Lucknow)
- Scheduled start: 13:30 UTC = 19:00 IST (TOI start time; Cricbuzz "Preview" until then).
- 17:14 IST: no T20 live on either source; fixture listed as upcoming.
- ~18:34 IST: `run_live_auto.py` attached via Cricbuzz; toss already known
  (India won, chose to bowl), XIs 11/15 confirmed each side.
- 18:57 IST: still no ball-by-ball data; driver printing "Deferred: No
  ball-by-ball state yet" every poll (expected before first ball).
- Exact first ball, innings break and finish times: not recorded live --
  read them from `data/live/commentary-151565.jsonl` (`timestamp_ms`)
  after the match.

## Console-noise issue to fix (cosmetic but misleading)
Before the first ball the driver prints "Deferred: No ball-by-ball state
yet ... innings likely hasn't started" on every poll, and counts each one
as a feed error, so it also prints "Reseeding Cricbuzz reader after 3
consecutive feed errors" every few polls. Nothing is actually wrong, but it
reads as continuous errors. Fix: treat "innings hasn't started" as a quiet
waiting state (print once, no reseed counting). Also seen once: a transient
404 from `cricbuzz.com/live-cricket-scores/151565/x?_cb=...` -- one-off,
recovered on the next poll, but worth knowing the cache-busting URL can 404.

## Fixed live at ~19:01 IST: pre-start error spam and the missing 1st-over prediction
- Before the first ball the Cricbuzz reader raised a generic feed error, so
  (a) every poll printed "Deferred", (b) the session reseeded the reader
  every 3 polls, (c) after 5 min the stuck-feed alert fired -- a false
  alarm that would have gone to Telegram in live mode -- and (d) the
  pipeline never received a "match live, no deliveries yet" snapshot, so
  the pre-innings (over 1) prediction could not fire on the Cricbuzz path
  at all (once ball 1 is accepted the pre-innings condition is false).
- Fix: `CricbuzzInningsNotStarted` (quiet wait, not counted as an error)
  until the scheduled start from the page's JSON-LD `startDate`; after the
  start, `CricbuzzLiveReader` returns an empty first-innings snapshot (batting
  side from the toss) so the pipeline fires the over-1 prediction.
  Verified live: "Recorded innings 1 over 1 prediction" (WI bat first,
  range 3-5, WI win prob 33.5%). Files: `app/live/cricbuzz_reader.py`,
  `app/live/cricbuzz_session.py` (uncommitted). Still to do: add unit tests
  for this path, and apply the same quiet-wait to the innings-2 break.

## Who is bowling / coming in to bat -- what the system has
- Over-1 prediction had bowler=None, bowler_source=None: the Cricbuzz path
  has no announced-bowler data (TOI had `announced_bowler`). Cricbuzz gives
  the bowler only once he bowls his first ball of an over.
- Playing XIs (with batting order) are in the briefing data and the page
  commentary ("India (Playing XI): ...").
- Non-striker is never filled (`non_striker=""` in the pipeline context).
- The page's pre-ball commentary announces the openers in prose ("West
  Indies openers Shai Hope and ...") but it is not parsed into data.

## Two more fixes made live (~19:15 IST), both uncommitted
1. Shadow mode now prints the full prediction message (the exact Telegram
   text) at all three "Recorded" sites -- `cricbuzz_session.py`,
   `run_live_auto.py`, `run_toi_auto.py`. Before, it printed only
   "Recorded innings 1 over 1 prediction." with nothing to read. Over-1 as
   recorded: Runs 3-5 (point 4.0), wicket risk 22.5%, West Indies 33.5% |
   India 66.5%, confidence 25%.
2. Batting-team flicker: non-ball commentary often has an empty `teamName`;
   when it was the newest object the reader fell back to teams[0] (India),
   so one poll showed "India 0/0" while West Indies batted -- the same bug
   family as the 09-15 team swap. Reader now uses the newest object that
   names a team. Need a unit test for this and for the pre-first-ball path.
Pattern worth acting on: every field test finds a team-identity bug in a new
spot. Add one guard that asserts the batting team never changes within an
innings and raise MatchStateAnomaly if it does.

## PRE-MATCH STATUS -- workstream for tomorrow (user: "we need to work on our prematch status")

What pre-match is today (verified tonight):
- Briefing: one message printed when the scanner first attaches a match
  (tonight ~18:34 IST, ~25 min before the 19:00 start): toss, venue, XI
  *counts* ("11/15 confirmed"), plus the new history block (H2H, last-5
  form, venue record). TOI path adds pitch/weather when TOI has them.
- Pre-match prediction: only the over-1 call, which fires at the scheduled
  start (19:00), not at the toss. No pre-toss prediction exists.

Gaps, in the order I'd fix them:
1. **Briefing is one-shot and goes stale.** If the match attaches before the
   toss, it prints "Toss: not yet available" and never updates when the toss
   or XIs arrive. Re-send/update it when toss or XIs change.
2. **XI is only a count.** Show the actual XIs (batting order), openers,
   captain/keeper, and any change from the last match. The data is on the
   page (commentary "India (Playing XI): ...", squads).
3. **No model-based pre-match win probability** (see the Elo section above:
   0.704 overall but only 0.638 on full-member matchups). Test form/H2H/
   venue/bat-first features against that baseline on the full-member split;
   ship only if it beats it, and print the measured accuracy next to it.
4. **Pitch/weather/dew:** Cricbuzz path has none; TOI had "batting friendly /
   not available" tonight. No historical source, so display-only unless we
   find one. Decide whether to keep TOI as a pre-match enrichment source even
   when Cricbuzz owns the match.
5. **Who bowls first / who opens:** not available before ball 1 on Cricbuzz
   (announced bowler absent; openers only in prose). Parse "X and Y are at
   the crease" and the first "comes into the attack" line into data.
6. **International H2H missing from the live model** (snapshot is IPL-only)
   and **no Afghanistan matches in the local T20I data** -- both undermine
   pre-match for Afghanistan fixtures.
7. **Pre-match outputs are never scored.** Log the over-1 / pre-match win
   probability against the final result so we know whether it carries signal.
8. **Visibility:** the over-1 message must print in the feed at the scheduled
   start (printing now fixed); on restart, reprint the latest recorded
   prediction so a restart can't hide it.
Tests to add with this: pre-first-ball path, quiet wait, briefing refresh.

## SPEC (from the user, 2026-10-06): pre-match win % for each team
Inputs to show and use, all restricted to the match's format (men's T20Is):
1. Last 5 matches of EACH team (result, opponent, date, margin).
2. Last 5 matches between the two teams, and the result pattern between them.
3. Venue: win % when batting first vs when chasing at this venue (and avg
   1st-innings score), used as the pitch signal.
4. Pitch note (TOI text) and weather where available, shown as context.
Output: "Win %: Team A xx% | Team B yy%" in the briefing, next to the lists
above so the reasoning is visible.

Design decisions to make when building (keep it honest):
- Weights: fit them (logistic regression on the same features, trained
  <=2024, tested on 2025+) rather than hand-picking, and compare against the
  Elo baseline (0.638 accuracy / Brier 0.2343 on full-member matchups).
  Ship only if it beats that; print the measured accuracy beside the number.
- Venue sample sizes are tiny (Lucknow: 3 men's T20Is). Shrink the venue
  bat-first/chase rates toward the global T20I rate by sample size, and
  show the sample size ("3 matches") with the percentage.
- Bat-first vs chase needs the toss decision: it is known after the toss, so
  compute each team's win % conditional on who bats first (tonight: India
  chose to bowl, so West Indies bat first) -- but earlier tests found toss
  itself carried no signal, so test this rather than assume it.
- "Last 5" is noisy and opponent-blind: weight by opponent strength (Elo of
  the opponent) or include it as a feature and let the fit decide.
- Pitch has no history; the venue split is the proxy. State that in the text.
- Teams with no history in the local data (Afghanistan) -> skip the block.

## SPEC addendum (user): when the T20I venue sample is small, add IPL at the same ground
Checked for Ekana/Lucknow (men's, first-innings totals from Cricsheet):
- T20I: 3 matches, avg 1st inns 164, batting first won 2/3 (67%)
- IPL 2023-2026: 27 matches, avg 1st inns 177, batting first won 11/27 (41%)
  -> chasing wins ~59% at this ground; by year avg 1st inns: 2023 152,
  2024 183, 2025 193, 2026 176.
The 3-match T20I figure pointed the wrong way; the IPL sample corrects it.
Build notes:
- Show the two sources separately with their sample sizes, then a combined
  figure -- do not blend silently.
- Same ground is the pitch proxy, but IPL conditions differ from T20Is
  (franchise teams, impact-player rule since 2023, season timing, dew), so
  test whether IPL-at-venue actually improves venue-based predictions on the
  T20I holdout before giving it equal weight; otherwise down-weight it.
- Weight recent seasons more: ground scoring has moved a lot year to year.
- Use IPL only for the venue (pitch) signal, not for team strength: the
  teams are different.

## ROOT CAUSE FOUND AND FIXED LIVE: overs with a wide/no-ball never completed (the "over skip")
Symptom 2026-10-06: over 4 never got a prediction ("OVER GAP DETECTED:
jumped from over 3 to 5") and over 3's review was lost. Over 3 had a no-ball.
Cause: Cricbuzz gives the re-bowled ball the SAME ballMetric as the
wide/no-ball it replaces, and the reader deduped on (innings, over, ball), so
the re-bowled ball was silently dropped. The over then had only 5 legal
balls, `completed_over` (needs 6 legal) never saw it finish, and the next
over's prediction plus the previous review were skipped.
Evidence it is systematic: in all 6 completed overs containing a wide or
no-ball across every saved match, the stored legal-ball count was not 6
(incl. 170110:2 over 13 -- the 09-15 "over 14 skip" -- and 170110:1 over 17).
The earlier 09-17 explanation (a later over's first ball arriving in the
same poll) may have been a second, separate cause; re-check after this fix.
Fix (uncommitted): `cricbuzz_reader._build_deliveries` now dedupes by each
entry's unique timestamp and numbers deliveries as metric ball + number of
illegal deliveries already seen in that over (re-bowled 2.1 after a no-ball
becomes ball 2, 2.6 becomes ball 7). Test:
`tests/test_cricbuzz_extras_numbering.py` (2 tests); full suite 299 pass.
To verify in the next over with an extra: the over must show 6 legal balls
and the following prediction must arrive with no OVER GAP line.
Open: wides should be checked the same way (same mechanism assumed, only a
no-ball seen tonight); old state files keep the old, wrong keys for past
overs.

## Dismissed batter is not recorded (run-out of the non-striker)
2026-10-06 over 6 ball 3: commentary "out Shimron Hetmyer Run Out" (Hetmyer
was the non-striker) while Shai Hope faced the ball; the record and the
console event line named Hope. Display fixed (`_dismissed_batter` in
`cricbuzz_session.py`, test added, restarted ~19:40). Still open: the data
model has `wicket_kind` but no dismissed-batter field, so the persisted
delivery and anything derived from "who is out" (new-batter / partnership
state when the non-striker goes) is not tracked. Models currently use only
"a wicket fell", so no prediction is affected, but add a `dismissed` field to
LiveDelivery (optional, defaulted so old state files still load) and check
that partnership features reset correctly when the non-striker is out.
Also check the over-5 "WICKET(run out)" case with the ball-number shift from
the extras fix.

## Found tonight: an existing "prematch status" Telegram command (I missed it earlier)
`backend/telegram_control.py` (launchd service `com.cricketbaba.telegram-control`,
running since Wednesday) answers "prematch status" / "/prematch" with, for up
to 5 TOI-listed live/upcoming matches: format, series, venue, start, TOI
status, weather, temperature, humidity, wind, rain, TOI pitch note, inferred
track and dew -- then "Feed / Candidate v3 / Telegram / watcher: armed".
Gaps for the pre-match workstream: TOI-only (TOI has missed two real
matches), no Cricbuzz, no toss/XIs, no H2H/form/venue history, no win %.
SIDE EFFECT TO FIX: `_prematch_status()` calls `_start_watcher()`, which
launches `run_toi_auto.py` in live-publishing mode (not shadow). Asking for
a status therefore silently starts a publishing watcher, and it launches the
TOI driver, not the unified Cricbuzz-primary `run_live_auto.py`. Make the
status read-only and point the watcher at `run_live_auto.py`.
Also: the user pasted an "Over 7 Prediction, Score: 0/2, West Indies 20.6% |
India 79.4%" message tonight; it does not match anything this run produced
(ours: 45/3, West Indies 35.8%) or any saved publication. Source unknown --
ask where it was seen.

## BUG (serious, fixed ~19:50): wrong "Score:" and wrong model inputs after a mid-innings attach/restart
Symptom: Over 8 message said "Score: 9/2" while the match was 53/3 (and an
earlier over-7 message said "0/2"; WI win prob fell to 21%). Cause:
`pipeline.py` computed score_before_over / wickets_before_over by summing
`verifier.over_runs/over_wickets`, which only contain balls THIS verifier
saw; after the restart that was overs 6-7 only. The same shortfall fed the
models wrong current run rate, wickets, and (via `len(legal_deliveries)`)
balls bowled/remaining, so every prediction after the restart was computed
on garbage state. Predictions from the over-7 call (post-restart) through the
fix should be discarded in any evaluation.
Fix: `LiveDeliveryVerifier.base_score/base_wickets` (set at the bootstrap)
added to the sums; legal balls = max(seen, completed*6). Test added. Any
future mid-innings attach (a restart, or joining a match late) hit this --
the 09-15/09-17 mid-innings runs were probably affected too, so re-check
those evaluations. Also still to check: recent-3-over window and partnership
features after an attach (they only see the balls the verifier saw).
Lesson: every restart tonight changed what the model saw; avoid restarting a
live driver, and add a guard that compares the pipeline's score with the
scoreboard score and flags a mismatch.

## Over-number display made consistent (~19:57)
Three numbering schemes were visible at once: the message ("Over 8 Prediction"
= the 8th over), the scorecard ("6.6 ov" = Cricbuzz's own 0-based metric for
the last ball of the 7th over) and event lines ("8.1" = 1-based over). The
user read "6.6 ov" under an "Over 8" message as "over not finished".
Now: scorecard and event lines use standard cricket notation (finished over =
"7.0", first ball of the 8th over = "7.1", legal balls only), messages still say
"Over 8 Prediction" for the upcoming over. Also fixed a latent bug from the
extras renumbering: `latest.ball` can exceed 6 after a wide/no-ball, which
would have made the reader think an innings ended a ball early (legal balls
= ball - illegal deliveries in the over). Confirmed live: after the score
baseline fix, the over-9 message said "Score: 61/3", matching the scoreboard.

## Scorecard line shows the LAST ball, not the next one (user: "why Kuldeep to Shai Hope?")
At an over boundary ("9.0 ov") the line still names the bowler/striker of the
last delivery. It reads as if it were the current/next ball. Relabel as
"last ball: X to Y", and at an over end say "next over: bowler not announced
yet" (Cricbuzz only shows the bowler on his first ball). Ties into the
bowler/openers/next-batter parsing task above.

## Commentary misreads fixed ~20:00 (user: "commentary is not read properly", "drink break ... you should have written it")
1. "leg byes, FOUR" was credited as 4 batter runs (8.6); now 4 leg-byes extras, 0 off the bat.
2. Cricbuzz rewrites an entry's prose after posting (same SIX had long prose in one fetch, short in the next);
   we stored the first text. Now the fullest text seen is kept, and the commentary log writes a delivery
   only once its over is finished, so the edge/mistimed/dropped-catch wording is not lost.
3. "mistimes" (verb form) was not recognised as a mistimed shot.
4. Non-ball commentary was discarded entirely. Drinks/timeouts, weather/delays, injuries/subs, reviews,
   ball changes and bowling changes now print as "[note] CATEGORY: text" and are saved to
   data/live/commentary-notes-<match>.jsonl. "X comes/is back into the attack" announces the next bowler
   before his first ball -- usable as the announced-bowler signal (not yet wired into predictions).
   Reseeding the reader forgets seen note timestamps, so a reseed can repeat recent notes.
5. Cannot screenshot cricbuzz.com: both the built-in browser and the Chrome extension refuse it
   (safety restriction). The driver reads the page over HTTP instead.

## The page has a structured "miniscore" block with the starred players (found ~20:05)
Visible text shows "Shai Hope *" and "Naman Dhir *" (star = on strike / bowling now). The RSC chunks
contain `"miniscore":{...}` with: batsmanStriker, batsmanNonStriker (id, name, runs, balls, fours, sixes,
strikeRate), bowlerStriker (current bowler: overs, maidens, economy, runs, wickets), bowlerNonStriker
(the other bowler), partnerShip {balls, runs}, currentRunRate, requiredRunRate, target,
latestPerformance (last 3/5 overs runs and wickets), lastWicket text (names the dismissed batter),
responseLastUpdated, batTeam score and wickets. Wire into the reader: fill non_striker and current
bowler on the snapshot, use partnerShip directly, use lastWicket for the dismissed batter, and compare
miniscore score/wickets with our running totals as an independent check every poll (would have caught
the "Score: 9/2" and "0/2" bugs). At an over break bowlerStriker probably still shows the finished
bowler; pair with the "comes into the attack" note for the next one.

## COMMENTARY COMMAND -- audit result and plan (user: "still not having good command on commentary")
Audit of all archived commentary (288 deliveries, 3 matches): every line's opening phrase is recognised
(0 unrecognised); parsed runs match the opening phrase except 2 old lines ("no ball, FOUR", "leg byes,
FOUR"), both since fixed. Weakness is the prose AFTER the opening phrase: only edge/mistimed/beaten/
well-timed/dropped-catch/conditions are read; shot type, direction/fielder zone, line & length, pace,
variations, fielding errors, overthrows, reviews and momentum language are ignored. Also: tiny archive
(3 matches + study index of 5 IDs with no logs), entry text is rewritten by Cricbuzz after posting,
and nothing cross-checks our parse against the miniscore block.
Plan, in order: (1) per-poll cross-check of our striker/non-striker/bowler/runs/wickets/partnership vs
`miniscore`; (2) richer prose extractor with a test per phrase taken from real lines; (3) archive every
reachable match's commentary (incl. matches we do not predict) and test which extracted signals relate
to next-over runs/wickets before any reach the models -- until then display-only.

## ROOT CAUSE #2 of "no prediction for this over" (20:35): prose read as an extra
17.2 "1 run, good yorker length ..., jams it down, wide of short fine for one" was parsed as a WIDE
(", wide" inside a fielding position). Over 18 then had 5 legal balls, never completed, and over 19's
prediction never fired. Cricbuzz's own over summary for that over was "1 1 2 6 1 4" (no wide). Fix: runs
and extras are now read from the OPENING PHRASE only (`_opening_phrase`), never from prose; tests added;
a bare "no ball" is now just the penalty run (old +1 guess removed). Over 19's prediction was lost for
tonight (the over was already in progress at restart). Over 17 was a separate skip (a stalled poll
spanning two overs; the unified driver blocks on a 12 s TOI timeout) -- make discovery/scans non-blocking.

## Live-board cross-check built tonight (user: "you are reading commentary and live board")
Snapshot now carries `board` (the page's miniscore). The session prints the starred players
("Mayank Yadav* bowling, Akeal Hosein* on strike, X at the other end") and "BOARD MISMATCH" when our
rebuilt score/wickets/overs/innings differ from the board. Not yet done: use the board to REPAIR state
(e.g. if board overs reach N.0 and our over N is incomplete, close it), and a guard that blocks a
prediction while a mismatch is open.

## Inventory of what the page carries that we do not use (user: "we are missing a lot of important data")
- `scorecard` block: full batting and bowling cards per player (runs, balls, 4s, 6s, SR, how out, bowler
  figures), extras and fall of wickets -- an independent, authoritative source; unused.
- `overSeparator` per over: overSummary ("1 1 2 6 1 4"), overRuns, striker/non-striker with scores, bowler
  figures -- ground truth for every over; only its score is used.
- Event tags beyond four/six/wicket: fifty, team_hundred, highscoring_over, maiden_over, two, over-break.
- miniscore: partnership, lastWicket (names the dismissed batter), latestPerformance (last 3/5 overs
  runs+wickets), recentOvsStats, matchUdrs (DRS reviews left), requiredRunRate, oversRem, remRunsToWin.
- squads/matchTeamInfo/tossResults: only partly used (briefing counts).
- Non-ball commentary: only the notes categories; trivia/stat lines ignored.
Next: pull the scorecard + overSummary into the reader and reconcile them against our deliveries every
poll; log them per ball/over so the archive has the full picture, not just commentary text.

## 2nd innings = data-quality session (user: "focus only on commentary and scoreboard perfection, learn how deliveries are being bowled")
Built before India batted (driver restarted 20:50, tab c11, shadow):
- `[audit] over N: OK/MISMATCH` per finished over, comparing our rebuilt runs and delivery count with
  Cricbuzz's own per-over summary; archived to `data/live/over-summaries-<match>.jsonl`.
  Overs we attached after print "not seen live" (not a false mismatch).
- Board cross-check each poll (`BOARD MISMATCH`) and starred players in the scorecard line.
- `app/live/delivery_detail.py`: pace, length, line, variation, shot, fielding area read from the prose;
  `[ball]` line per delivery in the feed and a `detail` field in the commentary log. First measurement on
  313 archived deliveries (125 with prose): length 61%, line 60%, shot 50%, variation 30%, area now
  114/313 (was 58), pace only 4 (rarely stated). Wicket lines carry the dismissal text but no detail.
  Only the opening phrase decides runs/extras; prose never does.
- The unified driver detaches at the innings break ("No longer live") and re-attaches when the chase starts;
  a not-yet-started later innings is now a quiet wait (no error spam / stuck alert).
After the match: score every [audit] line, count MISMATCHes, read the log's detail fields against the text
for 30 balls by hand, and list the prose phrasings the extractor missed. Do not feed any of it to a model yet.

## 2nd-innings audit result (21:57): 10/12 overs OK, 2 MISMATCH, 1 phantom wicket -- root cause: entries are NOT append-only
Cricbuzz rewrites entries after posting, in three ways: (1) prose is rewritten (long <-> short); (2) a re-bowled
ball reuses the wide/no-ball's ballMetric; (3) a REVIEW replaces an earlier call: at 10.5 (inn2) "out Caught by
Matthew Forde!!" was posted, the third umpire ruled not out ("gap between the fingers"), and the corrected entry
reused the ball number. We froze the first parse, stored the correction as an extra delivery (renumbering shifted
later balls), lost the SIX, and kept a phantom wicket (ours 3 vs board 2; Kishan still batting).
Audit: inn2 over 11 ours 6 runs vs Cricbuzz 12 ('0 4 1 0 1 6'); over 12 ours 5 runs/5 balls vs 7 runs/6 ('1 1 1 1 1 2').
Every prediction from over 12 on used wrong state (e.g. 108/3 vs board ~114/2) -- exclude them from evaluation.
Board noise: consecutive polls returned 102/2, 106/2, 107/2, 79/2, 117/2, 119/2, 121/2 (mixed-age cached copies);
only trust a board by `responseLastUpdated`, and compare only against the freshest one seen.
Design change (do first tomorrow): (1) slot-based latest-wins per (innings, over, ball slot); (2) when an
over's `overSummary` arrives, REBUILD that over from the tokens (runs, W, extras per delivery), commentary only
supplies text; (3) wickets/score repaired from the freshest board when we disagree for N polls; (4) collect every
token format seen in overSummary (digits, W, B1, ... wd/nb/LB unverified) into a catalog from the archive;
(5) treat "they are checking", "review", "umpire's call", "not out" in prose as a signal that the previous
entry may be revised -- hold that ball unsettled until the next entry or summary confirms it.

## Stale batter/bowler names (user: "still showing stale data of the batsman and bowler"; "at over end still previous bowler and batter until 1 ball")
Causes: (1) the scorecard line named the LAST ball's batter/bowler (one ball behind by definition); (2) the page
board flickered because Cricbuzz serves cached copies of different ages; (3) at an over end the board keeps the
finished over's bowler/positions until the next over's first ball (sometimes; at 14.0 it had already updated).
Fixes made (tests pass, 312): line now reads "last ball: X to Y" + "[NOW: bowler* bowling, striker* on strike, other
batter at the other end]" taken from the NEWEST board by `last_updated_s` (older cached copies ignored); at an
over end a "[NEXT OVER, expected: <announced bowler> bowling; <batter>* expected to face]" line, with the bowler from
the "comes/is back into the attack" note posted after the last ball and the striker derived from how the last ball
was scored (boundary/even -> other batter faces, odd -> same batter again; wicket on the last ball -> defer to board).
Still to do: put the announced bowler into `snapshot.announced_bowler/announced_bowler_over` so the bowler model and
prediction can use it (the fields already exist for TOI); verify the derived striker against the first ball of
each next over and count hits; the running driver was restarted before the NEXT OVER line existed.

## DONE 2026-10-09: live-data correctness fixes from the list above (uncommitted)
1. **Over rebuilt from Cricbuzz's own over summary** (`app/live/over_summary.py`, `CricbuzzLiveReader._reconcile_overs`).
   Token catalog verified against summed `overRuns` in the archived pages (matches 151565, 173123):
   digit = bat runs (legal); `W` = wicket, 0 runs; `N` = no-ball, 1 run; `N<k>` = no-ball + k off the bat (k+1 runs);
   `Wd` = wide, 1 run; `Wd<k>` = wide totalling k; `L<k>` = leg-byes totalling k. Anything else (`B<k>`, `W<k>`, ...)
   is NOT verified: the summary is treated as unreadable and nothing is rebuilt from it. When a finished over's
   deliveries disagree with the tokens (runs, wicket flag, legality), the over is rebuilt (commentary aligned to tokens
   by longest-common-subsequence; missing balls come from the tokens), stale entries are never re-added, and the
   running score/wickets are re-chained and re-anchored to the latest fully-known over's checkpoint.
   Replay of the two archived matches (`scratchpad/replay.py`): match 173123 hit a real gap (over 10 had 5 of 6 balls,
   score 77/2 vs board 90/2) and both were repaired; 151565's archive only starts at over 13, so the phantom-wicket
   case is covered by synthetic tests built from the real 10.5 sequence, not by a real page.
2. **Slot-based latest-wins for legal balls**: a LEGAL ball posted again in the same ballMetric slot is a revision (review
   replaced an OUT, etc.): the earlier entry is replaced, later totals shift, and the over cannot reach 6 legal balls a
   ball early. A wide/no-ball followed by the re-bowled ball (same slot, earlier one illegal) is still two deliveries.
3. Reader exposes `drain_repairs()`; the session prints `[repair] ...` and saves `data/live/repairs-<match>.jsonl`.
4. **Prediction hold**: `_prediction_hold` withholds the pipeline call while the page's board is at the SAME over/ball as
   ours but shows a different score or wicket count (prints `PREDICTION HELD`). An older cached board (fewer overs) or a
   newer one never holds; not applied once the innings is complete.
5. **Pipeline guard**: batting team changing inside an innings raises `MatchStateAnomaly` (poll skipped).
6. **Announced bowler** is put on the snapshot (`announced_bowler/_over`) when a "comes/is back into the attack" note was
   posted after the last ball. FINDING: Cricbuzz posts that note ~30-60 s AFTER the over ends (e.g. over 8 ended at
   ts ...863573, "Hosein is back into the attack" at ...909772), and the over-end prediction fires immediately, so the
   bowler model will usually still run without it. Using it needs a decision: delay the prediction up to ~60 s for the
   announcement, or keep predicting at once. Not changed.
7. **Next-over self-check**: the session stores the derived next-over striker/bowler and compares with the first real
   ball (`[check] over N: striker expected X, actual Y: HIT/MISS`, saved to `next-over-checks-<match>.jsonl`).
   Count the HITs after the next live match before trusting the parity rule.
8. **Discovery no longer blocks polling**: `run_live_auto.py` fetches both discovery sources in a background thread
   (`_ScanWorker`); the attach/detach logic still runs on the main thread. This was the suspected cause of the over-17 skip.
9. **Telegram `prematch status` is read-only** (`telegram_control.py`): it used to start the publishing TOI watcher and
   always said "armed". Now it reports whether the watcher is running. The launchd service must be restarted to load this.
   `match is live` still launches `run_toi_auto.py` (TOI, publishing), not the unified driver: decide whether to switch.

Still open from the list: scorecard parsing, our-vs-Cricbuzz win-probability comparison, `matchHeader.complete` as the
match-end signal, `dismissed` field on LiveDelivery, all pre-match work (win %, real XIs, briefing refresh, scoring
pre-match outputs, Afghanistan data hole), reprint latest prediction on restart.

## DONE 2026-10-09: pre-match win % in the briefing (uncommitted)
Code: `app/live/prematch_model.py` (point-in-time features), `evaluate_prematch_model.py` (backtest, `--save` writes
`models/prematch_t20i_logit.json`), `prematch_text(..., toss_won_by, toss_decision)` now appends "Win %: A xx% | B yy%",
the measured accuracy, T20I and IPL venue records shown separately, and who bats first. Tests: `tests/test_prematch_model.py`.
Features: Elo diff, recent-form residual (last 5, opponent-adjusted), H2H all + last 5 (shrunk), venue bat-first edge from
T20Is and from IPL in the same city (both x who bats first). Logistic regression, no intercept, antisymmetric.
Protocol: C chosen on 2023-24 (fit <2023); final fit <2025; scored once on 2025+ (n=830; 141 full-member games).
Holdout, both full members: raw Elo acc 0.638 / Brier 0.2343; fitted Elo 0.638 / 0.2291; elo+h2h 0.674 / 0.2199 (gain vs
fitted Elo +0.0092, 95% CI [+0.0002,+0.0178]); all features 0.660 / 0.2230 (gain +0.0061, CI [-0.0029,+0.0154], not
significant); form alone slightly HURTS (-0.0026); venue (T20I and IPL) adds nothing measurable (~0).
Saved model = "all" because it won the 2023-24 tune window (0.2137 vs elo+h2h 0.2140, a tie); elo+h2h scored better on the
holdout but was not chosen for that reason. The holdout was looked at before saving, so treat its numbers as slightly
optimistic. Associate games: 0.726 acc. Pitch/weather are NOT in the number (no history to train on).
Findings to act on: (a) the user's venue/IPL-at-ground idea showed no predictive gain on T20I winners -- keep it as context
text, not weight; (b) simplest honest model is elo+h2h; consider switching after a second holdout (2026 matches);
(c) Afghanistan has no local history so no win % is shown for its games (still the open data hole);
(d) model weights are trained through 2024 but ratings/h2h state use all local data to date.
Not done: real XI lists, briefing refresh when toss/XIs arrive, scoring pre-match outputs against results, pitch/weather text.

## DONE 2026-10-09 (later): conditions model -- dew, cloud, pitch type x team matchups (uncommitted)
User direction: model dew / overcast / spin-vs-pace conditions in general, not per-venue records.
Data: `fetch_match_weather.py` pulled Open-Meteo archive weather (free, keyless) for all 216 match cities / 3,724
city-dates -> `data/external/weather/{geocode,city_daily}.json` (rerun is resumable; aliases for Kigali etc.).
Match-day live use: `app/live/match_weather.py::match_day_weather(city)` (one forecast call; TOI shows weather/dew only
once play is near -- "Unavailable" pre-match on 10-09). `build_match_facts.py` -> `data/features/prematch_match_facts.json`
(each team's batting/bowling vs pace, finger-spin, wrist-spin from Cricsheet + player style CSV).
Code: `app/live/prematch_conditions.py`, `evaluate_prematch_conditions.py` (`--save-dew` writes `models/prematch_t20i_dew.json`).
Direct test of the user's dew hypothesis (pre-2025 men's T20Is, evening temperature minus dew point):
  gap < 2.5C: chasing side won 56.4% (n=496); gap >= 2.5C: ~49%; no gradient above the threshold. Within the 3 cities with
  both dewy and non-dewy matches the lift was +9.6 points. IPL shows no dew pattern (chase wins ~55% regardless).
  Cloud: NO relationship with chase wins in either IPL or T20Is.
Backtest (2025+, n=830; 141 full-member): dew adds Brier +0.0016 over Elo+H2H, 95% CI [+0.0004,+0.0028] on all games,
  +0.0007 (not significant) on full-member games. Cloud, spin matchups (finger, wrist), pace matchup, form: no gain
  (spin matchups and form slightly negative). Team-level windows (last 20 games) are a crude stand-in for the XI; a
  player-level matchup using the announced XIs is the untested next step.
Shipped in the briefing: after the toss + a forecast, "Win %" uses Elo + H2H + dew; a "Conditions (forecast, <city>)"
line shows temperature, dew point gap, humidity, cloud, rain and "DEW LIKELY"; cloud/pitch matchups are stated as tested
and not in the number. Caveats: Cricsheet has no start times, so every historical match is treated as an evening game
(day games dilute the dew estimate); dew threshold chosen on pre-2025 data only but the holdout was inspected earlier.
Not done: weather for a briefing before the toss (dew term needs who chases), player-level spin/pace matchups from XIs,
a pitch-type forecast from the TOI note, dew for IPL-style conditions.

## DONE 2026-10-09 (late): player-level spin/pace matchups from the actual XIs (uncommitted)
User ask: "spin and pace matchup". Team-level (last-20-games) version showed nothing; this version uses each match's real XI.
Code: `app/live/prematch_xi.py` (per-player record vs pace / finger-spin / wrist-spin from Cricsheet + style CSV, shrunk;
XI batting value vs the opponent's attack mix, tilted by the venue's pitch spin-lean; XI quality), `evaluate_prematch_xi.py`
(`--save` writes `models/prematch_t20i_xi.json`), live `predict_with_xi`, name resolver `resolve_player`
("Shai Hope" -> "SD Hope" by surname + first initial within the team's recent Cricsheet players; 27/30 on the 1st-T20I
squads, misses are debutants with no history and stay neutral).
Results (point-in-time, fit on earlier data only):
- 2025 holdout, all games (n=830): matchup gain vs Elo+H2H+dew +0.0037 Brier, CI [+0.0009,+0.0064]; full members (n=141) -0.0019, CI [-0.0090,+0.0055].
- Rolling 2021-25 (n=2,146), quality+matchup vs dew model: +0.0046 [+0.0019,+0.0073]; positive every year. Full members only
  (n=443): +0.0029 [-0.0019,+0.0081] -- right direction, not proven. XI quality alone helps full members a little (+0.0015, n.s.).
- Saved model = quality + pitch-tilted matchup + Elo + H2H + dew, C=0.3 (rolling-check strength); 2025 full-member accuracy 70% (n=141).
Shipped: briefing uses it once the toss is known AND both confirmed XIs (>=9 resolvable players each) exist; shows win %, the
attack mix per side (pace/finger/wrist share of overs), players matched, and XI-quality vs matchup shifts in points. Falls back
to the dew model, then the base model. Honest label in the text: gain proven across all T20Is, not for full-member games alone.
Limits: expected overs per bowler come from past balls-per-appearance (no knowledge of tonight's bowling plan); bowler styles
only ~80% covered; debutants neutral; the pitch tilt is mild (matchup with/without tilt correlates 0.986); briefing is still
one-shot (re-send when XIs arrive is not built).

## DONE 2026-10-09 (last): briefing re-sends when the toss and XIs arrive
`run_live_auto.py`: every attached pre-start match is re-checked every 60 s in a background thread (never blocks polling).
The briefing is re-sent only when its signature changes -- the "Toss:" line, or both teams reaching 11 confirmed players (a full line-up) --
so a moved percentage or a partial XI never triggers a message (max about 2 updates per match). Header says what changed
("toss now known", "full line-ups confirmed"); each update has its own Telegram key so de-duplication lets it through.
Stops once the first ball is bowled, once the toss is known and both XIs are confirmed, or after 8 h. Works for the Cricbuzz
path and the TOI backup path. Known gap: if the first briefing failed to fetch, the first successful one goes out labelled
"Updated briefing -- details changed". Restart the driver to load it (shadow mode prints the updates).

## FIXED 2026-10-10: mid-innings attach started from 0/0 (found in the 10-09 shadow run)
India v West Indies (151576) and Pakistan v Sri Lanka (170687) were attached mid-innings. The window had no over-separator
(Cricbuzz does not always send one), so the reader assumed "early in the innings" and counted from 0: our score ran 50-100
behind the board for the whole innings (e.g. 84/1 vs 141/1 at 13.2 ov). `PREDICTION HELD` caught it and nothing wrong was
published, but no prediction was made either. Fix in `cricbuzz_reader._build_deliveries`: if the window does not reach the
innings' first balls and has no checkpoint, take the starting score from the page's board, but ONLY when the board is in the
same innings and at the same over.ball as the newest delivery; otherwise raise and retry next poll. Entries older than the
window's first ball are cut off so an older cached copy cannot add them again. Replay of the archived pages now matches the
board at every continuous point. Also seen: the first archived page of each match is ~56 min after the start -- the driver was
probably suspended (laptop asleep?) for the first hour; keep the Mac awake during a match (mcp keep-awake) and consider an
explicit "no polls for N minutes" alert. Telegram `prematch status` fix still needs the launchd service restarted.

## 2026-10-10: briefing shortened (about 25 lines -> 10)
Format now: title + city; toss line with who bats first; XIs line; blank; Win % with its stage (pre-toss / after toss / with
XIs); one-line honesty note (model right X% on N unseen full-member games); "Why" (H2H, XI points, dew points); last-5 form;
dew call; ground record. Dropped from the message (documented here instead): methodology/disclaimer paragraphs, "tested and
left out" (cloud, team-level spin/pace), attack-mix percentages, input list, separate pitch/venue/weather lines (TOI keeps one
"TOI: pitch ...; weather ..." line). The re-send signature now reads the "Toss:" and "XIs:" lines.

## 2026-10-10: briefing header -- weather, pitch call, conditions advantage
Three lines directly under the title: Weather (evening temp, humidity, cloud, rain, DEW LIKELY flag from Open-Meteo),
Pitch (flat / green / turning with its evidence) and Advantage (team the conditions favour = dew effect + XI spin/pace
matchup effect, in win-probability points; before the toss: "dew would help whoever chases").
Evidence rules, from out-of-sample checks run today: ground scoring level predicts the next match's first-innings total
(corr 0.37), so "ground scoring high/average/low" is reported; the ground's spin-vs-pace wicket history does NOT
(corr 0.05), so it is not used to call a pitch turning. Green / turning come only from TOI's pitch descriptor; TOI's
"likely track ... grip for slower balls/spin" sentence is boilerplate printed for every dry surface (same text on four
grounds), so it is ignored; with no TOI note the line says "green / turning: no reliable source for this ground".
Cricbuzz path fetches TOI's pitch note by team names (10-min cache); TOI path passes its own.

## 2026-10-10: dew weight, chance and onset time in the briefing
`Dew: HEAVY -- 90% chance, sets in ~18:25, before the chase (~20:50) -- helps the chasing side` from the hour-by-hour
Open-Meteo forecast (+ sunset). Weight uses the temperature-minus-dew-point gap in the SECOND HALF of the chase: heavy <= 1.5C,
medium <= 2.5C, small <= 4C, little above (the backtest only showed a chasing advantage at <= 2.5C, so small/little are
described as little effect). Onset = first time after sunset the gap falls to <= 3.0C, interpolated between forecast hours.
Chase start = match start + 1h48 (start time read from the Cricbuzz page; 19:00 assumed on the TOI path, and said so).
CAVEATS: the "chance" uses an ASSUMED forecast error of +-1.5C on the gap (not measured against observed dew -- no
ground-truth dew record exists locally); weights/thresholds are on forecast air temperature and dew point, not the grass
surface temperature, which is what really decides dew.

## 2026-10-10: briefing trimmed to the final layout (user-specified)
Sent message = title with ground stats; Weather; "Dew ..., Pitch ..."; Toss; Win % + Form; then the XIs block last.
REMOVED from the message at the user's request: the "How:" build-up line (base % -> dew -> XI quality -> matchup = final)
and the "A lean, not a forecast: right X% on N unseen games" line, so H2H and the model's measured accuracy no longer
appear in the message. They are kept here: full-member holdout accuracy 66% pre-toss / 67% after toss (dew) / 70% with XIs
(n=141); build-up for India v West Indies was 62% base, dew -5, XI quality +5, matchup +3 = 65%. The decomposition code
(`p_base`, `dew_shift`, `quality_shift`, `matchup_shift` in `predict_with_xi` / `predict_with_dew`) is still in place if it
is wanted back.

## 2026-10-10: winning calculation -- one staged model replaces the three separate ones
Audit first: calibration is underconfident (said 85% -> favourite won 95%; said 74% -> 79%), but a calibration scale fitted
on prior data gains only +0.0013 Brier [-0.0004,+0.0030] and the best scale swings 0.95-1.55 between years, so no standing
fix. Full-member Brier is 0.225 vs 0.25 for a coin: T20 is mostly luck, so expect small gains.
Problem found: three separately fitted models gave India a base of 67% / 68% / 62% as stages changed. A single model with
the later features zeroed was significantly WORSE early (-0.0077 Brier: the XI weight is big and dropping it distorts the
Elo/H2H weights). Fix = sequential offset fit (`train_prematch_stages.py`, `models/prematch_t20i_stages.json`): base
(Elo + H2H) fitted first, dew fitted on top of the base logit, XI quality + matchup on top of that. Rolling-origin 2021-25
(fit on earlier years only): identical Brier to the separate models pre-toss and after toss; with XIs +0.0013 (all, n=2146,
CI [-0.0003,+0.0028]) / -0.0007 (full members, n=443, CI [-0.0044,+0.0030]) -- equal within noise. Live: `predict_stages`
in `prematch_xi.py`; the briefing uses only this. The base is constant across stages and the steps add exactly.
Fitted on ALL local matches (not just <2025) for live use, so numbers moved when it was refit: India v West Indies went
65% -> 69% (base 70.4%, dew -6.5, XI quality +2.6, matchup +2.9). Old models (`prematch_t20i_logit/dew/xi.json`) and their
evaluation scripts remain for comparison but are no longer used live.
Next levers, in order of expected value: (1) more player history -- CPL/PSL/BBL/SA20 from Cricsheet (needs a download; 1 of
11 West Indies XI players had no usable record); (2) margin-of-victory Elo and recency decay tuning; (3) separate bat/bowl
team ratings; (4) compare the in-play win % with the Cricbuzz win % recorded in data/live/board-*.jsonl.

## 2026-10-10: extra league data, and a better Elo (the part that actually helped)
**Downloaded** (Cricsheet, public, men's T20, ~21 MB zips; ignored by git under data/raw/cricsheet/): bbl 662, psl 357, cpl 445,
sat (SA20) 130, ilt (ILT20) 134, mlc 109, lpl 143, bpl 469, msl 56, ntb (T20 Blast) 1570, ssm (Super Smash) 467, sma (Syed
Mushtaq Ali) 695 matches. Player records grew from 4,965 to 7,535 players; e.g. Shamar Springer 126 -> 1,024 balls of evidence.
**Result: no gain** (`evaluate_prematch_leagues.py`, rolling origin 2021-25): franchise leagues at weight 1.0 -0.0005 Brier,
at 0.5 -0.0003, + domestic -0.0000, everything at 1.0 +0.0000 (CIs all include 0; full members the same). XI quality correlates
0.97 with the old one. So the live player records stay T20I + IPL (`DEFAULT_SOURCES`); the data and script stay for later.
**Elo upgrade (adopted)**: K now shrinks with a team's experience (96 -> 16, scale 30 games) and each result is scaled by margin
of victory (ln(1+margin/10)+0.5; wickets counted as 8 runs each). Tested ~30 variants on the same rolling window (so some
selection optimism). Chosen variant vs old fixed K=96: +0.0030 Brier [+0.0012,+0.0048] on all games (positive in 2021-23 AND
2024-25), +0.0018 on full members (CI includes 0, same direction in both halves). Staged model backtest after the change
(Brier, all / full members): pre-toss 0.2098 / 0.2270, after toss 0.2091 / 0.2262, with XIs 0.2056 / 0.2235 (were 0.2130 /
0.2289, 0.2122 / 0.2279, 0.2064 / 0.2258). Calibration of the new staged model is good: full members said 70-75% -> won 72.0%,
65-70% -> 69.8%. XI-quality weight fell 0.18 -> 0.10 because the better Elo already captures part of it.
Headline moves for India v West Indies as the model improved/refit today: 65% -> 69% (staged, refit on all data) -> 74%
(new Elo). Old saved models (prematch_t20i_logit/dew/xi.json) are legacy; they load the new Elo so their numbers are not
meaningful any more -- the live path uses only prematch_t20i_stages.json.
Still open: separate bat/bowl team ratings; compare the in-play win % with Cricbuzz's (data/live/board-*.jsonl).

## 2026-10-10: rain risk brought into the win %
Ground truth from Cricsheet's own outcomes (4,687 men's T20I + IPL matches): 143 decided by D/L or awarded, 86 "no result".
Recorded rain is a strong signal: daily rain < 0.1 mm -> 1.1% of matches interrupted, 0.4% washed out; 3-8 mm -> 16% / 5.8%;
8-20 mm -> 19% / 9.7%. Weather backfill re-pulled with match-window rain (17:00-midnight local: `eve_rain_mm`,
`eve_rain_hours`). `train_rain_risk.py` -> `models/rain_risk.json`; `app/live/rain_risk.py`.
Skill is modest: chronological test (fit <2024, score 2024+, n=1,694) interruption Brier 0.0440 vs 0.0466 for the constant
base rate; washout 0.0212 vs 0.0220. Calibrated except the top band (said 32% -> 17% actually, n=42), so risk is capped at 25%
interrupted / 15% washout. Applied to a FORECAST although trained on recorded rain, so real uncertainty is larger.
How it enters the win %: washout probability becomes a third share ("Win %: India 64% | West Indies 26% | No result 10%"), each
side's chance scaled by (1 - P(no result)); shown only when >= 2%. The Weather line shows "rain risk X% (washout Y%)".
What I tested and did NOT build: a rain effect on who wins. Among 85 completed D/L matches (rolling 2017-25) the chasing side won
51.8% vs 51.2% in normal matches, and the favourite's confidence scale was 0.99 [0.48,1.70] vs 1.19 [1.07,1.31] normal -- no
reliable shift toward the chaser, and only a hint (not significant) of more randomness. No day-night / reduced-overs modelling.

## 2026-10-10: India v West Indies (10-09) and the dew factor -- what the data says
Result: West Indies chased India's ~250 and won; pre-match we had India 65-74% depending on stage (a ~30% event; one match is
not evidence either way). Commentary notes show the dew was "beginning to set in" during INDIA's innings (~9th over, ~19:40) and
the forecast had dew from ~18:40, i.e. dew was present from before the match, not only in the chase.
Tested with hourly dew gaps (gap_h now stored per city-date; weather re-pulled): chasing side's win rate, D/L matches removed,
gap threshold 2.5C, assumed 19:00 start (Cricsheet has no start times, so day games add noise):
  no dew in the chase            T20I 49.5% (n=2398)   IPL 54.8% (n=1042)   both 51.1%
  dew only arrives by the chase  T20I 52.0% (n=352)    IPL 49.5% (n=107)    both 51.4% (n=459)
  dew from the start             T20I 56.0% (n=461)    IPL 56.5% (n=46)     both 56.0% (n=507, +/-4.3); pre-2025 only 54.1% vs 51.4%
=> the "dew arrives during the match" idea is NOT supported; if anything the chase edge sits where dew is there from the start,
about +3 to +5 points, uncertain. The model's dew term (about 5.4 points at heavy dew) is at the top of that range.
Scoring: dew-from-start matches score LOWER, not higher (T20I 1st innings 139 vs 151, 2nd 117 vs 129; within 5 cities with enough
matches: -11.6 and -8.6 runs) -- so dew does not inflate totals in this data (confounded by which grounds are humid). Yesterday's
250-run game was an outlier.
Open: show the dew effect in points on the Dew line; log each pre-match win % (by stage) with the result so the pre-match numbers are
scored on live matches (notes item 7 from 10-06, still not built).

## 2026-10-10: every pre-match win % is now logged and scored (closes notes item 7 from 10-06)
`app/live/prematch_log.py`: `data/live/prematch-predictions.jsonl` (one line whenever the stage or the printed numbers change:
stage, p_a/p_b/p_nr, base/dew/quality/matchup components, rain risk, model version) and `data/live/prematch-results.jsonl`
(winner recorded by the Cricbuzz session when the chase ends; ids are "cricbuzz:<id>" / "toi:<id>"). `python score_prematch_log.py`
joins them: right/wrong and Brier per stage, using the last prediction at each stage. TOI-only matches are logged but their results
are not recorded yet (no finish detection on that path). Data lives under data/live/ (git-ignored). Tests are isolated from the real
log by tests/conftest.py.
Backfilled with the numbers the driver printed on 10-09 (flagged `backfilled`, earlier model version): India v West Indies "with XIs"
India 66% -> West Indies won (reported by the user); Pakistan v Sri Lanka "after toss" Pakistan 73% -> Pakistan won.
Live record so far: 1/2 picks right; far too few to say anything. Backtest to compare against: ~68% all, ~64% full members.

## 2026-10-10: the seven open items, done
1. **Dew points on the Dew line**: `Dew: HEAVY -- 90%, sets in ~18:40 (WI +5 pts), Pitch: ...` -- the chaser's gain from the staged model,
   shown once the toss is known.
2. **Batting v bowling in both directions** (with full XIs): `India bat v WI bowl: +55 runs vs avg | WI attack pace 49% / spin 51%` and the
   mirror line, plus `| WI weak v wrist spin (22% of overs)` when a batting side is >= 10 net runs below its own average against a type
   the opposition bowls >= 15% of overs. Numbers are net runs per 20 overs vs an average T20 innings (a wicket costs 12), a rating scale
   not a predicted score.
3. **Commentary/scoreboard**: (a) `matchHeader` is now the match-end signal (`complete`/`state`) and decides the result text and winner,
   verified on 151576: "West Indies won by 6 wkts", India 249/5, WI 252/4. (b) `LiveDelivery.dismissed` (who went out; run-outs of the
   non-striker handled; defaults to "" for old saved states). (c) The scorecard is NOT in the match page (only a link): it is a separate page,
   `/live-cricket-scorecard/<id>/x`, now parsed (`cricbuzz_scorecard.py`: batting/bowling cards, extras, fall of wickets, header) and used as
   an independent cross-check once per over and at the finish: `[card] 14.0 ov: OK` or the mismatches, saved to card-checks-<match>.jsonl.
   Totals are always compared; per-batter runs and fall of wickets only when we saw the innings from ball 1.
4. **Poll-gap watchdog and keep-awake**: the driver prints (and sends to Telegram when live) "nothing was polled for X min" if the wall clock
   jumps >= 2 min between loops, and starts `caffeinate -i -s -w <pid>` so the Mac does not idle-sleep while it runs (opt out with
   --no-keep-awake). Closing the lid on battery still sleeps the Mac.
5. **Telegram control service restarted** (launchctl kickstart): `prematch status` is now read-only.
6. **In-play win % vs Cricbuzz's**: `compare_inplay_winprob.py` joins our GBM+NN shadow log with the `cb_win` percentages in the board logs;
   board log lines now name the batting team, so pairs accumulate from the next match. Existing data has almost no overlap (one chase,
   151565 India: ours 94% at over 10, 92% at 12, 97% at 14 vs Cricbuzz 91%, 98%, 99%; ours was ~40% at over 5 only because that over was
   computed on a wrongly restarted state). No scored comparison is possible yet.
7. **Afghanistan**: confirmed upstream -- Cricsheet has ZERO Afghanistan matches in any archive (0 of 5,775 international T20s, men's and women's,
   and none in the leagues). A re-download cannot fix it. The briefing now says "Win %: not available -- no match history for Afghanistan in our
   data" instead of silently dropping the line. Real fix needs another source (ESPNcricinfo/Cricbuzz history) for their results.

## 2026-10-10: stale bowler/batter at the over break -- fixed properly (user raised it again)
Cause #1: the scorecard line still printed "last ball: X to Y [NOW: bowler* ..., striker* ...]" at an over break, but the page's board
keeps the finished over's names until the next ball. Now, between the last ball of an over and the first of the next, the line reads
"over complete (last ball: Hosein to Kishan)" with no board names, followed by the "[NEXT OVER, expected: ...]" part.
Cause #2 (found in the first real next-over self-check, 151576 over 12: expected Ishan Kishan, actual Shivam Dube = MISS): the derived
striker came from the board's pair, and the stale board still listed Kishan after he was out. Now the pair at the crease comes from the
scorecard page (the two "not out" batters), used only when the card's total matches ours, and anyone our own deliveries show as dismissed
is removed; with fewer than two left it says "see board after the first ball" instead of guessing. The not-out pair is fetched once per
over end together with the card cross-check. Still to confirm on a real match: the `[check] over N: ... HIT/MISS` lines
(next-over-checks-<match>.jsonl) -- 1 check so far, 1 MISS (the case above).

## 2026-10-10: 1st-over prediction
Path (unchanged, verified in tests): Cricbuzz page with no ball-by-ball and the scheduled start passed -> `_pre_first_ball_snapshot` (batting side from the toss)
-> pipeline `_pre_innings_prediction` fires the over-1 run range / wicket risk / win % before the first ball. Re-checked against everything added since:
the prediction hold, scorecard check and announced-bowler code all skip an empty snapshot.
Two fixes: (1) the over-1 message's "Win Probability" came from the IN-PLAY match-winner model while the briefing showed the staged pre-match
model, so the two disagreed (10-06: over-1 message 33.5/66.5). Over 1 now uses the staged model (same number as the briefing; weather, XIs
and toss included), labelled "(pre-match model)", with the in-play value kept in metadata. The in-play shadow logs still record the in-play
model. (2) The new batting-team guard recorded the side from the pre-first-ball snapshot, which is only a guess from the toss; a wrong guess would
have blocked the whole innings. It now records the side only from a snapshot that has real deliveries. Not changed: the over-1 run range and wicket
risk (v3 engine, pre-innings context) and the bowler (none announced on Cricbuzz before ball 1).

## 2026-10-10: reading Cricbuzz's pages fully (user: "that's why we need to learn to read their page fully")
Mistake corrected: the page inventory said the main page carried the scorecard; it only links to it (found by accident). Wrote
`inspect_cricbuzz_pages.py` and read all eleven pages of a finished match; `docs/cricbuzz_page_data_inventory.md` is rewritten page by page.
Biggest finds: (1) the graphs page has Cricbuzz's own win probability after EVERY over of both innings -- the benchmark in one fetch, no live
logging needed; (2) the scorecard page has full cards, fall of wickets, partnerships and the result header; (3) the over-by-over page lists the
last 8 overs with scores (a possible mid-innings catch-up); (4) highlights/full-commentary/squads pages are saved but unused; facts/forecast are shells.
Built: `cricbuzz_capture.py` (+ `capture_cricbuzz_match.py`): at the end of every match the driver saves all eight data pages and the win curve; ran it for
the nine matches we had watched (win curves exist for the three most recent). `compare_inplay_winprob.py` now uses the curve: one match so far (151565,
29 over-forecasts): Brier ours 0.147 vs Cricbuzz 0.110 (overs 8+: 0.121 vs 0.091) -- one match, so an anecdote; ours was more jumpy early (13% vs 24% at over 7,
a restart-affected over) and they agree closely from over 9. Collect more matches before concluding anything.

## 2026-10-10: ROOT CAUSE #3 of the wrong 10-06 chase overs -- long commentary arrives as a pointer ("$26")
Found while showing the main page the driver reads. The page is a React Server Component stream: a long text is NOT embedded in the
entry; the entry holds a pointer ("commText":"$26") and the text comes separately as a header chunk "26:T501," (hex byte length) followed by
the text. Our reader kept the literal "$26", so such an entry had no runs, no wicket words, no prose. Real case: 151565 ball 10.5 (the
reviewed catch: "Springer to Kishan, 1 run, they are checking for a clean catch ... third ump feels there's a gap between the fingers ...
not out"). It was parsed as a dot/unknown ball -- this, not only the review mechanics, is why that over came out at 6 runs vs Cricbuzz's
12 and left the phantom wicket. Scan of every archived page: 6 ball entries affected (all 10.5, 151565), 0 elsewhere; post-match
chat/interview entries are pointers too. Fix: `_text_rows()` resolves pointers before parsing (byte-length aware, an unmatched pointer is
left as is). Likely also part of why only ~40% of deliveries carried prose in the delivery-detail audit -- re-measure on the next archive.
Lesson for the inventory: any field can be a pointer; the raw page archive is what let us find it after the fact.

## 2026-10-10: other pointer fields, and exact batter/bowler figures (user: "check other pointer fields ... the scorecard has bowler and batsman data")
Pointer scan of every saved page (main page at each poll, plus the eight pages captured per match): the only DATA field that is a "$<hex>" pointer is
`commText` (37 page-copies; 6 on the live page, 25 on full-commentary). The rest (`__html`, `error`, `globalErrorComponent`) are page plumbing. Resolved since 0643a51.
Batter/bowler data: the board (miniscore) and the scorecard page carry exact figures -- batters runs/balls/4s/6s/strike rate, bowlers overs-maidens-runs-wickets-
economy -- but we displayed names only. Now: mid-over `[NOW: Mayank Yadav* 3-0-36-3 bowling, Shai Hope* 38(19) on strike, Sherfane Rutherford 19(7) at the other end]`
(from the board; stats kept in the board dict), and at the over break `[JUST BOWLED: Hosein 4-0-37-1 | AT THE CREASE: Hope 102*(45), Shepherd 34*(13) | NEXT BOWLER: Forde 2-0-39-0]`
from the fresh scorecard page (used only when its total matches ours; otherwise just the bowler's name, nothing invented), followed by the expected next striker.

## 2026-10-11 -- scoring limited to the ICC top 10
- User decision: check pre-match results only for the ICC T20I top 10 (India, England, Australia, Pakistan, South Africa, New Zealand, West Indies, Sri Lanka, Afghanistan, Bangladesh). `prematch_log.ICC_TOP_10`; `score()` counts only matches where both teams are in it (`teams=None` scores everything). Update the set when the ranking changes.
- Today's TOI-only Asia-region matches (Malaysia v Maldives, China v Singapore, Hong Kong v Saudi) are outside it; Hong Kong's TOI name ("Hong Kong, China") does not match Cricsheet's ("Hong Kong"), left unfixed on purpose.

## 2026-10-11 -- ODI pre-match + ground win rates
- **ODI model** (`app/live/prematch_odi.py`, `train_prematch_odi.py`, `models/prematch_odi_base.json`): Elo + head-to-head from 2,552 decided men's ODIs (Cricsheet), same State/Elo as T20. Backtest rolling origin 2019-25 (fit on earlier years): all 741 games 64.5% right, Brier 0.2203 (coin 0.2500, plain Elo 0.2274); 442 full-member games 65.6% / 0.2203. Calibrated: 70-80% said -> 77% won. Six Elo K settings all within 0.0004 Brier, kept the T20 setting. Pre-toss only: no ODI dew block (no start times in Cricsheet) and no XI block (player records are T20-only).
- **Ground line** now: `T20 - 11 1st inns avg 153, 2nd inns avg 136, avg wkts - 6, pace - 3/spin - 4, win bat 1st 36%/chase 64% | ODI - 7 1st inns avg 277, 2nd inns avg 230, win bat 1st 50%/chase 50% [small sample]`. An ODI match leads with the ODI record in full and shows T20 short. Win rates are over decided matches. "2nd inns avg" is biased low (chases stop at the target).
- ODI ground facts: `python build_match_facts.py odi` -> data/features/prematch_match_facts_odi.json (T20 model never reads it).
- `python show_prematch_odi.py [--all]` prints ODI briefings for today's international ODI fixtures (nothing published). `score_prematch_log.py` reports T20I and ODI separately (`format` field; old records = T20).
- NOT done: the live driver still attaches only T20Is, so ODI briefings are not sent automatically (no toss refresh, no ODI result recording). Needs a decision before wiring.
- **Revised ground line (user, later 10-11):** scoring numbers (1st/2nd inns avg, wkts, pace/spin) are T20 only, no ODI segment; the results record is ONE pooled T20+ODI figure: `... | bat 1st wins 41% / chase wins 59% / toss winner wins 47% (T20+ODI, 17 matches)`. `venue_summary(city, fmt)` ignores fmt now (ODI briefings show the same line).
- Ground line also shows what toss winners usually choose there (pooled T20+ODI): final form (user-specified): `... | bat 1st wins 41% / chase wins 59% / toss winner wins 47%, toss winner chooses bowl 67%` -- only the more common choice, no match-count suffix; `[small sample]` only under 10 pooled matches.
- **Ground line, later 10-11:** "2nd inns avg" dropped (failed chases end short, so it read lower than the 1st innings even where chasers win). `pace - x/spin - y` are now average wickets per innings credited to pace / spin bowlers (build_match_facts.py records `innings_pace_wkts` / `innings_spin_wkts`; facts rebuilt, model inputs unchanged -- verified identical apart from the two new keys). They add up to a bit less than `avg wkts` (run-outs and bowlers with no known style are in neither). Ranchi: `T20 - 11 1st inns avg 153, avg wkts - 6, pace - 2.9/spin - 2.4 | bat 1st wins 41% / chase wins 59% / toss winner wins 47%, toss winner chooses bowl 67%`.

## 2026-10-11 -- batter "100% fifty record" stat (commentator: every time he gets to N balls he makes a fifty)
- User wants only perfect records, never a rate ("100% record that he hits a 50"). `build_batter_innings.py` -> data/features/batter_innings.json (74,005 men's T20I + IPL batter innings); `app/live/batter_settled.py`: a batter has a record at N balls (N in 10,15,..,45) when EVERY innings of his with N+ balls ended on 50+, with >= 6 such innings and fluke chance <= 1% (league rate of reaching 50 from N balls, to the power n). Claim = lowest such N.
- Findings: no 15-ball record exists (no batter 6-for-6). Today only 4 qualify: Priyansh Arya 6/6 from 25 balls, TP Ura 16/16 from 30, SK Raina 42/42 from 35 (retired), SA Yadav 26/26 from 40. Out of sample (record from earlier innings, tested on the next N+-ball innings) such records held 81% (70 tests; 88/88/91% at 30/35/40 balls; no better than league at 20-25). It is a fact about the past, not a guarantee; relaxing n or the fluke cap gives many more batters but mostly flukes.
- Live (cricbuzz_session): at an over break, for each batter at the crease (fresh scorecard) who has reached his record's ball count and is short of 50, print once per innings `[id] 📊 STAT: <name> 31*(42) -- scored 50+ in 26 of 26 innings where he faced 40+ balls`. Console only. NOT yet seen in a live match.
- The earlier rate-based version (1.5x league conversion; out-of-sample Brier gain +0.0069) was replaced; the evidence for it stays valid if a "usually" stat is wanted later.
- Not done: other milestones (30 after 10 balls, bowler stats), showing it mid-over, Telegram.

## 2026-10-11 -- 100% records for bowlers and venues (same rule as the batter stat)
- **Bowler** (`build_bowler_innings.py` -> data/features/bowler_innings.json, 55,578 innings; `app/live/bowler_records.py`): took k (1 or 2) wickets in EVERY innings in which he bowled N+ balls (6/12/18/24), >= 6 such innings, fluke chance <= 1% (league rate ** n). 9 bowlers qualify (J Coetzee 20/20 from 3 overs, G Sesay 19/19, Umair Tariq 18/18, Syazrul Idrus 17/17, MM Baig 14/14, JS Patel / SM Tupou / R Ankad 10/10 from 1 over, Abbas Afridi 2+ wkts in 9/9 from 4 overs). Out of sample held 70% of 138 tests (league 61%): modest. Live: at an over break, for the bowler who just bowled and the announced next bowler, once per innings, when he has bowled his record's balls without yet taking the wicket(s): `📊 STAT: J Coetzee (3.0 ov, 0 wkts) -- took a wicket in 20 of 20 innings where he bowled 3+ overs`. Console only; not seen live.
- **Venue** (`venue_record` in prematch_analysis, appended to the ground line as `| Record: ...`): the toss winner who chose to bowl / bat, the team batting first, or the team chasing has won EVERY decided T20+ODI match at the ground, >= 6 matches, fluke <= 1%. Across the whole data only ONE ground qualifies: Faisalabad, chasing team 8 of 8 (ODIs). The user's example is NOT true: at Ranchi the toss winner who chose to bowl won 6 of 11 (55%); Indore 9/15, Rawalpindi 12/16 (75%, not 100%).
- Loosening n or the fluke cap would add records that are mostly flukes.
- **Format-specific 100% records (user, later 10-11):** batter, bowler and venue records are built per format and never mixed: T20 = men's T20I + IPL, ODI = men's ODIs (`build_batter_innings.py odi`, `build_bowler_innings.py odi` -> *_odi.json; ODI ball thresholds: batters 20-100 balls, bowlers 12-60 balls). Every STAT line ends with its format. T20 now: batters Arya/Ura/Yadav/Raina, 9 bowlers, venue Birmingham (team batting first won 8 of 8). ODI: batters ME Trescothick 16/16 from 50 balls, ST Jayasuriya 26/26 from 60, V Sehwag 28/28 from 60, SR Watson 30/30 from 60, G Gambhir 40/40 from 60, V Kohli 107/107 from 70, H Klaasen 13/13 from 50, Tanzid Hasan 10/10 from 40; bowlers RJ Harris 19/19 and CJ Anderson 20/20 (wicket from 6 overs), Ehsan Khan, Shakeel Ahmed, S Shrivastava (wicket from 2 overs); venue Faisalabad (chasing team 8 of 8). `venue_summary(city, fmt)` appends the matching format's venue record as `| T20 record: ...` / `| ODI record: ...`. The ground line's pooled T20+ODI win rates (bat 1st / chase / toss winner) are unchanged, as the user specified earlier.
- The live driver is T20-only, so only T20 records print live today; the ODI ones are ready (fmt argument) for when ODIs are wired in.
- **Named "Fact-o-Meter"** (user's pick of Fact-o-Meter / Fact-a-mania; the name lives in one constant, `app/live/fact_o_meter.py`). Lines: `📊 Fact-o-Meter (T20): SA Yadav 33*(41) -- scored 50+ in 26 of 26 innings where he faced 40+ balls`; bowlers likewise; the ground line ends `| Fact-o-Meter (T20): the team batting first has won 8 of 8 here`. Note the ground line's pooled T20+ODI win rates can differ from a T20-only Fact-o-Meter (Birmingham: pooled bat 1st 58%, T20-only 8 of 8).
- **Fun look** (user: "show it in a fun way"): `app/live/fact_o_meter.py` renders each fact as a two-line card with a full meter bar and flames by sample size (1 under 20, 2 under 40, 3 from 40), e.g. `🎯 FACT-O-METER (T20) · SA Yadav 33*(41)` / `   Every single time he has faced 40+ balls, he has made a FIFTY  ▰▰▰▰▰▰▰▰▰▰ 26/26 🔥🔥`. Venue facts stay one line on the ground line: `| 🎯 Fact-o-Meter (T20): the team batting first has won 8 of 8 here ▰▰▰▰▰▰▰▰▰▰ 8/8 🔥`. Wording is playful; every number is still a real count. Console only.
- **Large-gap rule** (user: SA Yadav's 50 after 40+ balls "is not such a big fact"): a Fact-o-Meter is shown only if the average player does it at most 70% of the time (`MAX_BASELINE` in fact_o_meter.py; gap >= 30 points), and the card prints the average: `... ▰▰▰▰▰▰▰▰▰▰ 6/6 🔥 (avg batter 43%)`. League rates: reaching 50 from 40 balls is 83% in T20 (so Yadav drops out), from 60 balls 82% in ODIs (Sehwag, Watson, Gambhir, Jayasuriya), from 70 balls 91% (Kohli's 107/107). What remains -- T20 batters: TP Ura 16/16 from 30 balls (avg 57%), Priyansh Arya 6/6 from 25 (43%); T20 bowlers: JS Patel, SM Tupou, R Ankad 10/10 wicket from 1 over (59%), Abbas Afridi 2+ wkts 9/9 from 4 overs (40%); ODI batters: Trescothick 16/16 and Klaasen 13/13 from 50 balls (70%), Tanzid Hasan 10/10 from 40 (56%); ODI bowlers: Ehsan Khan 13/13, Shakeel Ahmed 22/22, S Shrivastava 13/13 wicket from 2 overs (64%); venues Birmingham T20 (48%) and Faisalabad ODI (52%). Bowlers at 71% (Coetzee, Sesay, Tariq...) just miss the cut; 70% is a judgement call and one constant.
- **Redo of the "large" rule (user 2026-10-11: "100% records only, but the ask is large"):** the first attempt (hide records the average player nearly matches) was wrong. The point is the ASK: SA Yadav on 31 off 41 balls "always makes 50" needs only 19 more -- not exciting; a batter on 41 balls who always goes on to 100 is. Batters: records now run up the target scale (T20 50/60/75/100, ODI 50/75/100/125/150): "every innings of N+ balls ended on X+", >= 6 innings, league rate for it <= 70%, fluke <= 1%, lowest N kept per target; the card is shown live only when he has faced N balls and still NEEDS >= 30 runs (`ASK_MIN_RUNS`), biggest target first, with "needs N more" on the card. Now (T20): GJ Maxwell 7/7 and PD Salt 9/9 of 45+ balls ended on 75+; SP Narine, Taranjeet Singh 6/6 of 30+ balls on 60+; Abhishek Sharma 12/12, FH Allen 9/9, B Zalmai 7/7, RD Rickelton 7/7 of 35+ balls on 60+; Priyansh Arya 6/6 of 25+ on 50+; TP Ura 16/16 of 30+ on 50+. (ODI): Fakhar Zaman 10/10 of 100+ balls on 100+; R Ravindra 7/7 of 60+ on 75+; JJ Roy 16/16 and DJ Malan 8/8 of 70+ on 75+; Tanzid Hasan 10/10 of 40+ on 50+; Trescothick 16/16, Klaasen 13/13 of 50+ on 50+. Yadav / Kohli (50 or 100 are routine at those ball counts) no longer qualify.
- Bowlers: same idea -- shown only when he still needs >= 2 wickets (`ASK_MIN_WICKETS`); the data has no 3-wicket record, so in practice only a 2+-wicket record qualifies (T20: Abbas Afridi, 9/9 innings of 4 overs). "A wicket" records (Patel, Tupou, Ankad, Ehsan Khan, Shakeel Ahmed, Shrivastava) are kept in the table but never printed -- loosen `ASK_MIN_WICKETS` to 1 to show them.
- **Card format fixed by the user (10-11):** `🎯 FACT-O-METER (T20) · PD Salt 40*(45) ` / `   Every single time he has faced 45+ balls, he has gone on to make 75+ runs  ▰▰▰▰▰▰▰▰▰▰  🔥` -- no "needs N more", no "9/9", no "(avg batter ...)" on the card; flames still scale with the sample. The "average" and "needs 30+ more" rules still decide WHICH facts show; they are just not printed. Bowlers and the venue line use the same bar + flames.

## 2026-10-11 -- ODI briefings wired into the driver
- `app/live/odi_watch.py` (`OdiBriefingWatcher`, ticked from run_live_auto.py's loop, own background thread, every 60 s over the Cricbuzz match list): men's international ODIs between two ICC top-10 sides (`--odi-all-teams` widens to every national side, `--no-odi` switches it off). Prints (publishes when not --shadow) the ODI briefing when the match shows as Preview, once more when the toss is known (then stops: no ODI toss/XI block), and records the result when Cricbuzz says Complete ("X won by ..." -> winner; tie/no result -> no winner) so `score_prematch_log.py` scores it under ODI. Joined mid-match -> no stale briefing. NO in-play ODI predictions. Pre-match predictions are logged with `"format": "odi"`.
- Not seen live yet (no top-10 ODI on 10-11). ODI sides outside the top 10 (e.g. Nepal v Canada, USA v UAE today) are skipped unless `--odi-all-teams`. TOI is not used for ODIs.
- Driver restarted in shadow mode in tab c44 (the older c43 is closed).
- **Data scope per format (user 10-11): ODI = men's ODIs; T20 = every men's T20 competition** (internationals + IPL, BBL, BPL, CPL, ILT20, LPL, MLC, MSL, T20 Blast, PSL, SA20, Syed Mushtaq Ali, Super Smash; all 20-over, male only). Applies to the ground line, the pooled win rates (T20 part), and every Fact-o-Meter. `build_match_facts.py t20` -> prematch_match_facts_t20.json (9,727 matches); batter_innings.json (153,244 innings) and bowler_innings.json (115,620) rebuilt from all T20 sources. The WIN MODEL and the pitch call keep reading `load_facts()` (T20I + IPL): the earlier test showed extra leagues add nothing to the win model, and changing its inputs was not asked for.
- The ground line follows the match's own format: an ODI match now leads with ODI scoring (`ODI - 8 1st inns avg 332, avg wkts - 8, pace - 4.3/spin - 2.9`); a T20 match with T20 (Ranchi: 22 matches, 1st inns avg 157). Pooled win rates and toss choice still pool T20 + ODI.
- Bigger samples shrink flukes: with all T20 data Salt, Maxwell, Narine, Abhishek, Arya, Yadav-type 100% records are gone. T20 batter facts now: CS Delport 11/11 and RD Rickelton 15/15 (60+ from 40 balls), FH Allen 25/25 (60+ from 35), MJ Owen 8/8 (50+ from 25), TP Ura 16/16 (50+ from 30), Zalmai 7/7, Mohammad Haris 7/7, Taranjeet Singh 6/6 (60+); no T20 bowler fact with a big enough ask; no T20 venue fact (Birmingham's 8/8 was a small-sample artefact). ODI unchanged (Fakhar Zaman, Roy, Malan, Ravindra, ...; Faisalabad).

## 2026-10-11 -- result facts: "when he does X his team ALWAYS wins / loses"
- User asked for player-performance -> team-result relations in both directions ("Kohli hits 100 in the 2nd innings and India still loses"). `build_player_results.py [odi]` -> data/features/player_results[_odi].json (T20: 199,933 player-matches over every men's T20 competition; ODI: 52,147; decided matches only). `app/live/result_facts.py`: per format, a record = in ALL n matches where he scored X+ (batters: T20 30/50/75/100, ODI 50/75/100/150) or took X+ wickets (bowlers 2/3/4/5, at least an over), optionally batting first / chasing, his team WON (or LOST) every time; n >= 5 and fluke <= 1% (league rate of that outcome ** n). 42 T20 claims (39 players) and 24 ODI claims (20 players) today.
- Out of sample (record from earlier matches, tested on his next): win records held 72% T20 / 82% ODI (league 63% / 64%); loss records repeated 49% / 61% (league 38% / 37%) -- real but modest, the card states a record.
- Kohli has NO such record: India won 24 of 29 ODIs when he made 100+ chasing (44 of 53 any innings; 20 of 24 batting first). Examples that qualify: ODI -- MP Stoinis 50+: Australia lost 7 of 7; BRM Taylor 100+ batting first: Zimbabwe lost 6 of 6; A Symonds 50+ batting first: Australia won 22 of 22; GB Hogg 2+ wkts bowling second: won 25 of 25; MJ Guptill 75+ batting first: NZ won 13 of 13. T20 -- AD Hales 75+ chasing: won 17 of 17; CJ Jordan 30+ chasing: lost 9 of 9; Mohammad Saifuddin 30+: lost 8 of 8.
- Live (cricbuzz_session, at over breaks, once per player per innings, console only): batters at the crease on 60%+ of a record's target and short of it, bowlers (just bowled / announced next) one wicket short of the record's wickets: `🎯 FACT-O-METER (ODI) · Marcus Stoinis 41*(38)` / `   Australia have LOST all 7 matches in which he scored 50+  ▰▰▰▰▰▰▰▰▰▰  🔥`. Not seen in a live match. Most records are for lesser-known players; few will fire on a given match.
- Not done: showing them in the pre-match briefing for the players in the XI, a "last N matches" streak version.
- **Team-specific (opponent) Fact-o-Meters (user: "overall or team specific, like ind vs wi, so WI specific stats"):** all three player fact types now exist overall AND against one opponent, same rules (the opponent record needs the same minimum innings/matches and fluke <= 1% against the league's OVERALL rate, so they are rarer). Tables got an `opponent` column (build_batter_innings.py / build_bowler_innings.py / build_player_results.py, all rebuilt). Live: batters get `opponent=snapshot.bowling_team`, bowlers/result facts the batting team; an opponent record shows only against that team and reads e.g. `Every single time he has faced 50+ balls vs West Indies, he has gone on to make a FIFTY` / `India have WON all 14 matches vs Sri Lanka in which he scored 75+`. Examples: ODI -- V Kohli 50+ vs West Indies in 20/20 innings of 50+ balls, V Kohli 75+ vs Sri Lanka India won 14/14, RA Jadeja 2+ wkts vs West Indies India won 13/13, B Lee 2+ wkts vs India Australia won 12/12, Shaheen Afridi 2+ wkts vs Australia in 9/9 innings of 8 overs; T20 -- AJ Finch 30+ vs India Australia lost 8/8, AU Rashid wicket vs West Indies in 25/25 innings (a "wicket" record never prints: small ask). Not done: team-at-a-ground (India at Ranchi) records.

## 2026-10-11 -- ground-and-innings Fact-o-Meters with averages ("his favourite ground")
- User: the 100% facts are rare, add run average / wicket average and venue + innings detail ("a bowler always 3 wickets bowling second at this venue", "a batter always 50 in the 1st innings at his favourite ground"). `app/live/venue_form.py`: per format (T20 = every men's T20 competition, ODI = ODIs), one ground (city key), innings 1st / 2nd / either: batter scored X+ in ALL n innings (T20 30/50/75/100, ODI 50/75/100/150), bowler took k+ wickets (2/3/4) in ALL n innings (>= an over); n >= 4, fluke <= 0.1% (stricter: thousands of batter-ground pairs searched). The card shows the count, the average and the best: `His favourite ground: 50+ in every first innings at Hove, 4 of 4 (avg 70, best 90)` / `Loves this ground: 3+ wickets every time he bowls first at Kigali City, 4 of 4 (avg 3.8 wkts)`. Live: batters at the crease (needs >= 20 runs more), bowlers (needs >= 2 more wickets), once per player per innings, using the match venue (`snapshot.venue_name`) -- console only, not seen live.
- Today: T20 25 batter records + 1 bowler; ODI 7 batters + 4 bowlers (e.g. JM Vince 4/4 first innings at Hove, V Kohli 30+ in 7/7 second innings at Mirpur avg 52, AB de Villiers 75+ in 4/4 first innings at Johannesburg avg 120, Fakhar Zaman 50+ in 4/4 chases at Rawalpindi avg 108 best 180).
- Persistence is weak (said plainly): records built from earlier innings held on the next innings 40-45% (T20 batters, league 9-21%), bowlers 21-27% (14 tests); a batter's average at a ground adds almost nothing beyond his career average (weight 0.12 in T20, 0.00 in ODI). They are true records of the past, not predictions.
- Tables gained `city` (batter and bowler) and the bowling innings number (bowler) -- all four rebuilt. Not done: pre-match "favourite ground" lines for the XI, average-only (non-100%) ground facts.
- **Average-gap ground facts + pre-match "players in this XI" lines (user: "yes add both"):**
  - Average-gap: `venue_form._average_claims` -- a batter's runs / a bowler's wickets per innings at one ground vs his own career average: >= 5 innings there, >= 20 career innings, >= 1.5x and z >= 3 (gap / (career spread / sqrt n)). Today T20 37 batters + 13 bowlers, ODI 7 + 5 (e.g. Imad Wasim averages 42 at Rawalpindi vs 14 overall in 8 innings; Pollard 50 vs 22 at St Lucia; Ponting 95 vs 37 at Johannesburg). The bar is PARTIAL (`ratio_fill`: 1.5x -> 3 of 10, 3x -> 7, never full -- a full bar means a 100% record). Out of sample (said plainly): the next innings comes back most of the way -- T20 batters claimed 46.9, next innings 27.5 vs career 23.2 (about 18% of the gap persists), bowlers 1.29 vs 1.06 vs claim 2.33; ODI no persistence. Live: shown only if the player has no 100% ground card, at over breaks, console only.
  - Pre-match: `app/live/prematch_facts.py` + `fact_o_meter_block` in the briefing builders (cricbuzz_session.briefing_text, run_toi_auto._briefing_text): once BOTH XIs are confirmed, up to 4 one-line Fact-o-Meters about the players in the XI, ranked by how unlikely luck is, one per player, from: 100% ground records, average-gap ground facts, and 100% result records (overall or vs this opponent, innings-specific only when the toss says who bats first). Placed after the win-% block, before the line-up block. T20 briefings only (the ODI briefing has no XIs). Records of the past, not forecasts. Not yet seen in a live briefing.
- **Team cards (user: "put toss winning and batting 1st/2nd winning under this"):** `app/live/team_records.py` -- a TEAM's 100% records in a situation (winning the toss, winning it and bowling / batting, batting first, chasing), per format (T20 = every men's T20 competition, ODI = ODIs), in three scopes (overall, at one ground, against one opponent); >= 7 matches and fluke <= 1% (league win rate for that situation is 48-52%, so ~7 matches); WON-all and LOST-all both. 41 T20 teams and 8 ODI teams have at least one (mostly associate sides; ODI e.g. India chasing vs Zimbabwe 12/12, Australia chasing vs Pakistan 14/14, Australia winning the toss at Melbourne 8/8, Zimbabwe chasing vs South Africa lost 15/15). The card is `🎯 FACT-O-METER (ODI) · Australia` + up to 3 lines like `   Chasing vs Pakistan: won 14 of 14  ▰▰▰▰▰▰▰▰▰▰  🔥`; the least likely scope per situation; pre-toss it lists winning the toss / batting first / chasing, after the toss only what applies. In the briefings (T20 via Cricbuzz and TOI, and the ODI briefing) the team cards come right after the win-% block, before the XI Fact-o-Meter lines (`team_card_block`). Not yet seen in a live briefing.
- **Fact-o-Meter is its own message (user 10-11: "keep fact o meter in diff cards, dont mix it with prematch"):** the briefing is back to the plain layout (no 🎯 lines; the ground line no longer carries the venue record). `fact_o_meter_message` (prematch_analysis) builds one standalone message: header `🎯 FACT-O-METER · A vs B -- City`, the team cards (toss winning / batting first / chasing), a `Ground record` card when the ground has a 100% record, then (T20, XIs confirmed) up to 4 player facts. Driver: `cricbuzz_session.fact_text` / `run_toi_auto._fact_text`; `LiveAutoState._emit_facts` prints it (publishes outside shadow mode as `factometer:<source>:<id>:<hash>`) at attach and again whenever its text changes (toss / XIs in) on the 60 s briefing refresh; the ODI watcher sends its card the same way (`odi_fact_message`). Live in-play cards (batter / bowler / result / ground at over breaks) were already separate console messages. tests/conftest.py stubs the card fetch so driver tests make no network calls.
- **Win rates are format specific, never pooled (user 10-11: "format specific and should match the format it is being played"):** the ground line's `bat 1st wins / chase wins / toss winner wins, toss winner chooses ...` now comes only from the match's own format: a T20 match uses every men's T20 competition together (T20I + IPL + the other leagues -- T20I and IPL matches both use this one T20 pool), an ODI match uses ODIs. This reverses the earlier "combined T20+ODI" instruction after it was shown to mislead: Rawalpindi T20 51% / 49% (toss winner 49%, 73% choose bowl) but ODI 15% / 85%, pooled 43% / 57%. Tonight's grounds (T20): Rawalpindi bat 1st 51% / chase 49%; Indore bat 1st 17% / chase 83% (24 matches), toss winner 54%, 71% choose bowl. The Fact-o-Meter team cards and ground records were already per format.
- **"Use full T20 and league data" (user 10-11), where it applies now:** ground line (scoring + win rates + toss), ground scoring level used by the pitch call (`_ground_totals`, switched from T20I+IPL to every men's T20 competition), every Fact-o-Meter (batter, bowler, result, ground, team, venue records). Deliberately NOT switched, with the evidence: the pre-match win model (Elo / head-to-head / dew / XI -- the 12-league test gave no gain and team strength is an international notion) and the in-play run/wicket models (extra leagues gave ~1% at best, not significance-tested, docs/finding_other_leagues_run_wicket.md).
- **Fact-o-Meter cards go to Telegram (user 10-11: "send the fact o meter cards to telegram too"):** `python run_live_auto.py --shadow --facts-telegram` sends ONLY the Fact-o-Meter cards (the pre-match card message, the ODI card message, and the live over-break cards) to the configured chat (@IntelligentCricBot, backend/.telegram.json) while predictions and briefings stay in shadow mode. Without `--facts-telegram` nothing changes. Cards are sent once: `SendOnceFile` keeps data/live/factometer-sent.json so a restart does not re-send, and keys are `factometer:<source>:<match>:<hash>` (pre-match; re-sent only when the text changes, e.g. toss / XIs in) and `factometer:live:<match>:<innings>:<kind>:<player>` (live). A failed send is printed and never disturbs the feed. In non-shadow mode the cards use the main publisher. Still not seen live.

## 2026-10-11 night -- first live matches with the new build (India v West Indies, Pakistan v Sri Lanka)
- **The Mac slept during the matches** (on battery 80%, lid closed: "Maintenance Sleep" 20:17:50, wake 20:25:30; watchdog logged 45.4 min and 8.1 min gaps). `caffeinate -i -s` does not stop lid-close sleep on battery. Keep the Mac on power with the lid open for live matches. After the gap the India v WI session fell behind the page (ours 164/6 vs board 197/6), PREDICTION HELD fired correctly, then the driver detached/re-attached and rebuilt from the board (`[repair] starting score 168/5`). The over-17 prediction (17th over = balls 16.1-16.6) was therefore made at 16.2 instead of at the over break, score 187/6 correct: India 85.4%.
- **Stale NOW tag fixed:** after Tilak Varma's run out at 16.4 the scorecard line still listed him "at the other end" because the cached board was at 16.3 (193/6). `_board_players` now prints the board's names only when the board is at the same innings / score / wickets / over.ball as our state (`_board_is_current`); otherwise it leaves the tag out. Regression test in test_cricbuzz_session.py. Not restarted into the running driver mid-match.
- One Fact-o-Meter card went to Telegram (Pakistan v SL pre-match: Shadab Khan averages 36 at Rawalpindi vs 17 overall). Nothing for India v WI. No live over-break cards fired yet in either match (checked console).
- To check after the matches: `score_prematch_log.py` (T20I), `compare_inplay_winprob.py`, the cb-win-curve files, whether any 📊/🎯 live card fired, the [check]/[card]/[repair] lines.
- **ROOT CAUSE of the missing over 17-19 predictions (India v WI, 10-11):** at 16.4 Tilak Varma was run out with a run completed ("...out Tilak Varma Run Out!! 1 run completed."). `_runs_and_extras` only read a ", 1 run" head, so the ball was 0 runs, our score ran 1 behind the page's board (195/7 vs 196/7) and PREDICTION HELD fired on every later poll (overs 18, 19, 20) -- the safeguard worked, the feed was wrong. Fixed in cricbuzz_reader._runs_and_extras (a "run out" text with "N run(s) completed" = N runs; regression test). The over-summary token for a run-out is "W" (no runs, unverified "W1"), so only the text parse can supply the run. Possible next step (NOT built): a drift self-heal -- when a CURRENT board (same over.ball, same wickets) differs from ours by a few runs on consecutive polls, shift the innings baseline by the difference and log a [repair], so one mis-read ball cannot cost the rest of the innings.
- **"comes to the crease" note now used (user 10-11, India v WI 19.0):** after Arshdeep Singh was out lbw off the last ball the scorecard page lagged and the line said "batters: see board after the first ball" although the commentary had said "Mayank Yadav, right handed bat, comes to the crease". `_NOTE_PATTERNS` gained `NEW BATTER`; the session keeps `_incoming[innings]` and the scorecard page's last not-out names (`_card_not_out`, even when stale); `_derived_crease` = those + the board's pair + incoming, minus everyone our own deliveries show dismissed, used when the fresh card pair is not available; `_expected_next_over` names the surviving batter to face and the new batter at the other end after a last-ball wicket (audited later by the existing HIT/MISS check). Console / expectation only; takes effect at the next restart.
- The `[NOW: ...]` tag comes from the page's `miniscore` (the same box the user pasted: IND 218-9 (19.3), Shamar Joseph 3.3-0-50-1, Prince Yadav 9(4), Mayank Yadav 0(0)) and matched it exactly; predictions use the ball-by-ball commentary state, with the miniscore as the cross-check (PREDICTION HELD when they disagree at the same over.ball). Poll interval is 10 s (`--poll-seconds`).
- **Sync = confirm, then re-read -- never block (user 10-11: "first match commentary and our recording in perfect sync ... dont block, read commentary, it will hardly take 1 second to reread previous commentary"):** before a prediction for the over after completed over C goes out, `_sync_status` asks whether an independent source AT THE SAME POINT confirms our record: the page's mini scorecard / board (`_board_verdict`: same over.ball and same score / wickets), Cricbuzz's own over summary (`_audit_ok`: runs and delivery count) or the scorecard page's total (looked at most every 12 s). ok -> predict at once. contradiction (a source disagrees, or the old board-mismatch hold) -> `_resync`: a FRESH `CricbuzzLiveReader` rebuilds the innings from everything the page shows now (the board as the baseline) and `poll_once` runs again immediately (about a second), at most `RESYNC_MAX_PER_OVER` = 2 times per over; if it still disagrees the prediction is held (`PREDICTION HELD (still disagrees after re-reading...)`). unconfirmed (no source at this point yet, e.g. the board is an older cached copy) -> up to 2 immediate looks at a fresher copy of the page, then it predicts anyway and prints `[sync] predicting UNCONFIRMED`. If the fresh reader cannot rebuild (no baseline) the previous reader is restored (`_previous_reader`). Nothing waits any more (the first version of this gate waited up to 120 s; replaced). Never seen live; after the first match count how often `[sync]` fired, and whether a re-read fixed the disagreement.
- **Chase over-1 prediction was never made on Cricbuzz (found 10-11, India v WI; "again we missed 1st over prediction"):** the reader handed out an empty pre-first-ball snapshot only for innings 1 (`_pre_first_ball_snapshot`); for a later innings it raised `CricbuzzInningsNotStarted` (a quiet wait), so the pipeline's pre-innings prediction (over 1) could not fire for any chase -- only the TOI path had it. Fix: at the innings break (the page's header state/status says "break"; `_at_innings_break`) the reader hands the finished first innings out ONCE (so the over-20 review still happens) and then `_pre_next_innings_snapshot`: innings 2, the other side batting, 0/0, target = page target or first-innings score + 1, no deliveries. The pipeline then makes the over-1 prediction as it does for innings 1. Also the staged pre-match win % in `_pre_innings_win_probability` now tells `predict_stages` that the OTHER side batted first when innings == 2. Smoke-tested on a faked finished innings (first poll innings 1 complete, then innings 2 / target 121); never seen live -- check at the next chase that "Recorded innings 2 over 1 prediction" appears before ball 1 and that nothing breaks when the first ball arrives.
- **Next bowler from the mini scorecard too (user 10-11: "still reading from main commentary, not going back to the mini scorecard for the bowler change"):** the announced bowler came only from the commentary note ("X comes into the attack"). `_board_announced_bowler`: at an over break the board lists (bowler, other bowler); the name counts as the next bowler only when it is NEITHER the bowler who just bowled NOR the previous over's bowler and `other_bowler` is the one who just bowled (a name equal to the previous over's bowler is ambiguous -- pairs that swap overs look the same -- and stays with the note). Used only when the note has not named one yet; printed `[note] BOWLING CHANGE (from the mini scorecard): X is next`. Each source's first naming is logged to data/live/announce-sources-<id>.jsonl so the two can be compared after a match. What tonight's logs showed (sparse): the board named the next bowler unambiguously in only 4 of ~25 over ends, and in all 4 the note had already been posted 9-80 s earlier -- so there is no evidence yet that the board is faster; the new log will tell. Takes effect at the next restart.

## 2026-10-11 late: console is summary only
User: "dont show full commentary only summary like 6.1 run 6.2 4 run and then in next over batsman and score and bowler".
Console now prints one short line per ball (`6.1  1 run`, `6.2  FOUR`, `6.3  WICKET -- X (caught)`) and one score line per over break
(score, just bowled, batters, next bowler/striker expected), not one per poll. Full commentary and shot-quality flags (STREAKY/BEATEN/
dropped catch) are no longer printed; they stay in data/live/commentary-<id>.jsonl. Takes effect at the next restart.
