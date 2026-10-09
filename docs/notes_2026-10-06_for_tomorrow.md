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
