# What the Cricbuzz match page contains vs what we read and save

Measured 2026-10-06 on match 151565 (India v West Indies, 1st T20I), live page.
Page: ~364 KB HTML, ~213 KB decoded payload, **213 distinct JSON keys**; our code
reads **33** of them in memory; before tonight we saved **none** of the raw
page (only fields we derived). Columns: READ = parsed by code, SAVED = written
to disk by the driver.

| Block | What it holds | READ | SAVED |
|---|---|---|---|
| Ball entries (`commentary`, 20 in the window) | ballMetric, commText, event tags, timestamp, teamName, batsmanDetails, bowlerDetails | yes (opening phrase for runs; prose for detail) | derived fields only: log + state (`commentary-<id>.jsonl`, `*.state.json`) |
| `overSeparator` (on the last ball of each over) | overNumber, overSummary ("0 4 1 0 1 6"), overRuns, team score, striker/non-striker with scores, bowler figures | yes (added tonight) | yes, `over-summaries-<id>.jsonl` |
| `miniscore` (the live board) | striker, non-striker, bowlerStriker (the starred bowler) + other bowler with figures, partnership, lastWicket text, latestPerformance (last 3/5 overs), recentOvsStats, currentRunRate, requiredRunRate, target, oversRem, remRunsToWin, matchUdrs, responseLastUpdated | yes (added tonight, subset) | yes (added tonight): `board-<id>.jsonl` |
| `matchHeader` | state, status text ("India need 32 runs in 42 balls"), complete flag, result, revisedTarget, toss, matchTeamInfo, start/complete times, dayNight, playersOfTheMatch/Series | partly (toss; state/status/complete/result added tonight) | board log (added tonight) |
| **`winProbability`** | Cricbuzz's own win % per team + `lastUpdatedOver` + odds | yes (added tonight) | board log (added tonight) -- **an external benchmark for our model, never recorded before** |
| `teamInningsArray` / `matchScoreDetails` | score, wickets, overs per innings | partly | no |
| `scorecard` | full batting and bowling cards: every batter's runs, balls, 4s, 6s, SR, how out; bowler figures; extras; fall of wickets | **no** | no (only inside the raw page archive) |
| `squads` / `matchTeamInfo` | playing XIs, roles | briefing counts only | no |
| Event tags | none, four, six, wicket, over-break, maiden_over, fifty, team_hundred, highscoring_over, two | four/six/wicket only | no |
| Non-ball commentary | bowling changes, drinks/timeouts, delays, trivia, team news | notes categories (added tonight) | yes, `commentary-notes-<id>.jsonl` |
| Other matches widgets (`matchesList`, `seriesMatches`) | live scores/status of other matches | discovery only | no |
| Video/news/ads/layout | not match data | no | no |

## Now recorded (from this restart)
- **Raw page archive:** `data/raw/cricbuzz-pages/<match>/<epoch_ms>-<hash>.html.gz`, written whenever the board,
  win-probability over, status text or delivery count changes (at most every 10 s, ~55 KB each). Anything the
  parser misses today can be recovered from it later.
- **Board log:** `data/live/board-<match>.jsonl`: our state next to the board, Cricbuzz's win %, status text.

## Known data-quality facts (all seen live tonight)
- Pages are served from caches of different ages: consecutive polls returned boards at 79/2, 102/2, 117/2, 121/2.
  Trust a board only by its own timestamp (`responseLastUpdated`).
- Ball entries are rewritten after posting: prose changes, re-bowled balls reuse a ball number, a review can
  replace an earlier OUT with a not-out entry. See `notes_2026-10-06_for_tomorrow.md`.

## Still to do
1. Parse `scorecard` (per-batter / per-bowler truth, how-out text, extras, fall of wickets) and reconcile every poll.
2. Compare our win probability with Cricbuzz's `winProbability` per over (the board log already has both sides).
3. Use `matchHeader.complete/state/result` as the match-end signal instead of our own heuristics.
4. Catalog every event tag from the archive. (`overSummary` tokens done 2026-10-09, see `app/live/over_summary.py`: digits, W, N, N<k>, Wd, Wd<k>, L<k>; B<k>/W<k> unverified.)
