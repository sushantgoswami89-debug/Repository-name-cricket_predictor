# What Cricbuzz publishes for a match -- every page, what it carries, what we read and save

Rewritten 2026-10-10. The earlier version of this file said the main match page held the full scorecard; it does not.
The main page only LINKS to the scorecard. A match has eleven pages; `backend/inspect_cricbuzz_pages.py <match_id>` lists the
data blocks on each (checked on the finished India v West Indies, 151576). URL pattern: `https://www.cricbuzz.com/<page>/<match_id>/x`.

| Page | Data it carries (JSON blocks) | Read live? | Saved |
|---|---|---|---|
| `live-cricket-scores` (main) | `commentaryPageData`: `matchCommentary` (the latest ~20 ball entries incl. `overSeparator` with the over's summary tokens), `miniscore` (striker, non-striker, bowler, partnership, last wicket, recent overs, `responseLastUpdated`), `matchHeader` (state, status, complete, result, toss), `winProbability` (Cricbuzz's current win %), `team1/team2.playerDetails` (squads, roles, captain/keeper), `partnersStrip`, `matchVideos`; also a ticker of every other live match | YES, every poll: deliveries, board, header, win %, toss, squads | raw page (gz) whenever the board/score changes; `board-<id>.jsonl`; commentary, notes, over summaries |
| `live-cricket-scorecard` | `scorecardApiData.scoreCard` per innings: every batter (runs, balls, 4s, 6s, strike rate, `outDesc`, `wicketCode`), every bowler (overs, maidens, runs, wickets, no-balls, wides, economy), `extrasData`, `ppData` (powerplay runs), `wicketsData` (fall of wickets with over and team score), `partnershipsData` (each stand incl. singles/twos/boundaries per batter), `matchHeader` incl. result and player of the match | YES since 2026-10-10: once per over end + at the finish, cross-checked against our rebuilt innings (`[card]` line); the not-out pair gives the next-over batters | `card-checks-<id>.jsonl`; raw page at match end |
| `live-cricket-graphs` | `graphsData`: **Cricbuzz's own win probability after every over of both innings** (`winProbabilityHorizontalChartData`, also a chart series with the wicket commentary at each fall), `runsChartData`, `runRateChartData`, `runsPerOverChartData`, `wicketOvers` | At match end (benchmark for our in-play model) | `cb-win-curve-<id>.json`; raw page |
| `live-cricket-over-by-over` | `overSummaryList`: the last 8 overs, each with summary tokens, runs, team score after the over, wickets, striker/non-striker with scores and balls, bowler with figures, timestamp | No (the main page's `overSeparator` covers it). Idea: a catch-up source for a mid-innings attach (score after each recent over) | raw page at match end |
| `live-cricket-full-commentary` | `commentary` per innings (paged), `matchPreviewFullComm` (pre-match preview text and commentary), miniscore, header | No | raw page at match end |
| `cricket-match-highlights` | `highlightsInitialData.commentaryList`: the ~50 key moments (fours, sixes, wickets) with over summaries; both squads | No | raw page at match end |
| `cricket-match-squads` | `team1/team2`: `players` = playing XI, `bench`, support staff; each with role, captain, keeper, substitute | Partly (the main page's squads are used) | raw page at match end |
| `cricket-match-facts` ("Info") | Navigation shell; visible text only: series, venue, date and local start time. No JSON data block | No | raw page at match end |
| `cricket-match-forecast` | Navigation shell, no match data found | No | raw page at match end |
| `cricket-match-news`, `cricket-match-videos` | Article and video cards | No (not match data) | not saved |

## What we record now
- **During the match:** raw main page (gz) on every change; the board log (our state next to the page's board, Cricbuzz's win %, status);
  commentary, notes, over summaries, repairs, card checks, next-over checks.
- **At the end of the match (automatic):** `capture_match()` saves the eight data pages (`data/raw/cricbuzz-pages/<id>/final/`) and writes
  Cricbuzz's win curve to `data/live/cb-win-curve-<id>.json`. `python capture_cricbuzz_match.py <ids>` does the same for any finished match,
  days later (done for the nine matches we had watched).

## Known data-quality facts
- Pages come from caches of different ages (boards of 79/2, 102/2, 117/2, 121/2 in consecutive polls); trust a board only by its own `responseLastUpdated`.
- Ball entries are rewritten after posting (prose, re-bowled balls reuse a ball number, a review replaces an earlier OUT); the over summary and the
  scorecard are the ground truth.
- The main page window has ~20 entries and may contain no `overSeparator`; attaching mid-innings then needs another baseline (the board, now).
- Cricbuzz's win curve starts at 50/50 and exists for recent matches only (9 matches checked: the three most recent have it, older ones return none).

## Still unused
- `over-by-over` as a catch-up checkpoint; `highlights` event list; `full-commentary` pre-match preview text; `ppData`/`partnershipsData` as model inputs;
  the page's ticker of other live matches as a third discovery source.
