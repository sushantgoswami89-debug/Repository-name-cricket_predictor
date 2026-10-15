# Live field test — 2026-09-15

First end-to-end live-match test of the prediction pipeline against a real
in-progress match (standing priority #4 from project status memory, blocked
for weeks on match availability). Running `run_toi_auto.py --shadow-t20i`
(shadow mode: predictions are generated and written to
`data/live/shadow-t20i/<match_id>/latest.json`, nothing is published to
Telegram) against whatever real T20I match TOI currently marks live.

Candidate match at kickoff: Afghanistan vs India (T20I), Arun Jaitley
Stadium, Delhi — the only full-member T20I live on TOI's board right now.

This file is a running log of things found during the test that need
correcting. Append as we go; don't fix silently without a note here first.

## Issues found

1. ~~**TOI discovery: `is_live`/`status` contradict each other for the same
   fixture.**~~ **Correction, ~15 min later: transient, not a code bug.**
   At kickoff `ToiMatchDiscovery.matches()` showed Afghanistan vs India with
   `is_live=True` but `status="Match yet to begin"`. Re-ran the same call
   fresh a few minutes later: all 8 fixtures now on the board (down from 17
   — the extra associate/women's matches that were live earlier have since
   finished and dropped off TOI's live-score page) consistently report
   `is_live=False` / `status="Match Yet To Begin"` for Afghanistan vs India,
   matching its real 14:00 UTC start time. Reads as TOI's own live-score
   page briefly mis-flagging the fixture (likely appeared on that page
   ahead of kickoff), not a discovery/parsing bug on our side. No code
   change made. Kept here rather than deleted, so the correction is visible.

2. **[FIXED] `run_toi_auto.py` crashed entirely on a transient TOI network
   timeout — first real failure of the field test.** A `ReadTimeoutError`
   from `toicri.timesofindia.indiatimes.com` during a routine discovery
   scan (`discovery.live_summaries()`, [run_toi_auto.py:92](../backend/run_toi_auto.py:92))
   killed the whole process. The loop's only error handling wrapped the
   per-match `reader.fetch()` call for `(ToiFeedError, VerificationError)`
   — the discovery scan itself had no protection against ordinary network
   flakiness, so one slow response from TOI ended the entire test, not
   just that scan cycle. **Fixed**: wrapped the discovery scan in
   `try/except requests.exceptions.RequestException`, logs and continues
   the loop instead of exiting. Verified: syntax + import check clean,
   monitor restarted successfully. This is infra hardening, not a
   modeling/promotion change — didn't touch prediction logic.

3. **User expectation gap: automated Telegram flow has no summary/lineups/
   pitch/weather.** `PredictionEngine`'s `_telegram_text()`
   ([pipeline.py:975](../backend/app/live/pipeline.py:975)) only ever sends
   over/score/run-range/wicket-risk/win-probability/confidence — nothing
   else. That content exists — `match_briefing.py` prints toss/lineups/
   pitch/weather from the same raw TOI feed — but by its own docstring it
   is explicitly a standalone manual CLI tool, "not wired into the
   prediction models." User expected this "as designed" for today's
   automated field test; it isn't there automatically. Needs a decision
   (see chat) on whether to wire a pre-match briefing message into
   `run_toi_auto.py` before today's real Telegram run, or run
   `match_briefing.py` manually alongside it.

4. **Backlog, decided not to build today: win-probability doesn't use pitch
   or team-level "last 5 matches" recent form.** Checked the real live
   feature set (`app/ml/match_winner_features.py`), not memory. Currently
   used: match state, this-pairing H2H win rate (shrunk), venue recency par
   score, home/away, toss (present but ranked 47th/~70, negligible).
   **Confirmed absent**: pitch (`pitch_type`/`pitch_surface` are fetched by
   `ToiSnapshot` but never threaded into `MatchContext` — zero grep hits for
   "pitch" in `pipeline.py`); team-level recent form independent of
   opponent (only exists as opponent-specific H2H + individual-player
   recency, no "how has this team played lately regardless of opponent"
   signal anywhere). **Decision**: don't build either mid-field-test —
   this project's standing practice is backtest-before-ship for every
   feature (toss, home advantage, team H2H all went through this; some
   were rejected). Wiring something new and untested into a live match
   right now would break that discipline. Revisit as a proper backtested
   candidate in a future session.

5. **Aside, found while checking this, not yet followed up**: a separate,
   older `app/ml/feature_builder.py`/`historical_feature_store.py` pair
   still references `pitch_type` and is still imported by several offline
   replay/diagnostic scripts (`run_match_card_replay.py`,
   `experiment_confidence_signal_search.py`, `tests/test_feature_builder.py`,
   etc.) — despite the 2026-08-22 promotion commit's message claiming the
   "entire pitch/weather/toss/context_builder/... module tree" was removed.
   Confirmed this legacy pair is NOT imported by the live win-probability
   path (`match_winner_engine.py`/`match_winner_features.py`), so it does
   not affect today's test — but the commit message's claim is broader than
   what's actually true on disk. Not investigated further; noting so it
   isn't mistaken for dead code and deleted without checking why several
   research scripts still depend on it.

6. **[URGENT, unresolved] TOI discovery does not list tonight's real match
   even after toss.** Confirmed via ESPNcricinfo (primary source, live
   commentary) that this is the real **2nd T20I** of a 3-match series
   (1st was 13 Sep, already finished — India won by 7 wickets, which is
   why it never appeared in our discovery at all). Toss happened ~7pm
   local (India won, chose to field), both XIs confirmed, ground time
   19:27 — the match is real and imminent. Directly fetched all three TOI
   pages `ToiMatchDiscovery` scrapes (`live-cricket-score`,
   `match-center/schedule`, and the series hub
   `sports/cricket/india-vs-afghanistan`) — **zero occurrences of
   `match-center-scorecard`, `scoreCardUrl`, or `defaultUrl` for this
   match on any of them**, even though the same scrape correctly returns
   8 *other* real fixtures (including the 3rd T20I, 17 Sep, same series).
   Two undistinguished possibilities: (a) TOI only publishes a match to
   these aggregator pages once the first ball is bowled, not at toss —
   the existing background monitor is already polling every 30s and
   should self-resolve this within a couple of minutes if so; (b) TOI
   changed something and the scraper is genuinely broken for this
   specific match/moment. Did not attempt a fix blind — need to see
   whether it resolves once play starts before diagnosing further.
   **This is the single most important open item tonight** — if (b), the
   whole point of today's field test (live-pipeline validation) is
   blocked regardless of every other fix made today.
   `app/live/cricketdata_reader.py` exists as an alternate CricketData.io-
   based discovery path (`find_match`/`current_matches`) with working
   local credentials (`.cricketdata.json`), but it's not wired into
   `VerifiedLivePredictionPipeline` at all (different snapshot type) —
   not a quick swap, noting as a possible fallback for a future session,
   not attempted tonight.

7. **Cricbuzz investigated as a fallback data source — real match confirmed,
   but no ball-by-ball access via plain HTTP.** Match is genuinely live on
   Cricbuzz (id `170110`, "IND opt to bowl"), confirming TOI's silence is a
   real data gap, not a fluke. Checked Cricbuzz's page directly: no
   `__NEXT_DATA__`, no JSON-LD score detail, no classic `cb-*` scraping
   hooks — modern React/Tailwind rewrite, content loads client-side. Only
   server-rendered live signal found: a `<meta description>` string like
   `"AFG 10/1 (2.3) (Ibrahim Zadran 9(11) Sediqullah Atal 0(1))"` — coarse
   score/overs/current-batters only, no per-ball bowler/extras/wicket-kind/
   timestamp data. This project's pipeline (`LiveDeliveryVerifier`,
   `recent_runs_per_ball`, `recent_dot_rate`, `recent_wicket_rate`, etc.)
   architecturally requires real per-ball deliveries — a coarse feed isn't
   a smaller version of that integration, it's a materially different
   (weaker) data shape that can't feed those features. Real ball-by-ball
   access would need either JS execution (not available) or Cricbuzz's
   private mobile-app API (undiscovered, possibly ToS-restricted). Paused
   here — presented options to the user (CricketData.io check / degraded
   coarse-score mode / stop for tonight); **user dismissed the question,
   asked to wait for further instruction.** Not proceeding further on any
   of the three paths until then.

8. **Cricbuzz ball-by-ball reader built and validated against tonight's
   real live match — `app/live/cricbuzz_reader.py`.** Real per-ball data
   IS available via plain HTTP (no JS needed): Cricbuzz's Next.js page
   embeds clean structured JSON per delivery inside React Server
   Component "flight" script chunks (`self.__next_f.push([id,"<escaped
   json>"])`) -- `commText`, `event` tags, `ballMetric` (over.ball, 0-
   indexed), `batsmanDetails`/`bowlerDetails`, and an `overSeparator`
   block with a real independently-reported cumulative score once every
   6 balls. Initial exploration wrongly concluded no such data existed
   (searched for legacy `cb-*` CSS classes instead of actual text
   content) -- corrected after the user pushed back with a direct
   observation.

   Real design difference from TOI, not an oversight: TOI's feed is the
   complete scorecard from ball 1; Cricbuzz's page is a rolling window of
   recent balls only, though older `overSeparator` checkpoints persist
   longer than the individual ball entries they summarize. A first real
   test (naive, assume-0/0-and-replay) produced an accumulated 9/0 against
   the feed's real 24/1 at the same point -- confirming this gap
   concretely, not hypothetically. Fixed by seeding from the latest
   available checkpoint on first attach rather than assuming 0/0, and
   returning the FULL accumulated per-match delivery history on every
   call (matching `LiveDeliveryVerifier.apply()`'s actual contract, which
   dedupes by key and expects the complete known history each time -- the
   reader's first draft wrongly filtered to new-only, a real design bug
   caught by reading `verification.py` rather than assuming).

   **Validated against the real live match across multiple real polls**
   (not just unit-tested against a static sample): correctly seeded from
   a live checkpoint, correctly reported "nothing new yet" through three
   real polling cycles without crashing, then correctly picked up and
   classified a genuine fresh wicket the moment it happened (over 9.1,
   Nitish Kumar Reddy to Ibrahim Zadran, caught, team 51/2) with the
   cumulative total matching Cricbuzz's own next checkpoint.

   **Explicitly NOT done / unvalidated** (see module docstring for the
   same list): team_players/toss/pitch/weather enrichment not ported;
   only "four"/"none"/"wicket"/"over-break"/"maiden_over" event tags
   observed -- six/wide/noball/bye handling is written by naming-
   convention analogy and commText-phrase-parsing, not confirmed against
   a real example of each; wicket-kind classifier validated against one
   real dismissal ("caught") only, not bowled/lbw/run-out/stumped; no
   second-innings/target example observed. **Not yet wired into
   `run_toi_auto.py`** -- that needs its own Cricbuzz-based match
   discovery (a different mechanism than TOI's), plus a decision on
   whether it runs as a parallel source or a manual fallback. Paused here
   to report back before going further tonight.

9. **Full pipeline dry-run against a real Cricbuzz snapshot found one more
   real gap: `LiveDeliveryVerifier` cannot start mid-innings.** It always
   constructs fresh at score=0/wickets=0 and expects to replay every ball
   from over 1 (`process()`'s `self._verifier = LiveDeliveryVerifier(...)`,
   no seed parameter) -- correct for TOI (always full history from ball 1)
   but breaks immediately for a Cricbuzz attach at over 9: first delivery
   handed to the fresh verifier has `total_runs=0, feed_total=51`, an
   unreconcilable mismatch (`VerificationError: Score mismatch at 9.1:
   accumulated 0, TOI reports 51`). Confirmed via `_restore()`
   ([pipeline.py:702](../backend/app/live/pipeline.py:702)): the only
   existing resume path replays this SAME pipeline's own prior persisted
   state, not an externally-known starting score -- there is no seed-mode
   today. **Two ways forward, presented to the user**: attach fresh at
   the start of tonight's 2nd innings (a genuine ball-1 boundary, zero
   changes to shared verifier code) vs. add a real seed-mode to
   `LiveDeliveryVerifier`/`VerifiedLivePredictionPipeline` (touches code
   the already-working live TOI bot also depends on). **User dismissed
   the question, asked to wait for further instruction — not proceeding
   with either until then.**

   **Where things stand overall**: `app/live/cricbuzz_reader.py` is real,
   built, and validated standalone against tonight's actual live match
   (see item 8) -- the gap found here is entirely in the surrounding
   pipeline's assumptions, not in the reader itself. Nothing has been
   published to Telegram for tonight's match; TOI's monitor
   (`run_toi_auto.py`) is still running in the background, unaffected by
   any of tonight's Cricbuzz work.

10. **Live pipeline validation achieved tonight, via Cricbuzz — real
    milestone.** After the verifier-bootstrap fix (item 9), the full
    chain now works end-to-end against tonight's real, already-in-
    progress match: `run_cricbuzz_match.py` (new,
    [backend/run_cricbuzz_match.py](../backend/run_cricbuzz_match.py))
    polls `CricbuzzLiveReader`, resumed correctly from a shadow-mode test
    run with zero state loss (persisted via the pipeline's existing
    `LiveStateStore`), and produced a real, sane prediction for over 17
    (expected runs 6.0/range 5-7, wicket probability 32.5%, correctly
    identified "Death Overs (Extreme Momentum)" phase, using the actual
    live models `run_range_v11_partnership_rate` /
    `contract22_wicket_v16_team_composition`) — no crash, no
    VerificationError. Switched to real Telegram publishing (explicit
    user request, twice) after the auto-mode permission classifier
    required a direct approval for the actual publishing command (a
    separate gate from the earlier conversational agreement) -- approved,
    running live.

    **A real scorecard-display bug found and fixed along the way**: the
    driver's first version computed current run rate from
    `len(snapshot.deliveries)` (Cricbuzz's own since-attach delivery
    count), showing an absurd RR of 630.00 -- fixed to derive balls-
    bowled from the innings-wide `snapshot.overs` figure instead. Caught
    immediately because the script was actually run against the real
    match rather than only unit-tested.

    **One open, unverified data-quality question, not blocking**:
    Cricbuzz's `teamName` gives short codes ("AFG"/"IND"), while TOI
    gives full names ("Afghanistan"/"India"). Checked `team_h2h_pairs.json`
    directly -- it contains neither format for this pairing at all
    (likely IPL-scope-only or T20I coverage gap in that snapshot), so
    H2H is probably falling back to neutral regardless of team-name
    format tonight specifically. Whether other name-keyed context
    features (venue context, home/away) are silently degraded by the
    short-code mismatch was not checked -- worth a proper look in a
    calmer session, not blocking tonight's validation.

11. **First real Telegram message sent for tonight's match — the field
    test's actual objective achieved.** "Sent innings 1 over 19
    prediction." fired after over 18 completed (AFG 142/6), the first
    live prediction published through the new Cricbuzz path to the real
    bot channel. End-to-end confirmed live, not just dry-run: discovery
    gap found (item 6) -> Cricbuzz reader built and validated (item 8) ->
    verifier mid-match-attach gap found and fixed (item 9) -> real
    scorecard bug found and fixed (item 10) -> real publish (this item).
    `run_cricbuzz_match.py` continues running against the live match.

12. **Two real, serious bugs found from the user directly questioning the
    output ("why so high?"/"where is India?") — both fixed, verified
    against real data.**

    a) **Win-probability display only showed the batting side's number**
       (by design, but poor UX) -- fixed `_telegram_text()`
       ([pipeline.py:988](../backend/app/live/pipeline.py:988)) to show
       both teams' percentages side by side (complementary, same
       underlying single-probability model, no model change).

    b) **`bowling_team` resolved to the SAME team as `batting_team`, just
       spelled differently** -- `CricbuzzLiveReader` compared the per-ball
       short code (`"AFG"`) directly against JSON-LD's full names
       (`["Afghanistan","India"]`); since neither equals `"AFG"`,
       `bowling_team` silently picked the first non-matching full name --
       which was Afghanistan itself. Confirmed live in tonight's actual
       output: `batting_team: "AFG"`, `bowling_team: "Afghanistan"`. This
       is why the shown win probability (64.5%) was suspect -- the model
       was fed a degenerate "Afghanistan vs Afghanistan" team-identity
       input, not a real, trustworthy number. **Fixed**: added
       `_resolve_full_team_name()` (short-code-to-full-name resolution,
       algorithmic first-letters/first-3-letters matching + a small
       override table for exceptions like USA/UAE/Hong Kong/PNG that
       don't follow the pattern) so both fields are always full names,
       consistently resolvable and matching TOI's own convention. This
       also resolves the earlier-flagged (item 10) "AFG vs Afghanistan"
       H2H-lookup concern as a side effect, not just the swap itself.

    **A third, independently-found bug while validating the above**:
    `run_cricbuzz_match.py` had already sent a real "over 21" prediction
    to Telegram for a 20-over innings that had actually finished at
    159/8. Root cause: `CricbuzzLiveReader` never set
    `innings_complete`/`match_complete` on `ToiSnapshot` (always default
    `False`), and `pipeline._is_complete()`'s own ball-count fallback
    counts only `verifier.accepted` deliveries -- which, for Cricbuzz's
    rolling window, never reaches the full 120-ball T20 count since it
    only reflects balls seen since attach. **Fixed**: `CricbuzzLiveReader`
    now detects completion directly (wickets>=10, or balls-bowled >=
    scheduled-overs*6, or a next-innings delivery appearing) and sets
    `innings_complete`/`scheduled_overs` itself. Live process was killed
    immediately on discovery (before further bad predictions), fix
    applied and verified against the real, now-completed 1st innings:
    `batting_team=Afghanistan, bowling_team=India, score=159/8,
    innings_complete=True` -- and a full pipeline dry-run against that
    exact snapshot now correctly returns no prediction (`None`) instead
    of a bogus "over 21" call. 29 tests still pass.

    **Fourth issue, found and fixed while testing the above**: a fresh
    reader instance (e.g. this driver restarting) landing during the
    innings break between innings previously couldn't report anything at
    all -- five consecutive real polls all raised "no ball-by-ball state
    yet," because the only fallback for an empty poll was the
    last-delivery seen by THIS reader instance, and a brand-new instance
    has none. Fixed by recording the seed checkpoint itself as a
    synthetic last-known-state the moment it's used to seed the running
    total, so a fresh attach can immediately report "innings complete,
    159/8" from the checkpoint alone rather than waiting for a ball that
    will never come (the innings is over).

    **Net effect**: three real production bugs (team-identity swap,
    missing completion detection, cold-start-during-break gap), all found
    by the user directly questioning output that looked wrong rather than
    accepting it, all fixed and verified against real data before
    resuming. Restarting `run_cricbuzz_match.py` now to watch for India's
    2nd innings (the chase) to begin.

13. **User asked "2nd innings started, no prediction?" — surfaced two more
    real bugs, both fixed and verified against the actual live chase
    before it could corrupt more output.**

    a) **All internal reader state was keyed only by `match_id`, not
       `(match_id, innings)`.** Confirmed happening live: the OLD process
       (not yet restarted with the earlier fixes) logged "Deferred:
       Cricbuzz over-checkpoint mismatch ... accumulated 171/8, feed
       reports 12/0 after over 1" -- 171 = Afghanistan's final 159 + 
       India's first-over 12; 8 = Afghanistan's leftover wicket count.
       The reader was about to keep accumulating India's fresh innings on
       top of Afghanistan's finished one. **Fixed**: every internal dict
       (`_running_score`, `_running_wickets`, `_all_deliveries`,
       `_seed_cutoff_ms`, `_last_delivery`) rekeyed to `(match_id,
       innings)`; `fetch()` now determines `current_innings` as the max
       `inningsId` present in the window and scopes everything to it,
       falling back to "no state yet" (not the previous innings' stale
       state) rather than silently mixing innings. Verified against the
       real chase: `innings=2, batting_team=India, bowling_team=Afghanistan,
       score=13/0` -- correctly reset, not 159+13.

    b) **`target` was never set at all -- always 0, even during the real
       chase.** Would have made `context.is_chase` False for the entire
       2nd innings, silently breaking every required-run-rate/chase
       feature for a whole innings' worth of predictions. Found Cricbuzz
       embeds a direct `"target":<int>` field -- but a first attempt
       regexed the raw HTML directly and matched the WRONG thing: an
       unrelated schema.org `"target":{"@type":"EntryPoint"...}` block,
       because the real cricket field lives escaped (`\"target\":160`)
       inside an RSC flight chunk, same as the delivery objects, and
       never matches an unescaped pattern against raw bytes. Fixed by
       searching the already-unescaped chunks instead of raw HTML.
       Verified against the real chase: `target=160` (Afghanistan's
       159 + 1), matching manual arithmetic.

    Both fixed and re-verified (including a full pipeline dry-run) before
    restarting the live driver -- the OLD process was killed the moment
    the checkpoint-mismatch error confirmed the innings-scoping bug live,
    before any further bad predictions could be generated. 29 tests still
    pass throughout.

14. **Most significant finding of the night: a real, standing bug in the
    core, already-deployed `PredictionEngine.predict()`
    ([prediction_engine.py:313](../backend/app/ml/prediction_engine.py:313))
    -- not a Cricbuzz issue at all.** Found while sanity-checking a real
    chase prediction's metadata (`batting_team: "Afghanistan"` on an
    India-batting over -- caught only because the value was checked
    against reality instead of trusted). During a chase, `batting_team`
    fell back to `context.bowling_first or context.team2` --
    `batting_first`/`bowling_first` are declared `MatchContext` fields
    (`app/models/match_context.py:31-32`) **never set anywhere in the
    codebase**, confirmed via grep (zero hits outside the declaration),
    so this always resolved to `context.team2` -- which `pipeline.py`
    always sets to the currently BOWLING team, for both innings. Real
    consequence, not cosmetic: `batting_team_players`/
    `bowling_team_players` derive from this same value and feed directly
    into `contract22_wicket_v16_team_composition`'s live feature
    computation. **Every chase prediction this model has ever served,
    since its 2026-08-23 promotion, computed the wicket model's
    team-composition feature with the two teams' roles reversed** -- a
    standing bug in the TOI-serving live bot too, not something tonight's
    Cricbuzz work introduced. This is the exact same bug class already
    found and fixed once, in a different file
    (`match_winner_engine.py`, `docs/finding_win_probability_chase_bug.md`,
    2026-08-23) -- missed here at the time.

    **Fixed**: removed the dead `is_chase`-branching fallback entirely;
    `batting_team = context.team1` always, matching the same precedent
    already validated in `match_winner_engine.py` and the training-side
    convention `match_winner_dataset.py` uses. Ran the **full** test suite
    (not just the live-pipeline subset) given this touches shared,
    core-engine code: **221 passed**, zero failures. Verified directly
    against the real live chase before restarting: `batting_team=India`
    (previously would have shown Afghanistan). Live driver restarted with
    the fix; next real send will be the first chase prediction ever
    computed with correct team roles.

15. **[Unresolved, lower severity, logged for follow-up] Over 14 was
    silently skipped -- no prediction, no review, no error.** User
    noticed the driver jumped straight from "Sent innings 2 over 13
    prediction" to "Sent innings 2 over 15 prediction" with nothing in
    between. Confirmed from the raw log: overs 13 and 14 both played out
    with clean, complete, gap-free ball-by-ball capture (every single
    ball from 12.1 through 13.6 visible in the scorecard log, no missing
    deliveries) and zero "Deferred"/error/Traceback lines anywhere in
    that window -- so this is not a Cricbuzz data-window gap like earlier
    findings. `review_history` confirms over 13 has no entry at all
    (jumps from `170110:2:12` straight to `170110:2:15`... actually to
    whatever comes after 12, skipping 13's own review). Root cause not
    yet identified -- suspect something in `pipeline.py`'s
    `verifier.last_key[1] != completed` boundary-matching
    ([pipeline.py:198](../backend/app/live/pipeline.py:198)) failing to
    match at exactly this transition, but not confirmed. Lower severity
    than tonight's other findings (no wrong prediction was shown --
    output was simply missing for one over, not incorrect), so not
    pursued further live mid-match. Worth deeper investigation with
    proper instrumentation in a calmer session.

16. **Match complete. Priorities for the next session, in order:**

    **Fix first — trust/correctness in the already-live system**
    1. Add self-checking guards to the live pipeline (`batting_team !=
       bowling_team`, `over <= scheduled_overs + 1`, team names resolve
       to a known-team list). Nothing caught tonight's bugs automatically
       -- a human noticing bad output was the only detection mechanism.
    2. Grep the whole codebase for the `batting_first`/`bowling_first`
       dead-field pattern (found in `match_winner_engine.py` in August,
       `prediction_engine.py` tonight -- check for a third instance).
    3. Root-cause the over-14 silent skip (item 15) -- unknown mechanism
       in `pipeline.py`'s boundary detection, may also affect the TOI bot.

    **Fix soon — real gaps in the live path**
    4. Diagnose why TOI's discovery silently didn't cover tonight's match
       at all (item 6) -- still unexplained, unclear if recurring.
    5. Validate the Cricbuzz reader's untested event types (six/wide/
       no-ball/bye, every wicket kind except "caught") against real
       archived matches, not just whatever happened to occur tonight.
    6. Build real Cricbuzz match discovery (currently hardcoded to one
       URL, only works as a manual one-off).

    **Product polish**
    7. Wire the pre-match briefing (toss/lineups/pitch/weather) into the
       Cricbuzz path too, matching tonight's TOI addition.
    8. Rename "System Health" to something honest like "Confidence" --
       current label actively misleads (it's model self-confidence, not
       uptime/server health).

    **Strategic**
    9. Stop recent feature-search churn (toss/home-advantage/venue-
       recency/team-H2H all rejected in a row -- diminishing returns);
       redirect effort to items 1-6 instead.
    10. Investigate the run-range model's ~2-over momentum lag (item,
        see chat 2026-09-15) -- a faster-reacting recency signal instead
        of the flat 3-over window is a real, scoped improvement.
    11. Add live observability/alerting so future bugs are caught by a
        monitor, not by a human watching a live match by hand.

17. **No match is currently live** (checked directly, not just inferred from
   the monitor's polling). Afghanistan vs India (T20I, Delhi) is the target
   fixture for today's test, scheduled 14:00 UTC. Shadow monitor is running
   and will auto-attach once TOI actually marks it live and it passes
   `_is_t20_international()` — flagging as a watch item since that filter
   checks `series` text for markers like "world cup"/"tour of"/"tri-series"/
   "international" and this series is literally named "India in Afghanistan,
   3 T20I Series, 2026" — worth re-checking once the match goes live that it
   actually gets picked up and isn't silently filtered out by that check.
