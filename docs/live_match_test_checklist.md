# Live match test checklist

Things to check the next time a real match is live, based on open
questions from the 2026-08-21 session. Update this file with results as
they're gathered — don't need a single match to answer everything at once.

## 1. Bowler announcement signal (original question)
- [ ] How often does TOI pre-announce the next over's bowler before the
      first ball lands? (hit rate, as a %)
- [ ] When it is announced early, how much lead time (seconds) is there
      between announcement and the over actually starting?
- Tool: `backend/monitor_announced_bowler.py` (already built, just run it)

## 2. Feed reliability by match tier
- [ ] Does the commentary-vs-scorecard reconciliation error (seen on the
      Bulgaria vs Czechia qualifier) also happen on bigger/better-covered
      matches (IPL, major international)? Or is it qualifier-specific?
- [ ] If it happens on well-covered matches too: how often, and does it
      self-resolve within a reasonable time (e.g. under 2 minutes)?
- This directly decides which of the three tier-handling options
  (allowlist / stuck-timeout alert / just document) is the right fix.

## 3. Full predict -> verify -> publish cycle
- [x] Has a single over ever gone cleanly through: prediction made before
      the over -> over completes -> actual result verified -> review
      scored -> next prediction made? **Confirmed offline, 2026-08-22, via
      `backend/run_live_pipeline_replay.py`** — a new tool that replays a
      pre-recorded Cricsheet match through the REAL `VerifiedLivePredictionPipeline`
      + `LiveDeliveryVerifier` (not a simplified stand-in), building
      cumulative `ToiSnapshot`/`ToiDelivery` objects ball-by-ball from
      historical data as a stand-in for a live feed. Ran clean (0
      `VerificationError`s, exactly one prediction published per
      over-completion, prior-over reviews scored correctly) across 2 full
      hand-picked matches (1 T20I, 1 IPL) and a further 16 randomly
      sampled matches (8 IPL + 8 T20I, `random.seed(7)`). Wicket
      probabilities varied sensibly throughout (14%-37% observed, not
      flat). **Caveat — this is not the same as item #2's question**: it
      proves the pipeline's own logic handles a full match cleanly when
      given internally-consistent delivery data; it does NOT test TOI's
      real-world feed messiness (out-of-order commentary, lag, score
      corrections), since the replay constructs already-consistent
      snapshots from ground-truth historical data. A genuinely live TOI
      test is still needed for that part. See
      `docs/candidate_ipl_wicket_v7_2_spell_features.md`-style session
      notes in project memory for the full run details.
- [ ] Does the wicket model v2 adjustment (bowler_spell_adjuster) actually
      fire in live conditions (i.e. does `context.live.bowler` resolve to
      a known player_id from real TOI data)? Not tested by the replay
      above — the replay leaves `announced_bowler` unset throughout (no
      simulated pre-over bowler announcement), so this is still genuinely
      open and needs either a live TOI test or extending the replay to
      simulate bowler announcements.

## 4. Telegram publishing, live (not just a standalone test send)
- [ ] Run with `--telegram` for real and confirm exactly one message per
      predicted over (no duplicates, no missed overs).
- [ ] Confirm message content is sensible (score, range, wicket risk
      label, confidence) against what actually happened.
- [ ] Time the gap between "over completes" and "Telegram message
      arrives" — is it fast enough to feel live?

## 5. Restart / persistence behavior
- [ ] If the process is killed and restarted mid-match, does it resume
      cleanly from `live_prediction.state.json` without re-publishing or
      losing state? (Tested only with synthetic fixtures in
      `tests/test_live_prediction_pipeline.py`, never against a real
      live session.)

## 6. The "flat 50%" wicket bug
- [x] Already fixed and regression-tested with synthetic match states
      (see commit 4ae6e33). Real-match sanity check done, 2026-08-22, via
      the same `run_live_pipeline_replay.py` runs as item #3: wicket
      probabilities varied sensibly across full real innings (observed
      range roughly 14%-37%, moving with match state), through the actual
      live pipeline, not just the unit test. Not a live TOI feed, but real
      historical ball-by-ball match states, which is the part this item
      was actually asking about.

---

**How to start watching:** `backend/monitor_announced_bowler.py` covers
#1 already. For #2-#6, run `backend/run_toi_live.py <url> --poll-seconds 8
--output live_test_output.json` (no `--telegram` until #3-#4 are ready to
test for real) against whatever match is live, and watch
`live_test_stdout.log`.
