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
- [ ] Has a single over ever gone cleanly through: prediction made before
      the over -> over completes -> actual result verified -> review
      scored -> next prediction made? (Never fully observed yet — every
      live test so far got stuck on feed reconciliation before reaching
      this.)
- [ ] Does the wicket model v2 adjustment (bowler_spell_adjuster) actually
      fire in live conditions (i.e. does `context.live.bowler` resolve to
      a known player_id from real TOI data)?

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
- [ ] Already fixed and regression-tested with synthetic match states
      (see commit 4ae6e33). Worth a quick real-match sanity check: do
      live wicket probabilities vary sensibly over the course of an
      innings, not just in the unit test?

---

**How to start watching:** `backend/monitor_announced_bowler.py` covers
#1 already. For #2-#6, run `backend/run_toi_live.py <url> --poll-seconds 8
--output live_test_output.json` (no `--telegram` until #3-#4 are ready to
test for real) against whatever match is live, and watch
`live_test_stdout.log`.
