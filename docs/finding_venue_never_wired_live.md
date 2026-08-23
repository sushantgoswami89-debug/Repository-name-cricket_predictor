# Venue was never wired into the live pipeline at all — fixed

Date: 2026-08-23. User asked to confirm pre-match win probability
genuinely considers venue and past matches. Checking this surfaced a
real, structural gap that predates today's win-probability work
entirely: `MatchContext.venue` has **never** been set anywhere in
`app/live/pipeline.py` -- not for win probability, not for run-range,
not for wicket. `grep -n "venue" app/live/pipeline.py` returned zero
matches before this fix. Every venue-dependent feature in every live
model (`venue_par_score`, `venue_scoring_regime`, `batting_team_venue_context`,
and now `venue_recency_par_score`) has been silently defaulting to
"unknown"/global-average in live serving this whole time, regardless of
which venue a match was actually played at.

## Root cause

`app/live/toi_reader.py`'s `ToiSnapshot` never had a venue field at
all -- `pitch_type`/`weather_condition`/`toss_won_by`/`team_players` all
exist, venue name never did. Confirmed the real TOI schema before
guessing at a field name: fetched a real cached match payload
(`Matchdetail.Venue`) and found `Name` ("Harare Sports Club, Harare"),
`City`, `Country` -- siblings of the `Pitch_Detail`/`Venue_Weather`
sub-objects that `_pitch()`/`_weather()` already extract from the same
parent dict. The data was always there in TOI's feed; nothing in this
codebase ever read it.

## Fix

- `ToiSnapshot` gained `venue_name: str = ""`.
- New `ToiLiveReader._venue_name()`, same defensive "never raises"
  pattern as `_pitch`/`_weather`: reads `Matchdetail.Venue.Name`, falls
  back to `City`, degrades to `""` on any missing/malformed structure.
  Verified against the real payload shape plus three degradation cases
  (missing Name, missing Venue, empty raw dict) -- all correct.
- Both `MatchContext(...)` construction sites in `app/live/pipeline.py`
  now pass `venue=snapshot.venue_name`.

## Verified the fix actually changes behavior, not just that it runs

Pre-match win probability for the same two teams at three different
venue inputs produced three different probabilities (0.350 / 0.341 /
0.374) -- confirming venue data now genuinely flows through to the
model's output, not just that the field exists. 208 tests pass; a real
match replay still completes with 0 errors and unchanged run/wicket
numbers (this fix doesn't touch run-range/wicket behavior in the replay
tool specifically, since that tool builds its own synthetic
`ToiSnapshot` without venue set -- a separate, pre-existing limitation of
`run_live_pipeline_replay.py`, not a regression from this fix).

## What this means for "predict before the match, considering venue and past matches"

This was already the intended design -- `MatchWinnerEngine` reads
`context.venue` for both `venue_recency_par_score`/`venue_prior_innings`
(recency-weighted venue scoring history) and
`batting_team_venue_context` (home/away), and reads team names for
`h2h_batting_team_win_rate_shrunk`/`h2h_matches_played` (team-vs-team
history) -- all of which are available and computed at the pre-innings
call site (before over 1), using team rosters and toss data that are
already threaded through. The gap was never in the win-probability
model's design; it was that the live venue name itself was silently
empty this whole session (and before), so those features were always
computing against a fallback, not the real venue. Now they use the real
one.
