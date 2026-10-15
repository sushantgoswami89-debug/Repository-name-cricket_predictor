# Finding: batter/bowler handedness matchup — tested, rejected for both models

**Date**: 2026-09-18
**Status**: Tested and rejected. Neither live model changed.

## Hypothesis

Checked first that this was genuinely new information, not something a
GBM could already reconstruct from existing columns: the live
`bowler_type` feature both models use is reduced all the way down to
just `"pace"`/`"spin"` (`WicketContract22FeatureComputer.
_classify_bowler_type`, `RunRangeV3FeatureComputer`'s equivalent) —
bowling arm (left/right) is discarded entirely. A matching normalizer,
`app/ml/player_style_registry.py`'s `normalize_bowling_style`, already
parses arm+type correctly (`left_arm_pace`/`right_arm_pace`/
`right_arm_offbreak`/`left_arm_orthodox`/`wrist_spin`/
`left_arm_wristspin`) but was used nowhere in the entire codebase before
this test. Same-handed vs opposite-handed is a real, distinct cricket
dynamic (natural angle into or away from the batter, especially
pronounced for spin) that no existing feature let the model see at all.

## Built

`app/ml/handedness_matchup_dataset.py`: resolves the current striker and
upcoming bowler to canonical player IDs per over (same registry-based
resolution every other per-match dataset builder here uses), looks up
each one's normalized batting/bowling style from
`data/external/cricsheet_player_styles.csv`, and derives `bowler_arm`
(left/right/unknown) and `handedness_match` (same/opposite/unknown).

Spot-checked against a real match (`data/raw/cricsheet/ipl/1136561.json`,
Mumbai Indians vs Chennai Super Kings, 2018-04-07) before trusting it:
Ravindra Jadeja's real bowling over (innings 1, over 8 — confirmed
directly from the raw delivery data) correctly resolved to
`bowler_arm=left`, matching his well-known real left-arm orthodox spin.

## Tested on both models

Identical architecture/features/split/calibration to each live model,
adding only `bowler_arm`/`handedness_match`.

**Wicket** (`train_contract22_wicket_v28_handedness_matchup.py`,
reference: `contract22_wicket_v16_team_composition`'s own real numbers):
worse or flat on every cut. Known: blended AUC 0.6123 vs 0.6126, IPL
0.6148 vs 0.6151, T20I 0.6109 vs 0.6113. Unknown: blended 0.6109 vs
0.6114, IPL 0.6105 vs 0.6131, T20I 0.6116 vs 0.6113 (the one cut with a
tiny, noise-level improvement).

**Run-range** (`train_run_range_v29_handedness_matchup.py`, reference:
30.04/33.25/29.45% blended/IPL/T20I): worse on every cut — 29.84%
(−0.20pp), 32.14% (−1.11pp), 29.43% (−0.02pp, effectively flat).

Feature importance was low in both models (52nd–57th of 63–65 features),
consistent with the negative holdout results.

## Conclusion

Rejected for both models, despite genuinely being new information the
model couldn't see before (unlike batting-order position, which likely
failed because it was largely redundant with existing player-identity
features). Two plausible explanations, not mutually exclusive: (1)
per-player prior/recency stats (a batter's own actual runs-per-ball
against pace vs spin, if that split existed, or simply their overall
career numbers) may already implicitly average in enough of this effect
per player that an explicit matchup flag adds little on top, or (2) the
real handedness effect may be smaller or noisier at the single-over
prediction granularity than commonly assumed, similar to several other
plausible-sounding contextual features already rejected in this
project's history.

Not promoted — `contract22_wicket_v16_team_composition` and
`run_range_v11_partnership_rate` remain live, unchanged.
`app/ml/handedness_matchup_dataset.py` kept as real, reusable, spot-
checked infrastructure — the arm/handedness derivation itself is
correct even though this specific hypothesis didn't pay off.

**Third feature-search rejection in a row this session** (momentum
window, batting position, handedness matchup), all clean and honestly
reported. Worth pausing to consider a different angle before testing
another individual feature idea — see this session's recommendation to
the user.
