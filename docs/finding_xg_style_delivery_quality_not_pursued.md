# Testing an xG-style "delivery quality" signal (line/length/shot) — real but small, and not currently usable in production

Date: 2026-08-23. Following research into what real cricket/football
prediction systems do differently, checked whether football's xG
(expected goals) concept — per-event quality-of-chance features,
independent of outcome — has an unused analog for wicket prediction:
does the ball's own line/length (a bowler-side property, known before
the batter reacts) carry genuine signal beyond what the live wicket
model already uses?

## Data source

[Kaggle: "The Ultimate Ball-by-Ball Cricket Dataset"](https://www.kaggle.com/datasets/ariadaikalam/the-ultimate-ball-by-ball-cricket-dataset)
(CC0, free, no restrictions) — Cricsheet ball-by-ball data enriched with
`ball_length`/`ball_line`/`shot_played`/`shot_direction` scraped and
parsed from ESPNcricinfo live commentary text. Downloaded and verified
directly, not assumed: spot-checked a real delivery (match 1082591,
over 2 ball 5, DA Warner caught off A Choudhary) against our own
Cricsheet source — exact match on player, bowler, ball notation, and
wicket outcome.

## Coverage — real constraint, checked precisely

- **IPL: 90.7%** of our 1,243-match corpus has full detail (all four
  fields present for at least one ball) — genuinely strong.
- **T20I: much weaker** (blended IPL+T20I coverage is only 40.1% of our
  6,767-match corpus — T20I associate-nation matches rarely have
  detailed ESPNcricinfo commentary).
- **The dataset is a frozen historical snapshot, not live-updated**:
  100% coverage every year 2008-2024, but only 45% for partial-year
  2025, and **0% for 2026** (confirmed directly, not inferred from the
  Kaggle page's "Expected update frequency: Never"). Even within a
  "covered" match, only 17.8% of individual rows have all four fields
  simultaneously present.

## The real methodology catch: shot_played/shot_direction are leaky

First test (all four fields as one group) showed an eye-catching
AUC 0.7738 for delivery-quality-only, and +0.196 AUC over a context
baseline when combined — high enough to be immediately suspicious
(the live wicket model's real production AUC is ~0.61 with 60
carefully-built features; a naive 4-field test beating that by 16
points warranted scrutiny before trusting it, not excitement).

Checked directly: `shot_played` values on wicket balls include "edge"
(261), "outside edge" (54), "inside edge" (54), "edged" (44), "beated"
[beaten] (40) — commentary language that describes *how the dismissal
happened*, not a pre-outcome property of the delivery. `shot_played`/
`shot_direction` describe what the **batter did in response** to the
ball -- fundamentally post-hoc, closer to football's "Post-Shot xG"
(computed after knowing shot placement) than standard pre-shot xG. Not
usable as a genuine predictive feature at all.

`ball_length`/`ball_line`, by contrast, are real bowler-side execution
properties (short/full/good-length/yorker; outside-off/leg-stump/etc.)
-- known before the batter reacts, the correct analog to xG's
distance/angle. Decomposed and re-tested separately.

## Honest result, decomposed

Same chronological out-of-sample discipline as everything else this
project does (train/calibration/holdout, no peeking) -- shifted back one
year from the usual train<=2023/cal=2024/holdout>=2025 split
specifically because this dataset's coverage ends in 2024:
train<=2022 / calibration=2023 / **holdout=2024** (100% coverage all
three years, a genuine full-size out-of-sample year, not the thin 33-match
2025 sample this dataset would otherwise force).

| | AUC (holdout=2024, IPL only) |
|---|---:|
| Base rate (sanity check) | 0.500 |
| `ball_length`+`ball_line` alone (genuine delivery quality) | 0.579 |
| `shot_played`+`shot_direction` alone (post-hoc, leaky) | 0.772 |
| All four combined | 0.774 |
| Context only (crude 4-feature proxy: over/phase/wickets-down/innings) | 0.587 |
| Context + clean length/line | 0.613 (**+0.026** over context) |
| Context + all four (includes leakage) | 0.783 (+0.196 -- not a real finding) |

**The genuine signal is real but small**: `ball_length`/`ball_line` adds
+0.026 AUC over a crude context baseline. This is very likely an
*overstatement* of the true incremental value against our actual
production wicket model, which already uses 60 features including
bowler-quality/recency signals that plausibly already capture some of
what "this bowler tends to bowl good-length deliveries" would add.

## Decision: not pursued

Three independent reasons, any one of which would be enough on its own:

1. **Small signal.** +0.026 AUC against a weak baseline is a modest
   effect size before even comparing to the real 60-feature model.
2. **No live-serving path exists.** The dataset is frozen (0% coverage
   for 2026); even a genuinely valuable signal here would need a new
   live-commentary-parsing pipeline (our live pipeline only ingests TOI
   today, never ESPNcricinfo commentary text) -- a real infrastructure
   project, not a feature-engineering task.
3. **Coverage gap even historically**: IPL-only, T20I too sparse to use,
   meaning even a research-only model would need graceful degradation
   for the majority of matches.

No code or model changes. No candidate directories created in
`models/candidates/`. Downloaded dataset kept in the session scratchpad
only, not added to the repo (133MB zip / 4.4GB extracted CSV, mostly
irrelevant ODI/Test rows for this project's scope, real coverage
limitations documented above) -- if revisited later, re-download from
the Kaggle link above rather than assuming this snapshot is still
current.
