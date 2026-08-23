# The 42%/27.86% wicket promotion gate blocks candidates against a product feature that doesn't exist

Date: 2026-08-23. Roadmap priority #5, flagged as a standing open question
since `docs/candidate_ipl_wicket_v7_2_spell_features.md`'s 2026-08-22
closing notes and `docs/CTO_HANDOVER_2026-07-24.md`: "a product decision on
whether the 42%/27.86% gate pair is achievable at all for this event rate."
Eleven candidates (`ipl_wicket_v7_1` through `v7_11`) were built and tested
against this pair and every one either missed it or landed within ~1 point
of clearing both simultaneously — never fully cleared.

## What the gate actually measures

`TARGET_PRECISION = 0.42` / `TARGET_RECALL = 0.2786`, hardcoded across all
eleven `train_ipl_wicket_v7_*.py` scripts: at some probability threshold,
predicted "wicket alert" rows must be right at least 42% of the time,
while still catching at least 27.86% of real wickets. This is a **binary
alert** framing — pick a threshold, call everything above it "high risk,"
and hold that decision to a precision/recall bar.

## What's actually live

Traced every consumer of `wicket_probability` end to end:

- `app/ml/prediction_engine.py` outputs a single calibrated float
  (`wicket_probability=round(evolved_wkt, 3)`) from
  `contract22_wicket_v2_batter_state` -- a continuous number, not a
  boolean.
- `app/live/pipeline.py`'s `_telegram_text` -- the only place that renders
  wicket risk to an end user -- prints it as plain text: `f"Wicket risk
  (phase baseline): {probability:.1f}%"`. No threshold, no "ALERT"
  label, no color coding tied to wicket risk (the emoji/color logic in
  that function is for `confidence_level`, a *run-range* confidence
  concept, unrelated to wicket risk at all).
- Grepped the whole `app/` tree and `dashboard.html` for
  `wicket_risk`/`wicket_alert`/`high_risk`/`risk_level`: zero matches
  anywhere outside the training/evaluation scripts themselves.

**There is no binary wicket alert feature anywhere in this product.** The
gate that blocked eleven candidates evaluates a UI decision that was never
built.

## How the model that's actually live got promoted instead

`contract22_wicket_v2_batter_state` (what's live today) was never
subjected to this gate at all -- its own training script
(`train_contract22_wicket_v2_batter_state.py`) has no
`TARGET_PRECISION`/`TARGET_RECALL` constants, no threshold search, no
promotion `decision` field. Per
`candidate_ipl_wicket_v7_2_spell_features.md`'s "2026-08-22 follow-up"
section, it (and its predecessor `contract22_wicket_rigorous`) were
promoted purely on **Brier skill score improvement over the currently-live
model**, the same style already standard for run-range promotions
(`beats_v7_blended_and_calibration_gate` in the run-range scripts). The
42%/27.86% gate was a parallel, stricter bar that only ever applied to the
separate CatBoost/`ipl_wicket_v7.x` research lineage -- which was
ultimately abandoned in favor of `contract22_wicket_v2_batter_state`
through a completely different, ungated evaluation.

This means the gate's practical effect wasn't "protect production quality"
-- it silently killed an entire research lineage (several of whose
candidates had *better* AUC/Brier than what shipped, per the same
follow-up section) against a bar the shipped model was never held to.

## Recommendation

**Retire the 42%/27.86% binary-alert gate as a promotion requirement.**
It's not calibrated to a real product surface, and holding future wicket
candidates to it (while the actual live model was promoted a different
way) is inconsistent and needlessly conservative. Going forward, evaluate
wicket candidates the same way `contract22_wicket_v2_batter_state` and
every run-range candidate already are: Brier skill score / AUC
improvement over the real current live baseline, calibration not
regressing, reported IPL-only and T20I-only per standing practice (see
`finding_wicket_v7x_t20i_lever_does_not_transfer.md`).

**If a binary wicket alert is ever actually built** (e.g. a Telegram
"⚠️ high wicket risk" flag), design its precision/recall target from that
feature's real cost/benefit trade-off at that time -- not by carrying
forward a number whose origin is no longer traceable in this codebase (no
docstring or commit explains why 0.42/0.2786 specifically, only that they
were used consistently across the v7.x line once set).

No code or model changes from this finding -- it's a promotion-policy
correction. No candidate is retroactively promoted; this only changes how
*future* wicket candidates should be evaluated.
