# Historical Replay Anomalies

This register records source matches that cannot pass CricketBaba's strict live
match rules. Entries are investigated individually. Historical facts are not
silently rewritten, and live-match safeguards remain enabled.

All matches in this register are excluded from prediction-model training and
evaluation. CricketBaba continues to apply its strict standard rule profile to
included ICC-recognized international and IPL data.

## 1179017.json — confirmed historical bowling-limit violation

- Match: Namibia Women vs Botswana Women, 6th T20I, 3 April 2019
- Source state: KA Green bowled 4.2 overs (26 legal deliveries) in a 20-over match.
- CricketBaba result: rejected when KA Green is selected for a fifth over after
  completing the normal four-over allocation.
- Classification: confirmed real-world umpiring error; Cricsheet accurately
  represents the recorded scorecard.
- Super Over check: not applicable. The match has two ordinary innings, was won
  by 86 runs, and contains no innings marked `super_over`. A legitimate Super
  Over is stored as a separate innings with a fresh bowling allocation.
- Decision: retain the source data unchanged and retain CricketBaba's strict
  four-over live-match rule. Exclude this match from strict-rule model evaluation
  unless a future archival-tolerance mode explicitly labels the violation.
- External confirmation: ESPN's scorecard notes that KA Green bowled an illegal
  over past the limit of four and identifies the standing umpire.

## 1267304.json — confirmed scorecard anomaly, cause unresolved

- Match: Greece vs Romania, Sofia Twenty20, 24 June 2021
- Source state: Asrar Ahmed is credited with five complete overs (30 legal
  deliveries) while Romania chased 158 in 12.5 overs.
- CricketBaba result: rejected when Asrar Ahmed is selected for a fifth over
  after completing the normal four-over allocation.
- Super Over check: not applicable. The match was won by three wickets and has
  only two ordinary innings.
- External confirmation: ESPN's current scorecard also lists Asrar Ahmed with
  figures of 5-0-68-2. Unlike match 1179017, it provides no umpiring note that
  explains the illegal allocation.
- Classification: confirmed historical scorecard anomaly. The underlying cause
  may be an on-field bowling-limit error or incorrect bowler attribution, but
  the available authoritative scorecard does not distinguish between them.
- Decision: retain the source unchanged, retain the strict four-over live rule,
  and exclude this match from strict-rule evaluation unless archival-tolerance
  mode explicitly labels the violation.

## 1283040.json — inconsistent bowler attribution across consecutive overs

- Match: Malawi vs Rwanda, ICC Men's T20 World Cup Sub Regional Africa
  Qualifier, 22 October 2021
- Source state: Sami Sohail is assigned the first five legal balls of over 3,
  M Abdulla is assigned its sixth ball, and M Abdulla then bowls all of over 4.
- CricketBaba result: rejected because a bowler may not complete part of one over
  and then bowl the immediately following over.
- Super Over check: not applicable. Malawi won by 24 runs and the file contains
  only two ordinary innings.
- External comparison: published scorecards disagree on the allocation. One
  lists M Abdulla at 3.3 overs while another lists 3.2; Sami Sohail is reported
  with four overs. This is consistent with a disputed or incorrectly attributed
  delivery at the end of over 3.
- Classification: source bowler-attribution inconsistency. The available records
  do not support silently choosing which bowler delivered the disputed ball.
- Decision: retain the source unchanged, retain the consecutive-over safeguard,
  and exclude this match from strict-rule evaluation unless archival-tolerance
  mode explicitly labels the inconsistency.

## 1419932.json — confirmed on-field bowling-limit violation

- Match: Malaysia Women vs Japan Women, Asian Cricket Council Women's Premier
  Cup quarter-final, 14 February 2024
- Official status: sanctioned women's T20 international, match type number 1780.
- Source state: Nur Dania Syuhada bowled 4.1 overs (25 legal deliveries) in a
  scheduled 20-over innings.
- Super Over check: not applicable. Malaysia won by 16 runs and the file contains
  only two ordinary innings.
- External confirmation: ESPN and CricketArchive both record 4.1 overs and
  explicitly state that she was erroneously allowed to exceed the four-over
  limit.
- Classification: confirmed on-field rule violation in an official match; not
  an informal-match playing-condition variation and not corrupted source data.
- Decision: keep the match eligible as official sanctioned cricket. Strict live
  mode must reject the fifth over; archival evaluation may reproduce it only
  with a `bowler_over_limit` exception label.

## 1490882.json — ball-by-ball bowler attribution conflicts with scorecard

- Match: France vs Malta, Continental Cup, 26 June 2025
- Official status: men's T20 international, match type number 3264.
- Local source state: Dawood Ahmadzai is assigned overs 3, 5, 7, 9, and 11,
  totalling 5-0-28-0.
- Super Over check: not applicable. Malta won by six wickets in 13.4 overs and
  the file contains only two ordinary innings.
- External comparison: current ESPN and other published scorecards list Dawood
  Ahmadzai at 4-0-23-0. The extra locally attributed over conceded five runs,
  exactly accounting for the difference between 23 and 28.
- Classification: ball-by-ball bowler-attribution error in the local revision,
  most likely affecting over 11. This is not evidence that Dawood was permitted
  to bowl a fifth over.
- Decision: do not silently rewrite the archived delivery data. Keep the match
  eligible as official cricket, label the affected replay row
  `bowler_over_limit`, and prefer a newer corrected source revision if one
  becomes available.

## 1499795.json — final-over bowler attribution conflicts with scorecard

- Match: Samoa Women vs Papua New Guinea Women, ICC Women's T20 World Cup East
  Asia Pacific Qualifier, 12 September 2025
- Official status: ICC women's T20 World Cup qualifier, match type number 2502.
- Local source state: AF Aoina is assigned overs 7, 9, 15, 17, plus the three
  legal deliveries of over 19, totalling 4.3 overs and 28 runs.
- Super Over check: not applicable. Samoa won by 13 runs and the file contains
  only two ordinary innings.
- External comparison: current ESPN scorecards list AF Aoina at 4-0-24-3. The
  locally attributed final three deliveries conceded four runs, exactly
  accounting for the difference between 24 and 28.
- Classification: ball-by-ball bowler-attribution error affecting over 19. The
  official aggregate figures do not support AF Aoina bowling a fifth over.
- Decision: keep this official match eligible, do not silently rewrite the local
  delivery data, label the affected replay row `bowler_over_limit`, and prefer a
  corrected source revision when available.

## Live TOI run — 26 July 2026

- TOI scorecard and commentary frequently updated out of sync, delaying
  otherwise completed-over predictions.
- TOI revised previously published deliveries, forcing verifier rebases.
- A verifier rebase reused an engine awaiting prior-over actuals and crashed
  with `FEED REQUIRED: Over 15 missing`; fixed by rebuilding the engine during
  a rebase.
- Restarting the service cleared the pending prediction, so the next Telegram
  message could not include the previous-over review.
- Some wicket probabilities repeated across adjacent states; continue tracing
  raw model output, calibration, and display rounding before promotion.
- Missing feature: generate and send a pre-innings prediction for over 1
  before play begins; the current pipeline predicts only after an over boundary.
- State-sensitivity defect: after two wickets fell in over 4, the next run
  prediction barely changed (10.2 to 10.1) and retained the same 8-11 range.
  Audit whether recent wickets materially influence the run model and widen or
  shift ranges after multi-wicket overs.
