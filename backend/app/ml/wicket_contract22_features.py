"""Live-time feature computation for contract22_wicket_v2_batter_state
(replaced contract22_wicket_rigorous 2026-08-22 -- see
docs/candidate_ipl_wicket_v7_2_spell_features.md's third 2026-08-22
update: adds new-batter/partnership-age signal for a real, measured
Brier skill score improvement, 3.52% -> 3.58%).

Reproduces the same features train_contract22_rigorous.py's build_enriched()
(plus v2's added partnership/new-batter state) compute offline per
training row, split into precomputed snapshots (too expensive to
recompute per prediction) plus cheap per-call state:

- Batter career stats, venue par scores, and name-alias resolution are
  reused directly from data/live/run_range_v3_*.json -- same underlying
  quantities as this model's bat_career_*/venue_* features, already built
  and validated for run_range_enriched_v3_batting_style.
- Bowler career stats, batter-vs-bowler head-to-head, and bowler-phase
  history: data/live/wicket_contract22_*.json
  (build_wicket_contract22_live_snapshots.py).
- Recency-weighted (match-EWMA, decay=0.95) batter/bowler form, added
  2026-08-23 for contract22_wicket_v9_recency_form: data/live/
  recency_form_*.json (build_recency_weighted_live_snapshots.py), same
  EWMA update as app/ml/recency_weighted_prior_dataset.py.
- New-batter/partnership-age state (active_batter_state, new_batter,
  partnership_legal_ball_age, striker_match_balls, partner_match_balls):
  computed fresh from the current innings' verified deliveries every
  call, identical logic to RunRangeV3FeatureComputer.compute() (same
  batter_match-balls-faced-this-innings tracking) -- no snapshot needed,
  it's purely current-match state.
- Player identity resolution follows the same pattern as
  RunRangeV3FeatureComputer: registry first (exact, what replay supplies),
  falling back to the identity_key-normalized name-alias snapshot when the
  registry doesn't have the name (the genuine live-TOI condition, no
  registry at all). Unresolved names degrade gracefully to zeroed
  stats/"__UNKNOWN__" categoricals, never raise.

Bowler-dependent features (BOWLER_FEATURES, MATCHUP_FEATURES) work whether
or not the current over's bowler is actually known -- validated both ways
(bowler_known vs bowler_unknown barely differ; see the doc above), so this
doesn't need the "leave bowler-dependent features off" complexity
run_range_enriched_v3 has -- if the caller doesn't know the bowler, pass
an empty string and this degrades to the unknown-bowler defaults
naturally, which the model already handles well.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.ml.ipl_identities import (
    UNKNOWN_PLAYER, build_full_name_alias_index, canonical_player_id,
    first_last_key, identity_key,
)
from app.ml.ipl_venue_regime_dataset import (
    DEFAULT_PAR_SCORE, MIN_PRIOR_INNINGS, VENUE_SHRINKAGE_INNINGS, venue_regime,
)
from app.ml.ipl_venues import normalize_ipl_venue

UNKNOWN_CATEGORY = "__UNKNOWN__"


def _empty_batter() -> dict[str, int]:
    return {"balls": 0}


class WicketContract22FeatureComputer:
    """Loads the offline snapshots once, computes live feature rows cheaply."""

    H2H_SHRINKAGE_OVERS = 4.0

    def __init__(self, project_root: Path | None = None) -> None:
        root = project_root or Path(__file__).resolve().parents[3]
        live_dir = root / "data/live"

        def _load(name: str) -> dict:
            path = live_dir / name
            if not path.is_file():
                raise FileNotFoundError(
                    f"Live snapshot missing -- run build_run_range_v3_live_snapshots.py, "
                    f"build_wicket_contract22_live_snapshots.py, and "
                    f"build_recency_weighted_live_snapshots.py first: {path}"
                )
            return json.loads(path.read_text(encoding="utf-8"))

        self._batter_stats: dict[str, dict[str, int]] = _load("run_range_v3_player_stats.json")
        self._venue_stats: dict[str, dict[str, float]] = _load("run_range_v3_venue_stats.json")
        self._name_aliases: dict[str, str] = _load("run_range_v3_name_aliases.json")
        self._bowler_stats: dict[str, dict[str, int]] = _load("wicket_contract22_bowler_stats.json")
        self._h2h_stats: dict[str, dict[str, int]] = _load("wicket_contract22_h2h_stats.json")
        self._bowler_phase_stats: dict[str, dict[str, int]] = _load("wicket_contract22_bowler_phase_stats.json")
        self._recency_batter_stats: dict[str, dict[str, float]] = _load("recency_form_batter_stats.json")
        self._recency_bowler_stats: dict[str, dict[str, float]] = _load("recency_form_bowler_stats.json")

        styles_path = root / "data/external/cricsheet_player_styles.csv"
        self._full_name_aliases: dict[str, str] = build_full_name_alias_index(styles_path)
        self._batting_style: dict[str, str] = {}
        self._bowler_type: dict[str, str] = {}
        if styles_path.is_file():
            import pandas as pd

            from app.ml.player_style_registry import normalize_batting_style

            styles = pd.read_csv(styles_path)
            for cid, batting, bowling in zip(
                styles["cricsheet_id"], styles["batting_style"], styles["bowling_style"]
            ):
                if not isinstance(cid, str):
                    continue
                player_id = f"player:{cid}"
                normalized = normalize_batting_style(batting)
                if normalized != "unknown":
                    self._batting_style[player_id] = normalized
                self._bowler_type[player_id] = self._classify_bowler_type(bowling)

    @staticmethod
    def _classify_bowler_type(style: Any) -> str:
        if not isinstance(style, str):
            return UNKNOWN_CATEGORY
        lowered = style.lower()
        if "spin" in lowered or "break" in lowered or "orthodox" in lowered or "chinaman" in lowered:
            return "spin"
        if "fast" in lowered or "medium" in lowered or "pace" in lowered:
            return "pace"
        return UNKNOWN_CATEGORY

    def _resolve_player_id(self, name: str, registry: Mapping[str, str]) -> str:
        if not name:
            return UNKNOWN_PLAYER
        player_id = canonical_player_id(name, registry)
        if not player_id.startswith("player:unresolved:") and player_id != UNKNOWN_PLAYER:
            return player_id
        key = identity_key(name)
        if key and key in self._name_aliases:
            return self._name_aliases[key]
        # Cricsheet's own naming convention varies by player -- some are
        # stored under their full name, others under initials (e.g.
        # "PJ Cummins", not "Pat Cummins"). Live feeds like TOI
        # consistently use the common first-name form, which the alias
        # snapshot above (Cricsheet-native names) won't match for the
        # initials case. See app.ml.ipl_identities.first_last_key.
        first_last = first_last_key(name)
        if first_last and first_last in self._full_name_aliases:
            return self._full_name_aliases[first_last]
        return player_id

    def _batter_career(self, player_id: str) -> dict[str, float]:
        profile = self._batter_stats.get(player_id)
        if profile is None:
            return {"bat_career_overs_faced": 0.0, "bat_hist_avg_runs_per_over": 0.0, "bat_hist_wicket_rate": 0.0}
        overs = profile["balls"] / 6.0
        return {
            "bat_career_overs_faced": overs,
            "bat_hist_avg_runs_per_over": (profile["runs"] / overs) if overs > 0 else 0.0,
            "bat_hist_wicket_rate": (profile["dismissals"] / profile["balls"]) if profile["balls"] > 0 else 0.0,
        }

    def _bowler_career(self, player_id: str, phase: str) -> dict[str, float]:
        profile = self._bowler_stats.get(player_id, {"balls": 0, "runs": 0, "wickets": 0})
        overs = profile["balls"] / 6.0
        phase_profile = self._bowler_phase_stats.get(f"{player_id}|{phase}", {"balls": 0, "runs": 0, "wickets": 0})
        phase_overs = phase_profile["balls"] / 6.0
        return {
            "bowl_career_overs_bowled": overs,
            "bowl_hist_avg_runs_conceded": (profile["runs"] / overs) if overs > 0 else 0.0,
            "bowl_hist_wicket_rate": (profile["wickets"] / profile["balls"]) if profile["balls"] > 0 else 0.0,
            "bowl_phase_avg_runs": (phase_profile["runs"] / phase_overs) if phase_overs > 0 else 0.0,
        }

    def _batter_recency(self, player_id: str) -> dict[str, float]:
        profile = self._recency_batter_stats.get(player_id)
        if profile is None:
            return {"balls": 0.0, "runs_per_ball": 0.0, "dot_rate": 0.0, "boundary_rate": 0.0, "dismissal_rate": 0.0}
        balls = max(1.0, profile["balls"])
        return {
            "balls": profile["balls"],
            "runs_per_ball": profile["runs"] / balls,
            "dot_rate": profile["dots"] / balls,
            "boundary_rate": profile["boundaries"] / balls,
            "dismissal_rate": profile["dismissals"] / balls,
        }

    def _bowler_recency(self, player_id: str) -> dict[str, float]:
        profile = self._recency_bowler_stats.get(player_id)
        if profile is None:
            return {"balls": 0.0, "economy": 0.0, "wicket_rate": 0.0}
        balls = max(1.0, profile["balls"])
        return {
            "balls": profile["balls"],
            "economy": 6.0 * profile["runs"] / balls,
            "wicket_rate": profile["wickets"] / balls,
        }

    def _h2h(self, striker_id: str, bowler_id: str, bowler_prior_avg: float) -> dict[str, float]:
        # Most tracked pairs have well under an over of shared history
        # (median 5 balls across the live snapshot) -- a raw small-sample
        # average is mostly noise (one boundary off 1 ball reads as a
        # 36-runs/over "matchup"). Shrink toward the bowler's own overall
        # average, weighted by how many overs of real head-to-head exist:
        # SHRINKAGE_OVERS of "trust" in the prior, same idea as a Bayesian
        # pseudo-count. overs=0 collapses cleanly to the bowler's average
        # instead of a misleading 0.0.
        profile = self._h2h_stats.get(f"{striker_id}|{bowler_id}", {"balls": 0, "runs": 0, "wickets": 0})
        overs = profile["balls"] / 6.0
        raw_avg = (profile["runs"] / overs) if overs > 0 else bowler_prior_avg
        shrunk_avg = (overs * raw_avg + self.H2H_SHRINKAGE_OVERS * bowler_prior_avg) / (
            overs + self.H2H_SHRINKAGE_OVERS
        )
        return {
            "h2h_overs": overs,
            "h2h_avg_runs": shrunk_avg,
        }

    def _venue_state(self, venue_name: str) -> dict[str, Any]:
        normalized = normalize_ipl_venue(venue_name)
        entry = self._venue_stats.get(normalized)
        prior_innings = int(entry["prior_innings"]) if entry else 0
        global_entry = self._venue_stats.get("__global__")
        global_avg = (
            global_entry["par_score"]
            if global_entry and global_entry["prior_innings"] > 0
            else DEFAULT_PAR_SCORE
        )
        if entry is not None and prior_innings > 0:
            # Shrink the venue's own average toward the global average,
            # weighted by prior_innings -- removes the hard cliff where a
            # venue with 1-4 prior innings got zero credit for its own
            # history. At MIN_PRIOR_INNINGS+ the venue's own average
            # already dominates the blend.
            par_score = (
                prior_innings * entry["par_score"] + VENUE_SHRINKAGE_INNINGS * global_avg
            ) / (prior_innings + VENUE_SHRINKAGE_INNINGS)
            source = "venue" if prior_innings >= MIN_PRIOR_INNINGS else "venue_blended"
        elif global_entry and global_entry["prior_innings"] > 0:
            par_score, source = global_avg, "global"
        else:
            par_score, source = DEFAULT_PAR_SCORE, "default"
        return {
            "venue_par_score": par_score,
            "venue_prior_innings": prior_innings,
            "venue_par_source": source,
            "venue_scoring_regime": venue_regime(par_score, prior_innings),
        }

    def _partnership_state(
        self, deliveries: Sequence[Any], registry: Mapping[str, str],
        striker_id: str, partner_id: str, bowler_id: str,
    ) -> dict[str, Any]:
        # Identical logic to RunRangeV3FeatureComputer.compute()'s
        # batter_match tracking -- balls faced THIS innings per batter,
        # from the verified deliveries so far. Bowler legal balls this
        # innings are tracked in the same pass (a bowler on their last
        # permitted over, or a part-timer pressed into an unusual over,
        # bowls differently than one mid-spell -- previously unmodeled).
        batter_match: dict[str, dict[str, int]] = {}
        bowler_balls_this_innings = 0
        # Spell tracking: walk overs in chronological order (first ball of
        # each new over number) and detect who bowled each one, to tell a
        # continuous spell apart from a bowler coming BACK after being
        # taken off -- a well-known tactical signal (captains hold a
        # strike bowler for a return spell at the death; batters respond
        # differently the second time facing the same bowler) that
        # bowler_match_overs_bowled alone can't distinguish (it just
        # counts total overs bowled this innings, spell or no spell).
        current_spell_bowler = ""
        current_spell_length = 0
        bowler_spell_count: dict[str, int] = {}
        seen_overs: set[int] = set()
        for delivery in deliveries:
            over_number = int(delivery.over)
            if over_number not in seen_overs:
                seen_overs.add(over_number)
                over_bowler = self._resolve_player_id(str(delivery.bowler), registry)
                if over_bowler == current_spell_bowler:
                    current_spell_length += 1
                else:
                    current_spell_bowler = over_bowler
                    current_spell_length = 1
                    bowler_spell_count[over_bowler] = (
                        bowler_spell_count.get(over_bowler, 0) + 1
                    )
            if not bool(getattr(delivery, "is_legal", True)):
                continue
            batter = self._resolve_player_id(str(delivery.striker), registry)
            batter_match.setdefault(batter, _empty_batter())["balls"] += 1
            if bowler_id != UNKNOWN_PLAYER:
                bowler = self._resolve_player_id(str(delivery.bowler), registry)
                if bowler == bowler_id:
                    bowler_balls_this_innings += 1

        if bowler_id == UNKNOWN_PLAYER:
            spell_over_number = 0
            is_return_spell = 0
        elif bowler_id == current_spell_bowler:
            # Continuing the active spell.
            spell_over_number = current_spell_length + 1
            is_return_spell = int(bowler_spell_count.get(bowler_id, 0) > 1)
        else:
            # Starting a new spell -- a return spell if they've bowled
            # before this innings, their first spell otherwise.
            spell_over_number = 1
            is_return_spell = int(bowler_spell_count.get(bowler_id, 0) >= 1)

        striker_balls = batter_match.get(striker_id, _empty_batter())["balls"]
        # Genuine live TOI has no non-striker field at all (ToiDelivery
        # carries only the current striker per ball) -- partner_id is
        # UNKNOWN_PLAYER whenever the caller can't supply one. Don't let
        # that manufacture a false 0-balls partner and force "new_batter"
        # on every single call regardless of the real match state: fall
        # back to judging newness from the striker alone.
        if partner_id == UNKNOWN_PLAYER:
            partner_balls = 0
            is_new = int(striker_balls <= 2)
        else:
            partner_balls = batter_match.get(partner_id, _empty_batter())["balls"]
            is_new = int(min(striker_balls, partner_balls) <= 2)
        return {
            "active_batter_state": "new_batter" if is_new else "established_pair",
            "new_batter": is_new,
            "partnership_legal_ball_age": striker_balls + partner_balls,
            "striker_match_balls": striker_balls,
            "partner_match_balls": partner_balls,
            "bowler_match_overs_bowled": bowler_balls_this_innings / 6.0,
            "bowler_spell_over_number": spell_over_number,
            "bowler_is_return_spell": is_return_spell,
        }

    def compute(
        self,
        *,
        registry: Mapping[str, str],
        striker_name: str,
        non_striker_name: str = "",
        bowler_name: str,
        venue_name: str,
        phase: str,
        deliveries: Sequence[Any] = (),
    ) -> dict[str, Any]:
        striker_id = self._resolve_player_id(striker_name, registry)
        bowler_id = self._resolve_player_id(bowler_name, registry) if bowler_name else UNKNOWN_PLAYER
        partner_id = self._resolve_player_id(non_striker_name, registry) if non_striker_name else UNKNOWN_PLAYER

        result: dict[str, Any] = {}
        result.update(self._batter_career(striker_id))
        result["striker_batting_style"] = self._batting_style.get(striker_id, UNKNOWN_CATEGORY)
        bowler_career = self._bowler_career(bowler_id, phase)
        result.update(bowler_career)
        result["bowler_type"] = self._bowler_type.get(bowler_id, UNKNOWN_CATEGORY)
        result.update(self._h2h(striker_id, bowler_id, bowler_career["bowl_hist_avg_runs_conceded"]))
        result.update(self._venue_state(venue_name))
        result.update(self._partnership_state(deliveries, registry, striker_id, partner_id, bowler_id))

        striker_recency = self._batter_recency(striker_id)
        result["striker_recency_balls"] = striker_recency["balls"]
        result["striker_recency_runs_per_ball"] = striker_recency["runs_per_ball"]
        result["striker_recency_dot_rate"] = striker_recency["dot_rate"]
        result["striker_recency_boundary_rate"] = striker_recency["boundary_rate"]
        result["striker_recency_dismissal_rate"] = striker_recency["dismissal_rate"]
        result["partner_recency_runs_per_ball"] = self._batter_recency(partner_id)["runs_per_ball"]
        bowler_recency = self._bowler_recency(bowler_id)
        result["bowler_recency_balls"] = bowler_recency["balls"]
        result["bowler_recency_economy"] = bowler_recency["economy"]
        result["bowler_recency_wicket_rate"] = bowler_recency["wicket_rate"]
        return result
