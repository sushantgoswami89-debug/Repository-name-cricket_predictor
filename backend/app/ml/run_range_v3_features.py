"""Live-time feature computation for run_range_enriched_v3_batting_style.

Reproduces, at prediction time, the same features
`build_ipl_phase_moe_features` / `build_ipl_venue_regime_dataset` compute
offline per training row -- split into two sources:

- Player career stats and venue par scores: precomputed snapshots
  (`build_run_range_v3_live_snapshots.py` -> `data/live/run_range_v3_*.json`),
  since these need the full historical match corpus and are too expensive
  to recompute per prediction. Re-run that script periodically to keep
  them current; this module only reads the snapshot files.
- In-match state (partnership age, batter balls faced this innings, recent
  12-ball momentum, wickets pressure, chase pressure): computed fresh from
  the current innings' verified deliveries every call, since it's cheap
  and must always reflect the live match exactly.

Player identity resolution (fixed 2026-08-22): tries the caller-supplied
registry first (exact, what replay uses -- real Cricsheet match JSON
carries its own name -> person UUID registry). If a name isn't in that
registry (the genuine-live-TOI case, where only a plain name string is
available, no registry at all), falls back to
`data/live/run_range_v3_name_aliases.json` -- an `identity_key()`-normalized
name -> canonical_player_id lookup built from every squad-list appearance
across all eligible historical matches (`build_run_range_v3_live_snapshots.py`).
Same technique `BowlerSpellAdjuster` already uses for the equivalent
wicket-model problem, applied here to this model's own IPL+T20I
population. Still not full fuzzy matching -- a TOI name that doesn't
normalize to any historical spelling (e.g. a brand-new international
debutant with zero prior Cricsheet appearances) won't resolve; this falls
back to "__UNKNOWN__"/zeroed prior stats gracefully (the existing
cold-start convention), it does not raise.
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
from app.ml.ipl_venues import normalize_ipl_venue, team_venue_context

UNKNOWN_CATEGORY = "__UNKNOWN__"


def _empty_batter() -> dict[str, int]:
    return {"balls": 0, "runs": 0, "dots": 0, "boundaries": 0, "dismissals": 0}


def _rate(profile: dict[str, int], event: str) -> float:
    return float(profile[event]) / max(1, int(profile["balls"]))


def _pressure_state(wickets_down: int, recent: list[dict[str, int]], innings_rate: float) -> str:
    recent_wickets = sum(item["wicket"] for item in recent)
    recent_rate = sum(item["runs"] for item in recent) / len(recent) if recent else innings_rate
    if recent_wickets >= 2 or wickets_down >= 7:
        return "wicket_pressure"
    if len(recent) >= 6 and recent_rate >= innings_rate + 0.35:
        return "accelerating"
    return "stable"


def _chase_pressure(required_rate: float, current_rate: float, is_chase: bool) -> str:
    if not is_chase:
        return "not_chasing"
    gap = required_rate - current_rate
    if gap >= 3:
        return "high"
    if gap >= 1:
        return "medium"
    return "low"


class RunRangeV3FeatureComputer:
    """Loads the offline snapshots once, computes live feature rows cheaply."""

    def __init__(self, project_root: Path | None = None) -> None:
        root = project_root or Path(__file__).resolve().parents[3]
        live_dir = root / "data/live"
        player_path = live_dir / "run_range_v3_player_stats.json"
        venue_path = live_dir / "run_range_v3_venue_stats.json"
        if not player_path.is_file() or not venue_path.is_file():
            raise FileNotFoundError(
                "Live snapshots missing -- run build_run_range_v3_live_snapshots.py first: "
                f"{player_path}, {venue_path}"
            )
        self._player_stats: dict[str, dict[str, int]] = json.loads(player_path.read_text(encoding="utf-8"))
        self._venue_stats: dict[str, dict[str, float]] = json.loads(venue_path.read_text(encoding="utf-8"))
        batter_phase_path = live_dir / "run_range_v3_batter_phase_stats.json"
        self._batter_phase_stats: dict[str, dict[str, int]] = (
            json.loads(batter_phase_path.read_text(encoding="utf-8"))
            if batter_phase_path.is_file() else {}
        )
        alias_path = live_dir / "run_range_v3_name_aliases.json"
        self._name_aliases: dict[str, str] = (
            json.loads(alias_path.read_text(encoding="utf-8")) if alias_path.is_file() else {}
        )
        # Bowler career/phase stats: same snapshots
        # build_wicket_contract22_live_snapshots.py already builds and the
        # wicket model already loads -- this model previously had none of
        # this at all, no need for a second copy of the same data.
        bowler_stats_path = live_dir / "wicket_contract22_bowler_stats.json"
        bowler_phase_path = live_dir / "wicket_contract22_bowler_phase_stats.json"
        self._bowler_stats: dict[str, dict[str, int]] = (
            json.loads(bowler_stats_path.read_text(encoding="utf-8"))
            if bowler_stats_path.is_file() else {}
        )
        self._bowler_phase_stats: dict[str, dict[str, int]] = (
            json.loads(bowler_phase_path.read_text(encoding="utf-8"))
            if bowler_phase_path.is_file() else {}
        )
        styles_path = root / "data/external/cricsheet_player_styles.csv"
        self._full_name_aliases: dict[str, str] = build_full_name_alias_index(styles_path)
        self._batting_style: dict[str, str] = {}
        self._bowler_type: dict[str, str] = {}
        if styles_path.is_file():
            import pandas as pd

            from app.ml.player_style_registry import normalize_batting_style

            styles = pd.read_csv(styles_path)
            for cricsheet_id, batting, bowling in zip(
                styles["cricsheet_id"], styles["batting_style"], styles["bowling_style"]
            ):
                if not isinstance(cricsheet_id, str):
                    continue
                normalized = normalize_batting_style(batting)
                if normalized != "unknown":
                    self._batting_style[f"player:{cricsheet_id}"] = normalized
                self._bowler_type[f"player:{cricsheet_id}"] = self._classify_bowler_type(bowling)

    @staticmethod
    def _classify_bowler_type(style: Any) -> str:
        # Same bucketing as WicketContract22FeatureComputer's own copy --
        # kept duplicated rather than shared, matching this pair of
        # feature computers' existing style (venue_state/batting_style
        # loading are already duplicated the same way).
        if not isinstance(style, str):
            return UNKNOWN_CATEGORY
        lowered = style.lower()
        if "spin" in lowered or "break" in lowered or "orthodox" in lowered or "chinaman" in lowered:
            return "spin"
        if "fast" in lowered or "medium" in lowered or "pace" in lowered:
            return "pace"
        return UNKNOWN_CATEGORY

    def _resolve_player_id(self, name: str, registry: Mapping[str, str]) -> str:
        """Registry-based resolution first (exact, what replay supplies);
        falls back to the identity_key-normalized name-alias snapshot for
        names the registry doesn't cover (the genuine-live-TOI case, where
        `registry` is typically empty)."""
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

    def _batter_prior(self, player_id: str) -> dict[str, float]:
        profile = self._player_stats.get(player_id)
        if profile is None:
            return {"balls": 0, "runs_per_ball": 0.0, "dot_rate": 0.0, "boundary_rate": 0.0, "dismissal_rate": 0.0}
        balls = max(1, profile["balls"])
        return {
            "balls": profile["balls"],
            "runs_per_ball": profile["runs"] / balls,
            "dot_rate": profile["dots"] / balls,
            "boundary_rate": profile["boundaries"] / balls,
            "dismissal_rate": profile["dismissals"] / max(1, profile["balls"]),
        }

    def _batter_phase(self, player_id: str, phase: str) -> dict[str, float]:
        # Mirrors _batter_prior's shape, keyed additionally by phase --
        # same convention as _bowler_career's phase lookup below.
        profile = self._batter_phase_stats.get(
            f"{player_id}|{phase}", {"balls": 0, "runs": 0, "dots": 0, "boundaries": 0, "dismissals": 0}
        )
        balls = max(1, profile["balls"])
        return {
            "batter_phase_balls": profile["balls"],
            "batter_phase_runs_per_ball": profile["runs"] / balls,
            "batter_phase_boundary_rate": profile["boundaries"] / balls,
            "batter_phase_dismissal_rate": profile["dismissals"] / balls,
        }

    def _venue_state(self, venue_name: str) -> dict[str, Any]:
        # Mirrors build_ipl_venue_regime_dataset's fallback exactly:
        # venue_prior_innings is always this VENUE's own count (0 if never
        # seen), regardless of which par_source ends up used.
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
            "normalized_venue": normalized,
            "venue_par_score": par_score,
            "venue_prior_innings": prior_innings,
            "venue_par_source": source,
            "venue_scoring_regime": venue_regime(par_score, prior_innings),
        }

    def _bowler_career(self, player_id: str, phase: str) -> dict[str, float]:
        # Same shape as WicketContract22FeatureComputer._bowler_career.
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

    def compute(
        self,
        *,
        deliveries: Sequence[Any],
        registry: Mapping[str, str],
        striker_name: str,
        non_striker_name: str,
        venue_name: str,
        batting_team: str,
        phase: str,
        over: int,
        wickets_down: int,
        current_rate: float,
        required_rate: float,
        is_chase: bool,
        bowler_name: str = "",
    ) -> dict[str, Any]:
        """`deliveries` are this innings' verified deliveries so far, sorted
        (over, ball) ascending -- e.g. `LiveDeliveryVerifier.accepted.values()`,
        or a replay's equivalent. Each must expose `.striker`, `.bowler`,
        `.over`, `.total_runs`, `.batter_runs`, `.wicket_kind`, `.is_legal`
        (matches `ToiDelivery`). `bowler_name` is the upcoming over's
        bowler if known -- often isn't in real live serving, in which case
        the bowler-dependent features below degrade to their unknown
        defaults, same convention as WicketContract22FeatureComputer.
        """

        striker_id = self._resolve_player_id(striker_name, registry)
        partner_id = self._resolve_player_id(non_striker_name, registry)
        bowler_id = self._resolve_player_id(bowler_name, registry) if bowler_name else UNKNOWN_PLAYER

        batter_match: dict[str, dict[str, int]] = {}
        recent: list[dict[str, int]] = []
        wickets_seen = 0
        bowler_balls_this_innings = 0
        # Spell tracking -- see WicketContract22FeatureComputer._partnership_state
        # for the full reasoning (this model had none of it before).
        current_spell_bowler = ""
        current_spell_length = 0
        bowler_spell_count: dict[str, int] = {}
        seen_overs: set[int] = set()
        for delivery in deliveries:
            over_number = int(getattr(delivery, "over", 0))
            if over_number not in seen_overs:
                seen_overs.add(over_number)
                over_bowler = self._resolve_player_id(str(getattr(delivery, "bowler", "")), registry)
                if over_bowler == current_spell_bowler:
                    current_spell_length += 1
                else:
                    current_spell_bowler = over_bowler
                    current_spell_length = 1
                    bowler_spell_count[over_bowler] = bowler_spell_count.get(over_bowler, 0) + 1
            batter = self._resolve_player_id(str(delivery.striker), registry)
            legal = bool(getattr(delivery, "is_legal", True))
            total_runs = int(delivery.total_runs)
            batter_runs = int(delivery.batter_runs)
            wicket = int(getattr(delivery, "wicket_kind", None) is not None)
            wickets_seen += wicket
            if legal:
                profile = batter_match.setdefault(batter, _empty_batter())
                profile["balls"] += 1
                profile["runs"] += batter_runs
                profile["dots"] += int(total_runs == 0)
                profile["boundaries"] += int(batter_runs in {4, 6})
                recent.append({
                    "runs": total_runs, "dot": int(total_runs == 0),
                    "boundary": int(batter_runs in {4, 6}), "wicket": wicket,
                })
                if bowler_id != UNKNOWN_PLAYER:
                    over_bowler = self._resolve_player_id(str(getattr(delivery, "bowler", "")), registry)
                    if over_bowler == bowler_id:
                        bowler_balls_this_innings += 1
        recent = recent[-12:]

        if bowler_id == UNKNOWN_PLAYER:
            spell_over_number = 0
            is_return_spell = 0
        elif bowler_id == current_spell_bowler:
            spell_over_number = current_spell_length + 1
            is_return_spell = int(bowler_spell_count.get(bowler_id, 0) > 1)
        else:
            spell_over_number = 1
            is_return_spell = int(bowler_spell_count.get(bowler_id, 0) >= 1)

        striker_match = batter_match.get(striker_id, _empty_batter())
        partner_match = batter_match.get(partner_id, _empty_batter())
        pair_age = striker_match["balls"] + partner_match["balls"]
        is_new = int(min(striker_match["balls"], partner_match["balls"]) <= 2)

        striker_prior = self._batter_prior(striker_id)
        partner_prior = self._batter_prior(partner_id)
        venue_state = self._venue_state(venue_name)
        bowler_career = self._bowler_career(bowler_id, phase)
        striker_phase = self._batter_phase(striker_id, phase)

        return {
            "active_batter_state": "new_batter" if is_new else "established_pair",
            "new_batter": is_new,
            "partnership_legal_ball_age": pair_age,
            "wickets_remaining_bucket": str(max(0, 10 - wickets_down)),
            "state_regime": _pressure_state(wickets_down, recent, current_rate / 6 if current_rate else 0.0),
            "chase_pressure": _chase_pressure(required_rate, current_rate, is_chase),
            "striker_match_balls": striker_match["balls"],
            "partner_match_balls": partner_match["balls"],
            "striker_prior_balls": striker_prior["balls"],
            "striker_prior_runs_per_ball": striker_prior["runs_per_ball"],
            "striker_prior_dot_rate": striker_prior["dot_rate"],
            "striker_prior_boundary_rate": striker_prior["boundary_rate"],
            "striker_prior_dismissal_rate": striker_prior["dismissal_rate"],
            "partner_prior_balls": partner_prior["balls"],
            "partner_prior_runs_per_ball": partner_prior["runs_per_ball"],
            "partner_prior_dot_rate": partner_prior["dot_rate"],
            "partner_prior_boundary_rate": partner_prior["boundary_rate"],
            "batter_phase_balls": striker_phase["batter_phase_balls"],
            "batter_phase_runs_per_ball": striker_phase["batter_phase_runs_per_ball"],
            "batter_phase_boundary_rate": striker_phase["batter_phase_boundary_rate"],
            "batter_phase_dismissal_rate": striker_phase["batter_phase_dismissal_rate"],
            "striker_batting_style": self._batting_style.get(striker_id, UNKNOWN_CATEGORY),
            "venue_par_score": venue_state["venue_par_score"],
            "venue_prior_innings": venue_state["venue_prior_innings"],
            "venue_scoring_regime": venue_state["venue_scoring_regime"],
            "venue_par_source": venue_state["venue_par_source"],
            "batting_team_venue_context": team_venue_context(batting_team, venue_name),
            "phase_venue_regime": f"{phase}|{venue_state['venue_scoring_regime']}",
            "bowl_career_overs_bowled": bowler_career["bowl_career_overs_bowled"],
            "bowl_hist_avg_runs_conceded": bowler_career["bowl_hist_avg_runs_conceded"],
            "bowl_hist_wicket_rate": bowler_career["bowl_hist_wicket_rate"],
            "bowl_phase_avg_runs": bowler_career["bowl_phase_avg_runs"],
            "bowler_type": self._bowler_type.get(bowler_id, UNKNOWN_CATEGORY),
            "bowler_match_overs_bowled": bowler_balls_this_innings / 6.0,
            "bowler_spell_over_number": spell_over_number,
            "bowler_is_return_spell": is_return_spell,
        }
