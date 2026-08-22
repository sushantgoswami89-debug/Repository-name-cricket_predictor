"""Fail-closed delivery reconciliation for live cricket feeds."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.live.toi_reader import ToiDelivery, ToiSnapshot


class VerificationError(ValueError):
    """A delivery conflicts with the independently accumulated match state."""


@dataclass(slots=True)
class LiveDeliveryVerifier:
    """Accept each delivery once and reconcile score/wickets after every ball."""

    match_id: str
    innings: int
    score: int = 0
    wickets: int = 0
    last_key: tuple[int, int, int] | None = None
    accepted: dict[tuple[int, int, int], ToiDelivery] = field(default_factory=dict)
    over_runs: dict[int, int] = field(default_factory=dict)
    over_wickets: dict[int, int] = field(default_factory=dict)

    @staticmethod
    def _same_cricket_state(first: ToiDelivery, second: ToiDelivery) -> bool:
        """Ignore TOI prose enrichment while protecting scoring facts."""
        return (
            first.match_id,
            first.innings,
            first.key,
            first.total_runs,
            first.batter_runs,
            first.extras,
            first.wicket_kind,
            first.feed_total,
            first.feed_wickets,
        ) == (
            second.match_id,
            second.innings,
            second.key,
            second.total_runs,
            second.batter_runs,
            second.extras,
            second.wicket_kind,
            second.feed_total,
            second.feed_wickets,
        )

    def apply(self, delivery: ToiDelivery) -> bool:
        """Apply one delivery. Return False only for an identical duplicate."""
        if delivery.match_id != self.match_id or delivery.innings != self.innings:
            raise VerificationError("Delivery belongs to a different match or innings.")
        existing = self.accepted.get(delivery.key)
        if existing is not None:
            if not self._same_cricket_state(existing, delivery):
                raise VerificationError(
                    f"Delivery {delivery.key} was silently revised by the feed."
                )
            # Retain later commentary/player enrichment without reapplying runs.
            self.accepted[delivery.key] = delivery
            return False
        if self.last_key is not None and delivery.key <= self.last_key:
            raise VerificationError(
                f"Out-of-order delivery {delivery.key} after {self.last_key}."
            )
        # A full-population replay of every offline IPL/T20I match (6,767
        # games) found 2 real historical overs that legitimately ran to 13
        # and 14 deliveries (several wides/no-balls in one over) -- a
        # ceiling of 12 silently dropped those overs' predictions even
        # though the data was genuine, not corrupted. 20 stays comfortably
        # defensive against actually-garbled feeds (duplicated/garbled ball
        # sequences) while covering realistic worst-case overs.
        if delivery.ball < 1 or delivery.ball > 20:
            raise VerificationError(
                f"Impossible ball number {delivery.over}.{delivery.ball}."
            )
        expected_score = self.score + delivery.total_runs
        expected_wickets = self.wickets + (delivery.wicket_kind is not None)
        if expected_score != delivery.feed_total:
            raise VerificationError(
                f"Score mismatch at {delivery.over}.{delivery.ball}: accumulated "
                f"{expected_score}, TOI reports {delivery.feed_total}."
            )
        if expected_wickets != delivery.feed_wickets:
            raise VerificationError(
                f"Wicket mismatch at {delivery.over}.{delivery.ball}: accumulated "
                f"{expected_wickets}, TOI reports {delivery.feed_wickets}."
            )
        self.score = expected_score
        self.wickets = expected_wickets
        self.last_key = delivery.key
        self.accepted[delivery.key] = delivery
        self.over_runs[delivery.over] = (
            self.over_runs.get(delivery.over, 0) + delivery.total_runs
        )
        self.over_wickets[delivery.over] = self.over_wickets.get(
            delivery.over, 0
        ) + int(delivery.wicket_kind is not None)
        return True

    def apply_snapshot(self, snapshot: ToiSnapshot) -> tuple[ToiDelivery, ...]:
        accepted: list[ToiDelivery] = []
        for delivery in snapshot.deliveries:
            if delivery.key not in self.accepted and self.apply(delivery):
                accepted.append(delivery)
            elif (
                delivery.key in self.accepted
                and not self._same_cricket_state(
                    self.accepted[delivery.key], delivery
                )
            ):
                raise VerificationError(
                    f"Delivery {delivery.key} changed between polls."
                )
            elif delivery.key in self.accepted:
                self.accepted[delivery.key] = delivery
        if snapshot.deliveries and self.last_key == snapshot.deliveries[-1].key:
            if (self.score, self.wickets) != (snapshot.score, snapshot.wickets):
                raise VerificationError(
                    "Delivery totals do not match the authoritative TOI scorecard: "
                    f"{self.score}/{self.wickets} vs "
                    f"{snapshot.score}/{snapshot.wickets}."
                )
        return tuple(accepted)

    @property
    def completed_over(self) -> int | None:
        if self.last_key is None:
            return None
        legal_by_over: dict[int, int] = {}
        for delivery in self.accepted.values():
            if delivery.is_legal:
                legal_by_over[delivery.over] = legal_by_over.get(delivery.over, 0) + 1
        completed = [
            over for over, legal_balls in legal_by_over.items() if legal_balls == 6
        ]
        return max(completed) if completed else None
