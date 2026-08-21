from __future__ import annotations

from app.ml.over_component_dataset import delivery_components


def test_delivery_components_reconcile_boundary_and_extras() -> None:
    four = delivery_components({"runs": {"batter": 4, "extras": 0, "total": 4}})
    assert four["fours"] == 1
    assert four["boundary_runs"] == 4
    assert four["total_runs"] == 4

    wide = delivery_components(
        {
            "extras": {"wides": 1},
            "runs": {"batter": 0, "extras": 1, "total": 1},
        }
    )
    assert wide["illegal_deliveries"] == 1
    assert wide["legal_balls"] == 0
    assert wide["extras_runs"] == 1


def test_delivery_components_counts_wicket_independently() -> None:
    wicket = delivery_components(
        {
            "runs": {"batter": 0, "extras": 0, "total": 0},
            "wickets": [{"kind": "bowled"}],
        }
    )
    assert wicket["dot_balls"] == 1
    assert wicket["wickets"] == 1
