"""
Tests for TossContext.
"""

from app.models.toss_context import TossContext


def test_toss_context_validation() -> None:
    toss = TossContext(
        winner="India",
        decision="Bat",
    )

    assert toss.validate() is True
    assert toss.decision == "bat"


def test_toss_context_to_dict() -> None:
    toss = TossContext(
        winner="Australia",
        decision="bowl",
    )

    data = toss.to_dict()

    assert data["winner"] == "Australia"
    assert data["decision"] == "bowl"
