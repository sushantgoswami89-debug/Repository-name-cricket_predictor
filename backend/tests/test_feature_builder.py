"""
Tests for FeatureBuilder.
"""

from app.ml.feature_builder import FeatureBuilder
from app.models.match_context import MatchContext


def test_feature_builder() -> None:
    """
    Ensure FeatureBuilder builds the expected feature dictionary.
    """

    context = MatchContext()

    builder = FeatureBuilder()

    features = builder.build(context)

    assert isinstance(features, dict)

    assert len(features) == 22

    assert features["over"] == context.live.over
    assert features["score_before_over"] == context.live.score_before_over
    assert features["wkts_down_before_over"] == context.live.wkts_down_before_over
    assert features["balls_faced_before_over"] == context.live.balls_faced_before_over
