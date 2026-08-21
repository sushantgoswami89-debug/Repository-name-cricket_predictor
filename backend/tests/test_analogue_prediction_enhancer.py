from __future__ import annotations

from app.ml.analogue_prediction_enhancer import EnhancedPrediction


def test_enhanced_prediction_is_json_ready() -> None:
    result = EnhancedPrediction(
        expected_runs=9.2,
        wicket_probability=0.3,
        range_low=4.0,
        range_high=15.0,
        confidence=0.82,
        similar_situations=384,
        effective_sample_size=350.0,
        similarity_percent=91,
        historical_expected_runs=9.0,
        historical_wicket_probability=0.28,
        most_common_runs_bucket="9-12",
        most_common_bucket_probability=0.37,
    )
    output = result.to_dict()
    assert output["most_common_runs_bucket"] == "9-12"
    assert output["similar_situations"] == 384
