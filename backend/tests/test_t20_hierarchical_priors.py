from build_ipl_t20_hierarchical_priors import _before


def test_prior_lookup_excludes_same_day_and_future_matches():
    history = [
        ("2024-01-01", {"balls": 10.0}),
        ("2024-02-01", {"balls": 20.0}),
    ]
    assert _before(history, "2024-01-01") == {}
    assert _before(history, "2024-01-15") == {"balls": 10.0}
    assert _before(history, "2024-02-01") == {"balls": 10.0}
