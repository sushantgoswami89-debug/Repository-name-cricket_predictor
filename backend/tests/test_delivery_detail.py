from app.live.delivery_detail import extract_delivery_detail, format_delivery_detail


def test_real_lines_are_read_correctly():
    d = extract_delivery_detail(
        "Arshdeep Singh to Sherfane Rutherford, 1 run, good yorker length delivery on off, "
        "Sherfane Rutherford picks it and jams it down, wide of short fine for one"
    )
    assert d["length"] == "yorker" and d["line"] == "off" and d["shot"] == "punch"

    d = extract_delivery_detail("Mayank Yadav to Shai Hope, FOUR, 147kph")
    assert d["pace_kph"] == 147

    d = extract_delivery_detail(
        "Kuldeep Yadav to Shai Hope, 2 runs, flat good length on off, Shai Hope goes on the back foot and sweeps to deep square leg"
    )
    assert d["length"] == "good_length" and d["variation"] == "flat_quick" and d["shot"] == "sweep"

    d = extract_delivery_detail("Naman Dhir to Shamar Springer, SIX, to deep mid-wicket")
    assert d["area"] == "deep mid-wicket"

    d = extract_delivery_detail(
        "Naman Dhir to Romario Shepherd, out Caught by Tilak Varma!! He bowls it slower and well outside off"
    )
    assert d["variation"] == "slower_ball" and d["line"] == "wide_outside_off"


def test_unknown_text_gives_none_not_guesses():
    d = extract_delivery_detail("Axar Patel to Shai Hope, no run")
    assert all(value is None for value in d.values())
    assert format_delivery_detail(d) == ""
