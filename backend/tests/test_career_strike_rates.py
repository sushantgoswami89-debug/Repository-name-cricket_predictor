import json
from types import SimpleNamespace

import numpy as np

from app.ml.career_strike_rates import PRIOR_BALLS, PRIOR_RPB, CareerStrikeRates, CareerTable, phase_of, save


def test_phase_boundaries():
    assert [phase_of(o) for o in (1, 6, 7, 15, 16, 20)] == ["powerplay", "powerplay", "middle", "middle", "death", "death"]


def test_unseen_batter_gets_the_typical_rate():
    table = CareerTable()
    career, in_phase, balls = table.batter("nobody", "death")
    assert career == PRIOR_RPB and in_phase == PRIOR_RPB and balls == 0
    assert "nobody" not in table.overall      # looking someone up does not create a record


def test_career_rate_shrinks_and_phase_rate_leans_on_career():
    table = CareerTable()
    table.add("a", "death", runs=120, balls=60)    # 2.0 a ball in the death
    career, death, balls = table.batter("a", "death")
    assert balls == 60
    assert career == (120 + PRIOR_RPB * PRIOR_BALLS) / (60 + PRIOR_BALLS)
    assert death == (120 + career * PRIOR_BALLS) / (60 + PRIOR_BALLS)
    assert table.batter("a", "powerplay")[1] == career      # no powerplay history: falls back to his career rate


def test_features_has_all_model_inputs():
    table = CareerTable()
    table.add("a", "powerplay", 30, 20)
    features = table.features("a", "b", over=3)
    assert set(features) == {"striker_career_sr", "partner_career_sr", "striker_phase_sr", "partner_phase_sr", "pair_phase_sr",
                             "striker_career_balls", "partner_career_balls"}
    assert features["striker_career_balls"] == 20 and features["partner_career_balls"] == 0
    assert features["pair_phase_sr"] == (features["striker_phase_sr"] + features["partner_phase_sr"]) / 2


def _match(date, batters):
    deliveries = [{"batter": b, "non_striker": "Other Player", "runs": {"batter": r, "total": r}} for b, r in batters]
    return {"info": {"gender": "male", "match_type": "T20", "dates": [date], "registry": {"people": {"V Kohli": "k1", "Other Player": "o1"}}},
            "innings": [{"team": "X", "overs": [{"over": 0, "deliveries": deliveries}]}]}


def test_saved_table_loads_and_matches_names(tmp_path):
    root = tmp_path / "root"
    (root / "data/raw/cricsheet/ipl").mkdir(parents=True)
    (root / "data/raw/cricsheet/ipl/1.json").write_text(json.dumps(_match("2025-01-01", [("V Kohli", 4), ("V Kohli", 6)])))
    save(tmp_path / "model", root)
    styles = tmp_path / "styles.csv"
    styles.write_text('"cricinfo_id","cricsheet_id","unique_name","full_name"\n"1","k1","V Kohli","Virat Kohli"\n')
    rates = CareerStrikeRates(tmp_path / "model", styles)
    assert rates.resolve("V Kohli") == ("k1", "native")
    assert rates.resolve("Virat Kohli") == ("k1", "full name")
    assert rates.resolve("Somebody Else") == (None, "unmatched")
    features, tiers = rates.features("Virat Kohli", "Somebody Else", over=2)
    assert tiers == {"striker": "full name", "partner": "unmatched"}
    assert features["striker_career_sr"] > PRIOR_RPB and features["partner_career_sr"] == PRIOR_RPB


def test_tampered_artifact_is_refused(tmp_path):
    root = tmp_path / "root"
    (root / "data/raw/cricsheet/ipl").mkdir(parents=True)
    (root / "data/raw/cricsheet/ipl/1.json").write_text(json.dumps(_match("2025-01-01", [("V Kohli", 1)])))
    save(tmp_path / "model", root)
    (tmp_path / "model" / "career_sr.json").write_text("{}")
    assert CareerStrikeRates.try_load(tmp_path / "model") is None


def test_outlook_extra_inputs_reach_the_model():
    from app.ml.over_outlook import OverOutlook

    outlook = OverOutlook.__new__(OverOutlook)
    outlook._features = ["over", "phase", "x_from_live", "career_input"]
    seen = {}

    class Model:
        def predict(self, frame):
            seen["row"] = frame.iloc[0].to_dict()
            return [7.0]

        def predict_proba(self, frame):
            return np.array([[0.5, 0.5]])

    outlook._models = {name: Model() for name in ("mean", "big", "quiet", "wicket")}
    outlook.big_runs, outlook.quiet_runs = 10, 4
    live = SimpleNamespace(balls_remaining=60, over=8, x_from_live=3)
    result = outlook.predict(live, extra={"career_input": 1.9})
    assert result["mean_runs"] == 7.0
    assert seen["row"]["career_input"] == 1.9 and seen["row"]["x_from_live"] == 3
