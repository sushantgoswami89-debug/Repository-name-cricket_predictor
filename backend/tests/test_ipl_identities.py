import json
from pathlib import Path

from app.ml.ipl_identities import (
    HOME_VENUES_2026,
    canonical_player_id,
    canonical_team_id,
)
from app.ml.ipl_venues import normalize_ipl_venue


def test_franchise_brand_aliases_share_stable_identity():
    assert canonical_team_id("Royal Challengers Bangalore") == canonical_team_id(
        "Royal Challengers Bengaluru"
    )
    assert canonical_team_id("Kings XI Punjab") == canonical_team_id("Punjab Kings")
    assert canonical_team_id("Delhi Daredevils") == canonical_team_id(
        "Delhi Capitals"
    )


def test_raipur_city_suffix_does_not_split_stadium():
    assert normalize_ipl_venue(
        "Shaheed Veer Narayan Singh International Stadium"
    ) == normalize_ipl_venue(
        "Shaheed Veer Narayan Singh International Stadium, Raipur"
    )


def test_2026_secondary_home_venues_are_explicit():
    assert "raipur_shaheed_veer_narayan" in HOME_VENUES_2026["team:rcb"]
    assert "dharamsala_hpca" in HOME_VENUES_2026["team:pbks"]
    assert "guwahati_barsapara" in HOME_VENUES_2026["team:rr"]


def test_real_cricsheet_registry_uuid_is_the_player_identity():
    root = Path(__file__).resolve().parents[2]
    path = root / "data/raw/cricsheet/ipl/1359475.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    registry = raw["info"]["registry"]["people"]
    delivery = raw["innings"][0]["overs"][0]["deliveries"][0]
    batter = delivery["batter"]

    assert canonical_player_id(batter, registry) == f"player:{registry[batter]}"
