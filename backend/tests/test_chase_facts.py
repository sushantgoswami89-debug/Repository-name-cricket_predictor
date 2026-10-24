"""Chase Fact-o-Meter cards from the historical record (user 2026-10-24)."""

from app.live.chase_facts import checkpoint_card, target_card


def test_target_card_says_how_often_a_big_total_was_defended():
    card = target_card(225, "New Zealand", "India")
    assert card and "India chase 226" in card and "225+" in card and "sides batting first" in card and "FACT-O-METER (T20)" in card
    assert target_card(150, "A", "B") is None or "chasing sides" in target_card(150, "A", "B") or "batting first" in target_card(150, "A", "B")


def test_checkpoint_card_counts_chasing_sides_in_the_same_spot():
    card = checkpoint_card(10, 119, 3, 226, "India")
    assert card and "after 10" in card and "3 down" in card and "went on to win" in card
    assert checkpoint_card(5, 82, 0, 226, "India") is None          # only after 6, 10 and 15 overs
    assert checkpoint_card(10, 119, 3, 0, "India") is None          # not a chase


def test_session_sends_each_chase_card_once(monkeypatch):
    from types import SimpleNamespace

    from app.live.cricbuzz_session import CricbuzzMatchSession

    sent = []
    session = CricbuzzMatchSession.__new__(CricbuzzMatchSession)
    session._settled_shown = set()
    session.tag_lines, session._tag = False, ""
    session._send_card = lambda stat, key: sent.append(key)
    delivery = SimpleNamespace(over=6, ball=6)
    snap = SimpleNamespace(innings=2, target=226, match_format="T20", bowling_team="NZ", batting_team="IND", score=94, wickets=1,
                           overs="6.0", deliveries=[delivery], innings_complete=False)
    session._chase_cards(snap)
    session._chase_cards(snap)
    assert sent == [(2, "target-card"), (2, "chase-card", 6)]
