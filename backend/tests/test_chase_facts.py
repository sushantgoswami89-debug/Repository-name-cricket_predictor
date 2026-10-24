"""The chase target Fact-o-Meter card: once, before the chase starts, one line (user 2026-10-24)."""

from types import SimpleNamespace

from app.live.chase_facts import target_card


def test_target_card_says_how_often_a_big_total_was_defended():
    card = target_card(225, "New Zealand", "India")
    assert card and "India chase 226" in card and "FACT-O-METER (T20)" in card
    assert "A total of 225+ is very hard to chase. It has happened just 12 times in 177 matches." in card
    assert "men's T20 since 2019" in card
    assert len(card.splitlines()) == 2                     # the title line and the one fact line
    assert "hard to chase" in target_card(195, "A", "B") and "only" in target_card(195, "A", "B")   # 16%: hard, not very hard
    assert target_card(150, "A", "B") is None              # not a big total: no card


def _session():
    from app.live.cricbuzz_session import CricbuzzMatchSession

    sent = []
    session = CricbuzzMatchSession.__new__(CricbuzzMatchSession)
    session._settled_shown = set()
    session.tag_lines, session._tag = False, ""
    session._send_card = lambda stat, key: sent.append(key)
    return session, sent


def _snap(deliveries):
    return SimpleNamespace(innings=2, target=226, match_format="T20", bowling_team="NZ", batting_team="IND", score=0, wickets=0,
                           overs="0.0", deliveries=deliveries, innings_complete=False)


def test_the_card_is_sent_once_before_the_chase_starts_and_never_during_it():
    session, sent = _session()
    session._chase_cards(_snap([]))
    session._chase_cards(_snap([]))
    assert sent == [(2, "target-card")]
    during, sent_during = _session()
    during._chase_cards(_snap([SimpleNamespace(over=1, ball=1)]))        # attached mid-chase: nothing
    assert sent_during == []
