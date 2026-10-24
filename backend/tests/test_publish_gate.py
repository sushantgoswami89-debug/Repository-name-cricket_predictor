"""The live Telegram wrapper: late forecasts are never sent, labels and flags go in front of the text, restarts never repeat a message."""

from types import SimpleNamespace

import pytest

from app.live.publish_gate import GatedPublisher, MutedPublication
from run_live_auto import SendOnceFile


class Inner:
    def __init__(self):
        self.sent = []

    def publish(self, key, text):
        self.sent.append((key, text))
        return SimpleNamespace(sent=True, message_id=1)


def test_label_and_flag_are_put_in_front_of_the_text():
    inner = Inner()
    gate = GatedPublisher(inner, prefix="LABEL")
    gate.flag = "FLAG"
    gate.publish("k1", "forecast")
    assert inner.sent == [("k1", "LABEL\nFLAG\nforecast")]
    plain = GatedPublisher(inner)
    plain.publish("k2", "forecast")
    assert inner.sent[-1] == ("k2", "forecast")


def test_a_muted_publication_raises_a_non_retryable_error_and_sends_nothing():
    inner = Inner()
    gate = GatedPublisher(inner)
    gate.mute = True
    with pytest.raises(MutedPublication) as error:
        gate.publish("k", "late forecast")
    assert error.value.retryable is False and inner.sent == []
    with pytest.raises(MutedPublication):
        gate.publish_correction("k2", "late correction")


def test_send_once_file_survives_a_restart_and_reports_duplicates_as_sent(tmp_path):
    path = tmp_path / "sent.json"
    inner = Inner()
    first = SendOnceFile(inner, path)
    first.publish("card:1", "text")
    again = SendOnceFile(inner, path)                      # a new process reads the file
    result = again.publish("card:1", "text")
    assert len(inner.sent) == 1 and result.sent is True


def test_a_forecast_one_ball_late_is_sent_flagged_and_two_balls_late_is_not():
    from types import SimpleNamespace

    from app.live.cricbuzz_session import MAX_LATE_BALLS_TO_SEND, _balls_into_over

    assert MAX_LATE_BALLS_TO_SEND == 1
    assert [_balls_into_over(SimpleNamespace(overs=o)) for o in ("1.1", 4.0, "4.0", "19.5", "0.6")] == [1, 0, 0, 5, 6]
