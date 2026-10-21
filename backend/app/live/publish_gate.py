"""Publisher wrapper for live Telegram forecasts (user 2026-10-21: all T20s live on Telegram).

The pipeline publishes the moment it has a forecast. Two things must never reach the chat as if they were normal live forecasts:
  * a LATE forecast (the next over is already under way when it comes out: "do not use it live"), and
  * a forecast for a match the models were not built for (women's: the men's-trained models), which carries its label in the text.
The session sets `mute` / `flag` just before each pipeline call; a muted publish raises a NON-retryable error, so the pipeline records the
forecast as failed_permanent instead of retrying it later when it would no longer be late.
"""

from __future__ import annotations


class MutedPublication(RuntimeError):
    retryable = False


class GatedPublisher:
    def __init__(self, inner, prefix: str = "") -> None:
        self._inner = inner
        self.prefix = prefix      # always put in front of the text (e.g. the EXPERIMENTAL label)
        self.flag = ""            # one-call extra line (e.g. UNCONFIRMED), cleared by the session
        self.mute = False

    def _text(self, text: str) -> str:
        head = "\n".join(line for line in (self.prefix, self.flag) if line)
        return f"{head}\n{text}" if head else text

    def publish(self, key: str, text: str):
        if self.mute:
            raise MutedPublication(f"{key}: not published (late forecast)")
        return self._inner.publish(key, self._text(text))

    def publish_correction(self, key: str, text: str):
        if self.mute:
            raise MutedPublication(f"{key}: not published (late forecast)")
        inner = getattr(self._inner, "publish_correction", self._inner.publish)
        return inner(key, self._text(text))

    def restore_published(self, keys: set[str]) -> None:
        restore = getattr(self._inner, "restore_published", None)
        if restore is not None:
            restore(keys)
