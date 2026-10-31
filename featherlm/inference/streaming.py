"""Streaming helpers: UTF-8 safe incremental decoding with stop-sequence handling."""
from __future__ import annotations

import json

from ..tokenizer.tokenizer import ByteBPETokenizer, IncrementalDecoder


class StopSequenceFilter:
    """Holds back just enough text to never emit part of a stop sequence."""

    def __init__(self, tok: ByteBPETokenizer, stops: list[str] | None) -> None:
        self.dec = IncrementalDecoder(tok)
        self.stops = [s for s in (stops or []) if s]
        self.hold = max((len(s) for s in self.stops), default=1) - 1
        self.pending = ""

    def push(self, token_id: int) -> tuple[str, bool]:
        """Returns (text_to_emit, stop_hit)."""
        self.pending += self.dec.push(token_id)
        for s in self.stops:
            i = self.pending.find(s)
            if i >= 0:
                out = self.pending[:i]
                self.pending = ""
                return out, True
        if self.hold == 0 or not self.stops:
            out, self.pending = self.pending, ""
            return out, False
        if len(self.pending) > self.hold:
            out = self.pending[: -self.hold]
            self.pending = self.pending[-self.hold :]
            return out, False
        return "", False

    def flush(self) -> str:
        out = self.pending + self.dec.flush()
        self.pending = ""
        return out


def sse_event(data: dict | str) -> str:
    """Format one server-sent event."""
    payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    return f"data: {payload}\n\n"
