"""SSE event parsing: Stateful StreamParser + stateless one-line helper.

Behavior-preserving extraction from protocol.py (structural split only).
Ban classification, retry codes, string matching, caps, and all
live-traffic quirk handling are unchanged — see protocol.py for the
authoritative ban helpers.

Ownership:
- This module owns SSE line/event parsing: StreamParser.feed, batched
  v-list, usage/finished/error envelopes, continuation chunks.
- Header builders and ban helpers stay in protocol.py (re-exported here
  only via deferred local import to avoid a module cycle).
- HTTP status/body/error-drain handling lives in transport_http.py.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Iterator, Literal

# Preserve the original log channel so existing filters see the same records.
logger = logging.getLogger("deepseaport.protocol")


@dataclass
class StreamEvent:
    kind: Literal["thinking", "content", "finished", "usage", "error"]
    text: str = ""
    code: str = ""
    message: str = ""
    # Ban expiry (mute_until unix ts) when the error carries a ban payload.
    # Lets callers cool the account down without an extra users/current HTTP.
    ban_until: float | None = None


class StreamParser:
    """Stateful SSE parser: continuation chunks arrive as bare {"v": "..."}
    with no "p" path, so the last content path is remembered."""

    CONTENT = "response/content"
    THINKING = "response/thinking_content"

    def __init__(self) -> None:
        self.last_path = ""

    def feed(self, line: str) -> Iterator[StreamEvent]:
        # Deferred import: protocol.py re-exports this module at top level,
        # so a top-level import would cycle. Resolved at call time when both
        # modules are fully loaded; behavior identical to direct globals.
        from .protocol import (
            ban_message,
            extract_ban_timestamp,
            format_ban_datetime,
            is_ban_payload,
        )

        line = line.strip()
        if not line or not line.startswith("data:"):
            return
        data = line[5:].strip()
        if data == "[DONE]":
            yield StreamEvent(kind="finished")
            return
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            logger.debug("skip non-JSON SSE line: %.80s", data)
            return
        if not isinstance(chunk, dict):
            return
        if "v" not in chunk:
            # Toast errors arrive as data: {"type":"error","content":"...","finish_reason":"..."}
            # with no "v"/"code" keys (e.g. rate_limit_reached). Surface them.
            if chunk.get("type") == "error" or "content" in chunk and "finish_reason" in chunk:
                code = str(chunk.get("finish_reason") or chunk.get("type") or "UPSTREAM_ERROR")
                yield StreamEvent(kind="error", code=code,
                                  message=str(chunk.get("content") or chunk.get("msg") or ""))
                return
            biz = chunk.get("data") if isinstance(chunk.get("data"), dict) else {}
            code = chunk.get("code", (biz or {}).get("biz_code", ""))
            # Top-level code 0 can still hide an inner ban (biz_code 5):
            # completion ban arrives as {code:0, data:{biz_code:5,...}}.
            inner_code = (biz or {}).get("biz_code", "")
            if code in (0, "0", None, "") and inner_code not in (0, "0", None, ""):
                code = inner_code
            if code not in (0, "0", None, "") or is_ban_payload(chunk):
                msg = str(chunk.get("msg") or (biz or {}).get("biz_msg") or "")
                ban_ts: float | None = None
                try:
                    if is_ban_payload(chunk):
                        ts = extract_ban_timestamp(chunk)
                        ban_ts = ts
                        if ts is not None:
                            msg = f"{msg} (suspended until {format_ban_datetime(ts)})".strip()
                        if not msg:
                            msg = ban_message(ts)
                        code = str(code) if code not in (0, "0", None, "") else "5"
                except Exception:
                    pass
                yield StreamEvent(kind="error", code=str(code), message=msg,
                                  ban_until=ban_ts)
            return
        path = chunk.get("p", "") or self.last_path
        value = chunk.get("v")
        if path == "response/status" and value == "FINISHED":
            self.last_path = ""
            yield StreamEvent(kind="finished")
            return
        if path == "response/search_status":
            return
        if path == "response/accumulated_token_usage" and isinstance(value, int):
            yield StreamEvent(kind="usage", text=str(value))
            return
        if isinstance(value, list):
            # Batched deltas: upstream can pack several {p,v} pairs (or raw
            # strings) in one "v" list. The old code only looked for FINISHED
            # and dropped any batched content, truncating the reply.
            for item in value:
                if isinstance(item, dict):
                    ip = item.get("p", "") or path
                    iv = item.get("v", "")
                    if iv == "FINISHED" and ip in ("status", "response/status"):
                        self.last_path = ""
                        yield StreamEvent(kind="finished")
                        return
                    if ip == "response/search_status":
                        continue
                    if (ip == "response/accumulated_token_usage"
                            and isinstance(iv, int)):
                        yield StreamEvent(kind="usage", text=str(iv))
                        continue
                    if isinstance(iv, list):
                        for sub in iv:
                            if isinstance(sub, str) and sub:
                                if ip == self.THINKING:
                                    self.last_path = ip
                                    yield StreamEvent(kind="thinking", text=sub)
                                elif ip == self.CONTENT:
                                    self.last_path = ip
                                    yield StreamEvent(kind="content", text=sub)
                        continue
                    if not isinstance(iv, str) or not iv:
                        continue
                    if ip == self.THINKING:
                        self.last_path = ip
                        yield StreamEvent(kind="thinking", text=iv)
                    elif ip == self.CONTENT:
                        self.last_path = ip
                        yield StreamEvent(kind="content", text=iv)
                elif isinstance(item, str) and item:
                    if path == self.THINKING:
                        self.last_path = path
                        yield StreamEvent(kind="thinking", text=item)
                    elif path == self.CONTENT:
                        self.last_path = path
                        yield StreamEvent(kind="content", text=item)
            return
        if not isinstance(value, str) or not value:
            return
        if path == self.THINKING:
            self.last_path = path
            yield StreamEvent(kind="thinking", text=value)
        elif path == self.CONTENT:
            self.last_path = path
            yield StreamEvent(kind="content", text=value)


def parse_sse_line(line: str) -> Iterator[StreamEvent]:
    """Stateless one-line parse (no continuation tracking)."""
    yield from StreamParser().feed(line)
