"""HTTP transport helpers: status/body/error-drain handling for completion.

Behavior-preserving extraction from client.stream_completion (structural
split only). No ban classification, retry-code, string-matching, or cap
changes here — caps (20k/4k) and truncations are preserved verbatim.

Ownership:
- This module owns HTTP-level concerns: content-type checks, .text vs
  iter_lines draining (stream=True leaves .text/.json() empty), 422
  detail surfacing, HTTP_403_WAF mapping, and the warning log shape.
- SSE line/event parsing lives in sse_parser.py.
- Ban-envelope mapping (_biz_error/_ban_stream_event) stays in client.py;
  this module only moves the byte-handling primitives so client.py keeps
  its _SESSION hook (tests monkeypatch client._SESSION).
"""

from __future__ import annotations

import json as _jsonlib
import logging
from typing import Any

from .sse_parser import StreamEvent

# Preserve the original log channel so existing filters see the same records.
logger = logging.getLogger("deepseaport.client")

# Drain caps (bytes-ish chars): 20000 for HTTP-200 JSON bodies arriving as
# raw JSON lines, 4000 for non-200 error drains. Do not change (later batch).
JSON_DRAIN_CAP = 20000
ERROR_DRAIN_CAP = 4000
# Truncation limits for surfaced error bodies. Do not change (later batch).
ERROR_BODY_MAX = 2000
DETAIL_SLICE_MAX = 1500
UPSTREAM_MESSAGE_MAX = 300


def decode_line(raw: bytes | str) -> str:
    """Decode one iter_lines item as utf-8/replace (bytes) or str()."""
    return raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)


def content_type_of(resp: Any) -> str:
    """Best-effort content-type fetch: "" on any error, never raises."""
    try:
        return resp.headers.get("content-type", "") or ""
    except Exception:
        return ""


def is_sse_response(resp: Any) -> bool:
    """True when the response claims text/event-stream (may still be empty)."""
    return "text/event-stream" in (content_type_of(resp) or "")


def drain_lines(resp: Any, cap: int) -> list[str]:
    """Drain iter_lines: skip empties, decode, stop once chars exceed cap.

    With stream=True the body is not buffered, so resp.json()/.text can be
    empty — callers drain via iter_lines instead. Cap preserves the
    20k/4k bounds from client.stream_completion.
    """
    out: list[str] = []
    for raw in resp.iter_lines():
        if not raw:
            continue
        out.append(decode_line(raw))
        if sum(len(c) for c in out) > cap:
            break
    return out


def read_text_body(resp: Any) -> str:
    """Best-effort resp.text: "" on any error, never raises."""
    try:
        return resp.text or ""
    except Exception:
        return ""


def read_error_body(resp: Any) -> tuple[str, str]:
    """Return (ctype, body) for non-200 paths: .text, else drain (cap 4k).

    stream=True can leave .text empty: drain a few lines instead.
    Body is stripped and truncated to 2000 chars (preserved).
    """
    ctype = content_type_of(resp)
    raw_body = read_text_body(resp)
    if not raw_body:
        # stream=True can leave .text empty: drain a few lines instead.
        try:
            raw_body = "".join(drain_lines(resp, ERROR_DRAIN_CAP))
        except Exception:
            pass
    return ctype, (raw_body or "").strip()[:ERROR_BODY_MAX]


def enrich_422_body(body: str) -> str:
    """Surface FastAPI-style 422 {"detail": [...]} compactly.

    Preserves: only when body starts with "{", detail appended as
    " | detail=<json>" with 1500-char slices, total capped at 2000.
    """
    # FastAPI-style 422 carries {"detail": [...]}: surface it compactly
    # so the server log shows the failing field, not just "HTTP_422:".
    if body.startswith("{"):
        try:
            parsed = _jsonlib.loads(body)
            if isinstance(parsed, dict) and parsed.get("detail") is not None:
                detail_s = _jsonlib.dumps(parsed.get("detail"), ensure_ascii=False)
                body = f"{body[:DETAIL_SLICE_MAX]} | detail={detail_s[:DETAIL_SLICE_MAX]}"[:ERROR_BODY_MAX]
        except Exception:
            pass
    return body


def http_error_code(status_code: Any) -> str:
    """Map HTTP status to event code: 403 -> HTTP_403_WAF, else HTTP_<n>."""
    return "HTTP_403_WAF" if status_code == 403 else f"HTTP_{status_code}"


def describe_payload(payload: Any) -> tuple[list[str], int]:
    """Return (sorted_keys, prompt_len) for the completion warning log."""
    try:
        payload_keys = sorted((payload or {}).keys()) if isinstance(payload, dict) else []
        prompt_len = len(str((payload or {}).get("prompt", ""))) if isinstance(payload, dict) else 0
    except Exception:
        payload_keys, prompt_len = [], 0
    return payload_keys, prompt_len


def log_http_error(status_code: Any, ctype: str, body: str, payload: Any) -> None:
    """Emit the completion warning with payload shape (preserved format)."""
    payload_keys, prompt_len = describe_payload(payload)
    logger.warning("completion HTTP %s (ct=%s) body=%.500s payload_keys=%s prompt_len=%d",
                   status_code, ctype, body, payload_keys, prompt_len)


def build_http_error_event(resp: Any, payload: Any) -> StreamEvent:
    """Full non-200 mapping: drain body, enrich 422, log, return error event.

    Caller owns resp.close()/yield (lifecycle stays in client.py so the
    client._SESSION monkeypatch hook keeps working).
    """
    ctype, body = read_error_body(resp)
    body = enrich_422_body(body)
    code = http_error_code(resp.status_code)
    log_http_error(resp.status_code, ctype, body, payload)
    return StreamEvent(kind="error", code=code, message=body)


def loads_lenient(body: str) -> Any:
    """Best-effort json.loads: {} on any error, never raises."""
    try:
        return _jsonlib.loads(body)
    except Exception:
        return {}


def truncate_message(text: str, limit: int = UPSTREAM_MESSAGE_MAX) -> str:
    """Truncate a surfaced upstream body (default 300, preserved)."""
    return (text or "")[:limit]
