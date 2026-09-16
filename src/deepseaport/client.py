"""DeepSeek web client: session, PoW, SSE completion (sync, thread-safe)."""

from __future__ import annotations

import logging
import time
from typing import Iterator

from curl_cffi import requests as crequests

from . import pow as PoW
from . import protocol as P
from . import transport_http as T
# Backward-compat re-exports: canonical SSE parsing lives in sse_parser.py,
# header/payload helpers in protocol.py — still importable from client.
from .protocol import base_headers, bridge_headers, completion_payload, parse_json  # noqa: F401
from .sse_parser import StreamEvent, StreamParser, parse_sse_line  # noqa: F401

logger = logging.getLogger("deepseaport.client")

# Backwards-compatible alias: the fingerprint lives in protocol as the single
# source shared with the account pool.
IMPERSONATE = P.IMPERSONATE
# Per-request stream timeout. curl_cffi treats this as a low-speed/stall
# timeout for streaming (read timeout), not a hard wall-clock cap, but keep it
# large enough for long reasoning generations. The server's consumer wait is
# derived from this value so the two cannot drift apart.
DEFAULT_TIMEOUT = 300

# Shared session: reuse HTTP/2 connections + TLS across calls. curl_cffi
# thread-safe: one curl handle per thread (thread-local), so concurrent
# callers get isolation while each thread reuses its connection. The cookie
# jar is discarded: the Cookie header is set manually per request and WAF
# Set-Cookie responses must not mutate a shared jar.
_SESSION = crequests.Session(impersonate=IMPERSONATE, discard_cookies=True)


def _json(resp) -> dict:
    return P.parse_json(resp)


def _biz_error(data: dict) -> tuple[str, str] | None:
    """Return (code, msg) when the envelope carries a biz error.

    Ban/mute envelopes (biz_code 5, "user is muted") embed
    biz_data {is_muted, mute_until}: the expiry is appended so callers
    surface "suspended until 16 September 2026 12:49" instead of a bare code.
    """
    if not isinstance(data, dict):
        return None
    inner = data.get("data")
    code = data.get("code")
    if isinstance(inner, dict):
        code = inner.get("biz_code", code)
        msg = inner.get("biz_msg", data.get("msg", ""))
    else:
        msg = data.get("msg", "")
    if code in (0, "0", None, ""):
        return None
    code_s, msg_s = str(code), str(msg or "")
    # Attach ban expiry when present (completion ban carries it here).
    try:
        if P.is_ban_payload(data):
            ts = P.extract_ban_timestamp(data)
            if ts is not None:
                msg_s = f"{msg_s} (suspended until {P.format_ban_datetime(ts)})".strip()
    except Exception:
        pass
    return code_s, msg_s


def fetch_current_user(headers: dict, timeout: int = 20) -> dict:
    """GET users/current. Raw envelope; caller parses ban/invalid itself."""
    resp = _SESSION.get(P.USERS_CURRENT_URL, headers=headers,
                        impersonate=IMPERSONATE, timeout=timeout)
    return _json(resp)


def check_ban(headers: dict, timeout: int = 20) -> tuple[bool, float | None]:
    """Return (is_banned, mute_until). Invalid token (40003) -> (False, None).

    Uses users/current chat.{is_muted,mute_until}. Best-effort: network or
    shape failures return (False, None), never raise.
    """
    try:
        data = fetch_current_user(headers, timeout=timeout)
        if not isinstance(data, dict):
            return False, None
        # Invalid/expired token is not a ban — just unauthenticated.
        code = data.get("code")
        inner = data.get("data")
        if code in (40003, "40003"):
            return False, None
        if isinstance(inner, dict) and inner.get("biz_code") in (40003, "40003"):
            return False, None
        if P.is_ban_payload(data):
            return True, P.extract_ban_timestamp(data)
        # Explicit non-muted chat shape also means not banned.
        try:
            biz = (inner or {}).get("biz_data", {}) if isinstance(inner, dict) else {}
            chat = biz.get("chat") if isinstance(biz, dict) else None
            if isinstance(chat, dict) and "is_muted" in chat:
                return bool(chat.get("is_muted") in (1, True, "1")), P.extract_ban_timestamp(data)
        except Exception:
            pass
        return False, None
    except Exception as exc:
        logger.debug("check_ban ignored: %s", exc)
        return False, None


def create_session(headers: dict) -> str:
    resp = _SESSION.post(P.SESSION_URL, headers=headers, json={"agent": "chat"},
                         impersonate=IMPERSONATE, timeout=30)
    data = _json(resp)
    err = _biz_error(data)
    if err:
        raise RuntimeError(f"create_session biz error {err[0]}: {err[1]}")
    inner = (data.get("data") or {}).get("biz_data", {}) if isinstance(data.get("data"), dict) else {}
    session_id = inner.get("id") or (inner.get("chat_session") or {}).get("id")
    if not session_id:
        raise RuntimeError(f"create_session: no id (HTTP {resp.status_code}): {resp.text[:200]}")
    return str(session_id)


def delete_session(headers: dict, session_id: str) -> None:
    try:
        _SESSION.post(P.DELETE_SESSION_URL, headers=headers,
                      json={"chat_session_id": session_id},
                      impersonate=IMPERSONATE, timeout=10)
    except Exception as exc:
        logger.debug("delete_session ignored: %s", exc)


def fetch_pow(headers: dict) -> dict:
    resp = _SESSION.post(P.POW_URL, headers=headers,
                         json={"target_path": P.COMPLETION_PATH},
                         impersonate=IMPERSONATE, timeout=30)
    data = _json(resp)
    err = _biz_error(data)
    if err:
        raise RuntimeError(f"pow challenge biz error {err[0]}: {err[1]}")
    inner = (data.get("data") or {}).get("biz_data", {}) if isinstance(data.get("data"), dict) else {}
    challenge = inner.get("challenge", inner)
    if not isinstance(challenge, dict) or "challenge" not in challenge:
        raise RuntimeError(f"pow challenge: bad shape: {resp.text[:200]}")
    return challenge


def solve_challenge(challenge: dict) -> str:
    """Solve an already-fetched PoW challenge; return the X-DS-PoW-Response value."""
    t0 = time.time()
    answer = PoW.solve(
        challenge.get("algorithm", PoW.ALGORITHM),
        challenge["challenge"], challenge["salt"],
        challenge.get("difficulty", 144000), challenge.get("expire_at", 0),
    )
    if answer is None:
        raise RuntimeError("PoW solver found no answer")
    challenge["answer"] = answer
    logger.debug("pow solved in %.1fs (difficulty %s)", time.time() - t0, challenge.get("difficulty"))
    return PoW.build_header(challenge)


def solve_pow(headers: dict) -> str:
    return solve_challenge(fetch_pow(headers))


def _ban_stream_event(data) -> P.StreamEvent | None:
    """Map a completion envelope to an error StreamEvent, else None.

    Handles the repeated biz-error / ban-payload sequence:
      - biz error present -> error event (ban expiry attached by _biz_error)
      - ban payload but no biz code -> explicit ban event (code "5")
    """
    if not isinstance(data, dict):
        return None
    err = _biz_error(data)
    if err:
        ban_ts = P.extract_ban_timestamp(data) if P.is_ban_payload(data) else None
        return P.StreamEvent(kind="error", code=err[0], message=err[1],
                             ban_until=ban_ts)
    if P.is_ban_payload(data):
        ts = P.extract_ban_timestamp(data)
        return P.StreamEvent(kind="error", code="5",
                             message=P.ban_message(ts), ban_until=ts)
    return None


def stream_completion(headers: dict, payload: dict,
                      timeout: int = DEFAULT_TIMEOUT) -> Iterator[P.StreamEvent]:
    """POST completion and yield parsed events. Caller must close on break."""
    resp = _SESSION.post(P.COMPLETION_URL, headers=headers, json=payload,
                         impersonate=IMPERSONATE, timeout=timeout, stream=True)
    if resp.status_code == 200 and not T.is_sse_response(resp):
        # Some errors arrive as JSON with HTTP 200 (e.g. ban:
        # biz_code 5 "user is muted" + mute_until). With stream=True the body
        # is not buffered, so resp.json() is empty — drain via iter_lines.
        data = _json(resp)
        event = _ban_stream_event(data) if data else None
        if event is not None:
            resp.close()
            yield event
            return
        # Streaming mode: body arrives as raw JSON lines, not SSE "data:".
        try:
            chunks = T.drain_lines(resp, T.JSON_DRAIN_CAP)
            resp.close()
            if chunks:
                body = "".join(chunks).strip()
                data2 = T.loads_lenient(body)
                event2 = _ban_stream_event(data2) if isinstance(data2, dict) else None
                if event2 is not None:
                    yield event2
                    return
                if body:
                    yield P.StreamEvent(kind="error", code="UPSTREAM_ERROR",
                                        message=T.truncate_message(body))
                    return
        except StopIteration:
            pass
        except Exception:
            pass
        try:
            resp.close()
        except Exception:
            pass
        # Fall through to SSE loop (empty body closes without events).
    if resp.status_code != 200:
        event = T.build_http_error_event(resp, payload)
        resp.close()
        yield event
        return
    try:
        parser = StreamParser()
        for raw in resp.iter_lines():
            if not raw:
                continue
            line = T.decode_line(raw)
            # Raw JSON ban can also arrive inside an event-stream body.
            stripped = line.strip()
            if not stripped.startswith("data:") and stripped.startswith("{"):
                data3 = T.loads_lenient(stripped)
                if isinstance(data3, dict):
                    event3 = _ban_stream_event(data3)
                    if event3 is not None:
                        yield event3
                        return
                continue
            for event in parser.feed(line):
                yield event
                if event.kind in ("finished", "error"):
                    return
    finally:
        resp.close()
