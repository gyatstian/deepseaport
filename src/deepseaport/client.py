"""DeepSeek web client: session, PoW, SSE completion (sync, thread-safe)."""

from __future__ import annotations

import logging
import time
from typing import Iterator

from curl_cffi import requests as crequests

from . import pow as PoW
from . import protocol as P

logger = logging.getLogger("deepseaport.client")

IMPERSONATE = "chrome"
DEFAULT_TIMEOUT = 120


def _json(resp) -> dict:
    try:
        return resp.json()
    except Exception:
        return {}


def _biz_error(data: dict) -> tuple[str, str] | None:
    """Return (code, msg) when the envelope carries a biz error."""
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
    return str(code), str(msg or "")


def create_session(headers: dict) -> str:
    resp = crequests.post(P.SESSION_URL, headers=headers, json={"agent": "chat"},
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
        crequests.post(P.DELETE_SESSION_URL, headers=headers,
                       json={"chat_session_id": session_id},
                       impersonate=IMPERSONATE, timeout=10)
    except Exception as exc:
        logger.debug("delete_session ignored: %s", exc)


def fetch_pow(headers: dict) -> dict:
    resp = crequests.post(P.POW_URL, headers=headers,
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


def solve_pow(headers: dict) -> str:
    challenge = fetch_pow(headers)
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


def stream_completion(headers: dict, payload: dict,
                      timeout: int = DEFAULT_TIMEOUT) -> Iterator[P.StreamEvent]:
    """POST completion and yield parsed events. Caller must close on break."""
    resp = crequests.post(P.COMPLETION_URL, headers=headers, json=payload,
                          impersonate=IMPERSONATE, timeout=timeout, stream=True)
    if resp.status_code == 200 and "text/event-stream" not in (resp.headers.get("content-type", "") or ""):
        # Some errors arrive as JSON with HTTP 200.
        data = _json(resp)
        err = _biz_error(data)
        resp.close()
        if err:
            yield P.StreamEvent(kind="error", code=err[0], message=err[1])
            return
    if resp.status_code != 200:
        body = resp.text[:300] if not resp.headers.get("content-type", "").startswith("text/") else ""
        code = "HTTP_403_WAF" if resp.status_code == 403 else f"HTTP_{resp.status_code}"
        resp.close()
        yield P.StreamEvent(kind="error", code=code, message=body)
        return
    try:
        parser = P.StreamParser()
        for raw in resp.iter_lines():
            if not raw:
                continue
            line = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
            for event in parser.feed(line):
                yield event
                if event.kind in ("finished", "error"):
                    return
    finally:
        resp.close()
