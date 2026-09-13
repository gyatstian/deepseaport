"""DeepSeek web protocol: headers, payloads, SSE parsing, error mapping."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Iterator, Literal

logger = logging.getLogger("deepseaport.protocol")

# Browser fingerprint used for every curl_cffi request (shared session).
IMPERSONATE = "chrome"

HOST = "chat.deepseek.com"
BASE = f"https://{HOST}"
LOGIN_URL = f"{BASE}/api/v0/users/login"
USERS_CURRENT_URL = f"{BASE}/api/v0/users/current"
SESSION_URL = f"{BASE}/api/v0/chat_session/create"
DELETE_SESSION_URL = f"{BASE}/api/v0/chat_session/delete"
POW_URL = f"{BASE}/api/v0/chat/create_pow_challenge"
COMPLETION_PATH = "/api/v0/chat/completion"
COMPLETION_URL = f"{BASE}{COMPLETION_PATH}"

CHROME_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

# Error families worth a single automatic retry.
RETRYABLE_BIZ = {"POW_HEADER_ERROR", "INVALID_POW_RESPONSE", "INVALID_SESSION_ID"}

# Ban family: completion returns biz_code 5 + "user is muted" with
# biz_data {"is_muted": 1, "mute_until": <unix ts>}; users/current returns
# the same shape under biz_data.chat when the token is still valid.
BANNED_BIZ_CODES = {"5", "5.0"}
BAN_KEYWORDS = ("muted", "mute", "suspend", "banned", "violation")


def _as_float(value) -> float | None:
    try:
        if value is None or isinstance(value, bool):
            return None
        num = float(value)
        # mute_until is seconds (float) — reject millisec typos, keep sane range.
        if num <= 0:
            return None
        return num
    except (TypeError, ValueError):
        return None


def extract_ban_timestamp(data: dict) -> float | None:
    """Return mute_until unix timestamp when payload carries a ban, else None.

    Handles both shapes:
      completion: data.data.biz_data.{is_muted,mute_until}
      users/current: data.data.biz_data.chat.{is_muted,mute_until}
    """
    try:
        if not isinstance(data, dict):
            return None
        inner = data.get("data")
        if not isinstance(inner, dict):
            return None
        biz = inner.get("biz_data")
        if not isinstance(biz, dict):
            return None
        # Direct shape (completion ban).
        if biz.get("is_muted") in (1, True, "1"):
            ts = _as_float(biz.get("mute_until"))
            if ts is not None:
                return ts
            # is_muted without timestamp still counts as banned, but no expiry.
            return None
        # Nested shape (users/current).
        chat = biz.get("chat")
        if isinstance(chat, dict) and chat.get("is_muted") in (1, True, "1"):
            ts = _as_float(chat.get("mute_until"))
            # Return ts (may be None when server omits expiry).
            return ts
        return None
    except Exception:
        return None


def is_ban_payload(data: dict) -> bool:
    """True when envelope is a ban/mute, even without a parseable timestamp."""
    try:
        if not isinstance(data, dict):
            return False
        if extract_ban_timestamp(data) is not None:
            return True
        inner = data.get("data") if isinstance(data.get("data"), dict) else {}
        code = str(data.get("code", "") or "")
        biz_code = str((inner or {}).get("biz_code", "") or "")
        msg = f"{data.get('msg', '')} {(inner or {}).get('biz_msg', '')}".lower()
        if biz_code in BANNED_BIZ_CODES or code in BANNED_BIZ_CODES:
            return True
        if any(k in msg for k in BAN_KEYWORDS):
            return True
        biz = (inner or {}).get("biz_data")
        if isinstance(biz, dict):
            if biz.get("is_muted") in (1, True, "1"):
                return True
            chat = biz.get("chat")
            if isinstance(chat, dict) and chat.get("is_muted") in (1, True, "1"):
                return True
        return False
    except Exception:
        return False


def is_ban_error_text(text: str) -> bool:
    low = (text or "").lower()
    return any(k in low for k in BAN_KEYWORDS)


def _ban_dt(mute_until: float):
    import datetime as _dt

    return _dt.datetime.fromtimestamp(float(mute_until))


def format_ban_day_month(mute_until: float) -> str:
    """'16 September' from unix ts (local time, matches web UI wording)."""
    dt = _ban_dt(mute_until)
    return f"{dt.day} {dt.strftime('%B')}"


def format_ban_datetime(mute_until: float) -> str:
    """'16 September 2026 12:49' from unix ts (local time)."""
    dt = _ban_dt(mute_until)
    return f"{dt.day} {dt.strftime('%B')} {dt.year} {dt.strftime('%H:%M')}"


def format_ban_label(mute_until: float | None) -> str:
    """'(BANNED: 16 September)' — red wrapper applied by caller."""
    if mute_until is None:
        return "(BANNED)"
    return f"(BANNED: {format_ban_day_month(mute_until)})"


def ban_message(mute_until: float | None, *, prefix: str = "") -> str:
    """Full human error: ban reason + expiry, mirrors web UI wording."""
    base = prefix or "Due to violation of user policies, your account has been suspended"
    if mute_until is None:
        return f"{base} (expiry unknown). If you have any questions, please contact us."
    return (f"{base} until {format_ban_datetime(mute_until)}. "
            "If you have any questions, please contact us.")


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
            for item in value:
                if isinstance(item, dict) and item.get("p") == "status" and item.get("v") == "FINISHED":
                    self.last_path = ""
                    yield StreamEvent(kind="finished")
                    return
            return
        if not isinstance(value, str) or not value:
            return
        if path == self.THINKING:
            self.last_path = path
            yield StreamEvent(kind="thinking", text=value)
        elif path == self.CONTENT:
            self.last_path = path
            yield StreamEvent(kind="content", text=value)


def parse_json(resp) -> dict:
    """Best-effort ``resp.json()``: {} on any error, never raises."""
    try:
        return resp.json()
    except Exception:
        return {}


def base_headers(user_agent: str = "", waf_cookies: str = "", bearer: str = "") -> dict[str, str]:
    headers = {
        "Accept": "application/json, text/event-stream",
        "Accept-Language": "en-US,en;q=0.9",
        "Content-Type": "application/json",
        "Origin": BASE,
        "Referer": f"{BASE}/",
        "User-Agent": user_agent or CHROME_UA,
    }
    if waf_cookies:
        headers["Cookie"] = waf_cookies
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"
    return headers


def bridge_headers(bearer: str = "") -> dict[str, str]:
    """Base headers enriched with a warm bridge's WAF cookies + browser UA.

    Single source for out-of-server callers (account pool ban probes): without
    the WAF cookie + matching UA, chat.deepseek.com blocks the request at the
    edge and a probe reads as a false "not banned". Falls back to plain
    base_headers when no bridge is registered (tests/offline).
    """
    try:
        from .obscura_bridge import get_default_bridge
        bridge = get_default_bridge()
    except Exception:
        bridge = None
    if bridge is None:
        return base_headers(bearer=bearer)
    return base_headers(user_agent=bridge.state.user_agent,
                        waf_cookies=bridge.cookie_header(), bearer=bearer)


def login_payload(email: str = "", mobile: str = "", password: str = "") -> dict:
    if email:
        return {"email": email, "password": password, "device_id": "deepseaport-web", "os": "web"}
    return {"mobile": mobile, "area_code": None, "password": password,
            "device_id": "deepseaport-web", "os": "web"}


def completion_payload(session_id: str, prompt: str, thinking: bool, search: bool,
                       model_type: str = "default") -> dict:
    return {
        "chat_session_id": session_id,
        "parent_message_id": None,
        "model_type": model_type,
        "prompt": prompt,
        "ref_file_ids": [],
        "thinking_enabled": thinking,
        "search_enabled": search,
        "source": "web",
        "action": None,
        "preempt": False,
    }


def parse_sse_line(line: str) -> Iterator[StreamEvent]:
    """Stateless one-line parse (no continuation tracking)."""
    yield from StreamParser().feed(line)
