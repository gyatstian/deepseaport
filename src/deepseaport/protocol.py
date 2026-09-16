"""DeepSeek web protocol: headers, payloads, SSE parsing, error mapping."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Iterator, Literal

logger = logging.getLogger("deepseaport.protocol")

# Structural split (behavior-preserving): SSE parsing lives in sse_parser.py;
# re-exported here so `deepseaport.protocol.StreamParser` etc. keep working.
from .sse_parser import StreamEvent, StreamParser, parse_sse_line  # noqa: F401,E402

__all__ = [
    "StreamEvent",
    "StreamParser",
    "parse_sse_line",
    "parse_json",
    "base_headers",
    "bridge_headers",
    "completion_payload",
    "is_ban_payload",
    "extract_ban_timestamp",
    "is_ban_error_text",
    "format_ban_day_month",
    "format_ban_datetime",
    "format_ban_label",
    "ban_message",
]

# Browser fingerprint used for every curl_cffi request (shared session).
IMPERSONATE = "chrome"

# Env override for host rotation survival (default preserves pinned host).
# Full discovery/fallback is deferred: this knob lets ops repoint without a
# release when chat.deepseek.com moves.
HOST = os.environ.get("DEEPSEAPORT_HOST", "chat.deepseek.com").strip() or "chat.deepseek.com"
BASE = os.environ.get("DEEPSEAPORT_BASE", f"https://{HOST}").strip() or f"https://{HOST}"
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

# Word-ish ban match: plain substring "mute" false-positives on "commute" /
# "transmute" and would 403 + cool down a healthy account, shrinking the
# pool. Letter boundaries (not \b: "_" must count as a boundary so
# "is_muted"/"mute_until" still hit) keep true bans while ignoring
# "commute". "unbanned" intentionally does NOT match (preceded by a letter).
_BAN_RE = re.compile(
    r"(?<![a-z])(muted?|suspend\w*|bann?ed?|violat\w*)(?![a-z])",
    re.IGNORECASE,
)


def _has_ban_word(text: str) -> bool:
    try:
        return bool(text and _BAN_RE.search(text))
    except Exception:
        return False


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
        msg = f"{data.get('msg', '')} {(inner or {}).get('biz_msg', '')}"
        if biz_code in BANNED_BIZ_CODES or code in BANNED_BIZ_CODES:
            return True
        if _has_ban_word(msg):
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
    return _has_ban_word(text or "")


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


def parse_json(resp) -> dict:
    """Best-effort ``resp.json()``: {} on any error, never raises."""
    try:
        return resp.json()
    except Exception:
        return {}


def _build_headers(*, user_agent: str = "", waf_cookies: str = "", bearer: str = "") -> dict[str, str]:
    """Single header builder shared by base_headers/bridge_headers."""
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


def base_headers(user_agent: str = "", waf_cookies: str = "", bearer: str = "") -> dict[str, str]:
    return _build_headers(user_agent=user_agent, waf_cookies=waf_cookies, bearer=bearer)


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


def completion_payload(session_id: str, prompt: str, thinking: bool, search: bool,
                       model_type: str = "default") -> dict:
    # Upstream 2026-09-15 rejects source="web" with 422:
    #   unknown variant `web`, expected `default` or `landing`.
    return {
        "chat_session_id": session_id,
        "parent_message_id": None,
        "model_type": model_type,
        "prompt": prompt,
        "ref_file_ids": [],
        "thinking_enabled": thinking,
        "search_enabled": search,
        "source": "default",
        "action": None,
        "preempt": False,
    }
