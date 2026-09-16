"""Browser-backed DeepSeek authentication.

Direct username/password POSTs are fingerprint-gated
(``RISK_DEVICE_DETECTED``); the only reliable non-automated path is to let
Obscura drive the real web login once, then copy ``localStorage.userToken``
into the account pool. This module is the single implementation of that flow
and is reused by the CLI, TUI, and the server's token-refresh path.

The flow is selector-driven instead of ref-driven: browser refs (``e1``,
``e2``, ...) are scoped to one MCP connection and are renumbered as soon as
the SPA changes. A stale ref used to mean "click whatever happens to be at
slot e8 now" (often Google/Apple/cookie UI). We instead use stable input
semantics plus a temporary marker on the actual "Log in" button.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import re
import time
from dataclasses import dataclass

from . import mcp_client
from .config import Settings
from .login_policy import (
    COOKIE_SELECTOR,
    EMAIL_SELECTOR,
    EMAIL_WAIT_TIMEOUT,
    LOGIN_BAN_PHRASES,
    LOGIN_CAPTCHA_PHRASES,
    LOGIN_CREDENTIAL_PHRASES,
    LOGIN_FALLBACK_SELECTOR,
    LOGIN_TIMEOUT,
    MARK_COOKIE as _MARK_COOKIE,
    MARK_LOGIN as _MARK_LOGIN,
    PAGE_PROBE as _PAGE_PROBE,
    PASSWORD_SELECTOR,
    POLL_BACKOFF,
    COOKIE_DISMISS_DELAY_MCP,
    REACT_COMMIT_DELAY,
    SIGN_IN_URL,
    JS_VISIBILITY as _JS_VISIBILITY,
    _POLL_MAX,
    _POLL_START,
    classify_page_state,
    collapse_text as _collapse,
)
from .obscura_bridge import ObscuraBridge, profile_lock
from .tokens import extract_token

logger = logging.getLogger("deepseaport.auth")

# Page policy (selectors, phrase lists, visibility/snapshot JS, poll/timeout
# constants, classify helpers) lives in ``login_policy``; names above are
# re-exported here for backward compat (``auth.classify_page_state``,
# ``auth._collapse``, phrase lists, probes, etc.).
#
# The sign-in page ships a hidden ``#cf-overlay`` containing the literal text
# "One more step before you proceed...". Monitoring raw body.innerText made
# every healthy login look like a captcha and discarded a valid token. These
# phrases are matched only against a visible-text projection plus an explicit
# visibility check for the challenge overlay (see ``login_policy``).

_MONTH_NUMBERS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5,
    "june": 6, "july": 7, "august": 8, "september": 9, "october": 10,
    "november": 11, "december": 12,
    # Polish web UI often shows "wrzesień 16, 2026 21:18".
    "styczen": 1, "stycznia": 1, "luty": 2, "lutego": 2,
    "marzec": 3, "marca": 3, "kwiecien": 4, "kwietnia": 4,
    "maj": 5, "maja": 5, "czerwiec": 6, "czerwca": 6,
    "lipiec": 7, "lipca": 7, "sierpien": 8, "sierpnia": 8,
    "wrzesien": 9, "wrzesnia": 9, "pazdziernik": 10,
    "pazdziernika": 10, "listopad": 11, "listopada": 11,
    "grudzien": 12, "grudnia": 12,
    "styczeń": 1, "stycznia": 1, "luty": 2, "lutego": 2,
    "marzec": 3, "marca": 3, "kwiecień": 4, "kwietnia": 4,
    "maj": 5, "maja": 5, "czerwiec": 6, "czerwca": 6,
    "lipiec": 7, "lipca": 7, "sierpień": 8, "sierpnia": 8,
    "wrzesień": 9, "września": 9, "październik": 10,
    "października": 10, "listopad": 11, "listopada": 11,
    "grudzień": 12, "grudnia": 12,
}
_MONTH_RE = "|".join(sorted((re.escape(k) for k in _MONTH_NUMBERS), key=len, reverse=True))


def parse_ban_until(text: str) -> float | None:
    """Best-effort parse of a human ban expiry from a login/page message.

    Supports English ``16 September 2026 12:49`` and Polish
    ``wrzesień 16, 2026 21:18``.  The exact API ``mute_until`` is preferred,
    but this keeps a human-readable date when the browser page is the only
    source of truth (for example a suspended account that still logs in).
    """
    try:
        low = (text or "").lower()
        day = month = year = None
        hour = minute = 0
        # English / most locales: day month [year] [HH:MM]
        m = re.search(
            rf"(\d{{1,2}})\s+({_MONTH_RE})(?:\s+(\d{{4}}))?"
            r"(?:\s+(\d{1,2}):(\d{2}))?",
            low,
        )
        if m:
            day = int(m.group(1))
            month = _MONTH_NUMBERS.get(m.group(2))
            year = int(m.group(3)) if m.group(3) else None
            hour = int(m.group(4) or 0)
            minute = int(m.group(5) or 0)
        if month is None:
            # Polish-ish: month day, year [HH:MM]
            m = re.search(
                rf"({_MONTH_RE})\s+(\d{{1,2}})(?:,)?(?:\s+(\d{{4}}))?"
                r"(?:\s+(\d{1,2}):(\d{2}))?",
                low,
            )
            if m:
                month = _MONTH_NUMBERS.get(m.group(1))
                day = int(m.group(2))
                year = int(m.group(3)) if m.group(3) else None
                hour = int(m.group(4) or 0)
                minute = int(m.group(5) or 0)
        if not day or not month:
            return None
        now = _dt.datetime.now()
        year = year or now.year
        dt = _dt.datetime(year, month, day, hour, minute)
        # No explicit year and the date has already passed: assume next year.
        if not m.group(3) and dt <= now:
            dt = dt.replace(year=year + 1)
        return dt.timestamp()
    except Exception:
        return None




@dataclass(frozen=True)
class LoginResult:
    token: str | None = None
    banned: bool = False
    detail: str = ""

    def as_tuple(self) -> tuple[str | None, bool, str]:
        return self.token, self.banned, self.detail


def unpack_login_result(res) -> tuple[str | None, bool, str]:
    """Normalize a LoginResult / legacy tuple / legacy string into a tuple."""
    if res is None:
        return None, False, ""
    if isinstance(res, LoginResult):
        return res.as_tuple()
    if isinstance(res, tuple):
        token = res[0] if len(res) > 0 else None
        banned = bool(res[1]) if len(res) > 1 else False
        detail = str(res[2] or "") if len(res) > 2 else ""
        token = token.strip() if isinstance(token, str) else None
        return (token or None), banned, detail
    if isinstance(res, str):
        return (res.strip() or None), False, ""
    return None, False, ""


def _tool_text(result) -> str:
    """Flatten an MCP tools/call result into its text payload."""
    if isinstance(result, str):
        return result
    if not isinstance(result, dict):
        return ""
    out: list[str] = []
    for part in (result.get("content") or []):
        if isinstance(part, dict) and part.get("type") == "text":
            out.append(str(part.get("text") or ""))
        elif isinstance(part, str):
            out.append(part)
    return "".join(out)


def _call(client, name: str, arguments: dict):
    """Call an MCP tool and turn ``isError`` tool results into exceptions.

    Obscura reports many failures (wait timeouts, missing selectors, bad
    refs) as normal JSON-RPC responses with an ``isError`` flag.  Treating
    those as success silently produced 0-field fills and wasted login runs.
    """
    result = client.call(name, arguments)
    if isinstance(result, dict) and result.get("isError"):
        raise RuntimeError(f"{name}: {_tool_text(result).strip()[:300]}")
    return result


def _json_eval(client, expression: str):
    """Call browser_evaluate and parse its JSON-encoded result."""
    raw = _tool_text(_call(client, "browser_evaluate", {"expression": expression}))
    try:
        value = json.loads(raw)
    except Exception:
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            return None
    return value


# --- DOM helpers: re-exported from login_policy (see imports) ---


class BrowserLogin:
    """Own one Obscura MCP session, drive sign-in, and capture userToken."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.bridge = ObscuraBridge(settings.obscura_bin, settings.obscura_profile)

    def run(self, email: str, password: str) -> LoginResult:
        email = (email or "").strip()
        password = password or ""
        if not email or not password:
            return LoginResult(detail="email and password are required")
        lock = profile_lock(self.bridge.profile)
        if not lock.acquire(blocking=False):
            print("waiting for Obscura profile lock (another warmup/login running)...")
            # Bounded wait: an indefinite block would hold the account slot
            # forever and starve the pool when the holder hangs. 180s covers
            # a full warmup + login cycle; on timeout fail fast so the
            # caller can fail over to another account.
            if not lock.acquire(timeout=180):
                return LoginResult(
                    detail=f"Obscura profile busy for {email} "
                    "(another warmup/login held the lock >180s)")
        try:
            try:
                client = mcp_client.McpClient(self.bridge.binary, self.bridge.profile)
            except Exception as exc:
                return LoginResult(
                    detail=f"could not start Obscura browser for {email}: {exc}")
            with client:
                return self._run(client, email, password)
        finally:
            try:
                lock.release()
            except Exception:
                pass

    def _run(self, client, email: str, password: str) -> LoginResult:
        try:
            _call(client, "browser_navigate", {"url": SIGN_IN_URL, "waitUntil": "networkidle0"})
        except Exception as exc:
            return LoginResult(detail=f"could not open sign_in page: {exc}")
        try:
            _call(client, "browser_wait_for", {"selector": EMAIL_SELECTOR, "timeout": EMAIL_WAIT_TIMEOUT})
        except Exception:
            logger.debug("email selector wait timed out; trying form anyway")

        try:
            marked = _tool_text(_call(client, "browser_evaluate", {"expression": _MARK_COOKIE}))
            if "ok" in marked.lower():
                try:
                    _call(client, "browser_click", {"selector": "[data-deepseaport-cookie='1']"})
                    time.sleep(COOKIE_DISMISS_DELAY_MCP)
                except Exception as exc:
                    logger.debug("cookie banner click ignored: %s", exc)
        except Exception:
            pass

        try:
            _call(client, "browser_evaluate", {
                "expression": "localStorage.removeItem('userToken'); 'cleared'"})
        except Exception:
            logger.debug("could not clear stale userToken")

        baseline = ""
        try:
            page = _json_eval(client, _PAGE_PROBE)
            if isinstance(page, dict):
                baseline = extract_token(page.get("tok"))
        except Exception:
            pass

        try:
            marked = _tool_text(_call(client, "browser_evaluate", {"expression": _MARK_LOGIN}))
            login_selector = ("[data-deepseaport-login='1']"
                              if "ok" in marked.lower() else LOGIN_FALLBACK_SELECTOR)
            # Fill and submit in two stages. A single fill_form with a
            # submit_selector clicks before React has committed the controlled
            # input state, so the form posts empty credentials and fails
            # without a network request (or with a generic "Login failed").
            try:
                filled = _tool_text(_call(client, "browser_fill_form", {
                    "fields": [
                        {"selector": EMAIL_SELECTOR, "value": email},
                        {"selector": PASSWORD_SELECTOR, "value": password},
                    ],
                }))
                low = filled.lower()
                if "filled 0 fields" in low or "error:" in low:
                    raise RuntimeError(filled[:300])
            except Exception:
                _call(client, "browser_fill", {"selector": EMAIL_SELECTOR, "value": email})
                _call(client, "browser_fill", {"selector": PASSWORD_SELECTOR, "value": password})
            # Let the SPA process onChange before clicking submit.
            time.sleep(REACT_COMMIT_DELAY)
            try:
                _call(client, "browser_click", {"selector": login_selector})
            except Exception:
                _call(client, "browser_press_key",
                      {"key": "Enter", "selector": PASSWORD_SELECTOR})
        except Exception as exc:
            detail = f"login form interaction failed (page refs changed?) for {email}: {exc}"
            logger.warning(detail)
            return LoginResult(detail=detail)

        deadline = time.monotonic() + LOGIN_TIMEOUT
        delay = _POLL_START
        ban_streak = 0
        last_url = ""
        last_body = ""
        last_error = ""
        while time.monotonic() < deadline:
            time.sleep(min(delay, max(0.05, deadline - time.monotonic())))
            delay = min(_POLL_MAX, delay * POLL_BACKOFF)
            try:
                page = _json_eval(client, _PAGE_PROBE)
            except Exception as exc:
                last_error = str(exc)
                continue
            if not isinstance(page, dict):
                continue
            last_url = str(page.get("url") or last_url)
            body = str(page.get("body") or "")
            last_body = body
            kind, phrase = classify_page_state(body, bool(page.get("captcha_visible")))
            if kind == "ban":
                ban_streak += 1
                if ban_streak >= 2:
                    detail = (f"login page shows suspension ({phrase}) "
                              f"url={last_url}: {_collapse(body, 300)}")
                    return LoginResult(banned=True, detail=detail)
                continue
            ban_streak = 0
            if kind == "captcha":
                detail = (f"login captcha for {email} (phrase={phrase!r} "
                          f"url={last_url}): {_collapse(body, 300)}")
                logger.info(detail)
                return LoginResult(detail=detail)
            if kind == "credential":
                detail = (f"login credential error for {email} (phrase={phrase!r} "
                          f"url={last_url}): {_collapse(body, 300)}")
                logger.info(detail)
                return LoginResult(detail=detail)
            token = extract_token(page.get("tok"))
            if not token:
                continue
            if not baseline or token != baseline or "sign_in" not in last_url.lower():
                return LoginResult(token=token)

        diag = (f"no token; last_url={last_url}; "
                f"last_error={last_error or 'none'}; body={_collapse(last_body, 800)}")
        logger.info("login failed for %s: %s", email, diag)
        return LoginResult(detail=diag)


def browser_login(email: str, password: str, settings: Settings) -> LoginResult:
    """Public entry point used by CLI/TUI/server refresh paths.

    A configured real browser (``browser_bin``) is preferred because it can
    initialize Shumei's device-id SDK. When it is absent/unset, fall back to
    the Obscura no-render flow (still useful on systems without a GUI browser).
    """
    try:
        from .real_browser import login_with_real_browser
        real = login_with_real_browser(email, password, settings)
    except Exception as exc:
        real = None
        logger.debug("real browser login unavailable: %s", exc)
    if real is not None:
        token, banned, detail = real
        # Real-browser localStorage returns the full JSON wrapper; normalize
        # before it can leak into an Authorization header.
        return LoginResult(token=extract_token(token) or None,
                           banned=banned, detail=detail)
    return BrowserLogin(settings).run(email, password)
