"""DeepSeek completion engine: failover, retries, PoW/session, token refresh.

Pure moves from deepseaport.server. deepseaport.server re-exports every
historical name. A few helpers are monkeypatched by tests on the
``deepseaport.server`` module, so the engine resolves them through a deferred
``_server()`` lookup at call time instead of a stale local binding.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException

from . import client as DS
from . import pow as PoW
from . import protocol as P
from .accounts import AccountPool, PooledAccount
from .config import Settings, load_settings
from .obscura_bridge import ObscuraBridge
from .tools_support import looks_truncated_tool_attempt
from .server_concurrency import (
    FAILOVER_ACQUIRE_TIMEOUT,
    INVALID_TOKEN_COOLDOWN_SECONDS,
    _CHALLENGE_EXECUTOR,
    _CLEANUP_EXECUTOR,
    _StreamCancelled,
    _await_cancelable,
)

logger = logging.getLogger("deepseaport.server")


def _server():
    """Late-bound deepseaport.server module for monkeypatch-compatible lookups."""
    from . import server as _s
    return _s


# Invalid/expired token signal (biz 40003 "Authorization Failed (invalid
# token)"). On this the request auto-refreshes via the Obscura browser login
# and retries once; only a second failure surfaces the error.
# Concurrent 40003s on the same account share one browser refresh.
_TOKEN_REFRESH_LOCK = threading.Lock()
_TOKEN_REFRESH_INFLIGHT: dict[str, threading.Event] = {}
# Login-page ban verdict from the leader refresh, shared with followers that
# waited on the same browser run (they never saw the page themselves).
_TOKEN_REFRESH_BAN_HINT: dict[str, tuple[bool, str]] = {}
# Backoff after a browser login that produced no token (captcha/credential/
# generic): hammering the login on every 40003 flags the IP/profile and makes
# the challenge permanent. Keyed by account, value (retry_after_ts, reason).
# Success, ban (own cooldown), or manual set-token/unblock clears the entry.
_LOGIN_FAIL_UNTIL: dict[str, tuple[float, str]] = {}


def _login_backoff_skip(key: str) -> str | None:
    """Remaining-skipping reason when a recent login failure backs off, else None."""
    try:
        with _TOKEN_REFRESH_LOCK:
            entry = _LOGIN_FAIL_UNTIL.get(key)
        if not entry:
            return None
        until, reason = entry
        now = time.time()
        if now >= until:
            try:
                with _TOKEN_REFRESH_LOCK:
                    _LOGIN_FAIL_UNTIL.pop(key, None)
            except Exception:
                pass
            return None
        return f"recent {reason} {int(until - now)}s ago"
    except Exception:
        return None


def _login_backoff_set(key: str, reason: str, seconds: float) -> None:
    try:
        with _TOKEN_REFRESH_LOCK:
            _LOGIN_FAIL_UNTIL[key] = (time.time() + max(0.0, seconds), reason)
    except Exception:
        pass


def clear_login_backoff(identifier: str | None = None) -> int:
    """Clear login backoff for one account (or all when None). Returns count."""
    try:
        with _TOKEN_REFRESH_LOCK:
            if identifier is None:
                n = len(_LOGIN_FAIL_UNTIL)
                _LOGIN_FAIL_UNTIL.clear()
                return n
            want = (identifier or "").strip().lower()
            dead = [k for k in list(_LOGIN_FAIL_UNTIL)
                    if k.strip().lower() == want]
            for k in dead:
                _LOGIN_FAIL_UNTIL.pop(k, None)
            return len(dead)
    except Exception:
        return 0


def _headers(app: FastAPI, token: str) -> dict:
    # getattr guard: embedded/test apps may have no lifespan bridge. Missing
    # bridge must not mask the real error with AttributeError (notably from
    # the session-cleanup finally, which must never replace the exception).
    bridge: ObscuraBridge | None = getattr(app.state, "bridge", None)
    if bridge is None:
        return P.base_headers(bearer=token)
    return P.base_headers(user_agent=bridge.state.user_agent,
                          waf_cookies=bridge.cookie_header(), bearer=token)


def _ensure_token(app: FastAPI, item: PooledAccount) -> str:
    """Return the account token, or a fast 401 that callers can fail over on.

    A missing token is a provisioning state, not something to solve by
    spawning a browser inside the first API request.  The explicit
    ``deepseaport login``/TUI flow populates it; the failover helper can move
    to another account that already has a token.  Expired tokens still take
    the one-time Obscura refresh path in _run_completion_core.
    """
    from .accounts import TOKEN_HELP, extract_token

    if item.cfg.token:
        token = extract_token(item.cfg.token)
        if token != item.cfg.token:
            item.cfg.token = token
        return token
    raise HTTPException(
        status_code=401,
        detail=f"account {item.cfg.identifier} has no token. " + TOKEN_HELP)


def _is_invalid_token_error(error: str | None) -> bool:
    """True when an attempt error is the expired/invalid token (biz 40003)."""
    low = (error or "").lower()
    return "40003" in low or (
        "authorization failed" in low and "token" in low)


def _is_transient_overload(error: str | None) -> bool:
    """True for retryable upstream overload (never a 500).

    e.g. "generation_timeout: Server busy, please try again later."
    Global throttle, not account-specific: must surface as 503 (retryable),
    never RuntimeError (unhandled 500 "Exception in ASGI application").
    """
    low = (error or "").lower()
    return any(k in low for k in (
        "generation_timeout", "server busy", "server_busy",
        "overloaded", "overload", "service unavailable",
        "temporarily unavailable", "try again", "timed out", "timeout",
        "bad gateway", "gateway timeout", "upstream timed out",
    ))


def _probe_ban(app: FastAPI, token: str, timeout: int = 5) -> tuple[bool, float | None]:
    """Best-effort users/current ban probe for one token. Never raises.

    Used to distinguish a muted/suspended account (must become 403 +
    cooldown + failover) from a merely stale token (401 + browser refresh).
    Banned accounts can surface as 40003 on session/create, so every 40003
    path probes here before concluding "invalid token".
    """
    try:
        return DS.check_ban(_headers(app, (token or "").strip()), timeout=timeout)
    except Exception:
        return False, None


def _ban_403_for_item(app: FastAPI, item: PooledAccount, ban_until: float | None,
                      error_text: str = "") -> HTTPException:
    """Mark + persist an account ban and build the 403.

    Persisting ``cfg.banned``/``cfg.banned_until`` is what makes the account
    show up as ``(BANNED: 16 September)`` in the CLI/TUI even after the server
    is restarted or the in-memory pool is rebuilt.  ``ban_until=None`` still
    marks the account (unknown expiry -> ``(BANNED)``) instead of losing the
    ban verdict.
    """
    try:
        AccountPool.mark_banned(item, until=ban_until)
        try:
            _app_settings(app).save()
        except Exception:
            pass
    except Exception:
        pass
    try:
        base = P.ban_message(ban_until) if ban_until else ""
        if not base:
            detail = (error_text or "account banned")[:300]
            if "banned" not in detail.lower() and "muted" not in detail.lower():
                detail = f"Account {item.cfg.identifier} banned. {detail}"
        else:
            detail = f"Account {item.cfg.identifier} banned. {base}"
    except Exception:
        detail = (error_text or f"Account {item.cfg.identifier} banned")[:300]
    return HTTPException(status_code=403, detail=detail)


def _refresh_token_via_obscura(app: FastAPI, item: PooledAccount,
                               cancel_event: threading.Event | None = None,
                               ) -> tuple[str | None, bool, str]:
    """Re-login through the Obscura browser and persist a fresh userToken.

    Returns (token_or_None, login_banned, ban_evidence). login_banned=True is
    ground truth from the login page suspension banner — the caller must treat
    the account as banned even when the API only ever said 40003.
    Singleflight per account: concurrent 40003s share one browser run.
    Never raises (except _StreamCancelled when the client vanished while waiting).
    """
    cfg = item.cfg
    if not (cfg.email or cfg.mobile) or not cfg.password:
        return None, False, ""
    settings = _app_settings(app)
    key = cfg.identifier or cfg.email or cfg.mobile

    with _TOKEN_REFRESH_LOCK:
        inflight = _TOKEN_REFRESH_INFLIGHT.get(key)
        if inflight is not None:
            leader = False
        else:
            inflight = threading.Event()
            _TOKEN_REFRESH_INFLIGHT[key] = inflight
            leader = True
    if not leader:
        # Chunked wait so a disconnected follower stops holding the account
        # slot instead of parking the full 180s. Leader browser run itself
        # is not killed; follower just aborts its wait promptly.
        deadline = time.monotonic() + 180
        while True:
            if cancel_event is not None and cancel_event.is_set():
                raise _StreamCancelled()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                if inflight.wait(timeout=min(0.2, remaining)):
                    break
            except Exception:
                pass
        if cancel_event is not None and cancel_event.is_set():
            raise _StreamCancelled()
        fresh = (item.cfg.token or "").strip()
        try:
            with _TOKEN_REFRESH_LOCK:
                banned, detail = _TOKEN_REFRESH_BAN_HINT.get(key, (False, ""))
        except Exception:
            banned, detail = False, ""
        return (fresh or None), banned, detail
    banned = False
    detail = ""
    # Backoff: a recent no-token login (captcha/credential) means the login
    # page is challenging this profile right now. Spawning the browser on
    # every 40003 flags it further — skip fast with the previous evidence.
    skip_reason = _login_backoff_skip(key)
    if skip_reason:
        msg = (f"browser login skipped for {key} ({skip_reason}; "
               f"manual set-token bypasses this)")
        logger.info(msg)
        return None, False, msg
    try:
        from .cli import _obscura_login_token, _unpack_login_token_result
        try:
            raw = _obscura_login_token(cfg.email or cfg.mobile,
                                       cfg.password, settings)
            new_token, banned, detail = _unpack_login_token_result(raw)
            try:
                from .accounts import extract_token as _extract_token
                new_token = _extract_token(new_token) or None
            except Exception:
                pass
        except Exception as exc:
            logger.warning("token refresh via Obscura failed for %s: %s", key, exc)
            new_token, banned, detail = None, False, ""
        try:
            with _TOKEN_REFRESH_LOCK:
                _TOKEN_REFRESH_BAN_HINT[key] = (bool(banned), str(detail or ""))
        except Exception:
            pass
        if banned:
            logger.warning("Obscura login page shows account %s BANNED: %s",
                           key, (detail or "")[:200])
            clear_login_backoff(key)
        if new_token:
            from .accounts import set_account_token
            item.cfg.token = new_token
            try:
                set_account_token(settings, key, new_token)
                settings.save()
            except Exception as exc:
                logger.warning("token refresh save failed for %s: %s", key, exc)
            if not banned:
                logger.info("token refreshed via Obscura for %s", key)
                clear_login_backoff(key)
            return new_token, bool(banned), str(detail or "")
        if not banned:
            # No token, no ban page: back off the browser so repeated 40003s
            # don't hammer the challenge endpoint. Manual set-token clears it.
            low = (detail or "").lower()
            if "captcha" in low:
                _login_backoff_set(key, "captcha-challenge", 180)
            elif "credential" in low or "incorrect" in low or "wrong password" in low:
                _login_backoff_set(key, "credential-error", 600)
            elif "form interaction failed" in low:
                _login_backoff_set(key, "login-form-changed", 600)
            else:
                _login_backoff_set(key, "no-token", 120)
        return None, bool(banned), str(detail or "")
    finally:
        with _TOKEN_REFRESH_LOCK:
            _TOKEN_REFRESH_INFLIGHT.pop(key, None)
            try:
                inflight.set()
            except Exception:
                pass


def _solve(app: FastAPI, token: str) -> dict[str, str]:
    """Fresh PoW header merged over base headers."""
    headers = _headers(app, token)
    return {**headers, "X-DS-PoW-Response": DS.solve_pow(headers)}


def _session_and_challenge(headers: dict, parallel: bool = True,
                            cancel_event: threading.Event | None = None,
                            ) -> tuple[str, dict]:
    """Create session + fetch PoW challenge concurrently (same headers).

    Sequential code paid 2 RTTs; these calls are independent. Session errors
    take priority to match old sequential ordering. When parallel=False the
    two calls run sequentially (session first, same error priority).

    cancel_event aborts the wait between SSE-independent setup calls so a
    disconnected client frees the account slot instead of running the full
    session→PoW→stream while the next request queues to 90s/429.
    """
    if cancel_event is not None and cancel_event.is_set():
        raise _StreamCancelled()
    if not parallel:
        session_id = DS.create_session(headers)
        if cancel_event is not None and cancel_event.is_set():
            try:
                DS.delete_session(headers, session_id)
            except Exception:
                pass
            raise _StreamCancelled()
        try:
            challenge = DS.fetch_pow(headers)
        except Exception:
            # Session created but challenge failed: don't leak it.
            try:
                DS.delete_session(headers, session_id)
            except Exception:
                pass
            raise
        if cancel_event is not None and cancel_event.is_set():
            try:
                DS.delete_session(headers, session_id)
            except Exception:
                pass
            raise _StreamCancelled()
        return session_id, challenge
    fut_session = _CHALLENGE_EXECUTOR.submit(DS.create_session, headers)
    fut_challenge = _CHALLENGE_EXECUTOR.submit(DS.fetch_pow, headers)
    try:
        session_id = _await_cancelable(fut_session, cancel_event)
    except _StreamCancelled:
        try:
            fut_challenge.cancel()
        except Exception:
            pass
        raise
    except Exception:
        try:
            _await_cancelable(fut_challenge, None)
        except Exception:
            pass
        raise
    try:
        challenge = _await_cancelable(fut_challenge, cancel_event)
    except _StreamCancelled:
        try:
            DS.delete_session(headers, session_id)
        except Exception:
            pass
        raise
    except Exception:
        # Session was created but challenge fetch failed: don't leak it.
        try:
            DS.delete_session(headers, session_id)
        except Exception:
            pass
        raise
    return session_id, challenge


def _header_from_challenge(challenge: dict) -> str:
    """Solve an already-fetched PoW challenge (no refetch)."""
    return DS.solve_challenge(challenge)


def _delete_session_bg(headers: dict, session_id: str | None, enabled: bool = True) -> None:
    """Best-effort session cleanup without blocking the response."""
    if not enabled or not session_id:
        return

    def _run() -> None:
        try:
            DS.delete_session(headers, session_id)
        except Exception:
            pass

    try:
        _CLEANUP_EXECUTOR.submit(_run)
    except Exception:
        pass


def _app_settings(app: FastAPI) -> Settings:
    settings = getattr(app.state, "settings", None)
    if settings is None:
        settings = load_settings()
        app.state.settings = settings
    return settings


def _refresh_waf_without_account_lock(app: FastAPI, item: PooledAccount,
                                        cancel_event: threading.Event | None = None) -> None:
    """Global WAF refresh without wasting the per-account slot.

    The caller holds item.lock (one in-flight per account). A WAF refresh is
    global (shared profile/cookies) and singleflight-deduped in the bridge,
    so release the account slot while waiting for it, then reacquire before
    the retry attempt. Safe when the lock isn't held (unit tests calling
    _run_completion_core with a fresh account): release fails -> just warm.
    Reacquire polls so a disconnected client aborts instead of blocking
    indefinitely under contention (raises _StreamCancelled).
    """
    released = False
    try:
        item.lock.release()
        released = True
    except RuntimeError:
        released = False
    except Exception:
        released = False
    try:
        try:
            bridge = getattr(app.state, "bridge", None)
            if bridge is not None:
                bridge.warmup()
        except Exception as exc:
            logger.warning("WAF warmup failed: %s", exc)
    finally:
        if released:
            # Cancelable reacquire: unbounded acquire() would pin an orphaned
            # producer indefinitely when another request holds the slot.
            # AccountPool.release() swallows RuntimeError, so leaving the
            # slot released on cancel is safe (outer release becomes no-op).
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    raise _StreamCancelled()
                try:
                    if item.lock.acquire(timeout=0.5):
                        break
                except Exception:
                    break


def _is_ban_http_exception(exc: BaseException) -> bool:
    """True when exc is the ban 403 raised by _run_completion_core."""
    return (isinstance(exc, HTTPException) and exc.status_code == 403
            and P.is_ban_error_text(str(getattr(exc, "detail", "") or "")))


def _switch_current_after_ban_failover(pool: AccountPool, settings: Settings,
                                       used: PooledAccount,
                                       failed: PooledAccount,
                                       reason: str = "unusable") -> None:
    """Move CURRENT to the working account after failover.

    Only fires when CURRENT pointed at the failed account: the request
    already proved it unusable (ban/auth cooldown), so future requests should
    prefer the healthy one first. Best-effort, never raises.
    """
    try:
        from .accounts import _matches as _m
        cur = pool.current
        if not cur or not _m(failed.cfg, cur):
            return  # auto-failover mode or CURRENT wasn't the failed one
        if _m(used.cfg, cur):
            return  # same account (shouldn't happen after mark_bad)
        if pool.set_current(used.cfg.identifier):
            try:
                settings.active_account = pool.current
                settings.save()
            except Exception:
                pass
            logger.warning("account %s %s, CURRENT auto-switched -> %s",
                           failed.cfg.identifier, reason, used.cfg.identifier)
    except Exception:
        pass


def _complete_with_failover_sync(app: FastAPI, pool: AccountPool, prep: dict,
                                 first_item: PooledAccount,
                                 cancel_event: threading.Event | None = None,
                                 on_content=None, on_thinking=None,
                                 allow_failover: bool | None = None) -> dict:
    """Run one completion, auto-failing over on ban or broken credentials.

    Owns first_item's lock: releases each attempt exactly once, acquires the
    next account after marking the failed one cooling (ban expiry or a short
    token-invalid cooldown). Returns the result of the first usable account.
    allow_failover=False (settings.use_multiple_accounts=False) surfaces the
    first error directly, never touching another account.
    None (default) reads the flag from app settings.
    """
    # Late-bound so tests can monkeypatch deepseaport.server._run_completion_core.
    from .server import _run_completion_core
    if allow_failover is None:
        try:
            allow_failover = bool(getattr(_app_settings(app), "use_multiple_accounts", True))
        except Exception:
            allow_failover = True
    cur = first_item
    first_exc: HTTPException | None = None
    failed_item: PooledAccount | None = None
    tries = 0
    max_tries = max(1, len(pool)) if allow_failover else 1
    while True:
        try:
            result = _run_completion_core(app, cur, prep, on_content=on_content,
                                          on_thinking=on_thinking,
                                          cancel_event=cancel_event)
        except _StreamCancelled:
            AccountPool.release(cur)
            raise
        except HTTPException as exc:
            AccountPool.release(cur)
            # Bans and 401 auth failures are account-specific; try a healthy
            # account before surfacing the error. Other HTTP failures (400,
            # 429, WAF, ...) belong to the caller/current request.
            retryable = _is_ban_http_exception(exc) or exc.status_code == 401
            if (not retryable
                    or (cancel_event is not None and cancel_event.is_set())
                    or tries + 1 >= max_tries):
                raise
            if first_exc is None:
                first_exc, failed_item = exc, cur
            tries += 1
            if exc.status_code == 401:
                AccountPool.mark_bad(cur, seconds=INVALID_TOKEN_COOLDOWN_SECONDS)
                logger.info("account %s has no usable token, failing over "
                            "(%d/%d)", cur.cfg.identifier, tries + 1, max_tries)
            else:
                logger.info("account %s banned, failing over to next account "
                            "(%d/%d)", cur.cfg.identifier, tries + 1, max_tries)
            try:
                cur = pool.acquire(timeout=FAILOVER_ACQUIRE_TIMEOUT)
            except Exception:
                raise first_exc
            continue
        except Exception:
            AccountPool.release(cur)
            raise
        if failed_item is not None:
            reason = "banned" if _is_ban_http_exception(first_exc) else "has no usable token"
            _switch_current_after_ban_failover(
                pool, _app_settings(app), cur, failed_item, reason=reason)
        AccountPool.release(cur)
        return result


def _run_completion_core(
    app: FastAPI,
    item: PooledAccount,
    prep: dict,
    on_content=None,
    on_thinking=None,
    cancel_event: threading.Event | None = None,
) -> dict:
    """Shared retry engine for buffered + live streaming.

    on_content/on_thinking are called incrementally (producer thread) for the
    live stream mode. Overload retries are skipped once partial deltas were
    already emitted (re-stream would duplicate text); overload retries use a
    fresh PoW header. PoW/session/WAF retries normally carry no content
    prefix, so the residual interleaving risk is minimal.
    Buffered callers pass no callbacks and get identical behaviour to before.

    cancel_event, when set, aborts the attempt at the next SSE event so a
    disconnected client does not keep the account lock for the whole reply.
    """
    # Late-bound so tests can monkeypatch deepseaport.server._attempt/_solve/
    # _delete_session_bg and have this engine pick up the replacements.
    from .server import _attempt, _delete_session_bg, _solve
    settings = _app_settings(app)
    max_retries = max(0, int(getattr(settings, "max_retries", 1)))
    parallel = bool(getattr(settings, "parallel_challenge_fetch", True))
    auto_delete = bool(getattr(settings, "auto_delete_session", True))
    token = _ensure_token(app, item)
    session_id: str | None = None
    token_refreshed = False
    # Diagnostic from the browser login (no-token reason, last URL/body) so
    # the terminal 401 tells exactly what the login saw. Never marks banned.
    login_note = ""

    def _cancelled() -> bool:
        return cancel_event is not None and cancel_event.is_set()

    def _refresh_token_once() -> bool:
        """Refresh the account token via Obscura, once per request.

        Returns True when a new token replaced the stale one. Shared by the
        session-create path (40003 raised) and the stream path (40003 event).
        Raises 403 immediately when the login page itself showed a suspension
        banner (deterministic ban, even if the API only ever said 40003).
        """
        nonlocal token, token_refreshed, login_note
        if token_refreshed:
            return False
        logger.info("token invalid for %s, refreshing via Obscura browser",
                    item.cfg.identifier)
        old_token = (token or "").strip()
        new_token, login_banned, login_detail = _refresh_token_via_obscura(
            app, item, cancel_event=cancel_event)
        if _cancelled():
            raise _StreamCancelled()
        if login_banned:
            # Ground truth from the browser: the login page showed
            # suspend/banned/violation. The API may only say 40003 for this
            # account, so the users/current probe cannot confirm it — trust
            # the page, mark + 403 for failover. Probe for mute_until first;
            # fall back to parsing the human date printed on the page.
            try:
                _, ban_until = _probe_ban(app, new_token or old_token)
            except Exception:
                ban_until = None
            if not ban_until:
                try:
                    from .auth import parse_ban_until
                    ban_until = parse_ban_until(login_detail)
                except Exception:
                    ban_until = None
            raise _ban_403_for_item(
                app, item, ban_until,
                f"Obscura login page shows account {item.cfg.identifier} "
                f"suspended: {(login_detail or '')[:200]}")
        if login_detail and not new_token:
            login_note = str(login_detail)[:500]
            logger.warning("Obscura login for %s produced no usable token: %s",
                           item.cfg.identifier, login_note[:300])
        if not new_token:
            return False
        if old_token and new_token.strip() == old_token:
            # Login demonstrably succeeded (left sign_in, else the login
            # helper would have ignored this value as stale), so the same
            # value is a stable token re-issue, not a failure. Retry with it;
            # a renewed 40003 then means account status, handled by the probe.
            logger.info("token refresh for %s returned same value after "
                        "verified login (stable token); retrying", item.cfg.identifier)
        token = new_token
        token_refreshed = True
        return True

    def _open_session() -> tuple[dict, str, dict]:
        """Create session + PoW; auto-refresh the token once on 40003.

        The 40003 surfaces as a RuntimeError from create_session. On the first
        hit the token is refreshed via Obscura and session creation retried;
        a second failure propagates to the caller untouched — except bans:
        a ban payload (biz 5 muted) or a users/current probe showing muted
        becomes 403 + cooldown (so failover + accounts list mark it) instead
        of a RuntimeError/401 that never marks the account. Banned accounts
        can present as 40003 on session/create, hence the probe on both
        attempts before concluding "invalid token".
        """
        nonlocal session_id
        last_exc: Exception | None = None
        for attempt in range(2):
            if _cancelled():
                raise _StreamCancelled()
            hdrs = _headers(app, token)
            try:
                sid, ch = _session_and_challenge(
                    hdrs, parallel=parallel, cancel_event=cancel_event)
                return hdrs, sid, ch
            except _StreamCancelled:
                raise
            except Exception as exc:
                last_exc = exc
                exc_s = str(exc)
                # Ban at session/create (biz 5 "user is muted"): mark + 403
                # so the failover helper retries on the next account. The
                # probe supplies mute_until for a precise cooldown; the ban
                # payload itself is trusted even when the probe has no ts.
                if P.is_ban_error_text(exc_s):
                    try:
                        _, ban_until = _probe_ban(app, token)
                    except Exception:
                        ban_until = None
                    raise _ban_403_for_item(app, item, ban_until, exc_s)
                if attempt == 0 and _is_invalid_token_error(exc_s):
                    # Old token 40003 may be a muted ban masquerading as
                    # invalid: probe before spending a ~2min browser run.
                    # Probe uses users/current: 40003 there means stale token
                    # (not banned); a mute payload means banned. An invalid
                    # token can never prove a ban — only the browser page can.
                    try:
                        banned, ban_until = _probe_ban(app, token)
                    except Exception:
                        banned, ban_until = False, None
                    if banned:
                        raise _ban_403_for_item(app, item, ban_until, exc_s)
                    logger.debug("ban probe for %s: not banned "
                                 "(old token invalid), trying browser refresh",
                                 item.cfg.identifier)
                    if _refresh_token_once():
                        continue
                if _is_invalid_token_error(exc_s):
                    # Fresh token also 40003 (or refresh impossible): probe
                    # once more — a ban that presents as 40003 must surface
                    # as 403 + cooldown, not a misleading 401. When the probe
                    # still says 40003 (invalid), there is no ban evidence, so
                    # 401 stays UNMARKED by design: marking it banned would
                    # false-positive wrong-password/captcha outages.
                    try:
                        banned, ban_until = _probe_ban(app, token)
                    except Exception:
                        banned, ban_until = False, None
                    if banned:
                        raise _ban_403_for_item(app, item, ban_until, exc_s)
                    note = f" Browser login note: {login_note[:300]}." if login_note else ""
                    raise HTTPException(
                        status_code=401,
                        detail=f"Authorization Failed: token invalid for "
                               f"{item.cfg.identifier} and could not be "
                               f"refreshed ({exc_s[:200]}).{note} Re-login the account. "
                               f"If the web UI shows suspension/violation, "
                               f"treat it as BANNED.")
                raise
        raise last_exc if last_exc else RuntimeError("session setup failed")

    # Fast path: skip cookie-header builds when wasm already cached.
    try:
        wp = PoW.wasm_path()
        if not (wp.exists() and wp.stat().st_size > 1000):
            _bridge = getattr(app.state, "bridge", None)
            PoW.ensure_wasm(cookies=_bridge.cookie_header() if _bridge else "",
                            user_agent=_bridge.state.user_agent if _bridge else "")
    except Exception as exc:
        logger.warning("wasm ensure failed: %s", exc)
    # Session creation + PoW solve sit inside the try so a solver failure
    # (answer None -> RuntimeError) still runs the finally cleanup below.
    try:
        if _cancelled():
            raise _StreamCancelled()
        headers, session_id, challenge = _open_session()
        if _cancelled():
            raise _StreamCancelled()
        headers = {**headers, "X-DS-PoW-Response": _header_from_challenge(challenge)}
        if _cancelled():
            raise _StreamCancelled()
        payload = P.completion_payload(session_id, prep["prompt"], prep["thinking"],
                                       prep["search"], prep["model_type"])
        # Ban expiry rides along with the completion error event (mute_until
        # from the ban envelope) so the ban path below needs no extra HTTP.
        # check_ban stays only as fallback when the stream carried no ts.
        stream_ban_until: float | None = None
        # Live mode already _put partial deltas to the client; a retry after
        # partial emit would re-stream from 0 (duplicated/interleaved text
        # and corrupt tool-call JSON). Track emits so overload retries skip
        # when output already left the server.
        emitted_live = False

        def _wrapped_content(t: str) -> None:
            nonlocal emitted_live
            if t:
                emitted_live = True
            if on_content is not None:
                try:
                    on_content(t)
                except Exception:
                    pass

        def _wrapped_thinking(t: str) -> None:
            nonlocal emitted_live
            if t:
                emitted_live = True
            if on_thinking is not None:
                try:
                    on_thinking(t)
                except Exception:
                    pass

        def _do_attempt(h: dict, p: dict) -> None:
            nonlocal content, think, error, usage_total, stream_ban_until
            outcome = _attempt(
                h, p,
                on_content=(_wrapped_content if on_content is not None else None),
                on_thinking=(_wrapped_thinking if on_thinking is not None else None),
                cancel_event=cancel_event)
            content, think, error, usage_total = outcome
            try:
                cur_ban = getattr(outcome, "ban_until", None)
            except Exception:
                cur_ban = None
            if cur_ban is not None:
                stream_ban_until = cur_ban

        content = think = ""
        error: str | None = None
        usage_total = 0
        _do_attempt(headers, payload)
        for _ in range(max_retries):
            if _cancelled() or not (error and "pow" in error.lower()):
                break
            logger.info("pow rejected, solving fresh (retries left %s)", max_retries)
            headers = _solve(app, token)
            _do_attempt(headers, payload)
        # Session recreate needs a fresh payload bound to the new session id
        # and a fresh PoW header (the old one may have expired by now).
        for _ in range(max_retries):
            if _cancelled() or not (error and error.startswith("INVALID_SESSION_ID")):
                break
            logger.info("session invalid, recreating (retries left %s)", max_retries)
            old_sid = session_id
            try:
                session_id = DS.create_session(_headers(app, token))
            except Exception as exc:
                # Recreated session can itself reveal a ban (biz 5): it must
                # mark + 403 for failover, not leak as RuntimeError.
                if P.is_ban_error_text(str(exc)):
                    try:
                        _, _ban_until = _probe_ban(app, token)
                    except Exception:
                        _ban_until = None
                    raise _ban_403_for_item(app, item, _ban_until, str(exc))
                raise
            # Old id was reported invalid upstream, but delete best-effort so
            # a stale-but-valid id never lingers when the error text lied.
            if old_sid and old_sid != session_id:
                _delete_session_bg(_headers(app, token), old_sid, enabled=auto_delete)
            headers = _solve(app, token)
            payload = P.completion_payload(session_id, prep["prompt"], prep["thinking"],
                                           prep["search"], prep["model_type"])
            _do_attempt(headers, payload)
        # Invalid token caught at completion time (not session create): the
        # reply arrives as a 40003 error event. Refresh once, recreate the
        # session with the new token, and retry before surfacing anything.
        for _ in range(max_retries):
            if _cancelled() or not _is_invalid_token_error(error):
                break
            old_sid = session_id
            old_headers = _headers(app, token)
            if not _refresh_token_once():
                break
            logger.info("retrying completion with refreshed token")
            try:
                session_id = DS.create_session(_headers(app, token))
            except Exception as exc:
                if P.is_ban_error_text(str(exc)):
                    try:
                        _, _ban_until = _probe_ban(app, token)
                    except Exception:
                        _ban_until = None
                    raise _ban_403_for_item(app, item, _ban_until, str(exc))
                if _is_invalid_token_error(str(exc)):
                    # Fresh token rejected at session create: may be a ban
                    # presenting as 40003 — probe before giving up.
                    try:
                        _banned, _ban_until = _probe_ban(app, token)
                    except Exception:
                        _banned, _ban_until = False, None
                    if _banned:
                        raise _ban_403_for_item(app, item, _ban_until, str(exc))
                raise
            # Old session was created with the previous token: delete with the
            # old headers (new token would 401 the delete). Best-effort.
            if old_sid and old_sid != session_id:
                _delete_session_bg(old_headers, old_sid, enabled=auto_delete)
            headers = _solve(app, token)
            payload = P.completion_payload(session_id, prep["prompt"], prep["thinking"],
                                           prep["search"], prep["model_type"])
            _do_attempt(headers, payload)
        for _ in range(max_retries):
            if _cancelled() or not (error and ("403" in error or "WAF" in error)):
                break
            logger.info("possible WAF block, refreshing cookies (retries left %s)", max_retries)
            # Global refresh (singleflight in bridge) must not hold the
            # per-account slot while all in-flight requests wait on it.
            _refresh_waf_without_account_lock(app, item, cancel_event=cancel_event)
            if _cancelled():
                break
            headers = _solve(app, token)
            _do_attempt(headers, payload)
        for _ in range(max_retries):
            if _cancelled() or not _is_transient_overload(error):
                break
            if emitted_live:
                # Partial deltas already sent: retry would duplicate text.
                # Surface the overload as-is instead of re-streaming.
                logger.info("upstream overloaded after partial emit, not retrying "
                            "to avoid duplication (%s)", (error or "")[:80])
                break
            logger.info("upstream overloaded (%s), retrying (retries left %s)",
                        (error or "")[:80], max_retries)
            # Fresh PoW: the prior header may have expired during a long
            # reasoning stream; reusing it risks POW_HEADER_ERROR on retry.
            try:
                headers = _solve(app, token)
            except Exception as exc:
                logger.warning("overload retry PoW refresh failed: %s", exc)
            _do_attempt(headers, payload)
        if _cancelled():
            raise _StreamCancelled()
        if error:
            low = error.lower()
            # Invalid/expired token: already refreshed once above; reaching
            # here means the fresh token also failed. A muted ban can present
            # as 40003 on completion, so probe users/current once before
            # concluding "invalid" — otherwise bans surface as 401 and the
            # account is never marked cooling/failing over.
            if _is_invalid_token_error(error):
                try:
                    _banned, _ban_until = _probe_ban(app, token)
                except Exception:
                    _banned, _ban_until = False, None
                if _banned:
                    raise _ban_403_for_item(app, item, _ban_until, error)
                _note = f" Browser login note: {login_note[:300]}." if login_note else ""
                raise HTTPException(
                    status_code=401,
                    detail=f"Authorization Failed: token invalid for "
                           f"{item.cfg.identifier} even after browser refresh "
                           f"({error[:200]}).{_note} Re-login the account. If the web "
                           f"UI shows suspension/violation, treat it as BANNED.")
            # Ban/auth family is account-specific: cool that account down.
            # Rate-limit family is usually a global throttle, so do NOT poison
            # a specific account (that would rotate through and shrink the pool).
            # P.is_ban_error_text uses letter-boundary matching so "commute"
            # does not false-positive on "mute".
            is_ban = (
                P.is_ban_error_text(error)
                or any(k in low for k in ("restricted", "unauthorized", "401"))
            )
            if is_ban:
                # Banned accounts stay unusable until mute_until: cool down for
                # the remaining suspension. The completion ban envelope already
                # carried mute_until (stream_ban_until) — use it directly and
                # skip the extra users/current HTTP. Fall back to check_ban
                # (short 5s timeout) only when the stream had no timestamp.
                ban_until: float | None = stream_ban_until
                if ban_until is None:
                    try:
                        _, ban_until = DS.check_ban(_headers(app, token), timeout=5)
                    except Exception:
                        ban_until = None
                raise _ban_403_for_item(app, item, ban_until, error)
            if any(k in low for k in ("rate_limit", "too frequent", "too_frequent", "429")):
                raise HTTPException(status_code=429,
                                    detail=f"DeepSeek rate limited: {error[:200]}. "
                                           "Wait 2-20 min, space requests, avoid parallel loops.")
            if _is_transient_overload(error):
                raise HTTPException(status_code=503,
                                    detail=f"DeepSeek server busy: {error[:200]}. "
                                           "Retry shortly (transient overload).")
            if "http_422" in low or "unprocessable" in low or "http_400" in low:
                logger.warning("upstream rejected completion payload (%s); "
                               "prompt_len=%d model=%s",
                               error[:500], len(prep.get("prompt", "")),
                               prep.get("model"))
                raise HTTPException(
                    status_code=502,
                    detail=f"DeepSeek rejected the request as invalid ({error[:800]}). "
                           "The prompt/payload failed upstream validation, not a ban or "
                           "rate limit. Retry with a non-empty prompt; if it persists, "
                           "the upstream API schema likely changed.")
            logger.warning("upstream completion error: %s", error[:500])
            raise HTTPException(
                status_code=502,
                detail=f"DeepSeek upstream error: {error[:800]}")
        if not content and not think:
            # Upstream closed without content/thinking/error (e.g. unparsed toast).
            # Never return silent empty 200: it surfaces as "tool stopped working".
            raise HTTPException(status_code=502,
                                detail="empty upstream response from DeepSeek "
                                       "(likely throttling; wait then retry)")
        return {"content": content, "thinking": think, "usage_total": usage_total}
    finally:
        try:
            # Guard session_id first so no header build happens when there is
            # nothing to delete; _headers itself is bridge-safe (getattr).
            if auto_delete and session_id:
                _delete_session_bg(_headers(app, token), session_id, enabled=True)
        except Exception:
            pass


class _AttemptOutcome(tuple):
    """4-tuple (content, thinking, error, usage) + ban_until attr.

    Subclass so existing unpacking `a, b, c, d = _attempt(...)` keeps
    working (tests + callers), while _run_completion_core reads
    `.ban_until` to skip the extra users/current HTTP on bans.
    Plain-tuple mocks simply have no attr -> getattr(..., None) fallback.
    """

    def __new__(cls, content: str, think: str, error: str | None,
                usage: int, ban_until: float | None = None):
        obj = super().__new__(cls, (content, think, error, usage))
        obj.ban_until = ban_until
        return obj


def _attempt(headers: dict, payload: dict, on_content=None, on_thinking=None,
             cancel_event: threading.Event | None = None,
             ) -> tuple[str, str, str | None, int]:
    content: list[str] = []
    think: list[str] = []
    error: str | None = None
    usage_total = 0
    ban_until: float | None = None
    # Explicit gen.close(): break out of a generator does NOT run its
    # finally (resp.close() in client.stream_completion) until close()/GC.
    # Close promptly so the curl connection returns to the pool instead of
    # lingering through retries (burst contention) or GC timing.
    gen = DS.stream_completion(headers, payload)
    try:
        for event in gen:
            if cancel_event is not None and cancel_event.is_set():
                # Client gone: stop reading and close below so the HTTP
                # response closes and the account lock releases promptly.
                break
            if event.kind == "content":
                content.append(event.text)
                if on_content is not None:
                    try:
                        on_content(event.text)
                    except Exception:
                        pass
            elif event.kind == "thinking":
                think.append(event.text)
                if on_thinking is not None:
                    try:
                        on_thinking(event.text)
                    except Exception:
                        pass
            elif event.kind == "usage":
                try:
                    usage_total = max(usage_total, int(event.text))
                except ValueError:
                    pass
            elif event.kind == "error":
                error = f"{event.code}: {event.message}"
                try:
                    ban_until = getattr(event, "ban_until", None)
                except Exception:
                    ban_until = None
                break
            elif event.kind == "finished":
                break
    finally:
        try:
            close = getattr(gen, "close", None)
            if callable(close):
                close()
        except Exception:
            pass
    return _AttemptOutcome("".join(content), "".join(think), error,
                           usage_total, ban_until)


def _openai_response(prep: dict, result: dict) -> dict:
    # Late-bound so tests can monkeypatch deepseaport.server.parse_tool_calls.
    from .server import parse_tool_calls
    cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    limit = prep.get("tool_args_max_chars")
    forgiving = bool(prep.get("forgiving_toolcalls", False))
    if prep["tools"]:
        if limit:
            calls, remaining = parse_tool_calls(
                result["content"], prep["tools"],
                max_args_chars=limit, forgiving=forgiving)
        else:
            calls, remaining = parse_tool_calls(
                result["content"], prep["tools"], forgiving=forgiving)
    else:
        calls, remaining = None, result["content"]
    if prep["tools"] and not calls:
        logger.debug("no tool call parsed model=%s preview=%.200s",
                     prep.get("model"), (result["content"] or "")[:200])
    message: dict[str, Any] = {"role": "assistant", "content": remaining}
    if result["thinking"]:
        message["reasoning_content"] = result["thinking"]
    finish = "stop"
    if calls:
        message["tool_calls"] = calls
        finish = "tool_calls"
    elif prep["tools"] and looks_truncated_tool_attempt(
            result["content"], prep["tools"], forgiving=forgiving):
        # The upstream web protocol has no finish_reason. An unbalanced tool
        # attempt means the reply was cut mid-call; report OpenAI's length
        # signal so the harness can retry/split instead of showing raw markup.
        finish = "length"
    pt = max(1, len(prep["prompt"]) // 4)
    ct = result.get("usage_total") or max(1, (len(result["content"]) + len(result["thinking"])) // 4)
    return {"id": cid, "object": "chat.completion", "created": created, "model": prep["model"],
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct}}
