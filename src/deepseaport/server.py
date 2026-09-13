"""FastAPI server: OpenAI-compatible endpoints over the DeepSeek web client.

Concurrency model: endpoints are async; all blocking DeepSeek I/O runs in
worker threads. Streaming responses bridge a producer thread (DeepSeek SSE)
to the response generator through a queue, so first tokens reach the
harness without waiting for the full reply. Tool calls are parsed from the
buffered full text at FINISHED and emitted as OpenAI tool_call deltas.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from . import client as DS
from . import pow as PoW
from . import protocol as P
from .accounts import AccountPool, PooledAccount, login
from .config import AccountConfig, Settings, load_settings
from .obscura_bridge import ObscuraBridge, register_default_bridge
from .tools_support import parse_tool_calls, render_prompt, tool_reminder_prompt, tool_system_prompt

logger = logging.getLogger("deepseaport.server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

MODELS = {
    "deepseek-flash": (False, False, "default"),
    "deepseek-flash-reasoner": (True, False, "default"),
    "deepseek-flash-search": (False, True, "default"),
    "deepseek-flash-reasoner-search": (True, True, "default"),
}

# Shared executors: replacing per-request ThreadPoolExecutor/thread churn.
# Challenge work (create_session + fetch_pow) is I/O-bound; sized generously
# so concurrent completions never serialize (previously one 2-worker executor
# per request). Cleanup (delete_session) is fire-and-forget best-effort.
_DEFAULT_POOL_SIZE = min(32, (os.cpu_count() or 1) + 4)
_CHALLENGE_EXECUTOR = ThreadPoolExecutor(
    max_workers=2 * _DEFAULT_POOL_SIZE, thread_name_prefix="ds-challenge")
_CLEANUP_EXECUTOR = ThreadPoolExecutor(
    max_workers=_DEFAULT_POOL_SIZE, thread_name_prefix="ds-cleanup")
# Stream producers (one per streamed request, each holding an account slot +
# PoW CPU): bounded so a parallel-subagent burst queues instead of piling
# unbounded raw threads.
_STREAM_EXECUTOR = ThreadPoolExecutor(
    max_workers=_DEFAULT_POOL_SIZE, thread_name_prefix="ds-stream")

# Consumer wait (seconds) for a producer event, on top of the curl stream
# timeout. Keeps the async reader from timing out before the blocking call.
_CONSUMER_GRACE_SECONDS = 10

# Failover wait (seconds) for a free account when the current one turns out
# banned mid-request. Short: healthy accounts are normally idle; long waits
# would stall the request that already paid for a failed ban attempt.
FAILOVER_ACQUIRE_TIMEOUT = 10

# Invalid/expired token signal (biz 40003 "Authorization Failed (invalid
# token)"). On this the request auto-refreshes via the Obscura browser login
# and retries once; only a second failure surfaces the error.
# Concurrent 40003s on the same account share one browser refresh.
_TOKEN_REFRESH_LOCK = threading.Lock()
_TOKEN_REFRESH_INFLIGHT: dict[str, threading.Event] = {}


class _StreamCancelled(Exception):
    """Internal: client disconnected, producer should stop at the next event."""


async def _watch_disconnect(request: Request, cancel_event: threading.Event) -> None:
    """Poll Request.is_disconnected() into cancel_event (non-stream path).

    The stream path sets cancel_event from the response generator's finally;
    non-stream has no generator, so a background poller watches the client
    socket. When the client goes away the blocking completion aborts at the
    next SSE event / setup checkpoint and releases the account slot instead
    of running the full session→PoW→stream while the next request queues.
    """
    try:
        while not cancel_event.is_set():
            try:
                if await request.is_disconnected():
                    cancel_event.set()
                    break
            except Exception:
                break
            await asyncio.sleep(0.15)
    except asyncio.CancelledError:
        pass


async def _acquire_slot_or_499(pool, request: Request,
                               cancel_event: threading.Event,
                               timeout: float, allow_failover: bool):
    """Acquire an account slot, aborting the queue wait on disconnect.

    aacquire() alone would park the endpoint task for the full 90s even
    after the client gave up. Chunk the wait so a disconnect surfaces as
    499 within ~2s without ever holding a slot.
    """
    from fastapi import HTTPException as _HTTPException

    deadline = time.monotonic() + max(0.0, timeout)
    while True:
        chunk = min(2.0, max(0.05, deadline - time.monotonic()))
        try:
            return await pool.aacquire(chunk, allow_failover=allow_failover)
        except ValueError:
            raise _HTTPException(status_code=503, detail="no DeepSeek accounts in pool")
        except TimeoutError:
            if cancel_event.is_set():
                raise _HTTPException(status_code=499, detail="client disconnected")
            try:
                if await request.is_disconnected():
                    raise _HTTPException(status_code=499, detail="client disconnected")
            except _HTTPException:
                raise
            except Exception:
                pass
            if time.monotonic() >= deadline:
                raise _HTTPException(status_code=429,
                                     detail="all DeepSeek accounts busy, retry later")
            continue


def apply_log_level(level: str) -> None:
    """Apply log level to deepseaport loggers (settings-driven, no handler reset)."""
    try:
        numeric = getattr(logging, str(level or "INFO").upper(), logging.INFO)
        for name in ("deepseaport.server", "deepseaport.client", "deepseaport.tools",
                     "deepseaport.pow", "deepseaport.accounts", "deepseaport.obscura",
                     "deepseaport.protocol"):
            logging.getLogger(name).setLevel(numeric)
    except Exception:
        pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: Settings = app.state.settings
    apply_log_level(getattr(settings, "log_level", "INFO"))
    bridge = ObscuraBridge(settings.obscura_bin, settings.obscura_profile)
    app.state.bridge = bridge
    register_default_bridge(bridge)
    if getattr(settings, "warmup_on_startup", True):
        threading.Thread(target=_startup_warm, args=(bridge,), daemon=True).start()
    yield
    # NOTE: module-level executors intentionally NOT shut down here: lifespan
    # can run multiple times per process (tests, reload) and a shutdown
    # executor raises RuntimeError on submit. Threads exit at process end.


def _startup_warm(bridge: ObscuraBridge) -> None:
    try:
        bridge.warmup()
    except Exception as exc:
        logger.warning("startup WAF warmup failed: %s", exc)
    try:
        PoW.ensure_wasm(cookies=bridge.cookie_header(), user_agent=bridge.state.user_agent)
    except Exception as exc:
        logger.warning("PoW wasm fetch failed (will retry on demand): %s", exc)


def create_app(settings: Settings | None = None) -> FastAPI:
    app = FastAPI(title="deepseaport", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings or load_settings()
    apply_log_level(getattr(app.state.settings, "log_level", "INFO"))
    app.state.pool = None

    @app.get("/health")
    async def health():
        bridge: ObscuraBridge | None = getattr(app.state, "bridge", None)
        return {"ok": True, "waf": bridge.has_waf_token() if bridge else False}

    @app.get("/v1/waf/status")
    async def waf_status():
        bridge: ObscuraBridge | None = getattr(app.state, "bridge", None)
        if bridge is None:
            return {"warmed": False, "error": "bridge not initialised yet"}
        return await asyncio.to_thread(bridge.status)

    @app.get("/v1/models")
    async def list_models():
        now = int(time.time())
        return {
            "object": "list",
            "data": [
                {"id": m, "object": "model", "created": now, "owned_by": "deepseek"}
                for m in MODELS
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(body: dict, request: Request,
                               authorization: str = Header(default="")):
        settings: Settings = app.state.settings
        _check_auth(settings, authorization)
        prep = _prepare(body, settings)  # raises 400/404 fast, before touching accounts
        pool = _pool(app)
        # Async poll (no executor thread held): blocking acquire would
        # occupy default-executor threads and starve other to_thread work.
        # use_multiple_accounts=False pins to CURRENT (queue on its lock);
        # True (default) fails over to another healthy account when
        # CURRENT is busy (per-account lock) or cooling down (banned).
        allow_multi = bool(getattr(settings, "use_multiple_accounts", True))
        if prep["stream"]:
            try:
                item = await pool.aacquire(90, allow_failover=allow_multi)
            except ValueError:
                raise HTTPException(status_code=503, detail="no DeepSeek accounts in pool")
            except TimeoutError:
                raise HTTPException(status_code=429, detail="all DeepSeek accounts busy, retry later")
            return StreamingResponse(
                _stream_completion(app, pool, item, prep),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        # Failover helper owns item's lock (releases each attempt exactly
        # once): a banned CURRENT transparently retries on the next unbanned
        # account instead of surfacing 403 for a usable pool.
        # Non-stream disconnect: poll the client socket into cancel_event so
        # a timed-out/cancelled client does not hold the slot for the full
        # session→PoW→stream (next requests would queue to 90s/429). The
        # watcher starts BEFORE acquire: a client that gives up while queued
        # must not then run a full orphaned session once a slot frees.
        cancel_event = threading.Event()
        watcher = asyncio.create_task(_watch_disconnect(request, cancel_event))
        try:
            # Chunked acquire: a disconnect while queued surfaces as 499
            # within ~2s (instead of running an orphaned session once freed).
            item = await _acquire_slot_or_499(pool, request, cancel_event, 90, allow_multi)
            if cancel_event.is_set():
                AccountPool.release(item)
                raise HTTPException(status_code=499, detail="client disconnected")
            try:
                result = await asyncio.to_thread(
                    _complete_with_failover_sync, app, pool, prep, item, cancel_event)
            except _StreamCancelled:
                raise HTTPException(status_code=499, detail="client disconnected")
        finally:
            watcher.cancel()
        return _openai_response(prep, result)

    @app.get("/v1/accounts")
    async def list_accounts(authorization: str = Header(default="")):
        settings: Settings = app.state.settings
        _check_auth(settings, authorization)
        pool = _pool(app)
        data = await asyncio.to_thread(pool.status)
        return {"object": "list", "current": pool.current or None, "data": data}

    @app.post("/v1/accounts/select")
    async def select_account(body: dict | None = None,
                             authorization: str = Header(default="")):
        settings: Settings = app.state.settings
        _check_auth(settings, authorization)
        ident = ((body or {}).get("identifier") or "").strip() if isinstance(body, dict) else ""
        pool = _pool(app)
        if not ident or ident.lower() in ("none", "all", "auto"):
            settings.active_account = ""
            pool.set_current(None)
            await asyncio.to_thread(settings.save)
            return {"ok": True, "current": None}
        ok = await asyncio.to_thread(pool.set_current, ident)
        if not ok:
            raise HTTPException(status_code=404, detail=f"account not found: {ident}")
        settings.active_account = pool.current
        await asyncio.to_thread(settings.save)
        return {"ok": True, "current": pool.current}

    @app.post("/v1/accounts", status_code=201)
    async def add_account(body: dict, authorization: str = Header(default="")):
        from .accounts import TOKEN_HELP, extract_token
        settings: Settings = app.state.settings
        _check_auth(settings, authorization)
        cfg = AccountConfig(email=str(body.get("email") or ""),
                            mobile=str(body.get("mobile") or ""),
                            password=str(body.get("password") or ""),
                            token=extract_token(str(body.get("token") or "")))
        if not (cfg.email or cfg.mobile or cfg.token):
            raise HTTPException(status_code=400,
                                detail="email, mobile, or token required. " + TOKEN_HELP)
        pool = _pool(app)
        try:
            from .accounts import add_account as _add
            item = await asyncio.to_thread(_add, settings, pool, cfg)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        await asyncio.to_thread(settings.save)
        resp: dict = {"ok": True, "account": item.status()}
        if not cfg.token:
            resp["warning"] = ("no token: password-only fails "
                               "(RISK_DEVICE_DETECTED). " + TOKEN_HELP)
        return resp

    @app.put("/v1/accounts/{identifier}/token")
    async def set_account_token(identifier: str, body: dict,
                                authorization: str = Header(default="")):
        from .accounts import TOKEN_HELP, extract_token, set_account_token
        settings: Settings = app.state.settings
        _check_auth(settings, authorization)
        token = extract_token(str((body or {}).get("token") or ""))
        if not token:
            raise HTTPException(status_code=400,
                                detail="token required (raw value or full JSON). " + TOKEN_HELP)
        pool = _pool(app)
        item = await asyncio.to_thread(pool.get, identifier)
        if item is None:
            raise HTTPException(status_code=404, detail=f"account not found: {identifier}")
        item.cfg.token = token
        await asyncio.to_thread(set_account_token, settings, identifier, token)
        await asyncio.to_thread(settings.save)
        await asyncio.to_thread(pool.clear_cooldown, identifier)
        return {"ok": True, "account": item.status()}

    @app.delete("/v1/accounts/{identifier}")
    async def delete_account(identifier: str, authorization: str = Header(default="")):
        from .accounts import remove_account
        settings: Settings = app.state.settings
        _check_auth(settings, authorization)
        pool = _pool(app)
        removed = await asyncio.to_thread(pool.remove, identifier)
        if not removed:
            raise HTTPException(status_code=404, detail=f"account not found: {identifier}")
        await asyncio.to_thread(remove_account, settings, identifier)
        await asyncio.to_thread(settings.save)
        return {"ok": True, "removed": identifier, "current": pool.current or None}

    @app.post("/v1/accounts/unblock")
    async def unblock_accounts(body: dict | None = None,
                              authorization: str = Header(default="")):
        settings: Settings = app.state.settings
        _check_auth(settings, authorization)
        identifier = (body or {}).get("identifier") if isinstance(body, dict) else None
        pool = _pool(app)
        cleared = await asyncio.to_thread(pool.clear_cooldown, identifier)
        return {"ok": True, "cleared": cleared}

    return app


def _check_auth(settings: Settings, authorization: str) -> None:
    if not settings.keys:
        return
    token = (authorization or "").removeprefix("Bearer ").strip()
    if token not in settings.keys:
        raise HTTPException(status_code=401, detail="invalid API key")


def _pool(app: FastAPI) -> AccountPool:
    from .accounts import sync_current_from_settings
    if app.state.pool is None:
        app.state.pool = AccountPool(app.state.settings.accounts,
                                     current=getattr(app.state.settings,
                                                     "active_account", ""))
    else:
        # Settings may change via API (add/remove/select): keep pool in sync.
        # Pool holds refs to same AccountConfig objects when created via API,
        # but CLI/TUI edits replace settings.accounts list: rebind not needed
        # for in-memory pool except CURRENT selection.
        try:
            if sync_current_from_settings(app.state.pool, app.state.settings):
                try:
                    app.state.settings.save()
                except Exception:
                    pass
        except Exception:
            pass
    return app.state.pool


def _prepare(body: dict, settings: Settings | None = None) -> dict:
    model = str(body.get("model") or "deepseek-flash")
    messages = body.get("messages", [])
    if not messages:
        raise HTTPException(status_code=400, detail="messages is required")
    if not isinstance(messages, list):
        raise HTTPException(status_code=400,
                            detail="messages must be a list of message objects")
    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise HTTPException(
                status_code=400,
                detail=f"messages[{idx}] must be an object with a 'role' field")
    spec = MODELS.get(model.lower())
    if spec is None:
        raise HTTPException(status_code=404, detail=f"model '{model}' not available")
    thinking, search, model_type = spec
    tools = body.get("tools") or []
    use_tools = bool(tools) and body.get("tool_choice", "auto") != "none"
    if settings is not None and not getattr(settings, "enable_tools", True):
        use_tools = False
    if use_tools:
        messages = [{"role": "system", "content": tool_system_prompt(tools)}, *messages]
    prompt = render_prompt(messages)
    if use_tools:
        # Recency reminder: long histories bury the start instruction.
        # Appended after user content; non-tool path unchanged.
        prompt = prompt + "\n\n" + tool_reminder_prompt(tools)
    return {
        "model": model, "stream": bool(body.get("stream", False)),
        "tools": tools if use_tools else [], "prompt": prompt,
        "thinking": thinking, "search": search, "model_type": model_type,
    }


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
    from .accounts import TOKEN_HELP
    if item.cfg.token:
        return item.cfg.token
    if not (item.cfg.email or item.cfg.mobile) or not item.cfg.password:
        raise HTTPException(
            status_code=401,
            detail=f"account {item.cfg.identifier} has no token. " + TOKEN_HELP)
    try:
        token = login(item.cfg, _headers(app, ""))
    except Exception as exc:
        raise HTTPException(
            status_code=401,
            detail=f"auto login failed for {item.cfg.identifier} ({exc}). "
                   "Direct password login hits RISK_DEVICE_DETECTED. " + TOKEN_HELP)
    app.state.settings.save()
    return token


def _is_invalid_token_error(error: str | None) -> bool:
    """True when an attempt error is the expired/invalid token (biz 40003)."""
    low = (error or "").lower()
    return "40003" in low or (
        "authorization failed" in low and "token" in low)


def _refresh_token_via_obscura(app: FastAPI, item: PooledAccount) -> str | None:
    """Re-login through the Obscura browser and persist a fresh userToken.

    Singleflight per account: concurrent 40003s share one browser run. Returns
    the new token or None when refresh is impossible/failed. Never raises
    (except _StreamCancelled when the client vanished while waiting).
    """
    cfg = item.cfg
    if not (cfg.email or cfg.mobile) or not cfg.password:
        return None
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
        try:
            inflight.wait(timeout=180)
        except Exception:
            pass
        fresh = (item.cfg.token or "").strip()
        return fresh or None
    try:
        from .cli import _obscura_login_token
        try:
            new_token = _obscura_login_token(cfg.email or cfg.mobile,
                                             cfg.password, settings)
        except Exception as exc:
            logger.warning("token refresh via Obscura failed for %s: %s", key, exc)
            new_token = None
        if new_token:
            from .accounts import set_account_token
            item.cfg.token = new_token
            try:
                set_account_token(settings, key, new_token)
                settings.save()
            except Exception as exc:
                logger.warning("token refresh save failed for %s: %s", key, exc)
            logger.info("token refreshed via Obscura for %s", key)
            return new_token
        return None
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


def _await_cancelable(fut, cancel_event: threading.Event | None):
    """Wait for a challenge-executor future, aborting promptly on disconnect.

    Polls with a short timeout so a client disconnect (cancel_event set by
    the Request watcher) raises _StreamCancelled instead of holding the
    account slot through the full session→PoW setup.
    """
    import concurrent.futures as _fut

    while True:
        if cancel_event is not None and cancel_event.is_set():
            try:
                fut.cancel()
            except Exception:
                pass
            raise _StreamCancelled()
        try:
            return fut.result(timeout=0.05)
        except _fut.TimeoutError:
            continue


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


def _refresh_waf_without_account_lock(app: FastAPI, item: PooledAccount) -> None:
    """Global WAF refresh without wasting the per-account slot.

    The caller holds item.lock (one in-flight per account). A WAF refresh is
    global (shared profile/cookies) and singleflight-deduped in the bridge,
    so release the account slot while waiting for it, then reacquire before
    the retry attempt. Safe when the lock isn't held (unit tests calling
    _run_completion_core with a fresh account): release fails -> just warm.
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
            try:
                item.lock.acquire()
            except Exception:
                pass


def _run_completion(app: FastAPI, item: PooledAccount, prep: dict,
                    cancel_event: threading.Event | None = None) -> dict:
    """Blocking full completion with retries per failure class."""
    return _run_completion_core(app, item, prep, cancel_event=cancel_event)


def _is_ban_http_exception(exc: BaseException) -> bool:
    """True when exc is the ban 403 raised by _run_completion_core."""
    return (isinstance(exc, HTTPException) and exc.status_code == 403
            and P.is_ban_error_text(str(getattr(exc, "detail", "") or "")))


def _switch_current_after_ban_failover(pool: AccountPool, settings: Settings,
                                       used: PooledAccount,
                                       banned: PooledAccount) -> None:
    """Move CURRENT to the working account after a ban failover.

    Only fires when CURRENT pointed at the banned account: the request
    already proved it unusable (ban marked it cooling until mute_until),
    so future requests should prefer the healthy one first. Best-effort,
    never raises.
    """
    try:
        from .accounts import _matches as _m
        cur = pool.current
        if not cur or not _m(banned.cfg, cur):
            return  # auto-failover mode or CURRENT wasn't the banned one
        if _m(used.cfg, cur):
            return  # same account (shouldn't happen after mark_bad)
        if pool.set_current(used.cfg.identifier):
            try:
                settings.active_account = pool.current
                settings.save()
            except Exception:
                pass
            logger.warning("account %s banned, CURRENT auto-switched -> %s",
                           banned.cfg.identifier, used.cfg.identifier)
    except Exception:
        pass


def _complete_with_failover_sync(app: FastAPI, pool: AccountPool, prep: dict,
                                 first_item: PooledAccount,
                                 cancel_event: threading.Event | None = None,
                                 on_content=None, on_thinking=None,
                                 allow_failover: bool | None = None) -> dict:
    """Run one completion, auto-failing over when the account is banned.

    Owns first_item's lock: releases each attempt exactly once, acquires the
    next account on ban (skipped via its fresh cooldown). Returns the result
    of the first unbanned account. Raises the ban 403 when every account is
    banned, or the original error when it isn't ban-related.
    allow_failover=False (settings.use_multiple_accounts=False) disables the
    ban hop: the ban 403 surfaces directly, never touching other accounts.
    None (default) reads the flag from app settings.
    """
    if allow_failover is None:
        try:
            allow_failover = bool(getattr(_app_settings(app), "use_multiple_accounts", True))
        except Exception:
            allow_failover = True
    cur = first_item
    ban_exc: HTTPException | None = None
    ban_item: PooledAccount | None = None
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
            if (not _is_ban_http_exception(exc)
                    or (cancel_event is not None and cancel_event.is_set())
                    or tries + 1 >= max_tries):
                raise
            if ban_exc is None:
                ban_exc, ban_item = exc, cur
            tries += 1
            logger.info("account %s banned, failing over to next account (%d/%d)",
                        cur.cfg.identifier, tries + 1, max_tries)
            try:
                cur = pool.acquire(timeout=FAILOVER_ACQUIRE_TIMEOUT)
            except Exception:
                raise ban_exc
            continue
        except Exception:
            AccountPool.release(cur)
            raise
        if ban_item is not None:
            _switch_current_after_ban_failover(pool, _app_settings(app), cur, ban_item)
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
    live stream mode. Retried attempts still invoke the callbacks as data
    arrives; retryable failures (PoW/session/WAF) normally carry no content
    prefix, so the risk of interleaving partial text across retries is minimal.
    Buffered callers pass no callbacks and get identical behaviour to before.

    cancel_event, when set, aborts the attempt at the next SSE event so a
    disconnected client does not keep the account lock for the whole reply.
    """
    settings = _app_settings(app)
    max_retries = max(0, int(getattr(settings, "max_retries", 1)))
    parallel = bool(getattr(settings, "parallel_challenge_fetch", True))
    auto_delete = bool(getattr(settings, "auto_delete_session", True))
    token = _ensure_token(app, item)
    session_id: str | None = None
    token_refreshed = False

    def _cancelled() -> bool:
        return cancel_event is not None and cancel_event.is_set()

    def _refresh_token_once() -> bool:
        """Refresh the account token via Obscura, once per request.

        Returns True when a new token replaced the stale one. Shared by the
        session-create path (40003 raised) and the stream path (40003 event).
        """
        nonlocal token, token_refreshed
        if token_refreshed:
            return False
        logger.info("token invalid for %s, refreshing via Obscura browser",
                    item.cfg.identifier)
        new_token = _refresh_token_via_obscura(app, item)
        if _cancelled():
            raise _StreamCancelled()
        if not new_token:
            return False
        token = new_token
        token_refreshed = True
        return True

    def _open_session() -> tuple[dict, str, dict]:
        """Create session + PoW; auto-refresh the token once on 40003.

        The 40003 surfaces as a RuntimeError from create_session. On the first
        hit the token is refreshed via Obscura and session creation retried;
        a second failure propagates to the caller untouched.
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
                if attempt == 0 and _is_invalid_token_error(str(exc)):
                    if _refresh_token_once():
                        continue
                if _is_invalid_token_error(str(exc)):
                    raise HTTPException(
                        status_code=401,
                        detail=f"Authorization Failed: token invalid for "
                               f"{item.cfg.identifier} and could not be "
                               f"refreshed ({str(exc)[:200]}). Re-login the account.")
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

        def _do_attempt(h: dict, p: dict) -> None:
            nonlocal content, think, error, usage_total, stream_ban_until
            outcome = _attempt(
                h, p, on_content=on_content, on_thinking=on_thinking,
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
            session_id = DS.create_session(_headers(app, token))
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
            if not _refresh_token_once():
                break
            logger.info("retrying completion with refreshed token")
            session_id = DS.create_session(_headers(app, token))
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
            _refresh_waf_without_account_lock(app, item)
            if _cancelled():
                break
            headers = _solve(app, token)
            _do_attempt(headers, payload)
        if _cancelled():
            raise _StreamCancelled()
        if error:
            low = error.lower()
            # Invalid/expired token: already refreshed once above; reaching
            # here means the fresh token also failed, so surface it clearly.
            if _is_invalid_token_error(error):
                raise HTTPException(
                    status_code=401,
                    detail=f"Authorization Failed: token invalid for "
                           f"{item.cfg.identifier} even after browser refresh "
                           f"({error[:200]}). Re-login the account.")
            # Ban/auth family is account-specific: cool that account down.
            # Rate-limit family is usually a global throttle, so do NOT poison
            # a specific account (that would rotate through and shrink the pool).
            is_ban = any(k in low for k in ("banned", "restricted", "401", "unauthorized",
                                            "muted", "mute", "suspend", "violation"))
            if is_ban:
                # Banned accounts stay unusable until mute_until: cool down for
                # the remaining suspension. The completion ban envelope already
                # carried mute_until (stream_ban_until) — use it directly and
                # skip the extra users/current HTTP. Fall back to check_ban
                # (short 5s timeout) only when the stream had no timestamp.
                cooldown = None
                ban_until: float | None = stream_ban_until
                if ban_until is None:
                    try:
                        _, ban_until = DS.check_ban(_headers(app, token), timeout=5)
                    except Exception:
                        ban_until = None
                if ban_until:
                    try:
                        cooldown = max(0.0, float(ban_until) - time.time())
                    except Exception:
                        cooldown = None
                if cooldown:
                    AccountPool.mark_bad(item, seconds=cooldown)
                else:
                    AccountPool.mark_bad(item)
                # New error message: tell exactly when the ban ends.
                try:
                    base = P.ban_message(ban_until) if ban_until else ""
                    if not base:
                        detail = error[:300]
                    else:
                        # Prefix account so multi-account setups know which one.
                        detail = f"Account {item.cfg.identifier} banned. {base}"
                except Exception:
                    detail = error[:300]
                raise HTTPException(status_code=403, detail=detail)
            if any(k in low for k in ("rate_limit", "too frequent", "too_frequent", "429")):
                raise HTTPException(status_code=429,
                                    detail=f"DeepSeek rate limited: {error[:200]}. "
                                           "Wait 2-20 min, space requests, avoid parallel loops.")
            raise RuntimeError(error[:300])
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
    for event in DS.stream_completion(headers, payload):
        if cancel_event is not None and cancel_event.is_set():
            # Client gone: stop reading so the generator's finally closes the
            # HTTP response and the account lock is released promptly.
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
    return _AttemptOutcome("".join(content), "".join(think), error,
                           usage_total, ban_until)


async def _stream_completion(app: FastAPI, pool: AccountPool, item: PooledAccount, prep: dict):
    """Async generator: buffered (default) or live deltas, tools parsed at end."""
    cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    model = prep["model"]
    out: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    settings = _app_settings(app)
    live = str(getattr(settings, "stream_mode", "live")).lower() == "live"
    # Consumer wait must outlast the blocking curl stream timeout, otherwise the
    # async reader times out while the producer is still working on a valid long
    # reply. Derive it so the two can never drift apart.
    consumer_timeout = float(DS.DEFAULT_TIMEOUT) + _CONSUMER_GRACE_SECONDS
    # Set when the client disconnects (generator GC/close): the producer checks
    # it between SSE events and stops, releasing the account lock promptly.
    cancel_event = threading.Event()

    def _emit(item: tuple) -> None:
        # call_soon_threadsafe raises RuntimeError once the loop is closed
        # (shutdown). Swallow: producer work is best-effort at that point.
        try:
            loop.call_soon_threadsafe(out.put_nowait, item)
        except RuntimeError:
            pass

    def _put(kind: str, text: str) -> None:
        if text:
            _emit((kind, text))

    def produce_buffered():
        # Ban failover lives in the helper (owns item's lock): a banned
        # account retries on the next unbanned one inside the same request.
        # Ban errors carry no content prefix, so nothing partial leaks.
        try:
            result = _complete_with_failover_sync(
                app, pool, prep, item, cancel_event=cancel_event)
            _emit(("done", result))
        except _StreamCancelled:
            pass
        except HTTPException as exc:
            _emit(("http_error", exc))
        except Exception as exc:  # noqa: BLE001
            _emit(("error", str(exc)))

    def produce_live():
        # Thinking always streams live (never contains tool JSON).
        # Content streams live only when no tools requested; with tools the
        # content is buffered so a tool_calls reply is not also emitted as
        # user-visible text.
        try:
            result = _complete_with_failover_sync(
                app, pool, prep, item,
                on_content=(None if prep.get("tools") else lambda t: _put("content", t)),
                on_thinking=lambda t: _put("thinking", t),
                cancel_event=cancel_event,
            )
            _emit(("done", result))
        except _StreamCancelled:
            pass
        except HTTPException as exc:
            _emit(("http_error", exc))
        except Exception as exc:  # noqa: BLE001
            _emit(("error", str(exc)))

    def chunk(delta: dict, finish: str | None = None) -> bytes:
        return ("data: " + json.dumps(
            {"id": cid, "object": "chat.completion.chunk", "created": created,
             "model": model, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]},
            ensure_ascii=False) + "\n\n").encode("utf-8")

    async def _parse_stream_calls(content: str):
        """Tool-parse buffered content (tools requested); log when none found."""
        calls, remaining = await asyncio.to_thread(parse_tool_calls, content, prep["tools"])
        if not calls:
            logger.debug("stream no tool call parsed model=%s preview=%.200s",
                         prep.get("model"), (content or "")[:200])
        return calls, remaining

    def _tool_finish_chunks(calls):
        """tool_call deltas + finish chunk; plain stop when no calls."""
        if calls:
            for call in calls:
                yield chunk({"tool_calls": [{
                    "id": call["id"], "type": "function",
                    "function": {"name": call["function"]["name"],
                                 "arguments": call["function"]["arguments"]}}]})
            yield chunk({}, "tool_calls")
        else:
            yield chunk({}, "stop")

    # Bounded producer: one account-holding thread per stream, capped by the
    # shared executor so a parallel-subagent burst queues instead of piling
    # unbounded raw threads (each with account + PoW CPU).
    try:
        _STREAM_EXECUTOR.submit(produce_live if live else produce_buffered)
    except RuntimeError:
        # Executor shut down (interpreter exit): fall back to a raw thread.
        threading.Thread(target=produce_live if live else produce_buffered, daemon=True).start()
    try:
        yield chunk({"role": "assistant"})
        if not live:
            # Buffered: producer sends one reply per request; emit thinking+content
            # once the reply lands, then tool deltas. First-byte ~= model latency.
            while True:
                try:
                    kind, payload = await asyncio.wait_for(out.get(), timeout=consumer_timeout)
                except asyncio.TimeoutError:
                    yield ("data: " + json.dumps({"error": {"message": "upstream timed out"}},
                                                ensure_ascii=False) + "\n\n").encode()
                    yield b"data: [DONE]\n\n"
                    return
                if kind == "done":
                    thinking, content = payload["thinking"], payload["content"]
                    if prep["tools"]:
                        calls, remaining = await _parse_stream_calls(content)
                    else:
                        calls, remaining = None, content
                    if thinking:
                        yield chunk({"reasoning_content": thinking})
                    if remaining:
                        yield chunk({"content": remaining})
                    for piece in _tool_finish_chunks(calls):
                        yield piece
                    yield b"data: [DONE]\n\n"
                    return
                if kind == "http_error":
                    err = payload
                    yield ("data: " + json.dumps({"error": {"message": err.detail, "code": err.status_code}},
                                                ensure_ascii=False) + "\n\n").encode()
                    yield b"data: [DONE]\n\n"
                    return
                yield ("data: " + json.dumps({"error": {"message": str(payload)}},
                                            ensure_ascii=False) + "\n\n").encode()
                yield b"data: [DONE]\n\n"
                return
        # Live: forward thinking/content deltas as they arrive from DeepSeek.
        while True:
            try:
                kind, payload = await asyncio.wait_for(out.get(), timeout=consumer_timeout)
            except asyncio.TimeoutError:
                yield ("data: " + json.dumps({"error": {"message": "upstream timed out"}},
                                            ensure_ascii=False) + "\n\n").encode()
                yield b"data: [DONE]\n\n"
                return
            if kind == "thinking":
                yield chunk({"reasoning_content": payload})
                continue
            if kind == "content":
                yield chunk({"content": payload})
                continue
            if kind == "done":
                content = payload["content"]
                if prep.get("tools"):
                    calls, remaining = await _parse_stream_calls(content)
                    # Thinking already streamed live; emit only the remainder.
                    if remaining:
                        yield chunk({"content": remaining})
                    for piece in _tool_finish_chunks(calls):
                        yield piece
                else:
                    # Content already streamed live; just close.
                    yield chunk({}, "stop")
                yield b"data: [DONE]\n\n"
                return
            if kind == "http_error":
                err = payload
                yield ("data: " + json.dumps({"error": {"message": err.detail, "code": err.status_code}},
                                            ensure_ascii=False) + "\n\n").encode()
                yield b"data: [DONE]\n\n"
                return
            yield ("data: " + json.dumps({"error": {"message": str(payload)}},
                                        ensure_ascii=False) + "\n\n").encode()
            yield b"data: [DONE]\n\n"
            return
    finally:
        # Generator closed (client disconnect, cancellation, completion): tell
        # the producer to stop at the next event so the account lock frees.
        cancel_event.set()


def _openai_response(prep: dict, result: dict) -> dict:
    cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    calls, remaining = (parse_tool_calls(result["content"], prep["tools"])
                        if prep["tools"] else (None, result["content"]))
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
    pt = max(1, len(prep["prompt"]) // 4)
    ct = result.get("usage_total") or max(1, (len(result["content"]) + len(result["thinking"])) // 4)
    return {"id": cid, "object": "chat.completion", "created": created, "model": prep["model"],
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct}}


def error_json(request, exc: HTTPException):  # helper for tests
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})


app = create_app()
