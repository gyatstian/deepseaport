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

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from . import client as DS
from . import pow as PoW
from . import protocol as P
from .accounts import AccountPool, PooledAccount, login
from .config import AccountConfig, Settings, load_settings
from .obscura_bridge import ObscuraBridge
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

# Consumer wait (seconds) for a producer event, on top of the curl stream
# timeout. Keeps the async reader from timing out before the blocking call.
_CONSUMER_GRACE_SECONDS = 10


class _StreamCancelled(Exception):
    """Internal: client disconnected, producer should stop at the next event."""


def apply_log_level(level: str) -> None:
    """Apply log level to deepseaport loggers (settings-driven, no handler reset)."""
    try:
        numeric = getattr(logging, str(level or "INFO").upper(), logging.INFO)
        for name in ("deepseaport.server", "deepseaport.client", "deepseaport.tools", "deepseaport.pow"):
            logging.getLogger(name).setLevel(numeric)
    except Exception:
        pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: Settings = app.state.settings
    apply_log_level(getattr(settings, "log_level", "INFO"))
    bridge = ObscuraBridge(settings.obscura_bin, settings.obscura_profile)
    app.state.bridge = bridge
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
    async def chat_completions(body: dict, authorization: str = Header(default="")):
        settings: Settings = app.state.settings
        _check_auth(settings, authorization)
        prep = _prepare(body, settings)  # raises 400/404 fast, before touching accounts
        pool = _pool(app)
        try:
            item = await asyncio.to_thread(pool.acquire, 90)
        except ValueError:
            raise HTTPException(status_code=503, detail="no DeepSeek accounts in pool")
        except TimeoutError:
            raise HTTPException(status_code=429, detail="all DeepSeek accounts busy, retry later")
        if prep["stream"]:
            return StreamingResponse(
                _stream_completion(app, pool, item, prep),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        try:
            result = await asyncio.to_thread(_run_completion, app, item, prep)
        finally:
            AccountPool.release(item)
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
            item = await asyncio.to_thread(pool.add, cfg)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        settings.accounts.append(cfg)
        if len(settings.accounts) == 1 and not getattr(settings, "active_account", ""):
            settings.active_account = cfg.identifier
            try:
                pool.set_current(cfg.identifier)
            except Exception:
                pass
        await asyncio.to_thread(settings.save)
        resp: dict = {"ok": True, "account": item.status()}
        if not cfg.token:
            resp["warning"] = ("no token: password-only fails "
                               "(RISK_DEVICE_DETECTED). " + TOKEN_HELP)
        return resp

    @app.put("/v1/accounts/{identifier}/token")
    async def set_account_token(identifier: str, body: dict,
                                authorization: str = Header(default="")):
        from .accounts import TOKEN_HELP, _matches as _m, extract_token
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
        for a in settings.accounts:
            if _m(a, identifier):
                a.token = token
                break
        await asyncio.to_thread(settings.save)
        await asyncio.to_thread(pool.clear_cooldown, identifier)
        return {"ok": True, "account": item.status()}

    @app.delete("/v1/accounts/{identifier}")
    async def delete_account(identifier: str, authorization: str = Header(default="")):
        from .accounts import _matches as _m
        settings: Settings = app.state.settings
        _check_auth(settings, authorization)
        pool = _pool(app)
        removed = await asyncio.to_thread(pool.remove, identifier)
        if not removed:
            raise HTTPException(status_code=404, detail=f"account not found: {identifier}")
        kept, dropped = [], False
        for a in settings.accounts:
            if not dropped and _m(a, identifier):
                dropped = True
                continue
            kept.append(a)
        settings.accounts = kept
        if settings.active_account and not any(
                _m(a, settings.active_account) for a in settings.accounts):
            settings.active_account = ""
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
    bridge: ObscuraBridge = app.state.bridge
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
        token = login(item.cfg, P.base_headers(
            user_agent=app.state.bridge.state.user_agent,
            waf_cookies=app.state.bridge.cookie_header()))
    except Exception as exc:
        raise HTTPException(
            status_code=401,
            detail=f"auto login failed for {item.cfg.identifier} ({exc}). "
                   "Direct password login hits RISK_DEVICE_DETECTED. " + TOKEN_HELP)
    app.state.settings.save()
    return token


def _solve(app: FastAPI, token: str) -> dict[str, str]:
    """Fresh PoW header merged over base headers."""
    headers = _headers(app, token)
    return {**headers, "X-DS-PoW-Response": DS.solve_pow(headers)}


def _session_and_challenge(headers: dict, parallel: bool = True) -> tuple[str, dict]:
    """Create session + fetch PoW challenge concurrently (same headers).

    Sequential code paid 2 RTTs; these calls are independent. Session errors
    take priority to match old sequential ordering. When parallel=False the
    two calls run sequentially (session first, same error priority).
    """
    if not parallel:
        session_id = DS.create_session(headers)
        try:
            challenge = DS.fetch_pow(headers)
        except Exception:
            # Session created but challenge failed: don't leak it.
            try:
                DS.delete_session(headers, session_id)
            except Exception:
                pass
            raise
        return session_id, challenge
    fut_session = _CHALLENGE_EXECUTOR.submit(DS.create_session, headers)
    fut_challenge = _CHALLENGE_EXECUTOR.submit(DS.fetch_pow, headers)
    try:
        session_id = fut_session.result()
    except Exception:
        try:
            fut_challenge.result()
        except Exception:
            pass
        raise
    try:
        challenge = fut_challenge.result()
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


def _run_completion(app: FastAPI, item: PooledAccount, prep: dict,
                    cancel_event: threading.Event | None = None) -> dict:
    """Blocking full completion with retries per failure class."""
    return _run_completion_core(app, item, prep, cancel_event=cancel_event)


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

    def _cancelled() -> bool:
        return cancel_event is not None and cancel_event.is_set()

    # Fast path: skip cookie-header builds when wasm already cached.
    try:
        wp = PoW.wasm_path()
        if not (wp.exists() and wp.stat().st_size > 1000):
            PoW.ensure_wasm(cookies=app.state.bridge.cookie_header(),
                            user_agent=app.state.bridge.state.user_agent)
    except Exception as exc:
        logger.warning("wasm ensure failed: %s", exc)
    # Session creation + PoW solve sit inside the try so a solver failure
    # (answer None -> RuntimeError) still runs the finally cleanup below.
    try:
        headers = _headers(app, token)
        session_id, challenge = _session_and_challenge(headers, parallel=parallel)
        headers = {**headers, "X-DS-PoW-Response": _header_from_challenge(challenge)}
        payload = P.completion_payload(session_id, prep["prompt"], prep["thinking"],
                                       prep["search"], prep["model_type"])
        content, think, error, usage_total = _attempt(
            headers, payload, on_content=on_content, on_thinking=on_thinking,
            cancel_event=cancel_event)
        for _ in range(max_retries):
            if _cancelled() or not (error and "pow" in error.lower()):
                break
            logger.info("pow rejected, solving fresh (retries left %s)", max_retries)
            headers = _solve(app, token)
            content, think, error, usage_total = _attempt(
                headers, payload, on_content=on_content, on_thinking=on_thinking,
                cancel_event=cancel_event)
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
            content, think, error, usage_total = _attempt(
                headers, payload, on_content=on_content, on_thinking=on_thinking,
                cancel_event=cancel_event)
        for _ in range(max_retries):
            if _cancelled() or not (error and ("403" in error or "WAF" in error)):
                break
            logger.info("possible WAF block, refreshing cookies (retries left %s)", max_retries)
            app.state.bridge.warmup()
            headers = _solve(app, token)
            content, think, error, usage_total = _attempt(
                headers, payload, on_content=on_content, on_thinking=on_thinking,
                cancel_event=cancel_event)
        if _cancelled():
            raise _StreamCancelled()
        if error:
            low = error.lower()
            # Ban/auth family is account-specific: cool that account down.
            # Rate-limit family is usually a global throttle, so do NOT poison
            # a specific account (that would rotate through and shrink the pool).
            if any(k in low for k in ("banned", "restricted", "401", "unauthorized")):
                AccountPool.mark_bad(item)
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
            if auto_delete:
                _delete_session_bg(_headers(app, token), session_id, enabled=True)
        except Exception:
            pass


def _attempt(headers: dict, payload: dict, on_content=None, on_thinking=None,
             cancel_event: threading.Event | None = None,
             ) -> tuple[str, str, str | None, int]:
    content: list[str] = []
    think: list[str] = []
    error: str | None = None
    usage_total = 0
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
            break
        elif event.kind == "finished":
            break
    return "".join(content), "".join(think), error, usage_total


async def _stream_completion(app: FastAPI, pool: AccountPool, item: PooledAccount, prep: dict):
    """Async generator: buffered (default) or live deltas, tools parsed at end."""
    cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    model = prep["model"]
    out: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    settings = _app_settings(app)
    live = str(getattr(settings, "stream_mode", "buffered")).lower() == "live"
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
        try:
            result = _run_completion(
                app, item, prep, cancel_event=cancel_event)
            _emit(("done", result))
        except _StreamCancelled:
            pass
        except HTTPException as exc:
            _emit(("http_error", exc))
        except Exception as exc:  # noqa: BLE001
            _emit(("error", str(exc)))
        finally:
            AccountPool.release(item)

    def produce_live():
        # Thinking always streams live (never contains tool JSON).
        # Content streams live only when no tools requested; with tools the
        # content is buffered so a tool_calls reply is not also emitted as
        # user-visible text.
        try:
            result = _run_completion_core(
                app, item, prep,
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
        finally:
            AccountPool.release(item)

    def chunk(delta: dict, finish: str | None = None) -> bytes:
        return ("data: " + json.dumps(
            {"id": cid, "object": "chat.completion.chunk", "created": created,
             "model": model, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]},
            ensure_ascii=False) + "\n\n").encode("utf-8")

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
                    calls, remaining = (await asyncio.to_thread(parse_tool_calls, content, prep["tools"])
                                        if prep["tools"] else (None, content))
                    if prep["tools"] and not calls:
                        logger.debug("stream no tool call parsed model=%s preview=%.200s",
                                     prep.get("model"), (content or "")[:200])
                    if thinking:
                        yield chunk({"reasoning_content": thinking})
                    if remaining:
                        yield chunk({"content": remaining})
                    if calls:
                        for call in calls:
                            yield chunk({"tool_calls": [{
                                "id": call["id"], "type": "function",
                                "function": {"name": call["function"]["name"],
                                             "arguments": call["function"]["arguments"]}}]})
                        yield chunk({}, "tool_calls")
                    else:
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
                    calls, remaining = await asyncio.to_thread(parse_tool_calls, content, prep["tools"])
                    if not calls:
                        logger.debug("stream no tool call parsed model=%s preview=%.200s",
                                     prep.get("model"), (content or "")[:200])
                    # Thinking already streamed live; emit only the remainder.
                    if remaining:
                        yield chunk({"content": remaining})
                    if calls:
                        for call in calls:
                            yield chunk({"tool_calls": [{
                                "id": call["id"], "type": "function",
                                "function": {"name": call["function"]["name"],
                                             "arguments": call["function"]["arguments"]}}]})
                        yield chunk({}, "tool_calls")
                    else:
                        yield chunk({}, "stop")
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
