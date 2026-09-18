"""FastAPI route wiring for deepseaport.server.

Pure move of the handlers that were previously defined inside
``deepseaport.server.create_app``, plus ``_check_auth``/``_pool``/``_prepare``.
``create_app`` stays in deepseaport.server as thin wiring that calls
``register_routes(app)``.

Handlers look up the completion/stream entry points through a deferred
``_server()`` reference so tests that monkeypatch them on
``deepseaport.server`` keep working.
"""

from __future__ import annotations

import asyncio
import threading
import time

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse

from .accounts import AccountPool
from .completion_service import clear_login_backoff
from .config import AccountConfig, Settings
from .obscura_bridge import ObscuraBridge
from .server_concurrency import (
    _StreamCancelled,
    _acquire_slot_or_499,
    _watch_disconnect,
)
from .tools_support import render_prompt, tool_reminder_prompt, tool_system_prompt


def _server():
    """Late-bound deepseaport.server module for monkeypatch-compatible lookups."""
    from . import server as _s
    return _s


def register_routes(app: FastAPI) -> None:
    @app.get("/health")
    async def health():
        # Liveness stays 200/ok:true so existing pollers (_wait_for_server,
        # simple uptime checks) keep working. Readiness details ride along so
        # an orchestrator can avoid routing to a live-but-dead instance
        # (empty pool / no tokens) without us 503ing liveness (which would
        # crash-loop a process that just needs config, not a restart).
        # No pool creation / settings.save here: health polls every few
        # seconds and must stay side-effect free.
        bridge: ObscuraBridge | None = getattr(app.state, "bridge", None)
        try:
            settings = getattr(app.state, "settings", None)
            accs = list(getattr(settings, "accounts", []) or [])
        except Exception:
            accs = []
        try:
            total = len(accs)
            with_token = sum(1 for a in accs if (getattr(a, "token", "") or "").strip())
            banned = sum(1 for a in accs if getattr(a, "banned", False))
        except Exception:
            total = with_token = banned = 0
        try:
            pool = getattr(app.state, "pool", None)
            pool_size = len(pool) if pool is not None else total
            current = getattr(pool, "current", None)
            if current is None:
                try:
                    current = (getattr(settings, "active_account", "") or "").strip() or None
                except Exception:
                    current = None
        except Exception:
            pool_size = total
            current = None
        waf = bool(bridge.has_waf_token()) if bridge else False
        ready = bool(total > 0 and with_token > 0 and with_token > banned)
        return {"ok": True, "ready": ready, "waf": waf,
                "accounts": total, "accounts_with_token": with_token,
                "banned": banned, "pool": pool_size, "current": current}

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
                for m in _server().MODELS
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(body: dict, request: Request,
                               authorization: str = Header(default="")):
        settings: Settings = app.state.settings
        _check_auth(settings, authorization)
        prep = _prepare(body, settings)  # raises 400/404 fast, before touching accounts
        pool = _server()._pool(app)
        # Async poll (no executor thread held): blocking acquire would
        # occupy default-executor threads and starve other to_thread work.
        # use_multiple_accounts=False pins to CURRENT (queue on its lock);
        # True (default) fails over to another healthy account when
        # CURRENT is busy (per-account lock) or cooling down (banned).
        allow_multi = bool(getattr(settings, "use_multiple_accounts", True))
        if prep["stream"]:
            # Same queued-acquire cancel as non-stream: a client that gives up
            # while queued must surface 499 without starting orphaned work
            # once a slot frees (previously parked full 90s here).
            cancel_event = threading.Event()
            watcher = asyncio.create_task(_watch_disconnect(request, cancel_event))
            try:
                item = await _acquire_slot_or_499(pool, request, cancel_event, 90, allow_multi)
                if cancel_event.is_set():
                    AccountPool.release(item)
                    raise HTTPException(status_code=499, detail="client disconnected")
            finally:
                watcher.cancel()
            return StreamingResponse(
                _server()._stream_completion(app, pool, item, prep),
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
                    _server()._complete_with_failover_sync, app, pool, prep, item, cancel_event)
            except _StreamCancelled:
                raise HTTPException(status_code=499, detail="client disconnected")
        finally:
            watcher.cancel()
        return _server()._openai_response(prep, result)

    @app.get("/v1/accounts")
    async def list_accounts(authorization: str = Header(default="")):
        settings: Settings = app.state.settings
        _check_auth(settings, authorization)
        pool = _server()._pool(app)
        data = await asyncio.to_thread(pool.status)
        return {"object": "list", "current": pool.current or None, "data": data}

    @app.post("/v1/accounts/select")
    async def select_account(body: dict | None = None,
                             authorization: str = Header(default="")):
        settings: Settings = app.state.settings
        _check_auth(settings, authorization)
        ident = ((body or {}).get("identifier") or "").strip() if isinstance(body, dict) else ""
        pool = _server()._pool(app)
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
        pool = _server()._pool(app)
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
        pool = _server()._pool(app)
        item = await asyncio.to_thread(pool.get, identifier)
        if item is None:
            raise HTTPException(status_code=404, detail=f"account not found: {identifier}")
        item.cfg.token = token
        await asyncio.to_thread(set_account_token, settings, identifier, token)
        await asyncio.to_thread(settings.save)
        await asyncio.to_thread(pool.clear_cooldown, identifier)
        await asyncio.to_thread(clear_login_backoff, identifier)
        try:
            await asyncio.to_thread(clear_login_backoff, item.cfg.identifier)
        except Exception:
            pass
        return {"ok": True, "account": item.status()}

    @app.delete("/v1/accounts/{identifier}")
    async def delete_account(identifier: str, authorization: str = Header(default="")):
        from .accounts import remove_account
        settings: Settings = app.state.settings
        _check_auth(settings, authorization)
        pool = _server()._pool(app)
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
        pool = _server()._pool(app)
        cleared = await asyncio.to_thread(pool.clear_cooldown, identifier)
        await asyncio.to_thread(clear_login_backoff, identifier)
        # Persist the ban-flag clear so labels survive the next restart too.
        await asyncio.to_thread(settings.save)
        return {"ok": True, "cleared": cleared}


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
    from .server import MODELS
    model = str(body.get("model") or "deepseek-flash")
    messages = body.get("messages", [])
    if not messages:
        raise HTTPException(status_code=400, detail="messages is required")
    if not isinstance(messages, list):
        raise HTTPException(status_code=400,
                            detail="messages must be a list of message objects")
    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict) or not msg.get("role"):
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
    tool_args_max_chars = None
    forgiving_toolcalls = False
    if settings is not None:
        try:
            tool_args_max_chars = int(
                getattr(settings, "tool_args_max_chars", 0) or 0) or None
        except (TypeError, ValueError):
            tool_args_max_chars = None
        forgiving_toolcalls = bool(getattr(settings, "forgiving_toolcalls", False))
    if use_tools:
        messages = [{"role": "system",
                     "content": tool_system_prompt(
                         tools, max_args_chars=tool_args_max_chars)},
                    *messages]
    prompt = render_prompt(messages)
    if use_tools:
        # Recency reminder: long histories bury the start instruction.
        # Appended after user content; non-tool path unchanged.
        prompt = prompt + "\n\n" + tool_reminder_prompt(tools)
    # User-configured wrapper: append_top at the absolute top (before even
    # the tool system prompt), append_bottom at the absolute bottom (after
    # the tool reminder). Gated by enable_append (False keeps texts stored
    # but sends nothing). Internal newlines preserved for multi-line blocks;
    # edge blank lines trimmed so joining never stacks empty gaps.
    # Whitespace-only counts as empty ("" disables). Checked on the final
    # prompt so appends alone can satisfy the non-empty requirement.
    appends_on = bool(getattr(settings, "enable_append", True)) if settings is not None else True
    raw_top = getattr(settings, "append_top", "") if (settings is not None and appends_on) else ""
    raw_bottom = getattr(settings, "append_bottom", "") if (settings is not None and appends_on) else ""
    top = str(raw_top or "").strip("\r\n")
    bottom = str(raw_bottom or "").strip("\r\n")
    if not top.strip():
        top = ""
    if not bottom.strip():
        bottom = ""
    if top or bottom:
        parts = ([top] if top else []) + ([prompt] if prompt.strip() else []) + ([bottom] if bottom else [])
        prompt = "\n\n".join(parts)
    if not prompt.strip():
        raise HTTPException(
            status_code=400,
            detail="messages must produce a non-empty prompt "
                   "(upstream rejects empty prompts with 422)")
    return {
        "model": model, "stream": bool(body.get("stream", False)),
        "tools": tools if use_tools else [], "prompt": prompt,
        "tool_args_max_chars": tool_args_max_chars,
        "forgiving_toolcalls": forgiving_toolcalls,
        "thinking": thinking, "search": search, "model_type": model_type,
    }
