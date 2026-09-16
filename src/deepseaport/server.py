"""FastAPI server: OpenAI-compatible endpoints over the DeepSeek web client.

Concurrency model: endpoints are async; all blocking DeepSeek I/O runs in
worker threads. Streaming responses bridge a producer thread (DeepSeek SSE)
to the response generator through a queue, so first tokens reach the
harness without waiting for the full reply. Tool calls are parsed from the
buffered full text at FINISHED and emitted as OpenAI tool_call deltas.

The endpoints live in ``deepseaport.server_routes``, the completion/retry
engine in ``deepseaport.completion_service``, and the executors + lifecycle
in ``deepseaport.server_concurrency``. This module keeps ``create_app`` as
thin wiring and re-exports every historical name for backward compatibility.
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
from .accounts import AccountPool, PooledAccount
from .config import AccountConfig, Settings, load_settings
from .obscura_bridge import ObscuraBridge, register_default_bridge
from .tools_support import parse_tool_calls, render_prompt, tool_reminder_prompt, tool_system_prompt

# --- Re-exports from the split modules (historical public surface) ---------
# Order matters only for readability; all names are importable from
# deepseaport.server exactly as before the split.
from .completion_service import (
    _AttemptOutcome,
    _LOGIN_FAIL_UNTIL,
    _TOKEN_REFRESH_BAN_HINT,
    _TOKEN_REFRESH_INFLIGHT,
    _TOKEN_REFRESH_LOCK,
    _app_settings,
    _attempt,
    _ban_403_for_item,
    _complete_with_failover_sync,
    _delete_session_bg,
    _ensure_token,
    _header_from_challenge,
    _headers,
    _is_ban_http_exception,
    _is_invalid_token_error,
    _is_transient_overload,
    _login_backoff_set,
    _login_backoff_skip,
    _openai_response,
    _probe_ban,
    _refresh_token_via_obscura,
    _refresh_waf_without_account_lock,
    _run_completion_core,
    _session_and_challenge,
    _solve,
    _switch_current_after_ban_failover,
    clear_login_backoff,
)
from .server_concurrency import (
    FAILOVER_ACQUIRE_TIMEOUT,
    INVALID_TOKEN_COOLDOWN_SECONDS,
    _CHALLENGE_EXECUTOR,
    _CLEANUP_EXECUTOR,
    _CONSUMER_GRACE_SECONDS,
    _DEFAULT_POOL_SIZE,
    _STREAM_EXECUTOR,
    _StreamCancelled,
    _acquire_slot_or_499,
    _await_cancelable,
    _startup_warm,
    _watch_disconnect,
    apply_log_level,
    lifespan,
)
from .server_routes import _check_auth, _pool, _prepare, register_routes

logger = logging.getLogger("deepseaport.server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

MODELS = {
    "deepseek-flash": (False, False, "default"),
    "deepseek-flash-reasoner": (True, False, "default"),
    "deepseek-flash-search": (False, True, "default"),
    "deepseek-flash-reasoner-search": (True, True, "default"),
}


def _run_completion(app: FastAPI, item: PooledAccount, prep: dict,
                    cancel_event: threading.Event | None = None) -> dict:
    """Blocking full completion with retries per failure class."""
    return _run_completion_core(app, item, prep, cancel_event=cancel_event)


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


def error_json(request, exc: HTTPException):  # helper for tests
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})


def create_app(settings: Settings | None = None) -> FastAPI:
    app = FastAPI(title="deepseaport", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings or load_settings()
    apply_log_level(getattr(app.state.settings, "log_level", "INFO"))
    app.state.pool = None
    register_routes(app)
    return app


app = create_app()
