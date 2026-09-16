"""Server concurrency primitives and process lifecycle.

Pure moves from deepseaport.server: the shared thread-pool executors, the
queued-slot acquisition / client-disconnect helpers, and the FastAPI
lifespan plus startup warmup. No DeepSeek completion logic lives here.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from . import pow as PoW
from .config import Settings
from .obscura_bridge import ObscuraBridge, register_default_bridge

logger = logging.getLogger("deepseaport.server")


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
# A 401 means either a missing token or a browser refresh that could not
# produce a usable one.  Cool that account briefly so failover can move to a
# healthy account instead of immediately reacquiring the same broken one.
INVALID_TOKEN_COOLDOWN_SECONDS = 60


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
        return
    # Pre-compile wasmtime Engine/Module so the first completion skips the
    # 100-300ms compile on the hot path (Store/instance stay per-solve).
    try:
        PoW.prewarm()
    except Exception as exc:
        logger.debug("PoW prewarm failed (lazy compile on demand): %s", exc)
