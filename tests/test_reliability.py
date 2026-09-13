"""Offline regression tests for the high-value reliability fixes.

Covers: session cleanup on PoW failure, atomic config writes, serialized
obscura warmups, safe DEEPSEAPORT_PORT parsing, fresh PoW on session retry,
and wasm magic validation. No network, no credentials.
"""

from __future__ import annotations

import json
import threading
import time
from subprocess import CompletedProcess

import pytest

from deepseaport import pow as PoW
from deepseaport.accounts import PooledAccount
from deepseaport.config import DEFAULT_PORT, AccountConfig, Settings
from deepseaport.obscura_bridge import ObscuraBridge
from deepseaport.protocol import StreamEvent
from deepseaport.server import (
    _attempt,
    _prepare,
    _run_completion_core,
    _session_and_challenge,
    _stream_completion,
    create_app,
)

CHALLENGE = {
    "algorithm": PoW.ALGORITHM,
    "challenge": "c",
    "salt": "s",
    "difficulty": 1000,
    "expire_at": 2**31,
    "signature": "sig",
    "target_path": "/api/v0/chat/completion",
}


class _FakeState:
    user_agent = "test-ua"
    cookies: dict = {}


class _FakeBridge:
    def __init__(self) -> None:
        self.state = _FakeState()

    def cookie_header(self) -> str:
        return ""

    def warmup(self) -> dict:
        return {}


def _settings(**kw) -> Settings:
    base = dict(
        keys=[],
        accounts=[AccountConfig(email="a@x.com", token="tok")],
        config_path="",
        warmup_on_startup=False,
        max_retries=1,
        log_level="WARNING",
        parallel_challenge_fetch=True,
    )
    base.update(kw)
    return Settings(**base)


def _app_with_bridge(settings: Settings):
    app = create_app(settings)
    app.state.bridge = _FakeBridge()
    return app


def _cached_wasm(monkeypatch, tmp_path):
    """Point PoW at an existing >1000 byte file so ensure_wasm is skipped."""
    wp = tmp_path / "wasm.bin"
    wp.write_bytes(b"x" * 2000)
    monkeypatch.setattr("deepseaport.server.PoW.wasm_path", lambda: wp)
    return wp


# 1. session leak on PoW failure -------------------------------------------


def test_session_cleaned_up_when_pow_solve_raises(monkeypatch, tmp_path):
    app = _app_with_bridge(_settings())
    _cached_wasm(monkeypatch, tmp_path)
    deleted: list = []
    monkeypatch.setattr("deepseaport.server._delete_session_bg",
                        lambda *a, **k: deleted.append(a))
    monkeypatch.setattr("deepseaport.server.DS.create_session", lambda h: "sess-1")
    monkeypatch.setattr("deepseaport.server.DS.fetch_pow", lambda h: dict(CHALLENGE))
    monkeypatch.setattr("deepseaport.server.PoW.solve", lambda *a, **k: None)
    prep = _prepare({"model": "deepseek-flash",
                     "messages": [{"role": "user", "content": "hi"}]})
    item = PooledAccount(cfg=AccountConfig(email="a@x.com", token="tok"))

    with pytest.raises(RuntimeError):
        _run_completion_core(app, item, prep)

    assert deleted, "session cleanup not scheduled when PoW solve failed"
    assert deleted[0][1] == "sess-1"


def test_session_and_challenge_deletes_on_pow_fetch_failure(monkeypatch):
    deleted: list = []
    monkeypatch.setattr("deepseaport.server.DS.create_session", lambda h: "sess-X")
    monkeypatch.setattr("deepseaport.server.DS.fetch_pow",
                        lambda h: (_ for _ in ()).throw(RuntimeError("pow fetch failed")))
    monkeypatch.setattr("deepseaport.server.DS.delete_session",
                        lambda h, sid: deleted.append(sid))

    with pytest.raises(RuntimeError):
        _session_and_challenge({"a": "b"}, parallel=True)
    assert deleted == ["sess-X"]


def test_session_and_challenge_deletes_on_fetch_failure_sequential(monkeypatch):
    deleted: list = []
    monkeypatch.setattr("deepseaport.server.DS.create_session", lambda h: "sess-Y")
    monkeypatch.setattr("deepseaport.server.DS.fetch_pow",
                        lambda h: (_ for _ in ()).throw(RuntimeError("pow fetch failed")))
    monkeypatch.setattr("deepseaport.server.DS.delete_session",
                        lambda h, sid: deleted.append(sid))

    with pytest.raises(RuntimeError):
        _session_and_challenge({"a": "b"}, parallel=False)
    assert deleted == ["sess-Y"]


# 2. atomic config writes ---------------------------------------------------


def test_save_is_atomic_and_leaves_no_tmp(tmp_path):
    cfg = tmp_path / "config.json"
    s = Settings(accounts=[AccountConfig(email="a@x.com", token="t")],
                 config_path=str(cfg))
    s.save()
    assert json.loads(cfg.read_text(encoding="utf-8"))["accounts"][0]["email"] == "a@x.com"
    assert list(tmp_path.glob("*.tmp")) == []


def test_concurrent_saves_never_corrupt(tmp_path):
    cfg = tmp_path / "config.json"
    s = Settings(accounts=[AccountConfig(email="a@x.com", token="t")],
                 config_path=str(cfg))
    errors: list = []

    def writer(i: int) -> None:
        try:
            for _ in range(25):
                s.active_account = f"acc-{i}"
                s.save()
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    data = json.loads(cfg.read_text(encoding="utf-8"))
    assert data["accounts"][0]["email"] == "a@x.com"
    assert list(tmp_path.glob("*.tmp")) == []


# 3. serialized obscura warmups --------------------------------------------


def test_concurrent_warmups_serialize_subprocesses(tmp_path):
    bridge = ObscuraBridge(binary="fake-obscura", profile=str(tmp_path / "prof"))
    guard = threading.Lock()
    active = 0
    peak = 0
    dump = json.dumps([{"name": "aws-waf-token", "value": "tok"}])

    def fake_run(*args, timeout=60):
        nonlocal active, peak
        with guard:
            active += 1
            peak = max(peak, active)
        time.sleep(0.05)
        with guard:
            active -= 1
        return CompletedProcess(args, 0, stdout=dump, stderr="")

    bridge._run = fake_run  # type: ignore[method-assign]
    threads = [threading.Thread(target=bridge.warmup) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert peak == 1, f"obscura subprocesses overlapped (peak={peak})"


def test_bridges_on_same_profile_share_lock(tmp_path):
    """Distinct instances on one --storage-dir must serialize on one lock.

    Token refresh / CLI login build their own ObscuraBridge; without a shared
    per-profile lock they could spawn obscura concurrently with an app-bridge
    warmup and corrupt the cookie jar.
    """
    from deepseaport.obscura_bridge import profile_lock

    prof = str(tmp_path / "prof")
    a = ObscuraBridge(binary="fake-obscura", profile=prof)
    b = ObscuraBridge(binary="fake-obscura", profile=prof)
    assert a._proc_lock is b._proc_lock
    assert profile_lock(prof) is a._proc_lock
    # A different profile gets a different lock.
    other = ObscuraBridge(binary="fake-obscura", profile=str(tmp_path / "other"))
    assert other._proc_lock is not a._proc_lock


def test_obscura_login_token_holds_profile_lock(monkeypatch, tmp_path):
    """Login's MCP subprocess must run under the shared profile lock."""
    from deepseaport import cli
    from deepseaport.config import Settings
    from deepseaport.obscura_bridge import profile_lock

    prof = str(tmp_path / "login-prof")
    settings = Settings(obscura_profile=prof, config_path="")
    entered = threading.Event()
    release = threading.Event()
    result = {}

    class FakeClient:
        def __init__(self, binary, profile):
            # Called on the login thread while it holds the profile lock.
            entered.set()
            release.wait(5)
            raise RuntimeError("stop after spawn check")

        def call(self, *a, **k):
            return {}

        def close(self):
            pass

    monkeypatch.setattr("deepseaport.mcp_client.McpClient", FakeClient)

    def run():
        try:
            cli._obscura_login_token("a@x.com", "pw", settings)
        except RuntimeError:
            pass

    t = threading.Thread(target=run)
    t.start()
    assert entered.wait(5), "login never reached MCP spawn"
    # Login holds the shared per-profile lock while spawning the subprocess.
    got = profile_lock(prof).acquire(blocking=False)
    result["locked"] = not got
    if got:
        profile_lock(prof).release()
    release.set()
    t.join(5)
    assert result["locked"] is True


# 4. malformed DEEPSEAPORT_PORT --------------------------------------------


def test_malformed_port_env_falls_back(monkeypatch, tmp_path):
    from deepseaport.config import load_settings

    cfg = tmp_path / "c.json"
    monkeypatch.setenv("DEEPSEAPORT_PORT", "not-a-number")
    assert load_settings(str(cfg)).port == DEFAULT_PORT
    monkeypatch.setenv("DEEPSEAPORT_PORT", "7000")
    assert load_settings(str(cfg)).port == 7000


# 5. fresh PoW header on session retry -------------------------------------


def test_session_retry_refreshes_pow_header(monkeypatch, tmp_path):
    app = _app_with_bridge(_settings())
    _cached_wasm(monkeypatch, tmp_path)
    monkeypatch.setattr("deepseaport.server._delete_session_bg", lambda *a, **k: None)
    sessions = iter(["sess-1", "sess-2"])
    monkeypatch.setattr("deepseaport.server.DS.create_session", lambda h: next(sessions))
    monkeypatch.setattr("deepseaport.server.DS.fetch_pow", lambda h: dict(CHALLENGE))
    monkeypatch.setattr("deepseaport.server.PoW.solve", lambda *a, **k: 42)
    monkeypatch.setattr("deepseaport.server._solve",
                        lambda a, token: {"X-DS-PoW-Response": "FRESH"})

    seen: list = []
    count = {"n": 0}

    def fake_attempt(headers, payload, on_content=None, on_thinking=None,
                     cancel_event=None):
        seen.append(headers)
        count["n"] += 1
        if count["n"] == 1:
            return "", "", "INVALID_SESSION_ID: gone", 0
        return "hi", "", None, 5

    monkeypatch.setattr("deepseaport.server._attempt", fake_attempt)
    prep = _prepare({"model": "deepseek-flash",
                     "messages": [{"role": "user", "content": "hi"}]})
    item = PooledAccount(cfg=AccountConfig(email="a@x.com", token="tok"))

    result = _run_completion_core(app, item, prep)

    assert result["content"] == "hi"
    assert len(seen) == 2
    assert seen[1].get("X-DS-PoW-Response") == "FRESH"


# 6. wasm magic validation --------------------------------------------------


class _FakeResp:
    def __init__(self, content: bytes) -> None:
        self.content = content

    def raise_for_status(self) -> None:
        return None


def test_ensure_wasm_rejects_non_wasm(monkeypatch, tmp_path):
    target = tmp_path / "sha3.wasm"
    monkeypatch.setattr("deepseaport.pow.wasm_path", lambda: target)
    monkeypatch.setattr("curl_cffi.requests.get",
                        lambda *a, **k: _FakeResp(b"<!doctype html><html>nope</html>"))

    with pytest.raises(RuntimeError):
        PoW.ensure_wasm()
    assert not target.exists()


def test_ensure_wasm_accepts_valid_module(monkeypatch, tmp_path):
    target = tmp_path / "sha3.wasm"
    payload = b"\x00asm\x01\x00\x00\x00" + b"z" * 2000
    monkeypatch.setattr("deepseaport.pow.wasm_path", lambda: target)
    monkeypatch.setattr("curl_cffi.requests.get", lambda *a, **k: _FakeResp(payload))

    path = PoW.ensure_wasm()
    assert path == target
    assert target.read_bytes().startswith(b"\x00asm")


# 7. client disconnect stops orphaned producer + frees account lock ---------


def test_attempt_stops_when_cancel_event_set(monkeypatch):
    """_attempt must stop consuming events once cancel_event is set."""
    consumed = {"n": 0}

    def fake_stream(headers, payload, **kw):
        for i in range(1000):
            consumed["n"] += 1
            yield StreamEvent(kind="content", text="x")
            time.sleep(0.001)

    monkeypatch.setattr("deepseaport.server.DS.stream_completion", fake_stream)
    cancel = threading.Event()

    def set_soon():
        time.sleep(0.02)
        cancel.set()

    threading.Thread(target=set_soon, daemon=True).start()
    content, _think, error, _usage = _attempt(
        {"h": "1"}, {"p": "1"}, cancel_event=cancel)
    assert error is None
    assert consumed["n"] < 1000, "producer kept streaming after cancellation"


def test_generator_close_releases_account_lock(monkeypatch, tmp_path):
    """Closing the SSE generator early must release the account lock."""
    import asyncio

    from deepseaport.accounts import AccountPool

    app = _app_with_bridge(_settings(stream_mode="live", auto_delete_session=False,
                                     max_retries=0))
    _cached_wasm(monkeypatch, tmp_path)
    monkeypatch.setattr("deepseaport.server.DS.create_session", lambda h: "sess-1")
    monkeypatch.setattr("deepseaport.server.DS.fetch_pow", lambda h: dict(CHALLENGE))
    monkeypatch.setattr("deepseaport.server.PoW.solve", lambda *a, **k: 42)

    state = {"closed": False, "emitted": 0}

    def fake_stream(headers, payload, **kw):
        try:
            for _ in range(1000):
                state["emitted"] += 1
                yield StreamEvent(kind="thinking", text="t")
                time.sleep(0.005)
        finally:
            state["closed"] = True

    monkeypatch.setattr("deepseaport.server.DS.stream_completion", fake_stream)

    item = PooledAccount(cfg=AccountConfig(email="a@x.com", token="tok"))
    pool = AccountPool([])
    item.lock.acquire()
    prep = _prepare({"model": "deepseek-flash-reasoner", "stream": True,
                     "messages": [{"role": "user", "content": "hi"}]})

    async def scenario() -> bool:
        agen = _stream_completion(app, pool, item, prep)
        await agen.__anext__()  # role chunk
        await agen.__anext__()  # first live delta
        await agen.aclose()
        for _ in range(300):
            if not item.lock.locked():
                return True
            await asyncio.sleep(0.01)
        return False

    released = asyncio.run(scenario())
    assert released, "account lock not released after generator close"
    assert state["closed"], "stream_completion generator was not closed"
    assert state["emitted"] < 1000, "producer ran to completion after disconnect"


# 8. banned account surfaces 403 with expiry + long cooldown -----------------


def test_banned_error_raises_403_with_expiry(monkeypatch, tmp_path):
    """Ban (biz 5 muted) must 403 with 'until <date>', not generic 500."""
    import time as _time

    from fastapi import HTTPException

    app = _app_with_bridge(_settings(max_retries=0))
    _cached_wasm(monkeypatch, tmp_path)
    monkeypatch.setattr("deepseaport.server.DS.create_session", lambda h: "sess-ban")
    monkeypatch.setattr("deepseaport.server.DS.fetch_pow", lambda h: dict(CHALLENGE))
    monkeypatch.setattr("deepseaport.server.PoW.solve", lambda *a, **k: 42)
    monkeypatch.setattr("deepseaport.server._delete_session_bg", lambda *a, **k: None)
    ban_until = _time.time() + 86400
    monkeypatch.setattr(
        "deepseaport.server._attempt",
        lambda *a, **k: ("", "", "5: user is muted (suspended until 16 September 2026 12:49)", 0),
    )
    monkeypatch.setattr("deepseaport.server.DS.check_ban", lambda h, timeout=15: (True, ban_until))
    prep = _prepare({"model": "deepseek-flash",
                     "messages": [{"role": "user", "content": "hi"}]})
    item = PooledAccount(cfg=AccountConfig(email="banned@x.com", token="tok"))

    with pytest.raises(HTTPException) as exc_info:
        _run_completion_core(app, item, prep)
    assert exc_info.value.status_code == 403
    assert "suspended until" in str(exc_info.value.detail)
    assert "banned" in str(exc_info.value.detail).lower()
    assert item.cooldown_remaining > 3600  # until expiry, not 120s default


# 9. same-request ban failover onto next unbanned account --------------------


def _ban_403(account: str) -> object:
    from fastapi import HTTPException

    return HTTPException(
        status_code=403,
        detail=f"Account {account} banned. Due to violation of user policies, "
               "your account has been suspended until 16 September 2026 12:49. "
               "If you have any questions, please contact us.")


def _two_account_app():
    from deepseaport.accounts import AccountPool

    settings = _settings(
        accounts=[AccountConfig(email="a@x.com", token="tok-a-1234567890"),
                  AccountConfig(email="b@x.com", token="tok-b-1234567890")],
        active_account="a@x.com")
    app = _app_with_bridge(settings)
    pool = AccountPool(settings.accounts, current=settings.active_account)
    return app, settings, pool


def test_ban_failover_uses_next_account_and_switches_current(monkeypatch):
    """Banned CURRENT must transparently retry on the next unbanned account."""
    from deepseaport.accounts import AccountPool
    from deepseaport.server import _complete_with_failover_sync

    app, settings, pool = _two_account_app()
    used: list = []

    def fake_core(app_, item_, prep_, **kw):
        used.append(item_.cfg.email)
        if item_.cfg.email == "a@x.com":
            AccountPool.mark_bad(item_, seconds=86400)  # as _run_completion_core does
            raise _ban_403("a@x.com")
        return {"content": "hi from b", "thinking": "", "usage_total": 5}

    monkeypatch.setattr("deepseaport.server._run_completion_core", fake_core)
    prep = _prepare({"model": "deepseek-flash",
                     "messages": [{"role": "user", "content": "hi"}]})
    first = pool.acquire(timeout=1)
    assert first.cfg.email == "a@x.com"  # CURRENT preferred first

    result = _complete_with_failover_sync(app, pool, prep, first)

    assert result["content"] == "hi from b"
    assert used == ["a@x.com", "b@x.com"]
    assert not pool.get("a@x.com").lock.locked()
    assert not pool.get("b@x.com").lock.locked()
    # CURRENT auto-switched to the working account (persisted).
    assert pool.current == "b@x.com"
    assert settings.active_account == "b@x.com"


def test_ban_failover_all_banned_raises_403(monkeypatch):
    """No healthy account left -> the ban 403 (with expiry) surfaces."""
    from fastapi import HTTPException

    from deepseaport.accounts import AccountPool
    from deepseaport.server import _complete_with_failover_sync

    app, settings, pool = _two_account_app()
    used: list = []

    def fake_core(app_, item_, prep_, **kw):
        used.append(item_.cfg.email)
        AccountPool.mark_bad(item_, seconds=86400)
        raise _ban_403(item_.cfg.email)

    monkeypatch.setattr("deepseaport.server._run_completion_core", fake_core)
    prep = _prepare({"model": "deepseek-flash",
                     "messages": [{"role": "user", "content": "hi"}]})
    first = pool.acquire(timeout=1)

    with pytest.raises(HTTPException) as exc_info:
        _complete_with_failover_sync(app, pool, prep, first)
    assert exc_info.value.status_code == 403
    assert "16 September" in str(exc_info.value.detail)
    assert sorted(used) == ["a@x.com", "b@x.com"]
    assert not pool.get("a@x.com").lock.locked()
    assert not pool.get("b@x.com").lock.locked()


def test_non_ban_error_does_not_failover(monkeypatch):
    """Generic failures must propagate without touching the next account."""
    from deepseaport.accounts import AccountPool
    from deepseaport.server import _complete_with_failover_sync

    app, settings, pool = _two_account_app()
    used: list = []

    def fake_core(app_, item_, prep_, **kw):
        used.append(item_.cfg.email)
        raise RuntimeError("boom")

    monkeypatch.setattr("deepseaport.server._run_completion_core", fake_core)
    prep = _prepare({"model": "deepseek-flash",
                     "messages": [{"role": "user", "content": "hi"}]})
    first = pool.acquire(timeout=1)

    with pytest.raises(RuntimeError, match="boom"):
        _complete_with_failover_sync(app, pool, prep, first)
    assert used == ["a@x.com"]
    assert pool.current == "a@x.com"  # selection untouched
    assert not pool.get("a@x.com").lock.locked()


# 10. fixes 2-4: disconnect cancel, bounded producer, no-bridge guard ------


def test_precancelled_event_aborts_session_setup_fast(monkeypatch):
    """Fix 2: disconnect before session setup must abort, not run full PoW."""
    import time as _time

    from deepseaport.server import _StreamCancelled, _session_and_challenge

    def slow_session(h):
        _time.sleep(5)
        return "sess-slow"

    monkeypatch.setattr("deepseaport.server.DS.create_session", slow_session)
    monkeypatch.setattr("deepseaport.server.DS.fetch_pow", lambda h: {"a": 1})
    cancel = threading.Event()
    cancel.set()
    t0 = _time.time()
    with pytest.raises(_StreamCancelled):
        _session_and_challenge({"h": "1"}, parallel=True, cancel_event=cancel)
    assert _time.time() - t0 < 1.0


def test_nonstream_endpoint_takes_request_param():
    """Fix 2: POST /v1/chat/completions must accept Request for polling."""
    import inspect

    from deepseaport.server import create_app

    app = create_app(_settings())
    route = next(r for r in app.routes
                 if getattr(r, "path", "") == "/v1/chat/completions")
    assert "request" in inspect.signature(route.endpoint).parameters


def test_nonstream_disconnect_while_queued_returns_499(monkeypatch):
    """Fix 2: disconnect during the account-queue wait -> fast 499, no slot."""
    import asyncio

    from deepseaport.accounts import AccountPool
    from deepseaport.server import _acquire_slot_or_499
    from fastapi import HTTPException

    pool = AccountPool([AccountConfig(email="a@x.com", token="tok")])
    held = pool.acquire(timeout=1)  # all slots busy
    try:
        class _Req:
            async def is_disconnected(self):
                return True

        async def scenario():
            with pytest.raises(HTTPException) as exc_info:
                await _acquire_slot_or_499(pool, _Req(), threading.Event(),
                                           90, True)
            return exc_info.value.status_code

        assert asyncio.run(scenario()) == 499
        assert held.lock.locked()  # never stole the busy slot
    finally:
        AccountPool.release(held)


def test_stream_producer_uses_bounded_executor():
    """Fix 3: stream producers submit to a bounded executor, not raw threads."""
    import inspect

    from deepseaport import server as S

    src = inspect.getsource(S._stream_completion)
    assert "_STREAM_EXECUTOR.submit" in src
    assert S._STREAM_EXECUTOR._max_workers <= 64


def test_no_bridge_surfaces_real_error(monkeypatch, tmp_path):
    """Fix 4: missing bridge must not mask the PoW error with AttributeError."""
    app = create_app(_settings())
    assert not hasattr(app.state, "bridge")
    _cached_wasm(monkeypatch, tmp_path)
    monkeypatch.setattr("deepseaport.server._delete_session_bg",
                        lambda *a, **k: None)
    monkeypatch.setattr("deepseaport.server.DS.create_session", lambda h: "sess-1")
    monkeypatch.setattr("deepseaport.server.DS.fetch_pow", lambda h: dict(CHALLENGE))
    monkeypatch.setattr("deepseaport.server.PoW.solve", lambda *a, **k: None)
    prep = _prepare({"model": "deepseek-flash",
                     "messages": [{"role": "user", "content": "hi"}]})
    item = PooledAccount(cfg=AccountConfig(email="a@x.com", token="tok"))
    with pytest.raises(RuntimeError, match="PoW solver found no answer"):
        _run_completion_core(app, item, prep)


def test_endpoint_failover_banned_current_serves_from_healthy(monkeypatch):
    """POST /v1/chat/completions with banned CURRENT returns 200 via next."""
    from fastapi.testclient import TestClient

    from deepseaport.accounts import AccountPool

    app, settings, pool = _two_account_app()
    app.state.pool = pool  # endpoint reuses this pool (syncs CURRENT from settings)

    def fake_core(app_, item_, prep_, **kw):
        if item_.cfg.email == "a@x.com":
            AccountPool.mark_bad(item_, seconds=86400)
            raise _ban_403("a@x.com")
        return {"content": "served by b", "thinking": "", "usage_total": 3}

    monkeypatch.setattr("deepseaport.server._run_completion_core", fake_core)
    resp = TestClient(app).post("/v1/chat/completions", json={
        "model": "deepseek-flash", "messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 200, resp.text
    assert resp.json()["choices"][0]["message"]["content"] == "served by b"
    assert settings.active_account == "b@x.com"


# 11. invalid-token (40003) auto-refresh via Obscura ------------------------


def _token_settings():
    return _settings(accounts=[AccountConfig(
        email="a@x.com", password="pw", token="stale-token-1234567890")])


def test_completion_40003_refreshes_token_and_retries(monkeypatch, tmp_path):
    """A 40003 error event must trigger one Obscura refresh + retry."""
    app = _app_with_bridge(_token_settings())
    _cached_wasm(monkeypatch, tmp_path)
    monkeypatch.setattr("deepseaport.server._delete_session_bg", lambda *a, **k: None)
    monkeypatch.setattr("deepseaport.server.DS.create_session", lambda h: "sess-1")
    monkeypatch.setattr("deepseaport.server.DS.fetch_pow", lambda h: dict(CHALLENGE))
    monkeypatch.setattr("deepseaport.server.PoW.solve", lambda *a, **k: 42)
    monkeypatch.setattr("deepseaport.server._solve",
                        lambda a, token: {"Authorization": f"Bearer {token}"})

    calls = {"n": 0, "refresh": 0}

    def fake_attempt(headers, payload, on_content=None, on_thinking=None,
                     cancel_event=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return "", "", "40003: Authorization Failed (invalid token)", 0
        return "hi", "", None, 5

    def fake_login(email, password, settings):
        calls["refresh"] += 1
        return "fresh-token-1234567890"

    monkeypatch.setattr("deepseaport.server._attempt", fake_attempt)
    monkeypatch.setattr("deepseaport.cli._obscura_login_token", fake_login)
    prep = _prepare({"model": "deepseek-flash",
                     "messages": [{"role": "user", "content": "hi"}]})
    item = PooledAccount(cfg=AccountConfig(
        email="a@x.com", password="pw", token="stale-token-1234567890"))

    result = _run_completion_core(app, item, prep)

    assert result["content"] == "hi"
    assert calls["n"] == 2
    assert calls["refresh"] == 1
    assert item.cfg.token == "fresh-token-1234567890"
    assert item.cooldown_remaining == 0  # not marked banned


def test_completion_40003_after_refresh_surfaces_401(monkeypatch, tmp_path):
    """A second 40003 (fresh token also rejected) must surface, not loop."""
    from fastapi import HTTPException

    app = _app_with_bridge(_token_settings())
    _cached_wasm(monkeypatch, tmp_path)
    monkeypatch.setattr("deepseaport.server._delete_session_bg", lambda *a, **k: None)
    monkeypatch.setattr("deepseaport.server.DS.create_session", lambda h: "sess-1")
    monkeypatch.setattr("deepseaport.server.DS.fetch_pow", lambda h: dict(CHALLENGE))
    monkeypatch.setattr("deepseaport.server.PoW.solve", lambda *a, **k: 42)
    monkeypatch.setattr("deepseaport.server._solve",
                        lambda a, token: {"Authorization": f"Bearer {token}"})
    monkeypatch.setattr("deepseaport.server._attempt",
                        lambda *a, **k: ("", "", "40003: Authorization Failed (invalid token)", 0))
    monkeypatch.setattr("deepseaport.cli._obscura_login_token",
                        lambda email, password, settings: "fresh-token-1234567890")
    prep = _prepare({"model": "deepseek-flash",
                     "messages": [{"role": "user", "content": "hi"}]})
    item = PooledAccount(cfg=AccountConfig(
        email="a@x.com", password="pw", token="stale-token-1234567890"))

    with pytest.raises(HTTPException) as exc_info:
        _run_completion_core(app, item, prep)
    assert exc_info.value.status_code == 401
    assert "token invalid" in str(exc_info.value.detail).lower()


def test_session_create_40003_refreshes_token(monkeypatch, tmp_path):
    """40003 raised at session create must refresh and retry, not 500."""
    app = _app_with_bridge(_token_settings())
    _cached_wasm(monkeypatch, tmp_path)
    monkeypatch.setattr("deepseaport.server._delete_session_bg", lambda *a, **k: None)

    sessions = {"n": 0}

    def fake_create(h):
        sessions["n"] += 1
        if sessions["n"] == 1:
            raise RuntimeError("create_session biz error 40003: "
                               "Authorization Failed (invalid token)")
        return "sess-fresh"

    monkeypatch.setattr("deepseaport.server.DS.create_session", fake_create)
    monkeypatch.setattr("deepseaport.server.DS.fetch_pow", lambda h: dict(CHALLENGE))
    monkeypatch.setattr("deepseaport.server.PoW.solve", lambda *a, **k: 42)
    monkeypatch.setattr("deepseaport.server._solve",
                        lambda a, token: {"Authorization": f"Bearer {token}"})
    monkeypatch.setattr("deepseaport.server._attempt",
                        lambda *a, **k: ("hi", "", None, 5))
    monkeypatch.setattr("deepseaport.cli._obscura_login_token",
                        lambda email, password, settings: "fresh-token-1234567890")
    prep = _prepare({"model": "deepseek-flash",
                     "messages": [{"role": "user", "content": "hi"}]})
    item = PooledAccount(cfg=AccountConfig(
        email="a@x.com", password="pw", token="stale-token-1234567890"))

    result = _run_completion_core(app, item, prep)

    assert result["content"] == "hi"
    assert sessions["n"] == 2
    assert item.cfg.token == "fresh-token-1234567890"

