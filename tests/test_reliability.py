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

