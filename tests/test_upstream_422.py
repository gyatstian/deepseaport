"""Regression tests for upstream HTTP 422 handling.

User report: chat mode crashed with
  RuntimeError: HTTP_422:
plus "Exception in ASGI application" and the chat REPL showed
  ! HTTP 500: Internal Server Error.

Root causes fixed:
  1. client.stream_completion dropped the upstream body when
     content-type started with text/* and truncated to 300 chars,
     so the log showed bare "HTTP_422:" with no detail.
  2. server._run_completion_core fell through to
     `raise RuntimeError(error[:300])` for unknown upstream errors,
     which Starlette turns into an unhandled 500 + ASGI traceback.
  3. server._prepare accepted empty prompts / role-less messages,
     a class of requests the upstream rejects with 422.

These tests lock in: body preserved, 422 -> 502 (never RuntimeError/
500), endpoint returns 502 JSON, empty prompt -> local 400.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from deepseaport import protocol as P
from deepseaport.accounts import PooledAccount
from deepseaport.config import AccountConfig, Settings
from deepseaport.server import _prepare, _run_completion_core, create_app
from deepseaport import pow as PoW


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
        max_retries=0,
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
    wp = tmp_path / "wasm.bin"
    wp.write_bytes(b"x" * 2000)
    monkeypatch.setattr("deepseaport.server.PoW.wasm_path", lambda: wp)
    return wp


def _stub_session_pow(monkeypatch):
    monkeypatch.setattr("deepseaport.server.DS.create_session", lambda h: "sess-1")
    monkeypatch.setattr("deepseaport.server.DS.fetch_pow", lambda h: dict(CHALLENGE))
    monkeypatch.setattr("deepseaport.server.PoW.solve", lambda *a, **k: 42)
    monkeypatch.setattr("deepseaport.server._delete_session_bg", lambda *a, **k: None)


# 1. client preserves 422 body even with text/* content-type -------------


class _FakeResp:
    def __init__(self, status_code: int, text: str, content_type: str):
        self.status_code = status_code
        self._text = text
        self.headers = {"content-type": content_type}

    @property
    def text(self) -> str:
        return self._text

    def iter_lines(self):
        return iter([])

    def close(self) -> None:
        return None


def test_stream_completion_preserves_422_text_body(monkeypatch):
    """text/plain 422 must not become bare 'HTTP_422:' with empty detail."""
    from deepseaport import client as C

    body = '{"detail":[{"loc":["body","prompt"],"msg":"field required"}]}'
    monkeypatch.setattr(
        C, "_SESSION",
        type("_S", (), {"post": staticmethod(
            lambda *a, **k: _FakeResp(422, body, "text/plain; charset=utf-8"))})(),
    )
    events = list(C.stream_completion({"h": "1"}, {"prompt": "hi"}))
    assert len(events) == 1 and events[0].kind == "error"
    assert events[0].code == "HTTP_422"
    assert "prompt" in events[0].message  # body survived, not ""


def test_stream_completion_preserves_422_json_detail(monkeypatch):
    """application/json 422 keeps the FastAPI detail payload."""
    from deepseaport import client as C

    body = '{"detail":[{"loc":["body","prompt"],"msg":"field required","type":"missing"}]}'
    monkeypatch.setattr(
        C, "_SESSION",
        type("_S", (), {"post": staticmethod(
            lambda *a, **k: _FakeResp(422, body, "application/json"))})(),
    )
    events = list(C.stream_completion({"h": "1"}, {"prompt": "hi"}))
    assert events[0].code == "HTTP_422"
    assert "detail" in events[0].message


# 2. server maps upstream 422 -> 502, never RuntimeError/500 ---------------


def test_run_core_maps_http_422_to_502(monkeypatch, tmp_path):
    """The exact user crash: 'HTTP_422: ' must raise 502, not RuntimeError."""
    app = _app_with_bridge(_settings())
    _cached_wasm(monkeypatch, tmp_path)
    _stub_session_pow(monkeypatch)
    monkeypatch.setattr(
        "deepseaport.server._attempt", lambda *a, **k: ("", "", "HTTP_422: ", 0))
    prep = _prepare({"model": "deepseek-flash",
                     "messages": [{"role": "user", "content": "hi"}]})
    item = PooledAccount(cfg=AccountConfig(email="a@x.com", token="tok"))

    with pytest.raises(HTTPException) as exc_info:
        _run_completion_core(app, item, prep)
    assert exc_info.value.status_code == 502
    assert "422" in str(exc_info.value.detail)


def test_run_core_maps_422_with_body_to_502(monkeypatch, tmp_path):
    app = _app_with_bridge(_settings())
    _cached_wasm(monkeypatch, tmp_path)
    _stub_session_pow(monkeypatch)
    monkeypatch.setattr(
        "deepseaport.server._attempt",
        lambda *a, **k: ("", "", 'HTTP_422: {"detail":[{"loc":["body"]}]}', 0))
    prep = _prepare({"model": "deepseek-flash",
                     "messages": [{"role": "user", "content": "hi"}]})
    item = PooledAccount(cfg=AccountConfig(email="a@x.com", token="tok"))

    with pytest.raises(HTTPException) as exc_info:
        _run_completion_core(app, item, prep)
    assert exc_info.value.status_code == 502
    assert "rejected" in str(exc_info.value.detail).lower()


def test_run_core_maps_generic_upstream_to_502_not_500(monkeypatch, tmp_path):
    """Any other unknown upstream error is a bad-gateway 502, not a 500."""
    app = _app_with_bridge(_settings())
    _cached_wasm(monkeypatch, tmp_path)
    _stub_session_pow(monkeypatch)
    monkeypatch.setattr(
        "deepseaport.server._attempt",
        lambda *a, **k: ("", "", "HTTP_500: Internal Server Error", 0))
    prep = _prepare({"model": "deepseek-flash",
                     "messages": [{"role": "user", "content": "hi"}]})
    item = PooledAccount(cfg=AccountConfig(email="a@x.com", token="tok"))

    with pytest.raises(HTTPException) as exc_info:
        _run_completion_core(app, item, prep)
    # 500 from upstream is in the transient-overload list -> 503; anything
    # else must still be an HTTPException (502), never a bare RuntimeError.
    assert exc_info.value.status_code in (502, 503)


# 3. endpoint returns 502 JSON (no ASGI traceback / 500) ------------------


def test_endpoint_returns_502_on_upstream_422(monkeypatch):
    from fastapi.testclient import TestClient

    settings = _settings()
    app = create_app(settings)
    app.state.bridge = _FakeBridge()

    def fake_core(app_, item_, prep_, **kw):
        raise HTTPException(status_code=502,
                            detail="DeepSeek rejected the request as invalid (HTTP_422: )")

    monkeypatch.setattr("deepseaport.server._run_completion_core", fake_core)
    # Stub the pool so no real account/network is touched.
    import deepseaport.server as S

    real_pool = S._pool

    class _FakePool:
        def __init__(self):
            self.item = PooledAccount(cfg=AccountConfig(email="a@x.com", token="tok"))

        def __len__(self):
            return 1

        async def aacquire(self, timeout, allow_failover=True):
            return self.item

    fake_pool = _FakePool()
    monkeypatch.setattr(S, "_pool", lambda app_: fake_pool)
    monkeypatch.setattr(S.AccountPool, "release", staticmethod(lambda item: None))

    resp = TestClient(app, raise_server_exceptions=False).post(
        "/v1/chat/completions",
        json={"model": "deepseek-flash",
              "messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 502
    assert "422" in resp.text


# 4. upstream source variant (live 2026-09-15 422) -----------------------


def test_completion_payload_source_is_upstream_variant():
    """Upstream rejects source='web': expected `default` or `landing` (422)."""
    payload = P.completion_payload("sess", "hi", False, False)
    assert payload["source"] in ("default", "landing"), payload["source"]


# 5. local validation prevents empty-prompt class of upstream 422 --------


def test_prepare_rejects_empty_prompt_as_400():
    with pytest.raises(HTTPException) as exc_info:
        _prepare({"model": "deepseek-flash",
                  "messages": [{"role": "user", "content": "   "}]})
    assert exc_info.value.status_code == 400
    assert "non-empty prompt" in str(exc_info.value.detail)


def test_prepare_rejects_missing_role_as_400():
    with pytest.raises(HTTPException) as exc_info:
        _prepare({"model": "deepseek-flash",
                  "messages": [{"content": "hi"}]})
    assert exc_info.value.status_code == 400
    assert "role" in str(exc_info.value.detail)
