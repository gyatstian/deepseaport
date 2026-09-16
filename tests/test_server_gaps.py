"""Offline gap tests for deepseaport.server (no network, no credentials).

Covers: _check_auth, _prepare (404/tools/settings=None), _headers,
_is_invalid_token_error/_is_transient_overload, _ensure_token,
use_multiple_accounts flag, offline endpoints (/health, /v1/models,
/v1/accounts auth, /v1/accounts/unblock), login backoff roundtrip,
_acquire_slot_or_499 paths, _ban_403_for_item ban_until=None,
_openai_response shapes.
"""

from __future__ import annotations

import asyncio
import threading

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from deepseaport.accounts import AccountPool, PooledAccount
from deepseaport.config import AccountConfig, Settings
from deepseaport.server import (
    _acquire_slot_or_499,
    _ban_403_for_item,
    _check_auth,
    _ensure_token,
    _headers,
    _is_invalid_token_error,
    _is_transient_overload,
    _login_backoff_set,
    _login_backoff_skip,
    _openai_response,
    _prepare,
    clear_login_backoff,
    create_app,
)


class _FakeState:
    user_agent = "test-ua"
    cookies: dict = {}


class _FakeBridge:
    def __init__(self, cookies: str = "") -> None:
        self.state = _FakeState()
        self._cookies = cookies

    def cookie_header(self) -> str:
        return self._cookies

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


def _tools_body(**kw):
    body = {
        "model": "deepseek-flash",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"function": {"name": "get_time"}}],
    }
    body.update(kw)
    return body


# _check_auth ---------------------------------------------------------------

def test_check_auth_empty_keys_allows_any():
    s = _settings(keys=[])
    _check_auth(s, "")
    _check_auth(s, "Bearer whatever")


def test_check_auth_valid_key_passes():
    s = _settings(keys=["secret"])
    _check_auth(s, "Bearer secret")


def test_check_auth_invalid_key_raises_401():
    s = _settings(keys=["secret"])
    with pytest.raises(HTTPException) as ei:
        _check_auth(s, "Bearer wrong")
    assert ei.value.status_code == 401
    with pytest.raises(HTTPException) as ei2:
        _check_auth(s, "")
    assert ei2.value.status_code == 401


# _prepare ------------------------------------------------------------------

def test_prepare_unknown_model_404():
    with pytest.raises(HTTPException) as ei:
        _prepare({"model": "gpt-99", "messages": [{"role": "user", "content": "hi"}]})
    assert ei.value.status_code == 404


def test_prepare_enable_tools_false_disables():
    s = _settings(enable_tools=False)
    prep = _prepare(_tools_body(), s)
    assert prep["tools"] == []


def test_prepare_tool_choice_none_disables():
    s = _settings()
    prep = _prepare(_tools_body(tool_choice="none"), s)
    assert prep["tools"] == []


def test_prepare_settings_none_keeps_tools():
    prep = _prepare(_tools_body(), None)
    assert len(prep["tools"]) == 1
    assert "get_time" in prep["prompt"]


def test_prepare_settings_none_tool_choice_none_disables():
    prep = _prepare(_tools_body(tool_choice="none"), None)
    assert prep["tools"] == []


# _headers ------------------------------------------------------------------

def test_headers_no_bridge_bearer_only():
    app = create_app(_settings())
    assert not hasattr(app.state, "bridge")
    h = _headers(app, "tok123")
    assert h["Authorization"] == "Bearer tok123"
    assert "Cookie" not in h


def test_headers_with_bridge_cookie_and_ua():
    app = create_app(_settings())
    app.state.bridge = _FakeBridge(cookies="waf=abc")
    h = _headers(app, "tok123")
    assert h["Authorization"] == "Bearer tok123"
    assert h["Cookie"] == "waf=abc"
    assert h["User-Agent"] == "test-ua"


# error classifiers ----------------------------------------------------------

def test_is_invalid_token_error_variants():
    assert _is_invalid_token_error("40003: biz error") is True
    assert _is_invalid_token_error("Authorization Failed (invalid token)") is True
    assert _is_invalid_token_error("AUTHORIZATION FAILED: bad TOKEN") is True
    assert _is_invalid_token_error("user is muted") is False
    assert _is_invalid_token_error("") is False
    assert _is_invalid_token_error(None) is False


def test_is_transient_overload_variants():
    assert _is_transient_overload("generation_timeout: Server busy, try again") is True
    assert _is_transient_overload("server busy") is True
    assert _is_transient_overload("service overloaded") is True
    assert _is_transient_overload("overload protection") is True
    assert _is_transient_overload("user is muted") is False
    assert _is_transient_overload("normal reply text") is False
    assert _is_transient_overload(None) is False


# _ensure_token ---------------------------------------------------------------

def test_ensure_token_normalizes_json_wrapper():
    app = create_app(_settings())
    item = PooledAccount(cfg=AccountConfig(email="a@x.com", token="tok"))
    raw = '{"value":"' + ("y" * 32) + '","__version":"0"}'
    object.__setattr__(item.cfg, "token", raw)
    assert item.cfg.token == raw
    assert _ensure_token(app, item) == "y" * 32
    assert item.cfg.token == "y" * 32


def test_ensure_token_missing_raises_401():
    app = create_app(_settings())
    item = PooledAccount(cfg=AccountConfig(email="a@x.com"))
    with pytest.raises(HTTPException) as ei:
        _ensure_token(app, item)
    assert ei.value.status_code == 401


# use_multiple_accounts flag ---------------------------------------------------

class _FakePool:
    def __init__(self) -> None:
        self.seen: list = []

    async def aacquire(self, timeout, allow_failover=True):
        self.seen.append({"timeout": timeout, "allow_failover": allow_failover})
        return object()


def _chat_post(app, monkeypatch, pool):
    monkeypatch.setattr("deepseaport.server._pool", lambda _app: pool)
    monkeypatch.setattr(
        "deepseaport.server._complete_with_failover_sync",
        lambda *a, **k: {"content": "hi", "thinking": "", "usage_total": 5},
    )
    resp = TestClient(app).post("/v1/chat/completions", json={
        "model": "deepseek-flash", "messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 200, resp.text
    return resp


def test_use_multiple_accounts_false_passes_allow_failover_false(monkeypatch):
    app = create_app(_settings(use_multiple_accounts=False))
    pool = _FakePool()
    _chat_post(app, monkeypatch, pool)
    assert pool.seen and pool.seen[0]["allow_failover"] is False


def test_use_multiple_accounts_defaults_true(monkeypatch):
    app = create_app(_settings())
    pool = _FakePool()
    _chat_post(app, monkeypatch, pool)
    assert pool.seen and pool.seen[0]["allow_failover"] is True


# offline endpoints -------------------------------------------------------------

def test_health_ready_true_shape():
    app = create_app(_settings(accounts=[AccountConfig(email="a@x.com", token="tok")]))
    data = TestClient(app).get("/health").json()
    assert data["ok"] is True
    assert data["ready"] is True
    for key in ("waf", "accounts", "accounts_with_token", "banned", "pool", "current"):
        assert key in data
    assert data["accounts"] == 1 and data["accounts_with_token"] == 1


def test_health_empty_pool_not_ready():
    app = create_app(_settings(accounts=[]))
    data = TestClient(app).get("/health").json()
    assert data["ok"] is True and data["ready"] is False
    assert data["accounts"] == 0


def test_health_tokenless_not_ready():
    app = create_app(_settings(accounts=[AccountConfig(email="a@x.com")]))
    data = TestClient(app).get("/health").json()
    assert data["ready"] is False
    assert data["accounts_with_token"] == 0


def test_health_banned_not_ready():
    app = create_app(_settings(accounts=[
        AccountConfig(email="b@x.com", token="tok", banned=True)]))
    data = TestClient(app).get("/health").json()
    assert data["ready"] is False
    assert data["banned"] == 1


def test_models_lists_four_ids():
    from deepseaport.server import MODELS

    data = TestClient(create_app(_settings())).get("/v1/models").json()
    assert data["object"] == "list"
    assert sorted(m["id"] for m in data["data"]) == sorted(MODELS)
    assert len(data["data"]) == 4


def test_accounts_invalid_key_401():
    app = create_app(_settings(keys=["secret"]))
    r = TestClient(app).get("/v1/accounts",
                            headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 401
    r2 = TestClient(app).get("/v1/accounts")
    assert r2.status_code == 401


def test_unblock_clears_cooldown():
    settings = _settings(accounts=[AccountConfig(email="a@x.com", token="tok")],
                         config_path="")
    app = create_app(settings)
    pool = AccountPool(settings.accounts)
    app.state.pool = pool
    item = pool.get("a@x.com")
    AccountPool.mark_bad(item, seconds=600)
    assert item.cooldown_remaining > 0
    r = TestClient(app).post("/v1/accounts/unblock", json={})
    assert r.status_code == 200, r.text
    assert r.json()["cleared"] >= 1
    assert item.cooldown_remaining == 0


# login backoff ---------------------------------------------------------------

def test_login_backoff_roundtrip():
    key = "gap-test-acct@example.com"
    clear_login_backoff(key)
    try:
        assert _login_backoff_skip(key) is None
        _login_backoff_set(key, "no-token", 120)
        reason = _login_backoff_skip(key)
        assert reason and "no-token" in reason
        assert clear_login_backoff(key) == 1
        assert _login_backoff_skip(key) is None
    finally:
        clear_login_backoff(key)


def test_login_backoff_expired_clears():
    key = "gap-test-expired@example.com"
    clear_login_backoff(key)
    try:
        _login_backoff_set(key, "no-token", 0)
        assert _login_backoff_skip(key) is None
    finally:
        clear_login_backoff(key)


# _acquire_slot_or_499 ----------------------------------------------------------

class _Req:
    def __init__(self, disconnected: bool = False) -> None:
        self._d = disconnected

    async def is_disconnected(self) -> bool:
        return self._d


def test_acquire_timeout_returns_429():
    pool = AccountPool([AccountConfig(email="a@x.com", token="tok")])
    held = pool.acquire(timeout=1)
    try:
        async def scenario():
            with pytest.raises(HTTPException) as ei:
                await _acquire_slot_or_499(pool, _Req(False), threading.Event(),
                                           0.1, True)
            return ei.value.status_code
        assert asyncio.run(scenario()) == 429
    finally:
        AccountPool.release(held)


def test_acquire_empty_pool_returns_503():
    pool = AccountPool([])

    async def scenario():
        with pytest.raises(HTTPException) as ei:
            await _acquire_slot_or_499(pool, _Req(False), threading.Event(),
                                       0.1, True)
        return ei.value.status_code

    assert asyncio.run(scenario()) == 503


# _ban_403_for_item ---------------------------------------------------------------

def test_ban_403_none_expiry_still_marks():
    settings = _settings(accounts=[AccountConfig(email="b@x.com", token="tok")],
                         config_path="")
    app = create_app(settings)
    item = PooledAccount(cfg=settings.accounts[0])
    exc = _ban_403_for_item(app, item, None, "something went wrong")
    assert exc.status_code == 403
    assert item.cfg.banned is True
    low = str(exc.detail).lower()
    assert "banned" in low or "muted" in low


def test_ban_403_keeps_muted_evidence():
    settings = _settings(accounts=[AccountConfig(email="b@x.com", token="tok")],
                         config_path="")
    app = create_app(settings)
    item = PooledAccount(cfg=settings.accounts[0])
    exc = _ban_403_for_item(app, item, None, "user is muted until Friday")
    assert exc.status_code == 403
    assert "muted" in str(exc.detail).lower()


# _openai_response ---------------------------------------------------------------

def _base_prep(**kw):
    prep = {"model": "deepseek-flash", "prompt": "hello world test prompt",
            "tools": [], "thinking": False, "search": False, "model_type": "default"}
    prep.update(kw)
    return prep


def test_openai_response_usage_and_thinking():
    prep = _base_prep()
    out = _openai_response(prep, {"content": "hi there", "thinking": "hmm",
                                  "usage_total": 7})
    msg = out["choices"][0]["message"]
    assert msg["content"] == "hi there"
    assert msg["reasoning_content"] == "hmm"
    assert out["choices"][0]["finish_reason"] == "stop"
    assert out["usage"] == {"prompt_tokens": out["usage"]["prompt_tokens"],
                            "completion_tokens": 7, "total_tokens": out["usage"]["total_tokens"]}
    assert out["usage"]["total_tokens"] == out["usage"]["prompt_tokens"] + 7


def test_openai_response_tool_calls_shape(monkeypatch):
    calls = [{"id": "call_1", "type": "function",
              "function": {"name": "get_time", "arguments": "{}"}}]
    monkeypatch.setattr("deepseaport.server.parse_tool_calls",
                        lambda content, tools: (calls, ""))
    prep = _base_prep(tools=[{"function": {"name": "get_time"}}])
    out = _openai_response(prep, {"content": "{}", "thinking": "",
                                  "usage_total": 3})
    msg = out["choices"][0]["message"]
    assert msg["tool_calls"] == calls
    assert out["choices"][0]["finish_reason"] == "tool_calls"
