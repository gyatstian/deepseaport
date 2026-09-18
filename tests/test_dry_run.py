"""Offline tests for dry-run mode (no network, no credentials, no browser).

Covers: dry app health/models, chat echo (raw + formatted print, prep JSON
response, append_top/bottom wiring, stream echo), auth parity, _prepare
error parity (400/404), and main-menu Dry run routing without preflight.
"""

from __future__ import annotations

import inspect

from fastapi.testclient import TestClient

from deepseaport import tui
from deepseaport.config import Settings
from deepseaport.dry_run import create_dry_app
from deepseaport.tui import main_menu


def _settings(**kw) -> Settings:
    kw.setdefault("config_path", "")
    return Settings(**kw)


def test_dry_health_and_models():
    c = TestClient(create_dry_app(_settings()))
    assert c.get("/health").json() == {"ok": True, "dry_run": True}
    ids = [m["id"] for m in c.get("/v1/models").json()["data"]]
    assert "deepseek-flash" in ids


def _content(data: dict) -> str:
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["finish_reason"] == "stop"
    return data["choices"][0]["message"]["content"]


def test_dry_echo_prints_raw_and_formatted(capsys):
    s = _settings(append_top="HDR", append_bottom="FTR")
    c = TestClient(create_dry_app(s))
    resp = c.post("/v1/chat/completions", json={
        "model": "deepseek-flash",
        "messages": [{"role": "user", "content": "hello"}],
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["dry_run"] is True
    content = _content(data)
    assert "HDR\n\nhello\n\nFTR" in content
    assert data["usage"]["total_tokens"] > 0
    out = capsys.readouterr().out
    assert "dry run #1" in out
    assert "NOT sent to DeepSeek" in out
    assert "(this gets sent to deepseek chat)" in out
    assert "HDR\n\nhello\n\nFTR" in out
    assert content in out  # frontend gets exactly what the terminal prints


def test_dry_multiline_appends_preserved():
    s = _settings(append_top="L1\nL2", append_bottom="F1\nF2\nF3")
    c = TestClient(create_dry_app(s))
    content = _content(c.post("/v1/chat/completions", json={
        "model": "deepseek-flash",
        "messages": [{"role": "user", "content": "hi"}],
    }).json())
    assert "L1\nL2\n\nhi\n\nF1\nF2\nF3" in content


def test_dry_stream_returns_sse_with_same_content():
    import json as _json

    c = TestClient(create_dry_app(_settings(append_top="HDR")))
    resp = c.post("/v1/chat/completions", json={
        "model": "deepseek-flash", "stream": True,
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    assert resp.text.rstrip().endswith("data: [DONE]")
    deltas = []
    saw_role = False
    for line in resp.text.splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        evt = _json.loads(line[len("data: "):])
        delta = evt["choices"][0]["delta"]
        if delta.get("role") == "assistant":
            saw_role = True
        deltas.append(delta.get("content", ""))
    assert saw_role
    assert "HDR\n\nhi" in "".join(deltas)


def test_dry_tools_inject_system_prompt():
    tools = [{"type": "function", "function": {
        "name": "read", "description": "d",
        "parameters": {"type": "object", "properties": {}}}}]
    c = TestClient(create_dry_app(_settings()))
    content = _content(c.post("/v1/chat/completions", json={
        "model": "deepseek-flash",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": tools,
    }).json())
    assert "Tool: read" in content


def test_dry_auth_parity():
    s = _settings(keys=["sk-test"])
    c = TestClient(create_dry_app(s), raise_server_exceptions=False)
    body = {"model": "deepseek-flash",
            "messages": [{"role": "user", "content": "hi"}]}
    assert c.post("/v1/chat/completions", json=body).status_code == 401
    ok = c.post("/v1/chat/completions", json=body,
                headers={"Authorization": "Bearer sk-test"})
    assert ok.status_code == 200
    assert "\nhi\n" in _content(ok.json())


def test_dry_prepare_error_parity():
    c = TestClient(create_dry_app(_settings()), raise_server_exceptions=False)
    msgs = {"messages": [{"role": "user", "content": "hi"}]}
    assert c.post("/v1/chat/completions",
                  json={"model": "gpt-99", **msgs}).status_code == 404
    assert c.post("/v1/chat/completions",
                  json={"model": "deepseek-flash", "messages": []}).status_code == 400


def test_dry_frontend_off_sends_note_but_prints_full(capsys):
    s = _settings(append_top="HDR", send_dry_run_to_frontend=False)
    c = TestClient(create_dry_app(s))
    content = _content(c.post("/v1/chat/completions", json={
        "model": "deepseek-flash",
        "messages": [{"role": "user", "content": "hello"}],
    }).json())
    assert "HDR" not in content
    assert "server terminal only" in content
    out = capsys.readouterr().out
    assert "HDR\n\nhello" in out
    assert "(this gets sent to deepseek chat)" in out


def test_dry_frontend_flag_roundtrip_and_env(tmp_path, monkeypatch):
    import json

    from deepseaport.config import load_settings

    for name in ("DEEPSEAPORT_SEND_DRY_RUN_TO_FRONTEND",):
        monkeypatch.delenv(name, raising=False)
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"send_dry_run_to_frontend": False}), encoding="utf-8")
    assert load_settings(str(cfg)).send_dry_run_to_frontend is False
    assert _settings().send_dry_run_to_frontend is True  # default
    cfg.write_text(json.dumps({}), encoding="utf-8")
    monkeypatch.setenv("DEEPSEAPORT_SEND_DRY_RUN_TO_FRONTEND", "0")
    assert load_settings(str(cfg)).send_dry_run_to_frontend is False


def test_main_menu_dry_routing_no_chat(monkeypatch):
    s = _fresh_no_accounts()
    called = {"n": 0}
    inputs = iter(["4", "q"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(inputs))
    rc = main_menu(s, lambda st: 0,
                   dry_fn=lambda st: called.__setitem__("n", 1) or 0)
    assert rc == 0 and called["n"] == 1


def test_main_menu_dry_routing_with_chat(monkeypatch):
    s = _fresh_no_accounts()
    called = {"n": 0}
    inputs = iter(["6", "q"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(inputs))
    rc = main_menu(s, lambda st: 0, lambda st: 0, lambda st: 0,
                   lambda st: called.__setitem__("n", 1) or 0)
    assert rc == 0 and called["n"] == 1


def test_main_menu_dry_needs_no_accounts(monkeypatch):
    """Empty pool must not block dry run (no preflight, no browser)."""
    s = _settings(accounts=[])
    called = {"n": 0}
    inputs = iter(["4", "q"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(inputs))
    monkeypatch.setattr(tui, "accounts_menu",
                        lambda st: (_ for _ in ()).throw(AssertionError("must not preflight")))
    rc = main_menu(s, lambda st: 0,
                   dry_fn=lambda st: called.__setitem__("n", 1) or 0)
    assert rc == 0 and called["n"] == 1


def test_main_menu_signature_has_dry_fn():
    assert "dry_fn" in inspect.signature(main_menu).parameters


def _fresh_no_accounts() -> Settings:
    return Settings(config_path="")
