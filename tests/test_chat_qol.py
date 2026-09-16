"""QoL tests: port auto-fix + server-with-chat helpers + menu routing."""

import json
import socket
import sys
import types
import urllib.error


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_is_port_free_true_then_false():
    from deepseaport.cli import _is_port_free

    port = _free_port()
    assert _is_port_free("127.0.0.1", port) is True
    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    holder.bind(("127.0.0.1", port))
    holder.listen(1)
    try:
        assert _is_port_free("127.0.0.1", port) is False
    finally:
        holder.close()
    assert _is_port_free("127.0.0.1", port) is True


def test_next_free_port_skips_busy():
    from deepseaport.cli import _is_port_free, _next_free_port

    port = _free_port()
    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    holder.bind(("127.0.0.1", port))
    holder.listen(1)
    try:
        nxt = _next_free_port("127.0.0.1", port, limit=10)
        assert nxt is not None and nxt != port
        assert _is_port_free("127.0.0.1", nxt) is True
    finally:
        holder.close()


def test_ensure_free_port_passthrough_when_free():
    from deepseaport.cli import _ensure_free_port
    from deepseaport.config import Settings

    port = _free_port()
    s = Settings(port=port, config_path="")
    assert _ensure_free_port(s, "127.0.0.1", port) == port
    assert s.port == port


def test_ensure_free_port_interactive_accept(monkeypatch):
    from deepseaport.cli import _ensure_free_port
    from deepseaport.config import Settings

    port = _free_port()
    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    holder.bind(("127.0.0.1", port))
    holder.listen(1)
    try:
        s = Settings(port=port, config_path="")
        monkeypatch.setattr(sys, "stdin",
                            types.SimpleNamespace(isatty=lambda: True))
        monkeypatch.setattr("builtins.input", lambda *a, **k: "")
        nxt = _ensure_free_port(s, "127.0.0.1", port)
        assert nxt != port and s.port == nxt
    finally:
        holder.close()


def test_ensure_free_port_interactive_decline(monkeypatch):
    from deepseaport.cli import _ensure_free_port
    from deepseaport.config import Settings

    port = _free_port()
    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    holder.bind(("127.0.0.1", port))
    holder.listen(1)
    try:
        s = Settings(port=port, config_path="")
        monkeypatch.setattr(sys, "stdin",
                            types.SimpleNamespace(isatty=lambda: True))
        monkeypatch.setattr("builtins.input", lambda *a, **k: "n")
        assert _ensure_free_port(s, "127.0.0.1", port) == port
        assert s.port == port
    finally:
        holder.close()


def test_ensure_free_port_noninteractive_autopicks_and_saves(tmp_path):
    from deepseaport.cli import _ensure_free_port
    from deepseaport.config import Settings, load_settings

    port = _free_port()
    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    holder.bind(("127.0.0.1", port))
    holder.listen(1)
    try:
        cfg = tmp_path / "c.json"
        cfg.write_text(json.dumps({"port": port}), encoding="utf-8")
        s = load_settings(str(cfg))
        assert s.port == port
        nxt = _ensure_free_port(s, "127.0.0.1", port)
        assert nxt != port and s.port == nxt
        assert load_settings(str(cfg)).port == nxt
    finally:
        holder.close()


def test_chat_build_payload_shape():
    from deepseaport.chat_ui import _build_payload

    msgs = [{"role": "user", "content": "hi"}]
    assert _build_payload("deepseek-flash", msgs) == {
        "model": "deepseek-flash", "messages": msgs, "stream": False}


def test_chat_resolve_bind_host():
    from deepseaport.chat_ui import _resolve_bind_host
    from deepseaport.config import Settings

    assert _resolve_bind_host(Settings(listen=False)) == "127.0.0.1"
    assert _resolve_bind_host(Settings(listen=True)) == "0.0.0.0"
    args = types.SimpleNamespace(host=" 1.2.3.4 ")
    assert _resolve_bind_host(Settings(listen=False), args) == "1.2.3.4"


class _FakeResp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status = status

    def read(self):
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_chat_post_completion_ok(monkeypatch):
    from deepseaport import chat_ui

    payload = {"choices": [{"message": {"role": "assistant", "content": "hello"}}]}
    monkeypatch.setattr("urllib.request.urlopen",
                        lambda *a, **k: _FakeResp(payload))
    assert chat_ui._post_completion("http://127.0.0.1:9", "", "m", []) == "hello"


def test_chat_post_completion_thinking_fallback(monkeypatch):
    from deepseaport import chat_ui

    payload = {"choices": [{"message": {"role": "assistant", "content": "",
                                        "reasoning_content": "hmm"}}]}
    monkeypatch.setattr("urllib.request.urlopen",
                        lambda *a, **k: _FakeResp(payload))
    assert chat_ui._post_completion("http://127.0.0.1:9", "", "m", []) == "[thinking]\nhmm"


def test_chat_post_completion_http_error(monkeypatch):
    import io

    from deepseaport import chat_ui

    def _boom(*a, **k):
        raise urllib.error.HTTPError("http://x", 429, "busy",
                                     {}, io.BytesIO(b'{"detail":"rate limited"}'))

    monkeypatch.setattr("urllib.request.urlopen", _boom)
    try:
        chat_ui._post_completion("http://127.0.0.1:9", "", "m", [])
    except RuntimeError as exc:
        assert "429" in str(exc) and "rate limited" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_chat_post_completion_bad_shape(monkeypatch):
    from deepseaport import chat_ui

    monkeypatch.setattr("urllib.request.urlopen",
                        lambda *a, **k: _FakeResp({"weird": 1}))
    try:
        chat_ui._post_completion("http://127.0.0.1:9", "", "m", [])
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError")


def test_chat_wait_for_server_ok_and_timeout(monkeypatch):
    from deepseaport import chat_ui

    monkeypatch.setattr("urllib.request.urlopen",
                        lambda *a, **k: _FakeResp({"ok": True}))
    assert chat_ui._wait_for_server("http://127.0.0.1:9", timeout=2) is True

    def _down(*a, **k):
        raise ConnectionRefusedError("down")

    monkeypatch.setattr("urllib.request.urlopen", _down)
    assert chat_ui._wait_for_server("http://127.0.0.1:9", timeout=1) is False


def test_chat_silence_hides_info_and_restores():
    import logging

    from deepseaport import chat_ui

    lg = logging.getLogger("deepseaport.obscura")
    root = logging.getLogger()
    old_level, old_root = lg.level, root.level
    lg.setLevel(logging.INFO)
    root.setLevel(logging.INFO)
    try:
        saved = chat_ui._silence_chat_logs()
        assert logging.getLogger("deepseaport.obscura").level == logging.ERROR
        assert logging.getLogger("uvicorn.access").level == logging.ERROR
        assert root.level == logging.ERROR
        assert lg.isEnabledFor(logging.INFO) is False
        chat_ui._restore_chat_logs(saved)
        assert logging.getLogger("deepseaport.obscura").level == logging.INFO
        assert root.level == logging.INFO
    finally:
        lg.setLevel(old_level)
        root.setLevel(old_root)


def test_preflight_blocks_empty_and_tokenless():
    from deepseaport.config import AccountConfig, Settings
    from deepseaport.tui import _preflight_reason

    assert _preflight_reason(Settings(accounts=[])) != ""
    reason = _preflight_reason(
        Settings(accounts=[AccountConfig(email="a@x.com", token="")]))
    assert reason != "" and "token" in reason.lower()
    assert _preflight_reason(
        Settings(accounts=[AccountConfig(email="a@x.com", token="t1")])) == ""


def test_main_menu_guards_start_when_no_accounts(monkeypatch):
    from deepseaport.config import Settings
    from deepseaport.tui import main_menu

    s = Settings(accounts=[], active_account="", config_path="")
    calls = {"serve": 0}
    # "1" blocked -> jumps to Accounts ("b" backs out), then quit.
    inputs = iter(["1", "b", "q"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(inputs))
    assert main_menu(s, lambda st: calls.__setitem__("serve", 1) or 0) == 0
    assert calls["serve"] == 0


def test_main_menu_guards_chat_when_tokenless(monkeypatch):
    from deepseaport.config import AccountConfig, Settings
    from deepseaport.tui import main_menu

    s = Settings(accounts=[AccountConfig(email="a@x.com", token="")],
                 active_account="", config_path="")
    calls = {"chat": 0}
    inputs = iter(["2", "b", "q"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(inputs))
    rc = main_menu(s, lambda st: 0, lambda st: calls.__setitem__("chat", 1) or 0)
    assert rc == 0 and calls["chat"] == 0


def test_main_menu_starts_when_token_present(monkeypatch):
    from deepseaport.config import AccountConfig, Settings
    from deepseaport.tui import main_menu

    s = Settings(accounts=[AccountConfig(email="a@x.com", token="t1")],
                 active_account="", config_path="")
    calls = {"serve": 0}
    inputs = iter(["1", "q"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(inputs))
    assert main_menu(s, lambda st: calls.__setitem__("serve", 1) or 0) == 0
    assert calls["serve"] == 1


def test_chat_model_memory_roundtrip(tmp_path):
    from deepseaport.config import load_settings

    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"chat_model": "deepseek-flash-search"}), encoding="utf-8")
    assert load_settings(str(cfg)).chat_model == "deepseek-flash-search"
    # Unknown file content falls back to default, never empty.
    cfg.write_text(json.dumps({}), encoding="utf-8")
    assert load_settings(str(cfg)).chat_model == "deepseek-flash"


def test_chat_initial_model_prefers_stored():
    from deepseaport import chat_ui
    from deepseaport.config import Settings

    models = ["deepseek-flash", "deepseek-flash-search"]
    assert chat_ui._initial_chat_model(
        Settings(chat_model="deepseek-flash-search"), models) == "deepseek-flash-search"
    assert chat_ui._initial_chat_model(
        Settings(chat_model="nope"), models) == "deepseek-flash"
    assert chat_ui._initial_chat_model(Settings(chat_model=""), models) == "deepseek-flash"


def test_chat_remember_model_saves(tmp_path):
    from deepseaport import chat_ui
    from deepseaport.config import Settings, load_settings

    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({}), encoding="utf-8")
    s = load_settings(str(cfg))
    chat_ui._remember_chat_model(s, "deepseek-flash-search")
    assert s.chat_model == "deepseek-flash-search"
    assert load_settings(str(cfg)).chat_model == "deepseek-flash-search"


def test_main_menu_routes_to_chat(monkeypatch):
    from deepseaport.config import AccountConfig, Settings
    from deepseaport.tui import main_menu

    s = Settings(accounts=[AccountConfig(email="a@x.com", token="t1")],
                 active_account="", config_path="")
    calls = {"serve": 0, "chat": 0}
    inputs = iter(["2", "q"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(inputs))
    rc = main_menu(s, lambda st: calls.__setitem__("serve", 1) or 0,
                   lambda st: calls.__setitem__("chat", 1) or 0)
    assert rc == 0 and calls == {"serve": 0, "chat": 1}


def test_main_menu_serve_still_first(monkeypatch):
    from deepseaport.config import AccountConfig, Settings
    from deepseaport.tui import main_menu

    s = Settings(accounts=[AccountConfig(email="a@x.com", token="t1")],
                 active_account="", config_path="")
    calls = {"serve": 0, "chat": 0}
    inputs = iter(["1", "q"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(inputs))
    rc = main_menu(s, lambda st: calls.__setitem__("serve", 1) or 0,
                   lambda st: calls.__setitem__("chat", 1) or 0)
    assert rc == 0 and calls == {"serve": 1, "chat": 0}


def test_settings_menu_groups_options_and_dispatches(monkeypatch, capsys):
    from deepseaport.config import Settings
    from deepseaport.tui import settings_menu

    s = Settings(config_path="", stream_mode="live")
    inputs = iter(["7", "q"])  # 7 = Reply streaming, q = back
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(inputs))
    settings_menu(s)

    out = capsys.readouterr().out
    assert "Request handling" in out
    assert "Chat & logs" in out
    assert "Server access" in out
    assert "Setup & defaults" in out
    assert "Accounts" in out
    assert s.stream_mode == "buffered"


def test_main_menu_flags_incomplete_setup(monkeypatch, capsys):
    from deepseaport.config import Settings
    from deepseaport.tui import main_menu

    s = Settings(accounts=[], active_account="", config_path="")
    monkeypatch.setattr("builtins.input", lambda *a, **k: "q")
    assert main_menu(s, lambda st: 0) == 0
    out = capsys.readouterr().out
    assert "no accounts yet" in out
    assert "Accounts (3)" in out


def test_load_settings_defaults_browser_bin_to_auto(tmp_path, monkeypatch):
    from deepseaport.config import load_settings

    monkeypatch.delenv("DEEPSEAPORT_BROWSER_BIN", raising=False)
    cfg = tmp_path / "c.json"
    cfg.write_text("{}", encoding="utf-8")
    assert load_settings(str(cfg)).browser_bin == "auto"
