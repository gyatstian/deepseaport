"""TUI gaps: helpers, settings_menu dispatch, accounts_menu, main_menu."""

import inspect
import sys
import types

import pytest

from deepseaport import tui
from deepseaport.config import AccountConfig, Settings
from deepseaport.tui import (
    _ask,
    _c,
    _edit_chat_model,
    _edit_choice,
    _edit_keys,
    _edit_obscura_bin,
    _edit_port,
    _edit_retries,
    _err,
    _has_token,
    _header,
    _info,
    _mask_key,
    _ok,
    _on_off,
    _preflight_reason,
    _refresh_from_disk,
    _section,
    _setting_line,
    _toggle_bool,
    _toggle_stream,
    _use_color,
    accounts_menu,
    main_menu,
    settings_menu,
)


def _fresh(**kw):
    kw.setdefault("config_path", "")
    return Settings(**kw)


# --- helpers ---

def test_on_off():
    assert _on_off(True) == "ON"
    assert _on_off(False) == "OFF"


def test_use_color_respects_no_color(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    assert _use_color() is False
    # empty value still counts as set
    monkeypatch.setenv("NO_COLOR", "")
    assert _use_color() is False


def test_use_color_tty(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setattr(sys, "stdout", types.SimpleNamespace(isatty=lambda: True))
    assert _use_color() is True
    monkeypatch.setattr(sys, "stdout", types.SimpleNamespace(isatty=lambda: False))
    assert _use_color() is False


def test_c_color_vs_no_color(monkeypatch):
    monkeypatch.setattr(tui, "_use_color", lambda: False)
    assert _c("hi", "31") == "hi"
    monkeypatch.setattr(tui, "_use_color", lambda: True)
    out = _c("hi", "31")
    assert "\x1b[31m" in out and out.endswith("\x1b[0m")


def test_mask_key():
    assert _mask_key("") == "***"
    assert _mask_key("12345678") == "***"
    assert _mask_key("1234567") == "***"
    assert _mask_key("  123  ") == "***"
    long_key = "1234567890"
    assert _mask_key(long_key) == "1234***90"
    assert _mask_key("abcdefghij-long-key") == "abcd***ey"


def test_ask_strips(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *a, **k: "  hi  ")
    assert _ask("Prompt") == "hi"


def test_ask_eof_raises(monkeypatch):
    # impl re-raises EOF (callers treat as cancel); no default-return.
    def _boom(*a, **k):
        raise EOFError
    monkeypatch.setattr("builtins.input", _boom)
    with pytest.raises(EOFError):
        _ask("Prompt", default="dflt")


def test_ask_keyboard_interrupt_raises(monkeypatch):
    def _boom(*a, **k):
        raise KeyboardInterrupt
    monkeypatch.setattr("builtins.input", _boom)
    with pytest.raises(KeyboardInterrupt):
        _ask("Prompt")


def test_has_token():
    s = _fresh(accounts=[AccountConfig(email="a@x.com", token="t1"),
                         AccountConfig(email="b@x.com", token="")])
    assert _has_token(s, "a@x.com") is True
    assert _has_token(s, "b@x.com") is False
    assert _has_token(s, "missing@x.com") is False
    assert _has_token(_fresh(), "a@x.com") is False


def test_toggle_bool_flips_and_saves():
    s = _fresh(enable_tools=True)
    calls = {"n": 0}
    orig = s.save
    s.save = lambda: calls.__setitem__("n", calls["n"] + 1) or orig()
    _toggle_bool(s, "enable_tools", "Tool calling")
    assert s.enable_tools is False
    assert calls["n"] == 1
    _toggle_bool(s, "enable_tools", "Tool calling")
    assert s.enable_tools is True
    assert calls["n"] == 2


def test_toggle_stream():
    s = _fresh(stream_mode="live")
    _toggle_stream(s)
    assert s.stream_mode == "buffered"
    _toggle_stream(s)
    assert s.stream_mode == "live"


def test_edit_choice_invalid_keeps(monkeypatch):
    s = _fresh(log_level="INFO")
    monkeypatch.setattr("builtins.input", lambda *a, **k: "bogus-choice")
    _edit_choice(s, "log_level", "Log level", ("DEBUG", "INFO", "WARNING", "ERROR"))
    assert s.log_level == "INFO"


def test_edit_choice_out_of_range_keeps(monkeypatch):
    s = _fresh(log_level="INFO")
    monkeypatch.setattr("builtins.input", lambda *a, **k: "99")
    _edit_choice(s, "log_level", "Log level", ("DEBUG", "INFO", "WARNING", "ERROR"))
    assert s.log_level == "INFO"


def test_edit_choice_cancel_keeps(monkeypatch):
    s = _fresh(log_level="INFO")
    monkeypatch.setattr("builtins.input", lambda *a, **k: "")
    _edit_choice(s, "log_level", "Log level", ("DEBUG", "INFO", "WARNING", "ERROR"))
    assert s.log_level == "INFO"


def test_edit_choice_valid_number_changes(monkeypatch):
    s = _fresh(log_level="INFO")
    monkeypatch.setattr("builtins.input", lambda *a, **k: "1")
    _edit_choice(s, "log_level", "Log level", ("DEBUG", "INFO", "WARNING", "ERROR"))
    assert s.log_level == "DEBUG"


def test_edit_choice_valid_text_changes(monkeypatch):
    s = _fresh(log_level="INFO")
    monkeypatch.setattr("builtins.input", lambda *a, **k: "error")
    _edit_choice(s, "log_level", "Log level", ("DEBUG", "INFO", "WARNING", "ERROR"))
    assert s.log_level == "ERROR"


def test_edit_retries_nan_keeps(monkeypatch):
    s = _fresh(max_retries=1)
    monkeypatch.setattr("builtins.input", lambda *a, **k: "abc")
    _edit_retries(s)
    assert s.max_retries == 1


def test_edit_retries_range_keeps(monkeypatch):
    s = _fresh(max_retries=1)
    monkeypatch.setattr("builtins.input", lambda *a, **k: "99")
    _edit_retries(s)
    assert s.max_retries == 1
    monkeypatch.setattr("builtins.input", lambda *a, **k: "-1")
    _edit_retries(s)
    assert s.max_retries == 1


def test_edit_retries_cancel_keeps(monkeypatch):
    s = _fresh(max_retries=1)
    monkeypatch.setattr("builtins.input", lambda *a, **k: "")
    _edit_retries(s)
    assert s.max_retries == 1


def test_edit_retries_valid_changes(monkeypatch):
    s = _fresh(max_retries=1)
    monkeypatch.setattr("builtins.input", lambda *a, **k: "3")
    _edit_retries(s)
    assert s.max_retries == 3


def test_edit_port_cancel_keeps(monkeypatch):
    s = _fresh(port=5001)
    monkeypatch.setattr("builtins.input", lambda *a, **k: "")
    _edit_port(s)
    assert s.port == 5001


def test_edit_port_nan_keeps(monkeypatch):
    s = _fresh(port=5001)
    monkeypatch.setattr("builtins.input", lambda *a, **k: "abc")
    _edit_port(s)
    assert s.port == 5001


def test_edit_port_range_keeps(monkeypatch):
    s = _fresh(port=5001)
    monkeypatch.setattr("builtins.input", lambda *a, **k: "99999")
    _edit_port(s)
    assert s.port == 5001
    monkeypatch.setattr("builtins.input", lambda *a, **k: "0")
    _edit_port(s)
    assert s.port == 5001


def test_edit_port_valid_with_mock(monkeypatch):
    s = _fresh(port=5001)
    monkeypatch.setattr("builtins.input", lambda *a, **k: "5002")
    monkeypatch.setattr("deepseaport.cli._resolve_bind_host", lambda *a, **k: "127.0.0.1")
    monkeypatch.setattr("deepseaport.cli._ensure_free_port", lambda st, h, p: int(p))
    _edit_port(s)
    assert s.port == 5002


def test_edit_keys_add(monkeypatch):
    s = _fresh(keys=[])
    inputs = iter(["a", "b"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(inputs))
    monkeypatch.setattr("getpass.getpass", lambda *a, **k: "newkey123456789")
    _edit_keys(s)
    assert s.keys == ["newkey123456789"]


def test_edit_keys_add_dupe_keeps_count(monkeypatch):
    s = _fresh(keys=["dupkey123456789"])
    inputs = iter(["a", "b"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(inputs))
    monkeypatch.setattr("getpass.getpass", lambda *a, **k: "dupkey123456789")
    _edit_keys(s)
    assert s.keys == ["dupkey123456789"]


def test_edit_keys_add_cancel_keeps(monkeypatch):
    s = _fresh(keys=[])
    inputs = iter(["a", "b"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(inputs))
    monkeypatch.setattr("getpass.getpass", lambda *a, **k: "   ")
    _edit_keys(s)
    assert s.keys == []


def test_edit_keys_delete_confirm_y(monkeypatch):
    s = _fresh(keys=["k1-long-enough-123", "k2-long-enough-456"])
    inputs = iter(["d 1", "y", "b"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(inputs))
    _edit_keys(s)
    assert s.keys == ["k2-long-enough-456"]


def test_edit_keys_delete_confirm_n_keeps(monkeypatch):
    s = _fresh(keys=["k1-long-enough-123", "k2-long-enough-456"])
    inputs = iter(["d 1", "n", "b"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(inputs))
    _edit_keys(s)
    assert s.keys == ["k1-long-enough-123", "k2-long-enough-456"]


def test_edit_keys_back_immediately(monkeypatch):
    s = _fresh(keys=["k1-long-enough-123"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: "b")
    _edit_keys(s)
    assert s.keys == ["k1-long-enough-123"]


def test_edit_obscura_bin_invalid_keeps(monkeypatch):
    s = _fresh(obscura_bin="")
    monkeypatch.setattr("builtins.input", lambda *a, **k: "/definitely/not/exist/xyz123")
    _edit_obscura_bin(s)
    assert s.obscura_bin == ""


def test_edit_obscura_bin_clear_resets(monkeypatch):
    s = _fresh(obscura_bin="/tmp/some.exe")
    monkeypatch.setattr("builtins.input", lambda *a, **k: "clear")
    _edit_obscura_bin(s)
    assert s.obscura_bin == ""


def test_edit_chat_model_invalid_keeps(monkeypatch):
    s = _fresh(chat_model="deepseek-flash")
    monkeypatch.setattr("builtins.input", lambda *a, **k: "bogus-model-xyz")
    _edit_chat_model(s)
    assert s.chat_model == "deepseek-flash"


def test_edit_chat_model_out_of_range_keeps(monkeypatch):
    s = _fresh(chat_model="deepseek-flash")
    monkeypatch.setattr("builtins.input", lambda *a, **k: "99")
    _edit_chat_model(s)
    assert s.chat_model == "deepseek-flash"


def test_preflight_empty_tokenless_ok():
    assert _preflight_reason(_fresh(accounts=[])) != ""
    r = _preflight_reason(_fresh(accounts=[AccountConfig(email="a@x.com", token="")]))
    assert r != "" and "token" in r.lower()
    assert _preflight_reason(_fresh(accounts=[AccountConfig(email="a@x.com", token="t1")])) == ""
    # impl has no banned-only branch: token present => ready even if banned flag set
    b = AccountConfig(email="a@x.com", token="t1")
    b.banned = True
    assert _preflight_reason(_fresh(accounts=[b])) == ""


def test_refresh_from_disk_missing_no_raise(tmp_path):
    s = Settings(active_account="keep", config_path=str(tmp_path / "missing.json"))
    _refresh_from_disk(s)  # must not raise
    # missing file reloads to defaults (clears stale selection), never raises
    assert s.active_account == ""
    s2 = _fresh()
    _refresh_from_disk(s2)  # empty path no-op
    assert s2.active_account == ""


def test_refresh_from_disk_reloads_active_account(tmp_path):
    import json
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({
        "accounts": [{"email": "a@x.com", "token": "t1"},
                     {"email": "b@x.com", "token": "t2"}],
        "active_account": "a@x.com",
    }), encoding="utf-8")
    s = Settings(accounts=[], active_account="", config_path=str(cfg))
    _refresh_from_disk(s)
    assert s.active_account == "a@x.com"
    assert {a.email for a in s.accounts} == {"a@x.com", "b@x.com"}
    cfg.write_text(json.dumps({
        "accounts": [{"email": "a@x.com", "token": "t1"},
                     {"email": "b@x.com", "token": "t2"}],
        "active_account": "b@x.com",
    }), encoding="utf-8")
    _refresh_from_disk(s)
    assert s.active_account == "b@x.com"


def test_display_helpers_smoke(capsys):
    _header("title")
    _section("sec")
    _setting_line(1, "Label", "val")
    _ok("okmsg")
    _info("infomsg")
    _err("errmsg")
    out = capsys.readouterr().out
    assert "title" in out and "Label" in out


# --- settings_menu dispatch ---

def test_settings_menu_toggle_enable_tools(monkeypatch):
    s = _fresh(enable_tools=True)
    inputs = iter(["1", "q"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(inputs))
    settings_menu(s)
    assert s.enable_tools is False


def test_settings_menu_toggle_warmup(monkeypatch):
    s = _fresh(warmup_on_startup=True)
    inputs = iter(["2", "q"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(inputs))
    settings_menu(s)
    assert s.warmup_on_startup is False


def test_settings_menu_toggle_auto_delete(monkeypatch):
    s = _fresh(auto_delete_session=True)
    inputs = iter(["3", "q"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(inputs))
    settings_menu(s)
    assert s.auto_delete_session is False


def test_settings_menu_retries_invalid_keeps(monkeypatch):
    s = _fresh(max_retries=1)
    inputs = iter(["4", "notanumber", "q"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(inputs))
    settings_menu(s)
    assert s.max_retries == 1


def test_settings_menu_retries_valid_changes(monkeypatch):
    s = _fresh(max_retries=1)
    inputs = iter(["4", "2", "q"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(inputs))
    settings_menu(s)
    assert s.max_retries == 2


def test_settings_menu_toggle_parallel(monkeypatch):
    s = _fresh(parallel_challenge_fetch=True)
    inputs = iter(["5", "q"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(inputs))
    settings_menu(s)
    assert s.parallel_challenge_fetch is False


def test_settings_menu_log_level_invalid_keeps(monkeypatch):
    s = _fresh(log_level="INFO")
    inputs = iter(["6", "bogus", "q"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(inputs))
    settings_menu(s)
    assert s.log_level == "INFO"


def test_settings_menu_log_level_valid_changes(monkeypatch):
    s = _fresh(log_level="INFO")
    inputs = iter(["6", "DEBUG", "q"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(inputs))
    settings_menu(s)
    assert s.log_level == "DEBUG"


def test_settings_menu_toggle_stream(monkeypatch):
    s = _fresh(stream_mode="live")
    inputs = iter(["7", "q"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(inputs))
    settings_menu(s)
    assert s.stream_mode == "buffered"


def test_settings_menu_toggle_listen(monkeypatch):
    s = _fresh(listen=False)
    inputs = iter(["8", "q"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(inputs))
    settings_menu(s)
    assert s.listen is True


def test_settings_menu_port_routed(monkeypatch):
    s = _fresh()
    called = {"n": 0}

    def _fake(st):
        called["n"] += 1

    monkeypatch.setattr(tui, "_edit_port", _fake)
    inputs = iter(["9", "q"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(inputs))
    settings_menu(s)
    assert called["n"] == 1


def test_settings_menu_keys_routed(monkeypatch):
    s = _fresh()
    called = {"n": 0}

    def _fake(st):
        called["n"] += 1

    monkeypatch.setattr(tui, "_edit_keys", _fake)
    inputs = iter(["10", "q"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(inputs))
    settings_menu(s)
    assert called["n"] == 1


def test_settings_menu_obscura_invalid_keeps(monkeypatch):
    s = _fresh(obscura_bin="")
    inputs = iter(["11", "/no/such/path/xyz123", "q"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(inputs))
    settings_menu(s)
    assert s.obscura_bin == ""


def test_settings_menu_chat_model_invalid_keeps(monkeypatch):
    s = _fresh(chat_model="deepseek-flash")
    inputs = iter(["12", "bogus-model-xyz", "q"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(inputs))
    settings_menu(s)
    assert s.chat_model == "deepseek-flash"


def test_settings_menu_toggle_failover(monkeypatch):
    s = _fresh(use_multiple_accounts=True)
    inputs = iter(["13", "q"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(inputs))
    settings_menu(s)
    assert s.use_multiple_accounts is False


def test_settings_menu_unknown_loops_then_back(monkeypatch):
    s = _fresh()
    before = (s.enable_tools, s.port, s.log_level)
    inputs = iter(["99", "q"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(inputs))
    settings_menu(s)
    assert (s.enable_tools, s.port, s.log_level) == before


# --- accounts_menu ---

def test_accounts_menu_offer_auto_login_declined(monkeypatch):
    s = _fresh(accounts=[AccountConfig(email="a@x.com", password="pw", token="")])

    def _fail(*a, **k):
        raise AssertionError("login must not run on decline")

    monkeypatch.setattr("deepseaport.cli._obscura_login_token", _fail)
    inputs = iter(["t", "a@x.com", "", "n", "b"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(inputs))
    accounts_menu(s)
    assert s.accounts[0].token == ""


def test_accounts_menu_offer_auto_login_confirmed(monkeypatch):
    s = _fresh(accounts=[AccountConfig(email="a@x.com", password="pw", token="")])
    monkeypatch.setattr("deepseaport.cli._obscura_login_token",
                        lambda email, pw, st: ("tok-confirmed", False, ""))
    inputs = iter(["t", "a@x.com", "", "", "b"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(inputs))
    accounts_menu(s)
    assert s.accounts[0].token == "tok-confirmed"


def test_accounts_menu_set_token_literal(monkeypatch):
    s = _fresh(accounts=[AccountConfig(email="a@x.com", password="pw", token="old")])

    def _fail(*a, **k):
        raise AssertionError("login must not run for literal token")

    monkeypatch.setattr("deepseaport.cli._obscura_login_token", _fail)
    inputs = iter(["t", "a@x.com", "newtok123", "b"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(inputs))
    accounts_menu(s)
    assert s.accounts[0].token == "newtok123"


def test_accounts_menu_set_token_not_found(monkeypatch):
    s = _fresh(accounts=[AccountConfig(email="a@x.com", token="t1")])
    inputs = iter(["t", "missing@x.com", "sometok", "b"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(inputs))
    accounts_menu(s)
    assert [a.email for a in s.accounts] == ["a@x.com"]
    assert s.accounts[0].token == "t1"


def test_accounts_menu_invalid_loops_then_back(monkeypatch):
    s = _fresh(accounts=[AccountConfig(email="a@x.com", token="t1")])
    inputs = iter(["zzz", "b"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(inputs))
    accounts_menu(s)
    assert len(s.accounts) == 1 and s.accounts[0].token == "t1"


# --- main_menu ---

def test_main_menu_invalid_then_quit(monkeypatch):
    s = _fresh(accounts=[AccountConfig(email="a@x.com", token="t1")])
    inputs = iter(["bogus", "q"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(inputs))
    assert main_menu(s, lambda st: 0) == 0


def test_main_menu_quit_on_eof(monkeypatch):
    s = _fresh(accounts=[AccountConfig(email="a@x.com", token="t1")])

    def _boom(*a, **k):
        raise EOFError

    monkeypatch.setattr("builtins.input", _boom)
    assert main_menu(s, lambda st: 0) == 0


def test_main_menu_quit_on_keyboard_interrupt(monkeypatch):
    s = _fresh(accounts=[AccountConfig(email="a@x.com", token="t1")])

    def _boom(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", _boom)
    assert main_menu(s, lambda st: 0) == 0


def test_main_menu_settings_routing(monkeypatch):
    s = _fresh(accounts=[AccountConfig(email="a@x.com", token="t1")])
    called = {"n": 0}
    monkeypatch.setattr(tui, "settings_menu", lambda st: called.__setitem__("n", 1))
    monkeypatch.setattr(tui, "accounts_menu", lambda st: (_ for _ in ()).throw(AssertionError("no accounts")))
    inputs = iter(["2", "q"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(inputs))
    assert main_menu(s, lambda st: 0) == 0
    assert called["n"] == 1


def test_main_menu_accounts_routing(monkeypatch):
    s = _fresh(accounts=[AccountConfig(email="a@x.com", token="t1")])
    called = {"n": 0}
    monkeypatch.setattr(tui, "accounts_menu", lambda st: called.__setitem__("n", 1))
    monkeypatch.setattr(tui, "settings_menu", lambda st: (_ for _ in ()).throw(AssertionError("no settings")))
    inputs = iter(["3", "q"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(inputs))
    assert main_menu(s, lambda st: 0) == 0
    assert called["n"] == 1


def test_main_menu_settings_routing_with_chat(monkeypatch):
    s = _fresh(accounts=[AccountConfig(email="a@x.com", token="t1")])
    called = {"n": 0}
    monkeypatch.setattr(tui, "settings_menu", lambda st: called.__setitem__("n", 1))
    inputs = iter(["4", "q"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(inputs))
    assert main_menu(s, lambda st: 0, lambda st: 0) == 0
    assert called["n"] == 1


def test_main_menu_accounts_routing_with_chat(monkeypatch):
    s = _fresh(accounts=[AccountConfig(email="a@x.com", token="t1")])
    called = {"n": 0}
    monkeypatch.setattr(tui, "accounts_menu", lambda st: called.__setitem__("n", 1))
    inputs = iter(["5", "q"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(inputs))
    assert main_menu(s, lambda st: 0, lambda st: 0) == 0
    assert called["n"] == 1


def test_main_menu_multi_chat_routing():
    sig = inspect.signature(main_menu)
    assert "multi_fn" in sig.parameters  # 3-fn signature supported
    assert "chat_fn" in sig.parameters


def test_main_menu_multi_chat_calls_multi_fn(monkeypatch):
    s = _fresh(accounts=[AccountConfig(email="a@x.com", token="t1")])
    calls = {"serve": 0, "chat": 0, "multi": 0}
    inputs = iter(["3", "q"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(inputs))
    rc = main_menu(s,
                   lambda st: calls.__setitem__("serve", 1) or 0,
                   lambda st: calls.__setitem__("chat", 1) or 0,
                   lambda st: calls.__setitem__("multi", 1) or 0)
    assert rc == 0
    assert calls == {"serve": 0, "chat": 0, "multi": 1}
