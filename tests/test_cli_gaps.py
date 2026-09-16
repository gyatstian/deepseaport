"""Offline gap tests for deepseaport.cli (no network, no browser, no server).

Covers cmd_login / cmd_chat / cmd_waf / cmd_models / cmd_accounts,
_scan_free_ports / _next_free_port / _ensure_free_port / _run_server /
_watch_stop_keys, and main() dispatch. All config goes to tmp_path.
"""

from __future__ import annotations

import argparse
import json
import time
import types

from deepseaport.cli import (
    _ensure_free_port,
    _next_free_port,
    _run_server,
    _scan_free_ports,
    _watch_stop_keys,
    cmd_accounts,
    main as cli_main,
)


def _write_cfg(tmp_path, payload: dict):
    p = tmp_path / "config.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return str(p)


def _load(path):
    from deepseaport.config import load_settings

    return load_settings(path)


# cmd_login -------------------------------------------------------------------

def test_login_success_saves_token_and_active(tmp_path, monkeypatch):
    import deepseaport.cli as cli_mod

    cfg = _write_cfg(tmp_path, {"keys": [], "accounts": []})
    monkeypatch.setattr(cli_mod, "_obscura_login_token",
                        lambda email, password, settings: ("tok123", False, ""))
    rc = cli_main(["--config", cfg, "login", "--email", "a@x.com", "--password", "pw"])
    assert rc == 0
    s = _load(cfg)
    assert len(s.accounts) == 1
    assert s.accounts[0].token == "tok123"
    assert s.accounts[0].email == "a@x.com"
    assert s.active_account == "a@x.com"


def test_login_banned_with_token_returns_3_and_persists_banned(tmp_path, monkeypatch, capsys):
    import deepseaport.accounts as acc_mod
    import deepseaport.cli as cli_mod

    cfg = _write_cfg(tmp_path, {"keys": [], "accounts": []})
    monkeypatch.setattr(cli_mod, "_obscura_login_token",
                        lambda email, password, settings: ("tok-banned", True, "suspended"))
    monkeypatch.setattr(acc_mod, "check_ban_for_token", lambda token, timeout=12: (True, None))
    rc = cli_main(["--config", cfg, "login", "--email", "b@x.com", "--password", "pw"])
    assert rc == 3
    s = _load(cfg)
    assert len(s.accounts) == 1
    assert s.accounts[0].token == "tok-banned"
    assert s.accounts[0].banned is True
    out = capsys.readouterr().out
    assert "BANNED" in out


def test_login_no_token_returns_3(tmp_path, monkeypatch):
    import deepseaport.cli as cli_mod

    cfg = _write_cfg(tmp_path, {"keys": [], "accounts": []})
    monkeypatch.setattr(cli_mod, "_obscura_login_token",
                        lambda email, password, settings: (None, False, ""))
    rc = cli_main(["--config", cfg, "login", "--email", "c@x.com", "--password", "pw"])
    assert rc == 3
    assert _load(cfg).accounts == []


def test_login_banned_no_token_returns_3(tmp_path, monkeypatch, capsys):
    import deepseaport.cli as cli_mod

    cfg = _write_cfg(tmp_path, {"keys": [], "accounts": []})
    monkeypatch.setattr(cli_mod, "_obscura_login_token",
                        lambda email, password, settings: (None, True, "suspended detail"))
    rc = cli_main(["--config", cfg, "login", "--email", "d@x.com", "--password", "pw"])
    assert rc == 3
    assert _load(cfg).accounts == []
    assert "BANNED" in capsys.readouterr().out


# cmd_chat --------------------------------------------------------------------

def _patch_chat(monkeypatch, thinking="think-stuff", content="hello"):
    import deepseaport.accounts as acc_mod
    import deepseaport.obscura_bridge as bridge_mod
    import deepseaport.server as srv_mod

    fake_app = types.SimpleNamespace(state=types.SimpleNamespace(bridge=None))
    monkeypatch.setattr(srv_mod, "create_app", lambda settings=None: fake_app)

    class _FakeBridge:
        def __init__(self, *a, **k):
            self.state = types.SimpleNamespace(user_agent="ua", cookies={})

        def warmup(self, timeout=45):
            return {}

    monkeypatch.setattr(bridge_mod, "ObscuraBridge", _FakeBridge)
    monkeypatch.setattr(bridge_mod, "register_default_bridge", lambda *a, **k: None)
    monkeypatch.setattr(acc_mod.AccountPool, "acquire", lambda self, timeout=30, allow_failover=True: object())
    monkeypatch.setattr(srv_mod, "_complete_with_failover_sync",
                        lambda app, pool, prep, item, **k: {"content": content, "thinking": thinking})


def test_chat_returns_0_and_prints_thinking(tmp_path, monkeypatch, capsys):
    cfg = _write_cfg(tmp_path, {"keys": [], "accounts": [{"email": "a@x.com", "token": "tok123"}]})
    _patch_chat(monkeypatch, thinking="think-stuff", content="hello")
    rc = cli_main(["--config", cfg, "chat", "--prompt", "hi"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "THINKING:" in out
    assert "think-stuff" in out
    assert "hello" in out


def test_chat_no_thinking_prints_content_only(tmp_path, monkeypatch, capsys):
    cfg = _write_cfg(tmp_path, {"keys": [], "accounts": [{"email": "a@x.com", "token": "tok123"}]})
    _patch_chat(monkeypatch, thinking="", content="just-content")
    rc = cli_main(["--config", cfg, "chat", "--prompt", "hi"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "THINKING" not in out
    assert "just-content" in out


# cmd_waf / cmd_models ----------------------------------------------------------

class _FakeWafBridge:
    def __init__(self, has_token: bool):
        self._has = has_token
        self.binary = "fake-obscura"
        self.state = types.SimpleNamespace(user_agent="fake-ua", cookies={})

    def version(self):
        return "9.9-test"

    def warmup(self, timeout=45):
        return {"aws-waf-token": "x"} if self._has else {}

    def has_waf_token(self):
        return self._has


def _patch_waf(monkeypatch, has_token: bool):
    import deepseaport.obscura_bridge as bridge_mod

    monkeypatch.setattr(bridge_mod, "ObscuraBridge",
                        lambda binary="", profile="": _FakeWafBridge(has_token))


def test_waf_warmup_true_returns_0(tmp_path, monkeypatch, capsys):
    cfg = _write_cfg(tmp_path, {"keys": [], "accounts": []})
    _patch_waf(monkeypatch, True)
    assert cli_main(["--config", cfg, "waf"]) == 0
    out = capsys.readouterr().out
    assert "fake-obscura" in out
    assert "cookies:" in out


def test_waf_no_token_returns_2(tmp_path, monkeypatch):
    cfg = _write_cfg(tmp_path, {"keys": [], "accounts": []})
    _patch_waf(monkeypatch, False)
    assert cli_main(["--config", cfg, "waf"]) == 2


def test_models_prints_4_ids(tmp_path, monkeypatch, capsys):
    from deepseaport.server import MODELS

    cfg = _write_cfg(tmp_path, {"keys": [], "accounts": []})
    assert cli_main(["--config", cfg, "models"]) == 0
    out = capsys.readouterr().out
    ids = json.loads(out)
    assert set(ids) == set(MODELS)
    assert len(ids) == 4


# cmd_accounts ------------------------------------------------------------------

def test_accounts_list_ban_label_and_missing_hint(tmp_path, monkeypatch, capsys):
    import deepseaport.accounts as acc_mod

    cfg = _write_cfg(tmp_path, {"keys": [], "accounts": [
        {"email": "a@x.com", "token": "tok-with-token-1234567890"},
        {"email": "b@x.com", "token": ""},
    ]})
    monkeypatch.setattr(acc_mod, "collect_ban_labels", lambda accounts, **k: {"a@x.com": None})
    assert cli_main(["--config", cfg, "accounts", "list"]) == 0
    out = capsys.readouterr().out
    assert "(BANNED)" in out
    assert "MISSING" in out
    assert "missing token: b@x.com" in out


def test_accounts_select_clear(tmp_path):
    cfg = _write_cfg(tmp_path, {"keys": [], "accounts": [
        {"email": "a@x.com", "token": "t1"},
    ], "active_account": "a@x.com"})
    assert cli_main(["--config", cfg, "accounts", "select", "--clear"]) == 0
    assert _load(cfg).active_account == ""


def test_accounts_add_missing_id_returns_2(tmp_path, capsys):
    cfg = _write_cfg(tmp_path, {"keys": [], "accounts": []})
    assert cli_main(["--config", cfg, "accounts", "add"]) == 2
    assert "need --email" in capsys.readouterr().out


def test_accounts_set_token_missing_args_returns_2(tmp_path):
    cfg = _write_cfg(tmp_path, {"keys": [], "accounts": [
        {"email": "a@x.com", "token": "t1"},
    ]})
    # Empty token value -> usage error.
    assert cli_main(["--config", cfg, "accounts", "set-token", "a@x.com", "--token", ""]) == 2
    # Missing identifier -> usage error (direct call; argparse would SystemExit).
    args = argparse.Namespace(config=cfg, accounts_action="set-token", identifier="", token="x")
    assert cmd_accounts(args) == 2


def test_accounts_remove_missing_returns_1(tmp_path, capsys):
    cfg = _write_cfg(tmp_path, {"keys": [], "accounts": [
        {"email": "a@x.com", "token": "t1"},
    ]})
    assert cli_main(["--config", cfg, "accounts", "remove", "missing@x.com"]) == 1
    assert "not found" in capsys.readouterr().out


def test_accounts_remove_no_identifier_returns_2(tmp_path):
    cfg = _write_cfg(tmp_path, {"keys": [], "accounts": []})
    args = argparse.Namespace(config=cfg, accounts_action="remove", identifier="")
    assert cmd_accounts(args) == 2


def test_accounts_unblock_clears_persisted_ban(tmp_path):
    cfg = _write_cfg(tmp_path, {"keys": [], "accounts": [
        {"email": "a@x.com", "token": "tok123", "banned": True, "banned_until": 0.0},
    ]})
    assert cli_main(["--config", cfg, "accounts", "unblock"]) == 0
    s = _load(cfg)
    assert s.accounts[0].banned is False
    assert float(s.accounts[0].banned_until or 0.0) == 0.0


def test_accounts_unknown_action_returns_2(tmp_path, capsys):
    cfg = _write_cfg(tmp_path, {"keys": [], "accounts": []})
    args = argparse.Namespace(config=cfg, accounts_action="bogus")
    assert cmd_accounts(args) == 2
    assert "unknown accounts action" in capsys.readouterr().out


# ports -------------------------------------------------------------------------

def test_scan_free_ports_bad_start_falls_back():
    ports = _scan_free_ports("127.0.0.1", "bad", 1, limit=5, check=lambda h, p: True)
    assert ports == [5001]


def test_scan_free_ports_limit_exhausted_short_list():
    ports = _scan_free_ports("127.0.0.1", 5000, 5, limit=2, check=lambda h, p: True)
    assert ports == [5000, 5001]


def test_scan_free_ports_check_raises_skipped():
    def _boom(host, port):
        raise OSError("busy")

    assert _scan_free_ports("127.0.0.1", 5000, 2, limit=5, check=_boom) == []


def test_next_free_port_none_when_all_busy(monkeypatch):
    import deepseaport.cli as cli_mod

    monkeypatch.setattr(cli_mod, "_is_port_free", lambda host, port: False)
    assert _next_free_port("127.0.0.1", 5000, limit=3) is None


def test_ensure_free_port_no_free_returns_original(tmp_path, monkeypatch, capsys):
    import deepseaport.cli as cli_mod
    from deepseaport.config import Settings

    s = Settings(port=5000, config_path=str(tmp_path / "c.json"))
    monkeypatch.setattr(cli_mod, "_is_port_free", lambda host, port: False)
    monkeypatch.setattr(cli_mod, "_next_free_port", lambda host, port, limit=50: None)
    assert _ensure_free_port(s, "127.0.0.1", 5000) == 5000
    assert "no free port" in capsys.readouterr().out


def test_run_server_workers_calls_uvicorn_run(tmp_path, monkeypatch):
    import deepseaport.cli as cli_mod
    import deepseaport.server as srv_mod
    import uvicorn

    from deepseaport.config import Settings

    calls = {}

    def _fake_run(app, host=None, port=None, log_level=None, workers=None):
        calls.update(host=host, port=port, workers=workers)

    monkeypatch.setattr(uvicorn, "run", _fake_run)
    monkeypatch.setattr(srv_mod, "create_app", lambda settings=None: object())
    monkeypatch.setattr(srv_mod, "apply_log_level", lambda *a, **k: None)
    monkeypatch.setattr(cli_mod, "_ensure_free_port", lambda s, h, p: p)

    s = Settings(port=5001, log_level="INFO", config_path=str(tmp_path / "c.json"))
    args = argparse.Namespace(host="", port=0, workers=2)
    assert _run_server(s, args) == 0
    assert calls.get("workers") == 2


def test_watch_stop_keys_never_raises_with_exited_server():
    srv = types.SimpleNamespace(should_exit=True)
    _watch_stop_keys(srv)  # must not raise
    assert srv.should_exit is True


def test_watch_stop_keys_broken_server_never_raises():
    class _Boom:
        @property
        def should_exit(self):
            raise RuntimeError("boom")

    _watch_stop_keys(_Boom())  # must not raise


# main dispatch -------------------------------------------------------------------

def test_main_no_subcommand_defaults_to_serve(tmp_path, monkeypatch):
    import deepseaport.cli as cli_mod

    seen = {}
    def _fake_serve(args):
        seen["cmd"] = args.cmd
        return 42
    monkeypatch.setattr(cli_mod, "cmd_serve", _fake_serve)
    cfg = _write_cfg(tmp_path, {"keys": [], "accounts": []})
    assert cli_mod.main(["--config", cfg]) == 42
    assert seen["cmd"] == "serve"


def test_main_login_dispatch(tmp_path, monkeypatch):
    import deepseaport.cli as cli_mod

    monkeypatch.setattr(cli_mod, "cmd_login", lambda args: 7)
    cfg = _write_cfg(tmp_path, {"keys": [], "accounts": []})
    assert cli_mod.main(["--config", cfg, "login", "--email", "a@x.com"]) == 7


def test_main_chat_dispatch(tmp_path, monkeypatch):
    import deepseaport.cli as cli_mod

    monkeypatch.setattr(cli_mod, "cmd_chat", lambda args: 8)
    cfg = _write_cfg(tmp_path, {"keys": [], "accounts": []})
    assert cli_mod.main(["--config", cfg, "chat", "--prompt", "hi"]) == 8


def test_main_unknown_cmd_returns_1(monkeypatch):
    import deepseaport.cli as cli_mod

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args",
                        lambda self, argv=None: argparse.Namespace(config=None, cmd="bogus"))
    try:
        assert cli_mod.main([]) == 1
    finally:
        # restore is handled by monkeypatch undo
        pass


def test_main_waf_models_accounts_dispatch(tmp_path, monkeypatch):
    import deepseaport.cli as cli_mod

    cfg = _write_cfg(tmp_path, {"keys": [], "accounts": []})
    monkeypatch.setattr(cli_mod, "cmd_waf", lambda args: 11)
    assert cli_mod.main(["--config", cfg, "waf"]) == 11
    monkeypatch.setattr(cli_mod, "cmd_models", lambda args: 12)
    assert cli_mod.main(["--config", cfg, "models"]) == 12
    monkeypatch.setattr(cli_mod, "cmd_accounts", lambda args: 13)
    assert cli_mod.main(["--config", cfg, "accounts", "list"]) == 13


def test_ensure_free_port_free_returns_same(tmp_path):
    import deepseaport.cli as cli_mod
    from deepseaport.config import Settings

    s = Settings(port=5010, config_path=str(tmp_path / "c.json"))
    # Loopback high ports are almost always free; but stub to be deterministic.
    orig = cli_mod._is_port_free
    try:
        cli_mod._is_port_free = lambda host, port: True
        assert _ensure_free_port(s, "127.0.0.1", 5010) == 5010
    finally:
        cli_mod._is_port_free = orig


def test_scan_free_ports_negative_start_clamped():
    ports = _scan_free_ports("127.0.0.1", -5, 1, limit=10, check=lambda h, p: True)
    assert ports == [1]


def test_login_upsert_path_uses_tmp_config(tmp_path, monkeypatch):
    """Second login for same email updates token, keeps single account."""
    import deepseaport.cli as cli_mod

    cfg = _write_cfg(tmp_path, {"keys": [], "accounts": []})
    monkeypatch.setattr(cli_mod, "_obscura_login_token",
                        lambda email, password, settings: ("first-tok", False, ""))
    assert cli_main(["--config", cfg, "login", "--email", "e@x.com", "--password", "pw"]) == 0
    monkeypatch.setattr(cli_mod, "_obscura_login_token",
                        lambda email, password, settings: ("second-tok", False, ""))
    assert cli_main(["--config", cfg, "login", "--email", "e@x.com", "--password", "pw"]) == 0
    s = _load(cfg)
    assert len(s.accounts) == 1
    assert s.accounts[0].token == "second-tok"
    # Token persists across reload.
    assert _load(cfg).accounts[0].token == "second-tok"


def test_unblock_single_identifier_only(tmp_path):
    future = time.time() + 3600
    cfg = _write_cfg(tmp_path, {"keys": [], "accounts": [
        {"email": "a@x.com", "token": "t1", "banned": True, "banned_until": future},
        {"email": "b@x.com", "token": "t2", "banned": True, "banned_until": future},
    ]})
    assert cli_main(["--config", cfg, "accounts", "unblock", "a@x.com"]) == 0
    s = _load(cfg)
    by_email = {a.email: a for a in s.accounts}
    assert by_email["a@x.com"].banned is False
    assert by_email["b@x.com"].banned is True
