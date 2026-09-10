"""Offline tests for AccountPool manager: add/remove/cooldown failover."""

import json

import pytest

from deepseaport.accounts import AccountPool
from deepseaport.config import AccountConfig


def _cfg(email, token="tok"):
    return AccountConfig(email=email, password="pw", token=f"{token}-{email}")


def test_add_and_remove():
    pool = AccountPool([])
    assert len(pool) == 0
    pool.add(_cfg("a@x.com"))
    pool.add(_cfg("b@x.com"))
    assert len(pool) == 2
    assert pool.remove("a@x.com") is True
    assert len(pool) == 1
    assert pool.remove("missing@x.com") is False


def test_add_rejects_empty_and_duplicate():
    pool = AccountPool([])
    with pytest.raises(ValueError):
        pool.add(AccountConfig())
    pool.add(_cfg("a@x.com"))
    with pytest.raises(ValueError):
        pool.add(_cfg("A@x.com"))  # case-insensitive duplicate


def test_empty_pool_acquire_raises():
    pool = AccountPool([])
    with pytest.raises(ValueError):
        pool.acquire(timeout=0)


def test_cooldown_failover():
    pool = AccountPool([_cfg("a@x.com"), _cfg("b@x.com")])
    a = pool.get("a@x.com")
    assert a is not None
    AccountPool.mark_bad(a, seconds=60)
    # Must skip a, return b.
    item = pool.acquire(timeout=1)
    try:
        assert item.cfg.email == "b@x.com"
    finally:
        AccountPool.release(item)
    # Both bad -> timeout.
    b = pool.get("b@x.com")
    AccountPool.mark_bad(b, seconds=60)
    with pytest.raises(TimeoutError):
        pool.acquire(timeout=0.2)
    # Clear one -> works again.
    assert pool.clear_cooldown("a@x.com") == 1
    with pool.slot(timeout=1) as item2:
        assert item2.cfg.email == "a@x.com"


def test_busy_failover_and_timeout():
    pool = AccountPool([_cfg("a@x.com"), _cfg("b@x.com")])
    first = pool.acquire(timeout=1)
    try:
        second = pool.acquire(timeout=1)
        try:
            assert second.cfg.email != first.cfg.email
            with pytest.raises(TimeoutError):
                pool.acquire(timeout=0.2)
        finally:
            AccountPool.release(second)
    finally:
        AccountPool.release(first)
    # After release, acquire works.
    with pool.slot(timeout=1):
        pass


def test_remove_in_flight_safe():
    pool = AccountPool([_cfg("a@x.com"), _cfg("b@x.com")])
    held = pool.acquire(timeout=1)
    assert held.cfg.email in ("a@x.com", "b@x.com")
    assert pool.remove(held.cfg.email) is True
    assert len(pool) == 1
    AccountPool.release(held)  # must not raise
    item = pool.acquire(timeout=1)
    try:
        assert item.cfg.email != held.cfg.email
    finally:
        AccountPool.release(item)


def test_status_shape_and_round_robin():
    pool = AccountPool([_cfg("a@x.com"), _cfg("b@x.com")])
    rows = pool.status()
    assert {r["identifier"] for r in rows} == {"a@x.com", "b@x.com"}
    assert all({"busy", "cooldown_remaining", "uses"} <= set(r) for r in rows)
    seen = set()
    for _ in range(4):
        with pool.slot(timeout=1) as item:
            seen.add(item.cfg.email)
    assert seen == {"a@x.com", "b@x.com"}


def test_cli_add_remove_syncs_config(tmp_path, monkeypatch):
    from deepseaport.cli import main as cli_main
    from deepseaport.config import load_settings

    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps({"keys": [], "accounts": []}), encoding="utf-8")
    rc = cli_main(["--config", str(cfg_path), "accounts", "add",
                   "--email", "c@x.com", "--token", "tok123"])
    assert rc == 0
    settings = load_settings(str(cfg_path))
    assert len(settings.accounts) == 1
    # Duplicate -> rc 1.
    rc = cli_main(["--config", str(cfg_path), "accounts", "add",
                   "--email", "c@x.com", "--token", "tok123"])
    assert rc == 1
    rc = cli_main(["--config", str(cfg_path), "accounts", "remove", "c@x.com"])
    assert rc == 0
    assert load_settings(str(cfg_path)).accounts == []


def test_extract_token_formats():
    import json as _json
    from deepseaport.accounts import TOKEN_HELP, extract_token

    assert extract_token("abc123") == "abc123"
    assert extract_token('{"value":"tok64","__version":"0"}') == "tok64"
    double = _json.dumps(_json.dumps({"url": "x", "tok": _json.dumps({"value": "tok99"})}))

    # Browser_evaluate nests token JSON as string; inner value still parses.
    inner = _json.dumps({"value": "tok99", "__version": "0"})
    assert extract_token(inner) == "tok99"
    assert "userToken" in TOKEN_HELP and "python -m deepseaport login" in TOKEN_HELP


def test_cli_set_token_and_json_token(tmp_path):
    import json as _json
    from deepseaport.cli import main as cli_main
    from deepseaport.config import load_settings

    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps({"keys": [], "accounts": []}), encoding="utf-8")
    full_json = _json.dumps({"value": "tok64abc", "__version": "0"})
    assert cli_main(["--config", str(cfg_path), "accounts", "add",
                     "--email", "d@x.com", "--token", full_json]) == 0
    assert load_settings(str(cfg_path)).accounts[0].token == "tok64abc"
    assert cli_main(["--config", str(cfg_path), "accounts", "set-token",
                     "d@x.com", "--token", "refreshed"]) == 0
    assert load_settings(str(cfg_path)).accounts[0].token == "refreshed"


def test_login_upsert_preserves_pool(tmp_path):
    from deepseaport.cli import _upsert_account
    from deepseaport.config import AccountConfig, Settings

    s = Settings(accounts=[AccountConfig(email="a@x.com", token="t1")],
                 config_path=str(tmp_path / "c.json"))
    _upsert_account(s, "b@x.com", "pw", "t2")
    assert len(s.accounts) == 2
    _upsert_account(s, "a@x.com", "pw", "t1-new")
    assert len(s.accounts) == 2
    assert [a for a in s.accounts if a.email == "a@x.com"][0].token == "t1-new"


def test_current_preferred_then_failover():
    pool = AccountPool([_cfg("a@x.com"), _cfg("b@x.com")])
    assert pool.set_current("b@x.com") is True
    assert pool.current == "b@x.com"
    with pool.slot(timeout=1) as item:
        assert item.cfg.email == "b@x.com"
    # Busy preferred -> fail over to other.
    held = pool.acquire(timeout=1)
    assert held.cfg.email == "b@x.com"
    try:
        with pool.slot(timeout=1) as item2:
            assert item2.cfg.email == "a@x.com"
    finally:
        AccountPool.release(held)
    # Cooldown preferred -> fail over.
    AccountPool.mark_bad(pool.get("b@x.com"), seconds=60)
    with pool.slot(timeout=1) as item3:
        assert item3.cfg.email == "a@x.com"


def test_current_invalid_and_remove_clears():
    pool = AccountPool([_cfg("a@x.com")])
    assert pool.set_current("missing@x.com") is False
    assert pool.current == ""
    assert pool.set_current("a@x.com") is True
    rows = pool.status()
    assert [r for r in rows if r["identifier"] == "a@x.com"][0]["current"] is True
    assert pool.remove("a@x.com") is True
    assert pool.current == ""


def test_sync_current_from_settings(tmp_path):
    from deepseaport.accounts import sync_current_from_settings
    from deepseaport.config import AccountConfig, Settings

    s = Settings(accounts=[AccountConfig(email="a@x.com", token="t")],
                 active_account="stale@x.com", config_path=str(tmp_path / "c.json"))
    pool = AccountPool(s.accounts, current=s.active_account)
    assert sync_current_from_settings(pool, s) is True
    assert s.active_account == "" and pool.current == ""


def test_cli_select_and_clear(tmp_path, capsys):
    import json as _json
    from deepseaport.cli import main as cli_main
    from deepseaport.config import load_settings

    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps({"keys": [], "accounts": []}), encoding="utf-8")
    assert cli_main(["--config", str(cfg_path), "accounts", "add",
                     "--email", "a@x.com", "--token", "t1"]) == 0
    assert cli_main(["--config", str(cfg_path), "accounts", "add",
                     "--email", "b@x.com", "--token", "t2"]) == 0
    assert cli_main(["--config", str(cfg_path), "accounts", "select", "b@x.com"]) == 0
    assert load_settings(str(cfg_path)).active_account == "b@x.com"
    assert cli_main(["--config", str(cfg_path), "accounts", "list"]) == 0
    out = capsys.readouterr().out
    assert "CURRENT: [b@x.com]" in out
    assert cli_main(["--config", str(cfg_path), "accounts", "select", "--clear"]) == 0
    assert load_settings(str(cfg_path)).active_account == ""
    # Deleting current clears selection.
    assert cli_main(["--config", str(cfg_path), "accounts", "select", "a@x.com"]) == 0
    assert cli_main(["--config", str(cfg_path), "accounts", "remove", "a@x.com"]) == 0
    assert load_settings(str(cfg_path)).active_account == ""


def test_api_select_current():
    from deepseaport.config import AccountConfig, Settings
    from deepseaport.server import create_app
    from fastapi.testclient import TestClient

    s = Settings(keys=[], accounts=[AccountConfig(email="a@x.com", token="t1"),
                                     AccountConfig(email="b@x.com", token="t2")])
    c = TestClient(create_app(s), raise_server_exceptions=False)
    r = c.post("/v1/accounts/select", json={"identifier": "b@x.com"})
    assert r.status_code == 200 and r.json()["current"] == "b@x.com"
    assert s.active_account == "b@x.com"
    lst = c.get("/v1/accounts").json()
    assert lst["current"] == "b@x.com"
    assert [d for d in lst["data"] if d["identifier"] == "b@x.com"][0]["current"] is True


def test_tui_number_selects_not_deletes(monkeypatch):
    from deepseaport.config import AccountConfig, Settings
    from deepseaport.tui import accounts_menu

    s = Settings(accounts=[AccountConfig(email="a@x.com", token="t1"),
                           AccountConfig(email="b@x.com", token="t2")],
                 active_account="", config_path="")
    inputs = iter(["1", "b"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(inputs))
    accounts_menu(s)
    assert s.active_account == "a@x.com"
    assert len(s.accounts) == 2  # nothing deleted


def test_tui_delete_needs_confirmation(monkeypatch):
    from deepseaport.config import AccountConfig, Settings
    from deepseaport.tui import accounts_menu

    s = Settings(accounts=[AccountConfig(email="a@x.com", token="t1"),
                           AccountConfig(email="b@x.com", token="t2")],
                 active_account="", config_path="")
    # Decline -> nothing deleted.
    inputs = iter(["d", "a@x.com", "n", "b"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(inputs))
    accounts_menu(s)
    assert len(s.accounts) == 2
    # Confirm -> deleted.
    inputs = iter(["d", "a@x.com", "y", "b"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(inputs))
    accounts_menu(s)
    assert [a.email for a in s.accounts] == ["b@x.com"]


def test_tui_add_skip_token_offers_auto_login(monkeypatch):
    from deepseaport.config import Settings
    from deepseaport.tui import accounts_menu

    s = Settings(accounts=[], active_account="", config_path="")
    # a -> add flow: email, password, token(skip), auto-login Y.
    inputs = iter(["a", "new@x.com", "pw", "", "y", "b"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(inputs))
    monkeypatch.setattr("deepseaport.cli._obscura_login_token",
                        lambda email, password, settings: "auto-tok-123")
    accounts_menu(s)
    assert len(s.accounts) == 1
    assert s.accounts[0].token == "auto-tok-123"
    assert s.active_account == "new@x.com"
