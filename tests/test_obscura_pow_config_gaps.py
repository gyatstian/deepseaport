"""Offline gap tests: obscura_bridge + pow + tokens + config (no network)."""

from __future__ import annotations

import base64
import json
import sys
import threading
import time
import types
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from deepseaport import pow as PoW
from deepseaport.config import (
    DEFAULT_PORT,
    AccountConfig,
    Settings,
    _app_dir,
    _default_config_path,
    load_settings,
)
from deepseaport.obscura_bridge import (
    ObscuraBridge,
    WafState,
    _profile_key,
    get_default_bridge,
    register_default_bridge,
)
from deepseaport.tokens import extract_token


def _bridge(tmp_path, name="prof", **kw):
    return ObscuraBridge(binary="fake-obscura", profile=str(tmp_path / name), **kw)


_CONFIG_ENV_KEYS = [
    "DEEPSEAPORT_CONFIG",
    "DEEPSEAPORT_KEYS",
    "DEEPSEAPORT_TOKEN",
    "DEEPSEEK_TOKEN",
    "DEEPSEAPORT_EMAIL",
    "DEEPSEEK_EMAIL",
    "DEEPSEAPORT_PASSWORD",
    "DEEPSEEK_PASSWORD",
    "DEEPSEAPORT_PORT",
    "DEEPSEAPORT_ACTIVE_ACCOUNT",
    "OBSCURA_BIN",
    "OBSCURA_PROFILE",
    "DEEPSEAPORT_BROWSER_BIN",
    "DEEPSEAPORT_BROWSER_HEADLESS",
    "DEEPSEAPORT_LISTEN",
    "DEEPSEAPORT_ENABLE_TOOLS",
    "DEEPSEAPORT_WARMUP_ON_STARTUP",
    "DEEPSEAPORT_AUTO_DELETE_SESSION",
    "DEEPSEAPORT_MAX_RETRIES",
    "DEEPSEAPORT_PARALLEL_FETCH",
    "DEEPSEAPORT_USE_MULTIPLE_ACCOUNTS",
    "DEEPSEAPORT_LOG_LEVEL",
    "DEEPSEAPORT_STREAM_MODE",
    "DEEPSEAPORT_CHAT_MODEL",
]


def _clean_env(monkeypatch):
    for k in _CONFIG_ENV_KEYS:
        monkeypatch.delenv(k, raising=False)


# --- obscura_bridge._parse_cookie_dump ---


def test_parse_cookie_dump_valid_list(tmp_path):
    b = _bridge(tmp_path, "p1")
    payload = json.dumps([{"name": "a", "value": "1"}, {"name": "aws-waf-token", "value": "tok"}])
    assert b._parse_cookie_dump(payload) == {"a": "1", "aws-waf-token": "tok"}


def test_parse_cookie_dump_log_prefix_ignored(tmp_path):
    b = _bridge(tmp_path, "p2")
    payload = "INFO some log line\n" + json.dumps([{"name": "a", "value": "b"}])
    assert b._parse_cookie_dump(payload) == {"a": "b"}


def test_parse_cookie_dump_no_bracket_bad_json_empty(tmp_path):
    b = _bridge(tmp_path, "p3")
    assert b._parse_cookie_dump("no bracket here") == {}
    assert b._parse_cookie_dump("{not json") == {}
    assert b._parse_cookie_dump("") == {}
    assert b._parse_cookie_dump(None) == {}  # type: ignore[arg-type]


def test_parse_cookie_dump_missing_name_value_skipped(tmp_path):
    b = _bridge(tmp_path, "p4")
    payload = json.dumps([
        {"name": "", "value": "x"},
        {"value": "y"},
        {"name": "c", "value": ""},
        {"name": "ok", "value": "v"},
    ])
    assert b._parse_cookie_dump(payload) == {"ok": "v"}


# --- _profile_key ---


def test_profile_key_same_input_same_key(tmp_path):
    p = str(tmp_path / "prof")
    assert _profile_key(p) == _profile_key(p)


def test_profile_key_different_profiles_different_keys(tmp_path):
    assert _profile_key(str(tmp_path / "a")) != _profile_key(str(tmp_path / "b"))


# --- _discover ---


def test_discover_path_hit(monkeypatch):
    monkeypatch.setattr("deepseaport.obscura_bridge.shutil.which",
                        lambda name: "/usr/bin/obscura" if name == "obscura" else None)
    assert ObscuraBridge._discover() == "/usr/bin/obscura"


def test_discover_local_glob_hit(monkeypatch, tmp_path):
    (tmp_path / "obscura").write_text("x", encoding="utf-8")
    monkeypatch.setattr("deepseaport.obscura_bridge.shutil.which", lambda name: None)
    monkeypatch.setattr(ObscuraBridge, "_search_roots",
                        staticmethod(lambda: [tmp_path]))
    found = ObscuraBridge._discover()
    assert found == str(tmp_path / "obscura")


def test_discover_fallback(monkeypatch):
    monkeypatch.setattr("deepseaport.obscura_bridge.shutil.which", lambda name: None)
    monkeypatch.setattr(ObscuraBridge, "_search_roots", staticmethod(lambda: []))
    assert ObscuraBridge._discover() == "obscura"


# --- version / cookie_header / has_waf_token / status ---


def test_version_exception_returns_unavailable(monkeypatch, tmp_path):
    b = _bridge(tmp_path, "ver")
    def _boom(*a, **k):
        raise RuntimeError("boom")
    monkeypatch.setattr("deepseaport.obscura_bridge.subprocess.run", _boom)
    assert b.version().startswith("unavailable:")


def test_cookie_header_join(tmp_path):
    b = _bridge(tmp_path, "hdr")
    b.state.cookies = {"a": "1", "b": "2"}
    assert b.cookie_header() == "a=1; b=2"
    b.state.cookies = {}
    assert b.cookie_header() == ""


def test_has_waf_token_true_false(tmp_path):
    b = _bridge(tmp_path, "waf")
    b.state.cookies = {"a": "1"}
    assert b.has_waf_token() is False
    b.state.cookies = {"aws-waf-token": "t"}
    assert b.has_waf_token() is True


def test_status_keys(monkeypatch, tmp_path):
    b = _bridge(tmp_path, "st")
    b.state = WafState(cookies={"aws-waf-token": "t"}, user_agent="ua", warmed=True)
    monkeypatch.setattr(b, "version", lambda: "v1")
    s = b.status()
    assert set(s) == {"binary", "version", "profile", "warmed",
                      "has_waf_token", "cookie_names", "user_agent"}
    assert s["warmed"] is True
    assert s["has_waf_token"] is True
    assert s["cookie_names"] == ["aws-waf-token"]


# --- warmup ---


def test_warmup_fresh_skip_no_run(tmp_path):
    b = _bridge(tmp_path, "fresh")
    b.state = WafState(cookies={"aws-waf-token": "tok", "a": "1"},
                       user_agent="ua", warmed=True)
    b._last_warmup_ok = time.time()
    calls = {"n": 0}

    def fake_run(*a, **k):
        calls["n"] += 1
        return CompletedProcess(a, 0, stdout="[]", stderr="")

    b._run = fake_run  # type: ignore[method-assign]
    out = b.warmup()
    assert out == {"aws-waf-token": "tok", "a": "1"}
    assert calls["n"] == 0


def test_warmup_pass1_empty_dump_fallback(tmp_path):
    b = _bridge(tmp_path, "fb")
    b.state = WafState(cookies={}, user_agent="cached-ua", warmed=False)
    b._last_warmup_ok = 0.0

    def empty_run(*a, **k):
        return CompletedProcess(a, 0, stdout="", stderr="")

    b._run = empty_run  # type: ignore[method-assign]
    b.dump_cookies = lambda timeout=30: {"aws-waf-token": "fallback"}  # type: ignore[method-assign]
    out = b.warmup()
    assert out == {"aws-waf-token": "fallback"}
    assert b.state.warmed is True


def test_warmup_ua_reuse_no_fetch(tmp_path):
    b = _bridge(tmp_path, "ua")
    b.state = WafState(cookies={}, user_agent="cached-ua", warmed=False)
    b._last_warmup_ok = 0.0
    dump = json.dumps([{"name": "aws-waf-token", "value": "tok"}])
    b._run = lambda *a, **k: CompletedProcess(a, 0, stdout=dump, stderr="")  # type: ignore[method-assign]
    called = {"n": 0}

    def fake_fetch(timeout=30):
        called["n"] += 1
        return "fresh-ua"

    b.fetch_ua = fake_fetch  # type: ignore[method-assign]
    out = b.warmup()
    assert out == {"aws-waf-token": "tok"}
    assert called["n"] == 0
    assert b.state.user_agent == "cached-ua"


def test_warmup_follower_shares_leader(tmp_path):
    b = _bridge(tmp_path, "single")
    b.state = WafState(cookies={}, user_agent="ua-preset", warmed=False)
    b._last_warmup_ok = 0.0
    dump = json.dumps([{"name": "aws-waf-token", "value": "slowtok"}])
    runs = {"n": 0}

    def slow_run(*a, **k):
        runs["n"] += 1
        time.sleep(0.3)
        return CompletedProcess(a, 0, stdout=dump, stderr="")

    b._run = slow_run  # type: ignore[method-assign]
    results: dict = {}

    def call(idx):
        results[idx] = b.warmup()

    t1 = threading.Thread(target=call, args=(1,))
    t2 = threading.Thread(target=call, args=(2,))
    t1.start()
    time.sleep(0.05)
    t2.start()
    t1.join(5)
    t2.join(5)
    assert results[1] == {"aws-waf-token": "slowtok"}
    assert results[2] == {"aws-waf-token": "slowtok"}
    assert runs["n"] == 1


def test_register_default_bridge_roundtrip(tmp_path):
    b = _bridge(tmp_path, "reg")
    try:
        register_default_bridge(b)
        assert get_default_bridge() is b
    finally:
        register_default_bridge(None)
    assert get_default_bridge() is None


# --- pow.py ---


def test_pow_build_header_roundtrip():
    ch = {"algorithm": PoW.ALGORITHM, "challenge": "c", "salt": "s",
          "answer": 42, "signature": "sig", "target_path": "/api/v0/chat/completion"}
    header = PoW.build_header(ch)
    data = json.loads(base64.b64decode(header).decode("utf-8"))
    assert data == ch
    assert set(data) == {"algorithm", "challenge", "salt", "answer", "signature", "target_path"}


def test_pow_solve_wrong_algorithm_raises():
    with pytest.raises(ValueError):
        PoW.solve("bad-algo", "c", "s", 1, 1)


def test_pow_solve_propagates_engine_error(monkeypatch):
    def _boom():
        raise FileNotFoundError("no wasm")
    monkeypatch.setattr("deepseaport.pow._get_engine_module", _boom)
    with pytest.raises(FileNotFoundError):
        PoW.solve(PoW.ALGORITHM, "c", "s", 1, 1)


def test_pow_wasm_path_caching_same():
    assert PoW.wasm_path() is PoW.wasm_path()


def test_pow_ensure_wasm_skips_download(monkeypatch, tmp_path):
    target = tmp_path / "wasm.bin"
    target.write_bytes(b"x" * 2000)
    monkeypatch.setattr("deepseaport.pow.wasm_path", lambda: target)

    def _boom(*a, **k):
        raise AssertionError("download must not be called")

    monkeypatch.setattr("curl_cffi.requests.get", _boom)
    assert PoW.ensure_wasm() == target


def test_pow_get_engine_module_cache_hit(monkeypatch, tmp_path):
    wp = tmp_path / "m.wasm"
    wp.write_bytes(b"\x00asm1234")
    monkeypatch.setattr("deepseaport.pow.wasm_path", lambda: wp)
    orig_e, orig_m, orig_k = PoW._ENGINE, PoW._MODULE, PoW._MODULE_KEY
    PoW._ENGINE = None
    PoW._MODULE = None
    PoW._MODULE_KEY = None
    counts = {"engine": 0, "module": 0}

    class FakeEngine:
        def __init__(self, *a, **k):
            counts["engine"] += 1

    class FakeModule:
        def __init__(self, engine, data):
            counts["module"] += 1

    fake = types.ModuleType("wasmtime")
    fake.Engine = FakeEngine  # type: ignore[attr-defined]
    fake.Module = FakeModule  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "wasmtime", fake)
    try:
        e1, m1 = PoW._get_engine_module()
        e2, m2 = PoW._get_engine_module()
        assert e1 is e2 and m1 is m2
        assert counts == {"engine": 1, "module": 1}
    finally:
        PoW._ENGINE, PoW._MODULE, PoW._MODULE_KEY = orig_e, orig_m, orig_k


# --- tokens.py ---


def test_extract_token_none_whitespace():
    assert extract_token(None) == ""
    assert extract_token("") == ""
    assert extract_token("   ") == ""


def test_extract_token_malformed_brace():
    assert extract_token("{") == ""
    assert extract_token("{not json") == ""


def test_extract_token_array_and_missing_value():
    # Top-level arrays are not token wrappers: passthrough documents current impl.
    assert extract_token('["a"]') == '["a"]'
    # Dict without "value" normalizes to "".
    assert extract_token('{"a":1}') == ""


def test_extract_token_single_quote_trim():
    assert extract_token("'tok64'") == "tok64"
    assert extract_token("  tok  ") == "tok"


def test_extract_token_outer_quote_json_literal():
    assert extract_token('"tok64"') == "tok64"
    assert extract_token(json.dumps("tok99")) == "tok99"


# --- config.py AccountConfig ---


def test_identifier_email_preferred():
    c = AccountConfig(email="a@x.com", mobile="555", token="123456789012345")
    assert c.identifier == "a@x.com"


def test_identifier_mobile_fallback():
    assert AccountConfig(mobile="555", token="123456789012345").identifier == "555"


def test_identifier_token_truncated():
    assert AccountConfig(token="123456789012345").identifier == "1234567890..."
    assert AccountConfig(token="abc").identifier == "abc..."


def test_identifier_empty_question():
    assert AccountConfig().identifier == "?"


def test_expired_ban_clears():
    c = AccountConfig(email="a@x.com", banned=True, banned_until=time.time() - 100)
    assert c.banned is False
    assert c.banned_until == 0.0


def test_invalid_banned_until_zero():
    assert AccountConfig(banned_until="not-a-number").banned_until == 0.0


# --- config.py Settings.save ---


def test_save_noop_empty_path(tmp_path):
    s = Settings(config_path="")
    s.save()  # must not raise, must not create a file
    assert list(tmp_path.glob("*.json")) == []


def test_save_payload_shape(tmp_path):
    cfg = tmp_path / "c.json"
    s = Settings(keys=["k"], accounts=[AccountConfig(email="a@x.com", token="t")],
                 active_account="a@x.com", port=5001, config_path=str(cfg))
    s.save()
    data = json.loads(cfg.read_text(encoding="utf-8"))
    for key in ("keys", "accounts", "active_account", "port", "log_level",
                "stream_mode", "chat_model", "max_retries"):
        assert key in data
    assert data["keys"] == ["k"]
    assert data["accounts"][0]["email"] == "a@x.com"


# --- config.py _default_config_path ---


def test_default_config_path_env_wins(monkeypatch, tmp_path):
    _clean_env(monkeypatch)
    custom = tmp_path / "custom.json"
    monkeypatch.setenv("DEEPSEAPORT_CONFIG", str(custom))
    assert _default_config_path() == Path(str(custom))


def test_default_config_path_cwd_wins(monkeypatch, tmp_path):
    _clean_env(monkeypatch)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    assert _default_config_path() == tmp_path / "config.json"


def test_default_config_path_app_dir_fallback(monkeypatch, tmp_path):
    _clean_env(monkeypatch)
    monkeypatch.chdir(tmp_path)
    assert _default_config_path() == _app_dir() / "config.json"


# --- config.py load_settings ---


def test_load_bad_json_defaults(monkeypatch, tmp_path):
    _clean_env(monkeypatch)
    p = tmp_path / "bad.json"
    p.write_text("{bad json", encoding="utf-8")
    s = load_settings(str(p))
    assert s.port == DEFAULT_PORT
    assert s.accounts == []


def test_keys_env_override(monkeypatch, tmp_path):
    _clean_env(monkeypatch)
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"keys": ["filekey"]}), encoding="utf-8")
    monkeypatch.setenv("DEEPSEAPORT_KEYS", " a ,b,, c ")
    assert load_settings(str(p)).keys == ["a", "b", "c"]


def test_env_token_replace_when_empty(monkeypatch, tmp_path):
    _clean_env(monkeypatch)
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"accounts": []}), encoding="utf-8")
    monkeypatch.setenv("DEEPSEAPORT_TOKEN", "envtok1234567890")
    s = load_settings(str(p))
    assert len(s.accounts) == 1
    assert s.accounts[0].token == "envtok1234567890"


def test_env_token_append_when_nonempty(monkeypatch, tmp_path):
    _clean_env(monkeypatch)
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"accounts": [{"email": "a@x.com", "token": "filetok1234567890"}]}),
                 encoding="utf-8")
    monkeypatch.setenv("DEEPSEAPORT_TOKEN", "envtok1234567890")
    s = load_settings(str(p))
    assert len(s.accounts) == 2
    assert s.accounts[0].email == "a@x.com"
    assert s.accounts[1].token == "envtok1234567890"


def test_bool_invalid_string_fallback(monkeypatch, tmp_path):
    _clean_env(monkeypatch)
    p = tmp_path / "c.json"
    p.write_text(json.dumps({}), encoding="utf-8")
    monkeypatch.setenv("DEEPSEAPORT_ENABLE_TOOLS", "maybe")
    assert load_settings(str(p)).enable_tools is False


def test_int_invalid_default(monkeypatch, tmp_path):
    _clean_env(monkeypatch)
    p = tmp_path / "c.json"
    p.write_text(json.dumps({}), encoding="utf-8")
    monkeypatch.setenv("DEEPSEAPORT_PORT", "not-a-number")
    monkeypatch.setenv("DEEPSEAPORT_MAX_RETRIES", "bad")
    s = load_settings(str(p))
    assert s.port == DEFAULT_PORT
    assert s.max_retries == 1


def test_str_case_insensitive_and_bad_default(monkeypatch, tmp_path):
    _clean_env(monkeypatch)
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"log_level": "info"}), encoding="utf-8")
    assert load_settings(str(p)).log_level == "INFO"
    p.write_text(json.dumps({"log_level": "NOPE"}), encoding="utf-8")
    assert load_settings(str(p)).log_level == "INFO"
    _clean_env(monkeypatch)
    p.write_text(json.dumps({}), encoding="utf-8")
    monkeypatch.setenv("DEEPSEAPORT_LOG_LEVEL", "info")
    assert load_settings(str(p)).log_level == "INFO"


def test_chat_model_env_override(monkeypatch, tmp_path):
    _clean_env(monkeypatch)
    p = tmp_path / "c.json"
    p.write_text(json.dumps({}), encoding="utf-8")
    monkeypatch.setenv("DEEPSEAPORT_CHAT_MODEL", "custom-model")
    assert load_settings(str(p)).chat_model == "custom-model"
