"""Offline gap tests for auth / real_browser / mcp_client / browser_paths.

Fully offline: fake subprocess/ws/browser. Never launches a real browser.
"""

import json
import queue
import subprocess
import sys
import types

import pytest

from deepseaport import auth
from deepseaport.auth import (
    BrowserLogin,
    LoginResult,
    _collapse,
    _json_eval,
    _tool_text,
    browser_login,
    classify_page_state,
    unpack_login_result,
)
from deepseaport.config import Settings


def _settings(**kw):
    base = {"obscura_bin": "dummy-obscura", "obscura_profile": "dummy-profile"}
    base.update(kw)
    return Settings(**base)


def _text_result(text):
    return {"content": [{"type": "text", "text": text}]}


# --- unpack_login_result ---

def test_unpack_login_result_none():
    assert unpack_login_result(None) == (None, False, "")


def test_unpack_login_result_login_result():
    assert unpack_login_result(LoginResult(token="t", banned=True, detail="d")) == ("t", True, "d")


def test_unpack_login_result_full_tuple():
    assert unpack_login_result(("  tok  ", True, "hi")) == ("tok", True, "hi")


def test_unpack_login_result_short_tuple():
    assert unpack_login_result(("tok",)) == ("tok", False, "")
    assert unpack_login_result(("tok", True)) == ("tok", True, "")


def test_unpack_login_result_two_elem_tuple_empty_token():
    assert unpack_login_result(("   ", False)) == (None, False, "")


def test_unpack_login_result_string():
    assert unpack_login_result("  abc  ") == ("abc", False, "")
    assert unpack_login_result("   ") == (None, False, "")


def test_unpack_login_result_invalid_type():
    assert unpack_login_result(12345) == (None, False, "")
    assert unpack_login_result({"token": "x"}) == (None, False, "")
    assert unpack_login_result(["tok"]) == (None, False, "")


# --- _tool_text ---

def test_tool_text_str_passthrough():
    assert _tool_text("hello") == "hello"


def test_tool_text_dict_text_parts():
    result = {"content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}
    assert _tool_text(result) == "ab"


def test_tool_text_dict_with_str_parts():
    result = {"content": [{"type": "text", "text": "a"}, "b", {"type": "other", "text": "x"}]}
    assert _tool_text(result) == "ab"


def test_tool_text_non_dict_empty():
    assert _tool_text(None) == ""
    assert _tool_text(123) == ""
    assert _tool_text([{"type": "text"}]) == ""


def test_tool_text_dict_no_content():
    assert _tool_text({}) == ""
    assert _tool_text({"content": None}) == ""


# --- _json_eval ---

class _JsonFake:
    def __init__(self, payload):
        self._payload = payload

    def call(self, name, arguments):
        assert name == "browser_evaluate"
        return self._payload


def test_json_eval_valid_json():
    assert _json_eval(_JsonFake(_text_result('{"a": 1}')), "expr") == {"a": 1}


def test_json_eval_double_encoded_string():
    inner = json.dumps({"tok": "x"})
    outer = json.dumps(inner)
    assert _json_eval(_JsonFake(_text_result(outer)), "expr") == {"tok": "x"}


def test_json_eval_invalid_returns_none():
    assert _json_eval(_JsonFake(_text_result("not-json{{")), "expr") is None


def test_json_eval_double_encoded_invalid_returns_none():
    assert _json_eval(_JsonFake(_text_result(json.dumps("{{bad"))), "expr") is None


# --- _collapse ---

def test_collapse_whitespace_and_truncation():
    assert _collapse("  a\n\t b   c  ") == "a b c"
    assert _collapse("x" * 600, 500) == "x" * 500
    assert _collapse("", 10) == ""


# --- classify_page_state ---

def test_classify_ban_phrase():
    kind, phrase = classify_page_state("your account has been suspended until tomorrow")
    assert kind == "ban"
    assert phrase != ""


def test_classify_captcha_phrase_list():
    kind, _ = classify_page_state("please slide to verify your identity")
    assert kind == "captcha"


def test_classify_priority_captcha_over_credential_and_ban():
    text = "slide to verify and incorrect password and account has been suspended"
    assert classify_page_state(text)[0] == "captcha"


def test_classify_priority_credential_over_ban():
    text = "incorrect password and account has been suspended"
    assert classify_page_state(text)[0] == "credential"


def test_classify_empty():
    assert classify_page_state("") == ("", "")
    assert classify_page_state("healthy home page", False) == ("", "")


def test_classify_captcha_visible_flag():
    assert classify_page_state("healthy page", True) == ("captcha", "visible challenge overlay")


# --- BrowserLogin.run ---

def test_browser_login_run_missing_credentials(monkeypatch):
    monkeypatch.setattr(auth, "ObscuraBridge", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no bridge")))
    bl = BrowserLogin.__new__(BrowserLogin)
    bl.settings = _settings()
    assert bl.run("", "pw").detail == "email and password are required"
    assert bl.run("a@x.com", "").detail == "email and password are required"


class _BusyLock:
    def acquire(self, blocking=False, timeout=None):
        return False

    def release(self):
        pass


def test_browser_login_run_profile_lock_busy_timeout(monkeypatch):
    bl = BrowserLogin.__new__(BrowserLogin)
    bl.settings = _settings()
    bl.bridge = types.SimpleNamespace(profile="p", binary="b")
    monkeypatch.setattr(auth, "profile_lock", lambda profile: _BusyLock())
    res = bl.run("a@x.com", "pw")
    assert res.token is None and res.banned is False
    assert "busy" in res.detail


def test_browser_login_run_mcp_spawn_fail(monkeypatch):
    bl = BrowserLogin.__new__(BrowserLogin)
    bl.settings = _settings()
    bl.bridge = types.SimpleNamespace(profile="p", binary="b")

    class _Lock:
        def acquire(self, blocking=False, timeout=None):
            return True

        def release(self):
            pass

    monkeypatch.setattr(auth, "profile_lock", lambda profile: _Lock())

    def _boom(*a, **k):
        raise RuntimeError("no binary")

    monkeypatch.setattr(auth.mcp_client, "McpClient", _boom)
    res = bl.run("a@x.com", "pw")
    assert "could not start" in res.detail


# --- BrowserLogin._run helpers ---

class FakeClient:
    """Scriptable MCP client fake: dispatches browser_evaluate by expression."""

    def __init__(self, monkeypatch=None, probes=(), mark_login="ok",
                 fill_mode="ok", click_mode="ok", navigate_exc=None):
        self.probes = list(probes)
        self.probe_calls = 0
        self.mark_login = mark_login
        self.fill_mode = fill_mode
        self.click_mode = click_mode
        self.navigate_exc = navigate_exc
        self.calls = []

    def call(self, name, arguments):
        self.calls.append((name, arguments))
        if name == "browser_navigate":
            if self.navigate_exc:
                raise self.navigate_exc
            return {}
        if name == "browser_wait_for":
            return {}
        if name == "browser_evaluate":
            expr = (arguments or {}).get("expression", "")
            if "data-deepseaport-cookie" in expr:
                return _text_result("not-found")
            if "data-deepseaport-login" in expr:
                return _text_result(self.mark_login)
            if "removeItem" in expr:
                return _text_result("cleared")
            # page probe
            idx = self.probe_calls
            self.probe_calls += 1
            if idx < len(self.probes):
                page = self.probes[idx]
            else:
                page = self.probes[-1] if self.probes else {}
            return _text_result(json.dumps(page))
        if name == "browser_fill_form":
            if self.fill_mode == "raise":
                raise RuntimeError("fill_form broken")
            if self.fill_mode == "zero":
                return _text_result("filled 0 fields")
            return _text_result("filled 2 fields")
        if name == "browser_fill":
            if self.fill_mode == "fill_raises":
                raise RuntimeError("fill broken")
            return {}
        if name == "browser_click":
            if self.click_mode == "raise":
                raise RuntimeError("click broken")
            return {}
        if name == "browser_press_key":
            if self.click_mode == "both_raise":
                raise RuntimeError("enter broken")
            return {}
        return {}


def _make_login(monkeypatch):
    monkeypatch.setattr(auth.time, "sleep", lambda *a, **k: None)
    bl = BrowserLogin.__new__(BrowserLogin)
    bl.settings = _settings()
    bl.bridge = types.SimpleNamespace(profile="p", binary="b")
    return bl


def _probe(url="https://chat.deepseek.com/", tok=None, body="home", captcha=False):
    return {"url": url, "tok": tok, "body": body, "captcha_visible": captcha}


def test_run_navigate_fail(monkeypatch):
    bl = _make_login(monkeypatch)
    fc = FakeClient(navigate_exc=RuntimeError("net down"))
    res = bl._run(fc, "a@x.com", "pw")
    assert "could not open sign_in page" in res.detail


def test_run_fill_fallback_to_browser_fill(monkeypatch):
    bl = _make_login(monkeypatch)
    monkeypatch.setattr(auth, "LOGIN_TIMEOUT", 0.0)
    fc = FakeClient(probes=[_probe(), _probe()], fill_mode="raise")
    res = bl._run(fc, "a@x.com", "pw")
    # fallback fill path used, then no-token timeout since no token probed
    assert "no token" in res.detail
    assert any(c[0] == "browser_fill" for c in fc.calls)


def test_run_fill_zero_fields_falls_back(monkeypatch):
    bl = _make_login(monkeypatch)
    monkeypatch.setattr(auth, "LOGIN_TIMEOUT", 0.0)
    fc = FakeClient(probes=[_probe(), _probe()], fill_mode="zero")
    res = bl._run(fc, "a@x.com", "pw")
    assert "no token" in res.detail
    assert any(c[0] == "browser_fill" for c in fc.calls)


def test_run_click_fallback_to_enter(monkeypatch):
    bl = _make_login(monkeypatch)
    monkeypatch.setattr(auth, "LOGIN_TIMEOUT", 0.0)
    fc = FakeClient(probes=[_probe(), _probe()], click_mode="raise")
    res = bl._run(fc, "a@x.com", "pw")
    assert "no token" in res.detail
    assert any(c[0] == "browser_press_key" for c in fc.calls)


def test_run_poll_captcha(monkeypatch):
    bl = _make_login(monkeypatch)
    monkeypatch.setattr(auth, "LOGIN_TIMEOUT", 5.0)
    fc = FakeClient(probes=[_probe(), _probe(body="home"), _probe(body="slide to verify now")])
    res = bl._run(fc, "a@x.com", "pw")
    assert res.token is None and res.banned is False
    assert "captcha" in res.detail


def test_run_poll_captcha_visible_flag(monkeypatch):
    bl = _make_login(monkeypatch)
    monkeypatch.setattr(auth, "LOGIN_TIMEOUT", 5.0)
    fc = FakeClient(probes=[_probe(), _probe(body="home"), _probe(body="home", captcha=True)])
    res = bl._run(fc, "a@x.com", "pw")
    assert "captcha" in res.detail


def test_run_poll_credential(monkeypatch):
    bl = _make_login(monkeypatch)
    monkeypatch.setattr(auth, "LOGIN_TIMEOUT", 5.0)
    fc = FakeClient(probes=[_probe(), _probe(body="home"), _probe(body="incorrect password try again")])
    res = bl._run(fc, "a@x.com", "pw")
    assert res.token is None and res.banned is False
    assert "credential" in res.detail


def test_run_poll_ban_twice(monkeypatch):
    bl = _make_login(monkeypatch)
    monkeypatch.setattr(auth, "LOGIN_TIMEOUT", 5.0)
    ban = _probe(body="account has been suspended until tomorrow")
    fc = FakeClient(probes=[_probe(), ban, ban])
    res = bl._run(fc, "a@x.com", "pw")
    assert res.banned is True
    assert "suspension" in res.detail


def test_run_poll_single_ban_not_enough_then_success(monkeypatch):
    bl = _make_login(monkeypatch)
    monkeypatch.setattr(auth, "LOGIN_TIMEOUT", 5.0)
    tok = json.dumps({"value": "tok123", "__version": "0"})
    fc = FakeClient(probes=[
        _probe(),
        _probe(body="account has been suspended"),
        _probe(url="https://chat.deepseek.com/", tok=tok, body="home"),
    ])
    res = bl._run(fc, "a@x.com", "pw")
    assert res.token == "tok123"


def test_run_token_no_baseline(monkeypatch):
    bl = _make_login(monkeypatch)
    monkeypatch.setattr(auth, "LOGIN_TIMEOUT", 5.0)
    tok = json.dumps({"value": "tok123", "__version": "0"})
    fc = FakeClient(probes=[_probe(tok=None), _probe(url="https://chat.deepseek.com/", tok=tok)])
    res = bl._run(fc, "a@x.com", "pw")
    assert res.token == "tok123"


def test_run_token_baseline_same_on_sign_in_keeps_polling_then_timeout(monkeypatch):
    bl = _make_login(monkeypatch)
    # baseline token == poll token while still on sign_in -> must not accept;
    # with LOGIN_TIMEOUT=0 the loop exits immediately as no-token.
    monkeypatch.setattr(auth, "LOGIN_TIMEOUT", 0.0)
    tok = json.dumps({"value": "tok123", "__version": "0"})
    fc = FakeClient(probes=[_probe(tok=tok, url="https://chat.deepseek.com/sign_in")])
    res = bl._run(fc, "a@x.com", "pw")
    assert res.token is None
    assert "no token" in res.detail


def test_run_timeout_detail(monkeypatch):
    bl = _make_login(monkeypatch)
    monkeypatch.setattr(auth, "LOGIN_TIMEOUT", 0.0)
    fc = FakeClient(probes=[_probe(body="home home")])
    res = bl._run(fc, "a@x.com", "pw")
    assert "no token" in res.detail


# --- browser_login dispatch ---

def test_browser_login_real_preferred_normalized(monkeypatch):
    import deepseaport.real_browser as rb

    raw = json.dumps({"value": "x" * 64, "__version": "0"})
    monkeypatch.setattr(rb, "login_with_real_browser", lambda *a, **k: (raw, False, "home"))
    res = browser_login("a@x.com", "pw", _settings())
    assert res.token == "x" * 64
    assert res.detail == "home"


def test_browser_login_real_none_falls_back(monkeypatch):
    import deepseaport.real_browser as rb

    monkeypatch.setattr(rb, "login_with_real_browser", lambda *a, **k: None)

    def _fake_run(self, email, password):
        return LoginResult(token="fallback-tok", detail="via-mcp")

    monkeypatch.setattr(BrowserLogin, "run", _fake_run)
    res = browser_login("a@x.com", "pw", _settings())
    assert res.token == "fallback-tok"


def test_browser_login_real_raises_falls_back(monkeypatch):
    import deepseaport.real_browser as rb

    def _boom(*a, **k):
        raise RuntimeError("cdp broken")

    monkeypatch.setattr(rb, "login_with_real_browser", _boom)

    def _fake_run(self, email, password):
        return LoginResult(token="fallback2")

    monkeypatch.setattr(BrowserLogin, "run", _fake_run)
    assert browser_login("a@x.com", "pw", _settings()).token == "fallback2"


# --- real_browser ---

def test_real_browser_clean():
    from deepseaport.real_browser import _clean

    assert _clean("  a\n\t b   c ") == "a b c"
    assert _clean(None) == ""


def _ws_module(monkeypatch):
    import websocket as _real  # noqa: F401  (only for exception type shape)

    class Timeout(Exception):
        pass

    # Reuse real exception type when available so `except` matches.
    try:
        import websocket as wsmod

        exc = wsmod.WebSocketTimeoutException
    except Exception:
        exc = Timeout
    fake = types.ModuleType("websocket")
    fake.WebSocketTimeoutException = exc
    fake.create_connection = lambda *a, **k: (_ for _ in ()).throw(AssertionError("no net"))
    monkeypatch.setitem(sys.modules, "websocket", fake)
    return fake


def test_real_browser_call_no_ws(monkeypatch):
    from deepseaport.real_browser import RealBrowser

    _ws_module(monkeypatch)
    rb = RealBrowser.__new__(RealBrowser)
    rb.ws = None
    rb._id = 0
    with pytest.raises(RuntimeError):
        rb.call("Page.enable")


def test_real_browser_call_cdp_error(monkeypatch):
    from deepseaport.real_browser import RealBrowser

    fake_mod = _ws_module(monkeypatch)

    class _WS:
        def send(self, payload):
            self.last = json.loads(payload)

        def recv(self):
            return json.dumps({"id": self.last["id"], "error": {"message": "boom"}})

    rb = RealBrowser.__new__(RealBrowser)
    rb.ws = _WS()
    rb._id = 0
    with pytest.raises(RuntimeError, match="CDP"):
        rb.call("Page.enable", timeout=1.0)


def test_real_browser_call_timeout(monkeypatch):
    from deepseaport.real_browser import RealBrowser

    fake_mod = _ws_module(monkeypatch)
    exc = fake_mod.WebSocketTimeoutException

    class _WS:
        def send(self, payload):
            pass

        def recv(self):
            raise exc("timed out")

    rb = RealBrowser.__new__(RealBrowser)
    rb.ws = _WS()
    rb._id = 0
    with pytest.raises(TimeoutError):
        rb.call("Page.enable", timeout=0.05)


def test_real_browser_call_success_skips_events(monkeypatch):
    from deepseaport.real_browser import RealBrowser

    _ws_module(monkeypatch)
    msgs = ["not-json", json.dumps({"id": 999, "result": {"x": 1}})]

    class _WS:
        def __init__(self):
            self.mid = None

        def send(self, payload):
            self.mid = json.loads(payload)["id"]

        def recv(self):
            if msgs:
                return msgs.pop(0)
            return json.dumps({"id": self.mid, "result": {"ok": True}})

    rb = RealBrowser.__new__(RealBrowser)
    rb.ws = _WS()
    rb._id = 0
    assert rb.call("Page.enable", timeout=2.0) == {"ok": True}


def test_real_browser_evaluate_exception_details(monkeypatch):
    from deepseaport.real_browser import RealBrowser

    rb = RealBrowser.__new__(RealBrowser)
    monkeypatch.setattr(rb, "call", lambda *a, **k: {"exceptionDetails": {"text": "bad js"}})
    with pytest.raises(RuntimeError, match="bad js"):
        rb.evaluate("1+1")


def test_real_browser_evaluate_success(monkeypatch):
    from deepseaport.real_browser import RealBrowser

    rb = RealBrowser.__new__(RealBrowser)
    monkeypatch.setattr(rb, "call", lambda *a, **k: {"result": {"value": 42}})
    assert rb.evaluate("1+1") == 42


def test_real_browser_fill_credentials_nofields(monkeypatch):
    from deepseaport.real_browser import RealBrowser

    rb = RealBrowser.__new__(RealBrowser)
    monkeypatch.setattr(rb, "evaluate", lambda *a, **k: "NOFIELDS")
    with pytest.raises(RuntimeError):
        rb.fill_credentials("a@x.com", "pw")


def test_real_browser_fill_credentials_ok(monkeypatch):
    from deepseaport.real_browser import RealBrowser

    rb = RealBrowser.__new__(RealBrowser)
    monkeypatch.setattr(rb, "evaluate", lambda *a, **k: "FILLED")
    rb.fill_credentials("a@x.com", "pw")


def test_real_browser_click_login_nobtn(monkeypatch):
    from deepseaport.real_browser import RealBrowser

    rb = RealBrowser.__new__(RealBrowser)
    monkeypatch.setattr(rb, "evaluate", lambda *a, **k: "NOBTN")
    with pytest.raises(RuntimeError):
        rb.click_login()


def test_real_browser_snapshot_bad_json(monkeypatch):
    from deepseaport.real_browser import RealBrowser

    rb = RealBrowser.__new__(RealBrowser)
    monkeypatch.setattr(rb, "evaluate", lambda *a, **k: "{{bad")
    assert rb.snapshot() == {}


def _stub_login_rb(monkeypatch, snaps, timeout=5.0):
    from deepseaport.real_browser import RealBrowser

    rb = RealBrowser.__new__(RealBrowser)
    rb.timeout = timeout
    monkeypatch.setattr(rb, "navigate", lambda url: None)
    monkeypatch.setattr(rb, "dismiss_cookie_banner", lambda: None)
    monkeypatch.setattr(rb, "fill_credentials", lambda e, p: None)
    monkeypatch.setattr(rb, "click_login", lambda: None)
    it = {"i": 0}

    def _eval(expr, timeout=20.0):
        return "cleared"

    def _snap():
        i = it["i"]
        it["i"] += 1
        if i < len(snaps):
            v = snaps[i]
        else:
            v = snaps[-1]
        if isinstance(v, Exception):
            raise v
        return v

    monkeypatch.setattr(rb, "evaluate", _eval)
    monkeypatch.setattr(rb, "snapshot", _snap)
    monkeypatch.setattr("deepseaport.real_browser.time.sleep", lambda *a, **k: None)
    return rb


def test_real_browser_login_captcha(monkeypatch):
    rb = _stub_login_rb(monkeypatch, [{"url": "u", "token": None, "body": "home", "captcha_visible": True}])
    tok, banned, detail = rb.login("a@x.com", "pw")
    assert tok is None and banned is False
    assert "captcha" in detail


def test_real_browser_login_ban_hint_with_token(monkeypatch):
    rb = _stub_login_rb(monkeypatch, [{"url": "u", "token": "tok1", "body": "account suspended violation"}])
    tok, banned, detail = rb.login("a@x.com", "pw")
    assert tok == "tok1" and banned is True


def test_real_browser_login_rejected(monkeypatch):
    rb = _stub_login_rb(monkeypatch, [{"url": "u", "token": None, "body": "incorrect password login"}])
    tok, banned, detail = rb.login("a@x.com", "pw")
    assert tok is None
    assert "rejected" in detail


def test_real_browser_login_timeout(monkeypatch):
    rb = _stub_login_rb(monkeypatch, [{"url": "u", "token": None, "body": "home"}], timeout=0.0)
    import deepseaport.real_browser as rbm

    monkeypatch.setattr(rbm.time, "monotonic", lambda: 100.0)
    tok, banned, detail = rb.login("a@x.com", "pw")
    assert tok is None
    assert "no token" in detail


def test_login_with_real_browser_no_binary(monkeypatch):
    import deepseaport.real_browser as rbm

    monkeypatch.setattr(rbm, "resolve_browser", lambda value: "")
    assert rbm.login_with_real_browser("a@x.com", "pw", _settings()) is None


def test_login_with_real_browser_start_exception(monkeypatch):
    import deepseaport.real_browser as rbm

    monkeypatch.setattr(rbm, "resolve_browser", lambda value: "/bin/chrome")
    monkeypatch.setattr(rbm.RealBrowser, "start", lambda self: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(rbm.RealBrowser, "close", lambda self: None)
    tok, banned, detail = rbm.login_with_real_browser("a@x.com", "pw", _settings())
    assert tok is None and banned is False
    assert "boom" in detail


def test_login_with_real_browser_headless_flag(monkeypatch):
    import deepseaport.real_browser as rbm

    seen = {}
    real_init = rbm.RealBrowser.__init__

    def _cap_init(self, binary, *, headless=False, timeout=75.0):
        seen["binary"] = binary
        seen["headless"] = headless
        self.binary = binary
        self.headless = headless
        self.timeout = timeout

    monkeypatch.setattr(rbm, "resolve_browser", lambda value: "/bin/chrome")
    monkeypatch.setattr(rbm.RealBrowser, "__init__", _cap_init)
    monkeypatch.setattr(rbm.RealBrowser, "start", lambda self: None)
    monkeypatch.setattr(rbm.RealBrowser, "login", lambda self, e, p: ("tok", False, "d"))
    monkeypatch.setattr(rbm.RealBrowser, "close", lambda self: None)
    out = rbm.login_with_real_browser("a@x.com", "pw", _settings(browser_headless=True))
    assert out == ("tok", False, "d")
    assert seen == {"binary": "/bin/chrome", "headless": True}


# --- mcp_client ---

def _mcp_client_new(**over):
    from deepseaport import mcp_client as mc

    c = mc.McpClient.__new__(mc.McpClient)
    import queue as _q
    import threading as _t

    c.binary = "fake"
    c.profile = "prof"
    c.timeout = 5.0
    c._id = 0
    c._closed = False
    c._state_lock = _t.Lock()
    c._write_lock = _t.Lock()
    c._pending = {}
    c._stderr_tail = []
    for k, v in over.items():
        setattr(c, k, v)
    return c


class _FakeStdin:
    def __init__(self, fail=None):
        self.written = []
        self.fail = fail

    def write(self, s):
        if self.fail:
            raise self.fail
        self.written.append(s)

    def flush(self):
        pass

    def close(self):
        pass


class _FakeProc:
    def __init__(self, stdout_lines=(), stderr_lines=(), poll_value=None, returncode=1):
        self.stdin = _FakeStdin()
        self.stdout = list(stdout_lines)
        self.stderr = list(stderr_lines)
        self._poll = poll_value
        self.returncode = returncode
        self.terminated = False
        self.killed = False
        self.wait_calls = 0
        self.wait_fail_first = False

    def poll(self):
        return self._poll

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        self.wait_calls += 1
        if self.wait_fail_first and self.wait_calls == 1:
            raise subprocess.TimeoutExpired("fake", timeout)
        return 0


def test_mcp_send_closed():
    c = _mcp_client_new(_closed=True, proc=_FakeProc())
    with pytest.raises(RuntimeError, match="closed"):
        c._send("tools/list")


def test_mcp_send_proc_exited_with_tail():
    c = _mcp_client_new(proc=_FakeProc(poll_value=1, returncode=3))
    c._stderr_tail = ["line1", "last-boom"]
    with pytest.raises(RuntimeError, match="last-boom"):
        c._send("tools/list")


def test_mcp_send_timeout(monkeypatch):
    c = _mcp_client_new(proc=_FakeProc())
    # never answer: box.get raises Empty
    monkeypatch.setattr(queue.Queue, "get", lambda self, timeout=None: (_ for _ in ()).throw(queue.Empty()))
    with pytest.raises(TimeoutError, match="timed out"):
        c._send("tools/list", timeout=0.01)


def test_mcp_send_error_in_msg():
    c = _mcp_client_new(proc=_FakeProc())
    box = queue.Queue()
    box.put({"error": {"message": "bad"}})
    c._pending = {}
    orig_send = c._send

    # drive _send but pre-seed the pending box by intercepting stdin write
    real_write = c.proc.stdin.write

    def _write_and_respond(payload):
        real_write(payload)
        # find the pending box and answer it
        mid = json.loads(payload)["id"]
        c._pending[mid].put({"error": {"message": "bad"}})

    c.proc.stdin.write = _write_and_respond
    with pytest.raises(RuntimeError, match="bad"):
        c._send("tools/list", timeout=2.0)


def test_mcp_send_success():
    c = _mcp_client_new(proc=_FakeProc())

    def _write_and_respond(payload):
        mid = json.loads(payload)["id"]
        c._pending[mid].put({"result": {"ok": 1}})

    c.proc.stdin.write = _write_and_respond
    assert c._send("tools/list", timeout=2.0) == {"ok": 1}


def test_mcp_send_none_on_death():
    c = _mcp_client_new(proc=_FakeProc())

    def _write_and_respond(payload):
        mid = json.loads(payload)["id"]
        c._pending[mid].put(None)

    c.proc.stdin.write = _write_and_respond
    with pytest.raises(RuntimeError, match="failed"):
        c._send("tools/list", timeout=2.0)


def test_mcp_read_loop_routes_and_wakes():
    import queue as _q

    c = _mcp_client_new(proc=_FakeProc())
    b1 = _q.Queue()
    b2 = _q.Queue()
    c._pending = {1: b1, 2: b2}
    c.proc.stdout = [
        "garbage line",
        json.dumps({"id": 1, "result": "r1"}),
        "{bad json",
        json.dumps({"no-id": True}),
        json.dumps({"id": 2, "result": "r2"}),
    ]
    c._read_loop()
    assert b1.get_nowait() == {"id": 1, "result": "r1"}
    assert b2.get_nowait() == {"id": 2, "result": "r2"}
    assert c._pending == {}


def test_mcp_read_loop_wakes_waiters_on_death():
    import queue as _q

    c = _mcp_client_new(proc=_FakeProc())
    b = _q.Queue()
    c._pending = {7: b}
    c.proc.stdout = []
    c._read_loop()
    assert b.get_nowait() is None


def test_mcp_close_terminate_and_kill_fallback():
    c = _mcp_client_new(proc=_FakeProc())
    c.proc._poll = None
    c.proc.wait_fail_first = True
    c.close()
    assert c.proc.terminated is True
    assert c.proc.killed is True
    assert c._closed is True
    # second close is a no-op
    c.close()


def test_mcp_context_manager_closes():
    c = _mcp_client_new(proc=_FakeProc(poll_value=0))
    with c as ctx:
        assert ctx is c
    assert c._closed is True


# --- browser_paths ---

def test_resolve_browser_empty():
    from deepseaport.browser_paths import resolve_browser

    assert resolve_browser("") == ""
    assert resolve_browser("   ") == ""


def test_resolve_browser_auto_uses_discover(monkeypatch):
    import deepseaport.browser_paths as bp

    monkeypatch.setattr(bp, "discover_browser", lambda: "/found/chrome")
    assert bp.resolve_browser("auto") == "/found/chrome"
    assert bp.resolve_browser("Default") == "/found/chrome"


def test_resolve_browser_exact_file(tmp_path):
    from deepseaport.browser_paths import resolve_browser

    f = tmp_path / "chrome.exe"
    f.write_text("x")
    assert resolve_browser(str(f)) == str(f)


def test_resolve_browser_via_which(monkeypatch):
    import deepseaport.browser_paths as bp

    monkeypatch.setattr(bp.Path, "is_file", lambda self: False)
    monkeypatch.setattr(bp.shutil, "which", lambda cmd: "/usr/bin/chrome" if cmd == "mychrome" else None)
    assert bp.resolve_browser("mychrome") == "/usr/bin/chrome"


def test_resolve_browser_missing(monkeypatch):
    import deepseaport.browser_paths as bp

    monkeypatch.setattr(bp.Path, "is_file", lambda self: False)
    monkeypatch.setattr(bp.shutil, "which", lambda cmd: None)
    assert bp.resolve_browser("no-such-browser-xyz") == ""


def test_discover_browser_path_hit(monkeypatch):
    import deepseaport.browser_paths as bp

    monkeypatch.setattr(bp.shutil, "which", lambda cmd: "/usr/bin/helium" if cmd == "helium" else None)
    assert bp.discover_browser() == "/usr/bin/helium"


def test_discover_browser_candidates_fallback(monkeypatch):
    import deepseaport.browser_paths as bp

    monkeypatch.setattr(bp.shutil, "which", lambda cmd: None)
    monkeypatch.setattr(bp, "_candidates", lambda: ["/c1/chrome", "/c2/chrome"])

    def _is_file(self):
        return str(self).replace("\\", "/").endswith("/c2/chrome")

    monkeypatch.setattr(bp.Path, "is_file", _is_file)
    assert bp.discover_browser() == "/c2/chrome"


def test_discover_browser_none(monkeypatch):
    import deepseaport.browser_paths as bp

    monkeypatch.setattr(bp.shutil, "which", lambda cmd: None)
    monkeypatch.setattr(bp, "_candidates", lambda: [])
    assert bp.discover_browser() == ""
