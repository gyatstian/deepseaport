"""Gap tests for client.py + protocol.py: offline only, no network."""

from __future__ import annotations

import json

import pytest

from deepseaport import protocol as P
from deepseaport import pow as PoW


_MUTE_TS = 1789555791.11
_BAN_COMPLETION = {
    "code": 0, "msg": "",
    "data": {"biz_code": 5, "biz_msg": "user is muted",
             "biz_data": {"is_muted": 1, "mute_until": _MUTE_TS}},
}
_BAN_NO_CODE = {
    "code": 0, "msg": "",
    "data": {"biz_code": 0, "biz_msg": "",
             "biz_data": {"is_muted": 1, "mute_until": _MUTE_TS}},
}
_VALID_USER = {
    "code": 0, "msg": "",
    "data": {"biz_code": 0, "biz_msg": "",
             "biz_data": {"chat": {"is_muted": 0, "mute_until": None}}},
}

_UNSET = object()


class _FakeResp:
    """Mirrors tests/test_upstream_422.py pattern + json()/close tracking."""

    def __init__(self, status_code=200, text="", content_type="application/json",
                 lines=None, json_data=_UNSET, json_raises=False):
        self.status_code = status_code
        self._text = text
        self.headers = {"content-type": content_type}
        self._lines = list(lines) if lines is not None else []
        self._json_data = json_data
        self._json_raises = json_raises
        self.closed = False

    @property
    def text(self) -> str:
        return self._text

    def json(self):
        if self._json_raises:
            raise ValueError("bad json")
        if self._json_data is not _UNSET:
            return self._json_data
        return json.loads(self._text)

    def iter_lines(self):
        return iter(self._lines)

    def close(self) -> None:
        self.closed = True
        return None


def _sess(monkeypatch, *, get=None, get_exc=None, post=None, post_exc=None):
    from deepseaport import client as C

    attrs = {}
    if get_exc is not None:
        def _raise_get(*a, **k):
            raise get_exc
        attrs["get"] = staticmethod(_raise_get)
    elif get is not None:
        attrs["get"] = staticmethod(lambda *a, **k: get)
    if post_exc is not None:
        def _raise_post(*a, **k):
            raise post_exc
        attrs["post"] = staticmethod(_raise_post)
    elif post is not None:
        attrs["post"] = staticmethod(lambda *a, **k: post)
    monkeypatch.setattr(C, "_SESSION", type("_S", (), attrs)())


# client._biz_error ---------------------------------------------------------


def test_biz_error_non_ban_passthrough():
    from deepseaport.client import _biz_error

    err = _biz_error({"code": 40003, "msg": "Auth failed", "data": None})
    assert err == ("40003", "Auth failed")
    err = _biz_error({"code": 0, "msg": "",
                      "data": {"biz_code": "POW_HEADER_ERROR", "biz_msg": "bad pow"}})
    assert err == ("POW_HEADER_ERROR", "bad pow")


def test_biz_error_code_zero_none():
    from deepseaport.client import _biz_error

    assert _biz_error({"code": 0, "msg": "", "data": None}) is None
    assert _biz_error({"code": "0", "msg": "x", "data": None}) is None
    assert _biz_error({"code": None, "msg": "x", "data": None}) is None
    assert _biz_error({"code": "", "msg": "x", "data": None}) is None
    assert _biz_error({"code": 0, "msg": "",
                       "data": {"biz_code": 0, "biz_msg": ""}}) is None
    assert _biz_error(None) is None
    assert _biz_error([]) is None


# fetch_current_user / check_ban --------------------------------------------


def test_fetch_current_user_returns_envelope(monkeypatch):
    from deepseaport import client as C

    resp = _FakeResp(200, json_data=dict(_VALID_USER))
    _sess(monkeypatch, get=resp)
    assert C.fetch_current_user({"h": "1"}) == _VALID_USER


def test_check_ban_top_level_40003(monkeypatch):
    from deepseaport import client as C

    resp = _FakeResp(200, json_data={"code": 40003, "msg": "Auth failed", "data": None})
    _sess(monkeypatch, get=resp)
    assert C.check_ban({"h": "1"}) == (False, None)


def test_check_ban_inner_biz_code_40003(monkeypatch):
    from deepseaport import client as C

    data = {"code": 0, "msg": "",
            "data": {"biz_code": 40003, "biz_msg": "invalid", "biz_data": {}}}
    _sess(monkeypatch, get=_FakeResp(200, json_data=data))
    assert C.check_ban({"h": "1"}) == (False, None)


def test_check_ban_ban_payload(monkeypatch):
    from deepseaport import client as C

    _sess(monkeypatch, get=_FakeResp(200, json_data=dict(_BAN_COMPLETION)))
    banned, ts = C.check_ban({"h": "1"})
    assert banned is True
    assert ts == _MUTE_TS


def test_check_ban_explicit_non_muted(monkeypatch):
    from deepseaport import client as C

    _sess(monkeypatch, get=_FakeResp(200, json_data=dict(_VALID_USER)))
    assert C.check_ban({"h": "1"}) == (False, None)


def test_check_ban_exception(monkeypatch):
    from deepseaport import client as C

    _sess(monkeypatch, get_exc=RuntimeError("net down"))
    assert C.check_ban({"h": "1"}) == (False, None)


# create_session / delete_session / fetch_pow --------------------------------


def test_create_session_biz_error_raises(monkeypatch):
    from deepseaport import client as C

    body = json.dumps(_BAN_COMPLETION)
    resp = _FakeResp(200, text=body, json_data=dict(_BAN_COMPLETION))
    _sess(monkeypatch, post=resp)
    with pytest.raises(RuntimeError, match="biz error"):
        C.create_session({"h": "1"})


def test_create_session_no_id_raises(monkeypatch):
    from deepseaport import client as C

    data = {"code": 0, "msg": "", "data": {"biz_code": 0, "biz_msg": "", "biz_data": {}}}
    resp = _FakeResp(200, text=json.dumps(data), json_data=data)
    _sess(monkeypatch, post=resp)
    with pytest.raises(RuntimeError, match="no id"):
        C.create_session({"h": "1"})


def test_delete_session_swallows_exception(monkeypatch):
    from deepseaport import client as C

    _sess(monkeypatch, post_exc=RuntimeError("net down"))
    C.delete_session({"h": "1"}, "sess-1")  # must not raise


def test_fetch_pow_biz_error_raises(monkeypatch):
    from deepseaport import client as C

    data = {"code": 0, "msg": "",
            "data": {"biz_code": "POW_HEADER_ERROR", "biz_msg": "bad pow"}}
    resp = _FakeResp(200, text=json.dumps(data), json_data=data)
    _sess(monkeypatch, post=resp)
    with pytest.raises(RuntimeError, match="biz error"):
        C.fetch_pow({"h": "1"})


def test_fetch_pow_bad_shape_raises(monkeypatch):
    from deepseaport import client as C

    data = {"code": 0, "msg": "", "data": {"biz_code": 0, "biz_msg": "", "biz_data": {}}}
    resp = _FakeResp(200, text=json.dumps(data), json_data=data)
    _sess(monkeypatch, post=resp)
    with pytest.raises(RuntimeError, match="bad shape"):
        C.fetch_pow({"h": "1"})


# solve_challenge ------------------------------------------------------------


def _challenge() -> dict:
    return {"algorithm": PoW.ALGORITHM, "challenge": "c", "salt": "s",
            "difficulty": 1000, "expire_at": 2 ** 31,
            "signature": "sig", "target_path": "/api/v0/chat/completion"}


def test_solve_challenge_none_raises(monkeypatch):
    from deepseaport import client as C

    monkeypatch.setattr(C.PoW, "solve", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="no answer"):
        C.solve_challenge(_challenge())


def test_solve_challenge_success(monkeypatch):
    from deepseaport import client as C

    monkeypatch.setattr(C.PoW, "solve", lambda *a, **k: 42)
    monkeypatch.setattr(C.PoW, "build_header", lambda ch: f"hdr-{ch['answer']}")
    ch = _challenge()
    assert C.solve_challenge(ch) == "hdr-42"
    assert ch["answer"] == 42


# _ban_stream_event ----------------------------------------------------------


def test_ban_stream_event_without_biz_code():
    from deepseaport.client import _ban_stream_event

    ev = _ban_stream_event(dict(_BAN_NO_CODE))
    assert ev is not None and ev.kind == "error"
    assert ev.code == "5"
    assert ev.message == P.ban_message(_MUTE_TS)
    assert ev.ban_until == _MUTE_TS


def test_ban_stream_event_non_ban_none():
    from deepseaport.client import _ban_stream_event

    assert _ban_stream_event(dict(_VALID_USER)) is None
    assert _ban_stream_event({"code": 0, "data": None}) is None
    assert _ban_stream_event(None) is None


# stream_completion ----------------------------------------------------------


def test_stream_completion_200_json_ban(monkeypatch):
    from deepseaport import client as C

    resp = _FakeResp(200, content_type="application/json",
                     json_data=dict(_BAN_COMPLETION), lines=[])
    _sess(monkeypatch, post=resp)
    events = list(C.stream_completion({"h": "1"}, {"prompt": "hi"}))
    assert len(events) == 1 and events[0].kind == "error"
    assert events[0].code == "5"
    assert resp.closed is True


def test_stream_completion_200_empty_falls_through(monkeypatch):
    from deepseaport import client as C

    resp = _FakeResp(200, text="", content_type="application/json",
                     lines=[], json_raises=True)
    _sess(monkeypatch, post=resp)
    assert list(C.stream_completion({"h": "1"}, {"prompt": "hi"})) == []
    assert resp.closed is True


def test_stream_completion_403_waf(monkeypatch):
    from deepseaport import client as C

    resp = _FakeResp(403, text="blocked", content_type="text/html", lines=[])
    _sess(monkeypatch, post=resp)
    events = list(C.stream_completion({"h": "1"}, {"prompt": "hi"}))
    assert len(events) == 1
    assert events[0].code == "HTTP_403_WAF"
    assert "blocked" in events[0].message


def test_stream_completion_422_detail(monkeypatch):
    from deepseaport import client as C

    body = '{"detail":[{"loc":["body","prompt"],"msg":"field required"}]}'
    resp = _FakeResp(422, text=body, content_type="application/json", lines=[])
    _sess(monkeypatch, post=resp)
    events = list(C.stream_completion({"h": "1"}, {"prompt": "hi"}))
    assert events[0].code == "HTTP_422"
    assert "detail" in events[0].message
    assert "prompt" in events[0].message


def test_stream_completion_raw_json_ban_in_stream(monkeypatch):
    from deepseaport import client as C

    line = json.dumps(_BAN_COMPLETION)
    resp = _FakeResp(200, content_type="text/event-stream", lines=[line])
    _sess(monkeypatch, post=resp)
    events = list(C.stream_completion({"h": "1"}, {"prompt": "hi"}))
    assert len(events) == 1 and events[0].kind == "error"
    assert events[0].code == "5"


def test_stream_completion_finished_closes(monkeypatch):
    from deepseaport import client as C

    resp = _FakeResp(200, content_type="text/event-stream", lines=[
        'data: {"p":"response/status","v":"FINISHED"}',
        'data: {"p":"response/content","v":"late"}',
    ])
    _sess(monkeypatch, post=resp)
    events = list(C.stream_completion({"h": "1"}, {"prompt": "hi"}))
    assert len(events) == 1 and events[0].kind == "finished"
    assert resp.closed is True


def test_stream_completion_error_closes(monkeypatch):
    from deepseaport import client as C

    resp = _FakeResp(200, content_type="text/event-stream", lines=[
        'data: {"type":"error","content":"boom","finish_reason":"rate_limit_reached"}',
        'data: {"p":"response/content","v":"late"}',
    ])
    _sess(monkeypatch, post=resp)
    events = list(C.stream_completion({"h": "1"}, {"prompt": "hi"}))
    assert len(events) == 1 and events[0].kind == "error"
    assert events[0].code == "rate_limit_reached"
    assert resp.closed is True


# protocol._BAN_RE -----------------------------------------------------------


def test_ban_word_commute_transmute_no_match():
    assert P._has_ban_word("commute to work") is False
    assert P._has_ban_word("transmute the sample") is False
    assert P.is_ban_error_text("commute") is False
    assert P.is_ban_error_text("transmute") is False


def test_ban_word_unbanned_no_match():
    assert P._has_ban_word("you are unbanned now") is False
    assert P.is_ban_error_text("unbanned") is False


def test_ban_word_matches():
    for text in ("is_muted", "mute_until", "user is muted", "suspended",
                 "muted", "mute", "banned", "violation"):
        assert P._has_ban_word(text) is True, text
        assert P.is_ban_error_text(text) is True, text


# protocol helpers -----------------------------------------------------------


def test_extract_ban_timestamp_muted_no_ts():
    data = {"code": 0, "msg": "",
            "data": {"biz_code": 0, "biz_msg": "", "biz_data": {"is_muted": 1}}}
    assert P.extract_ban_timestamp(data) is None
    assert P.is_ban_payload(data) is True


def test_extract_ban_timestamp_non_dict():
    assert P.extract_ban_timestamp(None) is None
    assert P.extract_ban_timestamp("x") is None
    assert P.extract_ban_timestamp([]) is None
    assert P.extract_ban_timestamp({}) is None


def test_parse_json_exception():
    class _Bad:
        def json(self):
            raise ValueError("nope")

    assert P.parse_json(_Bad()) == {}


def test_base_headers_defaults():
    h = P.base_headers()
    assert h["User-Agent"] == P.CHROME_UA
    assert "Cookie" not in h
    assert "Authorization" not in h


def test_base_headers_cookie_bearer():
    h = P.base_headers(user_agent="UA-X", waf_cookies="a=b", bearer="tok")
    assert h["User-Agent"] == "UA-X"
    assert h["Cookie"] == "a=b"
    assert h["Authorization"] == "Bearer tok"


def test_bridge_headers_fallback(monkeypatch):
    monkeypatch.setattr("deepseaport.obscura_bridge.get_default_bridge",
                        lambda: None)
    assert P.bridge_headers(bearer="t") == P.base_headers(bearer="t")


def test_bridge_headers_uses_bridge(monkeypatch):
    class _State:
        user_agent = "bridge-ua"

    class _Bridge:
        state = _State()

        def cookie_header(self):
            return "c=d"

    monkeypatch.setattr("deepseaport.obscura_bridge.get_default_bridge",
                        lambda: _Bridge())
    h = P.bridge_headers(bearer="t")
    assert h["User-Agent"] == "bridge-ua"
    assert h["Cookie"] == "c=d"
    assert h["Authorization"] == "Bearer t"


# StreamParser ----------------------------------------------------------------


def test_stream_parser_toast_error():
    evs = list(P.StreamParser().feed(
        'data: {"type":"error","content":"oops","finish_reason":"rate_limit_reached"}'))
    assert len(evs) == 1 and evs[0].kind == "error"
    assert evs[0].code == "rate_limit_reached"
    assert evs[0].message == "oops"


def test_stream_parser_search_status_ignored():
    assert list(P.StreamParser().feed(
        'data: {"p":"response/search_status","v":"searching"}')) == []


def test_stream_parser_batched():
    line = ('data: {"p":"response/content","v":['
            '{"p":"response/content","v":"hi"},'
            '{"p":"response/thinking_content","v":"hmm"},'
            '{"p":"response/accumulated_token_usage","v":123},'
            '{"p":"response/status","v":"FINISHED"}]}')
    evs = list(P.StreamParser().feed(line))
    by_kind = [(e.kind, e.text) for e in evs]
    assert ("content", "hi") in by_kind
    assert ("thinking", "hmm") in by_kind
    assert ("usage", "123") in by_kind
    assert evs[-1].kind == "finished"


def test_stream_parser_non_string_v_ignored():
    parser = P.StreamParser()
    assert list(parser.feed('data: {"p":"response/content","v":123}')) == []
    assert list(parser.feed('data: {"p":"response/content","v":null}')) == []
    assert list(parser.feed('data: {"p":"response/content","v":""}')) == []


def test_stream_parser_continuation_reset_after_finished():
    parser = P.StreamParser()
    assert list(parser.feed('data: {"p":"response/content","v":"a"}'))[0].text == "a"
    assert list(parser.feed('data: {"v":"b"}'))[0].text == "b"
    assert list(parser.feed('data: {"p":"response/status","v":"FINISHED"}'))[0].kind == "finished"
    assert parser.last_path == ""
    assert list(parser.feed('data: {"v":"c"}')) == []
