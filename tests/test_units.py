"""Offline unit tests: no network, no credentials."""

import json

from deepseaport import protocol as P
from deepseaport import tools_support as T
from deepseaport.server import MODELS


def test_model_table():
    assert MODELS["deepseek-flash"] == (False, False, "default")
    assert MODELS["deepseek-flash-reasoner"][0] is True
    assert MODELS["deepseek-flash-search"][1] is True
    assert MODELS["deepseek-flash-reasoner-search"] == (True, True, "default")
    assert len(MODELS) == 4


def test_render_prompt_roles_and_markers():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "again"},
    ]
    prompt = T.render_prompt(msgs)
    assert prompt.startswith("<System>sys")
    assert "<User>hi" in prompt
    assert "<Assistant>hello<endofsentence>" in prompt


def test_render_prompt_tool_cycle():
    msgs = [
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "call_001", "type": "function",
                         "function": {"name": "get_weather", "arguments": '{"a": 1}'}}]},
        {"role": "tool", "tool_call_id": "call_001", "name": "get_weather", "content": "24C"},
    ]
    prompt = T.render_prompt(msgs)
    assert "get_weather" in prompt and "24C" in prompt


def test_render_prompt_list_content():
    msgs = [{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}]
    assert T.render_prompt(msgs) == "a\nb"


def test_tool_system_prompt_lists_schema():
    tools = [{"type": "function", "function": {
        "name": "get_weather", "description": "desc",
        "parameters": {"type": "object", "properties": {
            "location": {"type": "string", "description": "city"}}, "required": ["location"]}}}]
    prompt = T.tool_system_prompt(tools)
    assert "get_weather" in prompt and "location" in prompt and "required" in prompt


def test_parse_tool_calls_object_format():
    text = 'thinking...\n{"tool_calls": [{"id": "call_001", "type": "function", "function": {"name": "get_weather", "arguments": "{\\"location\\": \\"Beijing\\"}"}}]}\ntail'
    tools = [{"type": "function", "function": {"name": "get_weather"}}]
    calls, rest = T.parse_tool_calls(text, tools)
    assert calls and calls[0]["function"]["name"] == "get_weather"
    assert json.loads(calls[0]["function"]["arguments"])["location"] == "Beijing"
    assert "tool_calls" not in rest


def test_parse_tool_calls_parallel_and_unknown_dropped():
    text = '{"tool_calls": [{"function": {"name": "a", "arguments": "{}"}}, {"function": {"name": "nope", "arguments": "{}"}}, {"function": {"name": "b", "arguments": {"x": 1}}}]}'
    tools = [{"type": "function", "function": {"name": "a"}}, {"type": "function", "function": {"name": "b"}}]
    calls, _ = T.parse_tool_calls(text, tools)
    assert [c["function"]["name"] for c in calls] == ["a", "b"]
    assert calls[0]["id"] != calls[1]["id"]


def test_parse_tool_calls_tag_and_fence():
    tools = [{"type": "function", "function": {"name": "calc"}}]
    calls, _ = T.parse_tool_calls('<tool_call>{"name": "calc", "arguments": {"e": "1+1"}}</tool_call>', tools)
    assert calls and calls[0]["function"]["name"] == "calc"
    calls, _ = T.parse_tool_calls('```json\n{"name": "calc", "arguments": {"e": "2"}}\n```', tools)
    assert calls and json.loads(calls[0]["function"]["arguments"])["e"] == "2"


def test_parse_tool_calls_none_for_plain_text():
    assert T.parse_tool_calls("just a normal answer")[0] is None


def test_sse_parser_content_and_thinking():
    lines = [
        'data: {"p": "response/thinking_content", "v": "hmm"}',
        'data: {"p": "response/content", "v": "hi"}',
        'data: {"p": "response/status", "v": "FINISHED"}',
    ]
    kinds = []
    for line in lines:
        kinds += [(e.kind, e.text) for e in P.parse_sse_line(line)]
    assert kinds[0] == ("thinking", "hmm")
    assert kinds[1] == ("content", "hi")
    assert kinds[2][0] == "finished"


def test_sse_parser_list_finish_and_done():
    assert list(P.parse_sse_line('data: {"p": "x", "v": [{"p": "status", "v": "FINISHED"}]}'))[0].kind == "finished"
    assert list(P.parse_sse_line("data: [DONE]"))[0].kind == "finished"
    assert list(P.parse_sse_line("data: not-json")) == []
    assert list(P.parse_sse_line(": keep-alive")) == []


def test_stream_parser_bare_continuation_and_usage():
    parser = P.StreamParser()
    events = []
    for line in ['data: {"p":"response/content","o":"APPEND","v":"web"}',
                 'data: {"v":"2"}',
                 'data: {"v":"api"}',
                 'data: {"p":"response/accumulated_token_usage","o":"SET","v":47}',
                 'data: {"p":"response/status","v":"FINISHED"}']:
        events += list(parser.feed(line))
    texts = [e.text for e in events if e.kind == "content"]
    assert "".join(texts) == "web2api"
    assert [e.text for e in events if e.kind == "usage"] == ["47"]
    assert events[-1].kind == "finished"


def test_completion_payload_shape():
    payload = P.completion_payload("sess", "hi", True, False)
    assert payload["chat_session_id"] == "sess"
    assert payload["parent_message_id"] is None
    assert payload["thinking_enabled"] is True
    assert payload["model_type"] == "default"


def test_parse_dsml_ascii():
    text = ('<||DSML|| calls>\n<||DSML|| invoke name="read">\n'
            '<||DSML|| parameter name="filePath" string="true">a/b.md</||DSML|| parameter>\n'
            '<||DSML|| parameter name="limit" string="false">120</||DSML|| parameter>\n'
            '</||DSML|| invoke>\n</||DSML|| calls>')
    tools = [{"type": "function", "function": {"name": "read"}}]
    calls, rest = T.parse_tool_calls(text, tools)
    assert calls and calls[0]["function"]["name"] == "read"
    args = json.loads(calls[0]["function"]["arguments"])
    assert args["filePath"] == "a/b.md"
    assert args["limit"] == 120
    assert "DSML" not in rest


def test_parse_dsml_fullwidth():
    text = ('<\uFF5C\uFF5CDSML\uFF5C\uFF5C calls>\n'
            '<\uFF5C\uFF5CDSML\uFF5C\uFF5C invoke name="read">\n'
            '<\uFF5C\uFF5CDSML\uFF5C\uFF5C parameter name="filePath" string="true">x.md</\uFF5C\uFF5CDSML\uFF5C\uFF5C parameter>\n'
            '</\uFF5C\uFF5CDSML\uFF5C\uFF5C invoke>\n</\uFF5C\uFF5CDSML\uFF5C\uFF5C calls>')
    tools = [{"type": "function", "function": {"name": "read"}}]
    calls, _ = T.parse_tool_calls(text, tools)
    assert calls and calls[0]["function"]["name"] == "read"


def test_parse_thought_prefix_with_json():
    text = 'Thought: I should read the file.\n{"tool_calls": [{"function": {"name": "read", "arguments": "{}"}}]}'
    tools = [{"type": "function", "function": {"name": "read"}}]
    calls, rest = T.parse_tool_calls(text, tools)
    assert calls and calls[0]["function"]["name"] == "read"
    assert "Thought" not in rest


def test_parse_bare_object_gated():
    tools = [{"type": "function", "function": {"name": "calc"}}]
    calls, _ = T.parse_tool_calls('please {"name": "calc", "arguments": {"e": "1+1"}} done', tools)
    assert calls and calls[0]["function"]["name"] == "calc"
    # Unknown bare JSON must not fire.
    assert T.parse_tool_calls('{"name": "nope", "arguments": {}}', tools)[0] is None
    # Normal code sample without valid name must not fire.
    assert T.parse_tool_calls('here is code {"foo": 1}', tools)[0] is None


def test_parse_case_insensitive_maps_canonical():
    tools = [{"type": "function", "function": {"name": "read"}}]
    calls, _ = T.parse_tool_calls('{"tool_calls": [{"function": {"name": "Read", "arguments": "{}"}}]}', tools)
    assert calls and calls[0]["function"]["name"] == "read"


def test_parse_unknown_dsml_dropped_no_false_positive():
    text = '<||DSML|| invoke name="nope"><||DSML|| parameter name="x" string="true">1</||DSML|| parameter></||DSML|| invoke>'
    tools = [{"type": "function", "function": {"name": "read"}}]
    calls, rest = T.parse_tool_calls(text, tools)
    assert calls is None
    assert rest == text


def test_system_prompt_forbids_dsml_and_lists_names():
    tools = [{"type": "function", "function": {"name": "read"}}]
    prompt = T.tool_system_prompt(tools)
    assert "DSML" in prompt and "read" in prompt
    reminder = T.tool_reminder_prompt(tools)
    assert "tool_calls" in reminder and "read" in reminder


def test_prepare_appends_reminder_only_with_tools():
    from deepseaport.server import _prepare
    body = {"model": "deepseek-flash", "messages": [{"role": "user", "content": "hi"}]}
    assert "Reminder" not in _prepare(body)["prompt"]
    body_tools = {**body, "tools": [{"type": "function", "function": {"name": "read"}}]}
    prep = _prepare(body_tools)
    assert "Reminder" in prep["prompt"]
    assert prep["prompt"].endswith(T.tool_reminder_prompt(body_tools["tools"]))
    # tool_choice none disables tools entirely (no prompt change).
    body_none = {**body_tools, "tool_choice": "none"}
    assert _prepare(body_none)["tools"] == []


def test_prepare_rejects_malformed_messages_as_400():
    """Non-list / non-dict messages must 400, not crash render_prompt (500)."""
    import pytest
    from fastapi import HTTPException

    from deepseaport.server import _prepare

    for bad in ({"role": "user"}, ["hi"], "hello", [["x"]], [None], 123):
        with pytest.raises(HTTPException) as exc_info:
            _prepare({"model": "deepseek-flash", "messages": bad})
        assert exc_info.value.status_code == 400
    # Valid shape still renders.
    ok = _prepare({"model": "deepseek-flash",
                   "messages": [{"role": "user", "content": "hi"}]})
    assert ok["prompt"] == "hi"


def test_ban_extract_completion_shape():
    # Live shape: POST completion -> biz_code 5 "user is muted".
    data = {"code": 0, "msg": "", "data": {"biz_code": 5, "biz_msg": "user is muted",
            "biz_data": {"is_muted": 1, "mute_until": 1789555791.11}}}
    assert P.extract_ban_timestamp(data) == 1789555791.11
    assert P.is_ban_payload(data) is True


def test_ban_extract_users_current_shape():
    # Live shape: GET users/current -> chat.{is_muted,mute_until}.
    data = {"code": 0, "msg": "", "data": {"biz_code": 0, "biz_msg": "",
            "biz_data": {"chat": {"is_muted": 1, "mute_until": 1789555791.11}}}}
    assert P.extract_ban_timestamp(data) == 1789555791.11
    assert P.is_ban_payload(data) is True
    # Valid account: not banned.
    valid = {"code": 0, "msg": "", "data": {"biz_code": 0, "biz_msg": "",
             "biz_data": {"chat": {"is_muted": 0, "mute_until": None}}}}
    assert P.extract_ban_timestamp(valid) is None
    assert P.is_ban_payload(valid) is False
    # Invalid token is not a ban.
    invalid = {"code": 40003, "msg": "Authorization Failed (invalid token)", "data": None}
    assert P.is_ban_payload(invalid) is False


def test_ban_format_day_month():
    label = P.format_ban_label(1789555791.11)
    assert label.startswith("(BANNED:")
    assert "16" in label and "September" in label
    assert P.format_ban_day_month(1789555791.11) == "16 September"
    assert "16 September 2026" in P.format_ban_datetime(1789555791.11)
    assert "16 September" in P.ban_message(1789555791.11)
    assert P.format_ban_label(None) == "(BANNED)"


def test_biz_error_includes_ban_expiry():
    from deepseaport.client import _biz_error
    data = {"code": 0, "msg": "", "data": {"biz_code": 5, "biz_msg": "user is muted",
            "biz_data": {"is_muted": 1, "mute_until": 1789555791.11}}}
    err = _biz_error(data)
    assert err is not None and err[0] == "5"
    assert "16 September" in err[1]


def test_sse_parser_surfaces_ban():
    line = ('data: {"code":0,"msg":"","data":{"biz_code":5,"biz_msg":"user is muted",'
            '"biz_data":{"is_muted":1,"mute_until":1789555791.11}}}')
    evs = list(P.parse_sse_line(line))
    assert evs and evs[0].kind == "error"
    assert "16 September" in evs[0].message


def test_ban_check_skips_short_tokens_no_network():
    from deepseaport.accounts import check_ban_for_token, collect_ban_labels
    from deepseaport.config import AccountConfig
    # Short/test tokens never touch network.
    assert check_ban_for_token("t1") == (False, None)
    assert check_ban_for_token("") == (False, None)
    assert collect_ban_labels([AccountConfig(email="a@x.com", token="t1")]) == {}


def test_extract_token_normalizes_json_shapes():
    from deepseaport.tokens import extract_token

    assert extract_token('{"value":null,"__version":"0"}') == ""
    assert extract_token('{"value":"tok64","__version":"0"}') == "tok64"
    assert extract_token('"tok64"') == "tok64"
    double = json.dumps(json.dumps({"value": "tok99", "__version": "0"}))
    assert extract_token(double) == "tok99"


def test_account_config_normalizes_token_at_boundary():
    from deepseaport.config import AccountConfig

    assert AccountConfig(token='{"value":null,"__version":"0"}').token == ""
    assert AccountConfig(token='{"value":"tok64"}').token == "tok64"


def test_login_state_uses_visible_text_and_overlay_flag():
    from deepseaport.auth import classify_page_state

    # A healthy sign-in page with no visible challenge is not a captcha.
    assert classify_page_state(
        "By signing up or logging in... Forgot password? Sign up Log in", False
    ) == ("", "")
    # The hidden overlay phrase only counts when the overlay is actually visible.
    assert classify_page_state("One more step before you proceed...", True)[0] == "captcha"
    # The generic login failure is a credential/rejected state, so callers can
    # stop polling instead of waiting the full login timeout.
    assert classify_page_state("Log in ... Login failed.", False)[0] == "credential"


def test_mcp_tool_iserror_becomes_exception():
    from deepseaport.auth import _call

    class _Fake:
        def call(self, name, arguments):
            return {"isError": True, "content": [{"type": "text", "text": "Error: nope"}]}

    try:
        _call(_Fake(), "browser_fill", {})
    except RuntimeError as exc:
        assert "nope" in str(exc)
    else:
        raise AssertionError("isError MCP result was treated as success")


def test_parse_ban_until_from_human_login_text():
    import time as _time

    from deepseaport import protocol as P
    from deepseaport.auth import parse_ban_until

    until = parse_ban_until(
        "Due to violation of user policies, your account has been suspended "
        "until 16 September 2026 12:49. If you have any questions, please contact us.")
    assert until is not None and until > _time.time()
    assert P.format_ban_datetime(until) == "16 September 2026 12:49"
    until_pl = parse_ban_until(
        "Twoje konto zostało zawieszone do września 16, 2026 21:18.")
    assert until_pl is not None
    assert P.format_ban_datetime(until_pl) == "16 September 2026 21:18"


def test_persisted_banned_account_label_needs_no_token():
    import time as _time

    from deepseaport.accounts import collect_ban_labels
    from deepseaport.config import AccountConfig

    until = _time.time() + 86400
    acc = AccountConfig(email="banned@x.com", banned=True, banned_until=until)
    labels = collect_ban_labels([acc], force_refresh=True)
    assert labels["banned@x.com"] == until


def test_settings_roundtrips_persisted_ban(tmp_path):
    import time as _time

    from deepseaport.config import AccountConfig, Settings, load_settings

    until = _time.time() + 86400
    path = tmp_path / "config.json"
    settings = Settings(accounts=[AccountConfig(email="b@x.com", banned=True,
                                                banned_until=until)],
                        config_path=str(path))
    settings.save()
    loaded = load_settings(str(path))
    assert loaded.accounts[0].banned is True
    assert loaded.accounts[0].banned_until == until


def test_real_browser_disabled_without_configuration(monkeypatch):
    import deepseaport.real_browser as rb
    from deepseaport.config import Settings

    monkeypatch.setattr(rb, "resolve_browser", lambda value: "")
    assert rb.login_with_real_browser("a@x.com", "pw", Settings(browser_bin="")) is None


def test_settings_roundtrips_browser_fields(tmp_path):
    from deepseaport.config import Settings, load_settings

    path = tmp_path / "config.json"
    settings = Settings(browser_bin="auto", browser_headless=True,
                        config_path=str(path))
    settings.save()
    loaded = load_settings(str(path))
    assert loaded.browser_bin == "auto"
    assert loaded.browser_headless is True


def test_real_browser_login_token_is_normalized(monkeypatch):
    import deepseaport.real_browser as rb
    from deepseaport.auth import browser_login
    from deepseaport.config import Settings

    raw = '{"value":"' + ("x" * 64) + '","__version":"0"}'
    monkeypatch.setattr(rb, "login_with_real_browser",
                        lambda *a, **k: (raw, False, "home page detail"))
    result = browser_login("a@x.com", "pw", Settings(browser_bin="dummy"))
    assert result.token == "x" * 64
    assert result.detail == "home page detail"


def test_set_account_token_normalizes_json_wrapper():
    from deepseaport.accounts import set_account_token
    from deepseaport.config import AccountConfig, Settings

    settings = Settings(accounts=[AccountConfig(email="a@x.com", token="old")])
    acc = set_account_token(settings, "a@x.com",
                            '{"value":"fresh-token","__version":"0"}')
    assert acc is not None and acc.token == "fresh-token"


def test_account_config_normalizes_token_on_every_assignment():
    from deepseaport.config import AccountConfig

    acc = AccountConfig(email="a@x.com")
    acc.token = '{"value":"runtime-token","__version":"0"}'
    assert acc.token == "runtime-token"
    acc.token = '{"value":null,"__version":"0"}'
    assert acc.token == ""
