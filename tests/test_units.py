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
    assert prompt.startswith("sys")
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
