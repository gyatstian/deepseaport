"""Offline gap tests for tools_support (no network)."""

import json
import logging

import pytest

from deepseaport import tools_support as T


# --- _int_env ---

def test_int_env_missing_returns_default(monkeypatch):
    monkeypatch.delenv("DEEPSEAPORT_TEST_INT_GAP", raising=False)
    assert T._int_env("DEEPSEAPORT_TEST_INT_GAP", 7) == 7


def test_int_env_bad_string_returns_default(monkeypatch):
    monkeypatch.setenv("DEEPSEAPORT_TEST_INT_GAP", "notanint")
    assert T._int_env("DEEPSEAPORT_TEST_INT_GAP", 7) == 7


def test_int_env_empty_returns_default(monkeypatch):
    monkeypatch.setenv("DEEPSEAPORT_TEST_INT_GAP", "")
    assert T._int_env("DEEPSEAPORT_TEST_INT_GAP", 7) == 7


def test_int_env_zero_and_negative_return_default(monkeypatch):
    monkeypatch.setenv("DEEPSEAPORT_TEST_INT_GAP", "0")
    assert T._int_env("DEEPSEAPORT_TEST_INT_GAP", 7) == 7
    monkeypatch.setenv("DEEPSEAPORT_TEST_INT_GAP", "-5")
    assert T._int_env("DEEPSEAPORT_TEST_INT_GAP", 7) == 7


def test_int_env_valid_returns_int(monkeypatch):
    monkeypatch.setenv("DEEPSEAPORT_TEST_INT_GAP", "42")
    assert T._int_env("DEEPSEAPORT_TEST_INT_GAP", 7) == 42


# --- _valid_names ---

def test_valid_names_none_returns_empty():
    assert T._valid_names(None) == set()


def test_valid_names_malformed_returns_empty():
    malformed = [None, "str", 123, [], {}, {"function": None},
                 {"function": {}}, {"function": {"name": ""}},
                 {"function": {"name": None}}, {"function": "notdict"},
                 {"no_function": 1}]
    assert T._valid_names(malformed) == set()


def test_valid_names_mixed_keeps_only_valid():
    tools = [None, {"type": "function", "function": {"name": "read"}},
             {"function": {"name": "write"}}, {"function": {}}]
    assert T._valid_names(tools) == {"read", "write"}


# --- tool_system_prompt ---

def test_system_prompt_empty_still_has_rules_and_contract():
    prompt = T.tool_system_prompt([])
    assert "Rules:" in prompt
    assert "tool_calls" in prompt
    assert "arguments" in prompt


def test_system_prompt_with_tools_lists_required_marks():
    tools = [{"type": "function", "function": {
        "name": "get_weather", "description": "desc",
        "parameters": {"type": "object",
                       "properties": {
                           "location": {"type": "string", "description": "city"},
                           "unit": {"type": "string", "description": "unit"}},
                       "required": ["location"]}}}]
    prompt = T.tool_system_prompt(tools)
    assert "get_weather" in prompt
    assert "location" in prompt
    assert "(required)" in prompt
    # non-required must not carry mark
    unit_line = [ln for ln in prompt.splitlines() if "unit:" in ln][0]
    assert "(required)" not in unit_line


# --- _message_text ---

def test_message_text_list_skips_non_text():
    content = [
        {"type": "text", "text": "a"},
        {"type": "image_url", "image_url": "http://x"},
        {"type": "input_text", "text": "skip"},
        "notadict",
        None,
        {"type": "text"},
        {"type": "text", "text": "b"},
    ]
    assert T._message_text(content) == "a\n\nb"


def test_message_text_non_str_fallback():
    assert T._message_text(None) == ""
    assert T._message_text(123) == "123"
    assert T._message_text(0) == ""
    d = {"a": 1}
    assert T._message_text(d) == str(d)


# --- render_prompt ---

def test_render_prompt_same_role_merge_separator():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "a"},
        {"role": "user", "content": "b"},
    ]
    prompt = T.render_prompt(msgs)
    assert "--- user ---" in prompt
    assert "a" in prompt and "b" in prompt


def test_render_prompt_first_user_block_no_prefix_but_merge():
    msgs = [
        {"role": "user", "content": "a"},
        {"role": "user", "content": "b"},
    ]
    prompt = T.render_prompt(msgs)
    assert prompt == "a\n\n--- user ---\n\nb"


def test_render_prompt_unknown_role_passthrough():
    assert T.render_prompt([{"role": "weird", "content": "hello"}]) == "hello"


def test_render_prompt_tool_without_name_uses_tool():
    prompt = T.render_prompt([{"role": "tool", "content": "out"}])
    assert prompt == "<Tool>Tool tool returned: out"


def test_render_prompt_assistant_malformed_tool_calls_skipped():
    msgs = [{"role": "assistant", "content": "hi",
             "tool_calls": [None, 123, "bad",
                            {"id": "call_001", "type": "function",
                             "function": {"name": "f", "arguments": "{}"}}]}]
    prompt = T.render_prompt(msgs)
    assert "f({})" in prompt
    assert "hi" in prompt


# --- _balanced_json ---

def test_balanced_json_unterminated_returns_none():
    assert T._balanced_json('{"a": 1', 0) is None
    assert T._balanced_json('{"a": "x', 0) is None


def test_balanced_json_escaped_quotes():
    text = '{"a": "b\\"c"} rest'
    assert T._balanced_json(text, 0) == '{"a": "b\\"c"}'


def test_balanced_json_nested_objects():
    text = '{"a": {"b": 1, "c": {"d": 2}}} tail'
    got = T._balanced_json(text, 0)
    assert got == '{"a": {"b": 1, "c": {"d": 2}}}'
    assert json.loads(got)["a"]["c"]["d"] == 2


# --- _normalize_args ---

def test_normalize_args_oversize_dict_raises():
    with pytest.raises(T._OversizeArgs):
        T._normalize_args({"a": "x" * 100}, 0, max_args_chars=10)


def test_normalize_args_oversize_str_raises():
    with pytest.raises(T._OversizeArgs):
        T._normalize_args("x" * 100, 0, max_args_chars=10)


def test_normalize_args_non_json_str_wraps_raw():
    got = T._normalize_args("hello world", 0, max_args_chars=1000)
    assert json.loads(got) == {"_raw": "hello world"}


def test_normalize_args_valid_json_str_passthrough():
    got = T._normalize_args('  {"a": 1}  ', 0, max_args_chars=1000)
    assert got == '{"a": 1}'
    assert json.loads(got) == {"a": 1}


# --- _normalize_calls ---

def test_normalize_calls_unknown_dropped_with_indexes():
    raw = [{"function": {"name": "a", "arguments": "{}"}},
           {"function": {"name": "nope", "arguments": "{}"}},
           {"function": {"name": "b", "arguments": {"x": 1}}}]
    calls, dropped = T._normalize_calls(raw, {"a", "b"})
    assert [c["function"]["name"] for c in calls] == ["a", "b"]
    assert dropped == [1]


def test_normalize_calls_oversize_dropped():
    raw = [{"function": {"name": "a", "arguments": {"big": "x" * 100}}}]
    calls, dropped = T._normalize_calls(raw, {"a"}, max_args_chars=10)
    assert calls == []
    assert dropped == [0]


def test_normalize_calls_case_insensitive_maps_canonical():
    calls, dropped = T._normalize_calls(
        [{"function": {"name": "Read", "arguments": "{}"}}], {"read"})
    assert calls[0]["function"]["name"] == "read"
    assert dropped == []


def test_normalize_calls_missing_id_auto():
    calls, _ = T._normalize_calls(
        [{"function": {"name": "a", "arguments": "{}"}}], {"a"})
    assert calls[0]["id"] == "call_001"
    calls2, _ = T._normalize_calls(
        [{"function": {"name": "a", "arguments": "{}"}},
         {"function": {"name": "a", "arguments": "{}"}}], {"a"})
    assert [c["id"] for c in calls2] == ["call_001", "call_002"]


def test_normalize_calls_non_dict_and_missing_name_dropped():
    raw = [None, "x", {}, {"function": {}}, {"function": {"name": ""}}]
    calls, dropped = T._normalize_calls(raw, {"a"})
    assert calls == []
    assert dropped == [0, 1, 2, 3, 4]


# --- _clean_rest ---

def test_clean_rest_no_preamble_just_strips():
    assert T._clean_rest("  hello  ") == "hello"
    assert T._clean_rest("hello\nworld") == "hello\nworld"


def test_clean_rest_thought_lines_removed():
    assert T._clean_rest("Thought: foo\nreal answer") == "real answer"
    assert T._clean_rest("Thinking: x\nanswer") == "answer"
    assert T._clean_rest("\n\nThought: blah\n  answer  ") == "answer"


def test_clean_rest_stray_dsml_markers_removed():
    assert "DSML" not in T._clean_rest("hi <||DSML|| calls> there")
    assert T._clean_rest("a <||DSML|| invoke name=x> b").strip() != ""


# --- _normalize_dsml_text / _coerce_dsml_value ---

def test_normalize_dsml_text_fullwidth_to_ascii():
    text = "<\uFF5C\uFF5CDSML\uFF5C\uFF5C invoke>"
    assert T._normalize_dsml_text(text) == "<||DSML|| invoke>"
    assert T._normalize_dsml_text("<\uFF1Ctest\uFF1E>") == "<<test>>"


def test_coerce_dsml_value_branches():
    assert T._coerce_dsml_value("true", None) is True
    assert T._coerce_dsml_value("False", None) is False
    assert T._coerce_dsml_value("null", None) is None
    assert T._coerce_dsml_value("none", None) is None
    assert T._coerce_dsml_value("~", None) is None
    assert T._coerce_dsml_value("42", None) == 42
    assert T._coerce_dsml_value("3.14", None) == 3.14
    assert T._coerce_dsml_value('"hi"', None) == "hi"
    assert T._coerce_dsml_value("'hi'", None) == "hi"
    assert T._coerce_dsml_value("hello", None) == "hello"
    # string=true forces raw string even for bool/int-looking values
    assert T._coerce_dsml_value("true", "true") == "true"
    assert T._coerce_dsml_value("120", "True") == "120"


# --- _parse_dsml_calls / _dsml_spans / _strip_spans ---

def test_parse_dsml_calls_missing_name_skipped():
    text = ('<||DSML|| invoke>'
            '<||DSML|| parameter name="x">1</||DSML|| parameter>'
            '</||DSML|| invoke>')
    assert T._parse_dsml_calls(text) == []


def test_dsml_spans_finds_calls_and_invoke():
    text = ('<||DSML|| calls>\n<||DSML|| invoke name="read">\n'
            '<||DSML|| parameter name="x">1</||DSML|| parameter>\n'
            '</||DSML|| invoke>\n</||DSML|| calls>')
    spans = T._dsml_spans(text)
    assert len(spans) >= 3


def test_strip_spans_overlapping_skipped():
    text = "abcdefghij"
    assert T._strip_spans(text, [(0, 5), (2, 8)]) == "fghij"
    assert T._strip_spans(text, []) == text


# --- _looks_like_tool_attempt ---

def test_looks_like_tool_attempt_word_boundary():
    valid = {"read"}
    assert T._looks_like_tool_attempt("already done", valid) is False
    assert T._looks_like_tool_attempt("bread time", valid) is False
    assert T._looks_like_tool_attempt("please read file", valid) is True


def test_looks_like_tool_attempt_substrings_fire():
    assert T._looks_like_tool_attempt('has "tool_calls" here', None) is True
    assert T._looks_like_tool_attempt("<tool_call>{}</tool_call>", None) is True
    assert T._looks_like_tool_attempt("<function_call>{}</function_call>", None) is True
    assert T._looks_like_tool_attempt("some DSML block", None) is True
    assert T._looks_like_tool_attempt("<invoke>hi</invoke>", None) is True
    assert T._looks_like_tool_attempt("just a normal answer", {"read"}) is False


# --- parse_tool_calls limits ---

def test_parse_tool_calls_max_args_chars_tiny_drops_to_none():
    tools = [{"type": "function", "function": {"name": "calc"}}]
    text = '{"tool_calls": [{"function": {"name": "calc", "arguments": {"e": "123456789"}}}]}'
    calls, rest = T.parse_tool_calls(text, tools, max_args_chars=5)
    assert calls is None
    assert rest == text


def test_parse_tool_calls_max_bare_objects_zero_disables_bare():
    tools = [{"type": "function", "function": {"name": "calc"}}]
    text = 'please {"name": "calc", "arguments": {"e": "1+1"}} done'
    calls, _ = T.parse_tool_calls(text, tools)
    assert calls is not None
    calls2, rest2 = T.parse_tool_calls(text, tools, max_bare_objects=0)
    assert calls2 is None
    assert rest2 == text


def test_parse_tool_calls_fenced_tool_calls_array():
    tools = [{"type": "function", "function": {"name": "calc"}}]
    text = '```json\n{"tool_calls": [{"function": {"name": "calc", "arguments": {"e": "2"}}}]}```'
    calls, rest = T.parse_tool_calls(text, tools)
    assert calls and calls[0]["function"]["name"] == "calc"
    assert json.loads(calls[0]["function"]["arguments"])["e"] == "2"
    assert "tool_calls" not in rest


def test_parse_tool_calls_debug_miss_returns_unchanged(caplog):
    tools = [{"type": "function", "function": {"name": "calc"}}]
    text = '{"tool_calls": [{"function": {"name": "nope", "arguments": "{}"}}]}'
    with caplog.at_level(logging.DEBUG, logger="deepseaport.tools"):
        calls, rest = T.parse_tool_calls(text, tools)
    assert calls is None
    assert rest == text


# --- new reliability fixes (large args, lenient JSON, tolerant DSML) ---

def test_parse_tool_calls_accepts_large_write_arguments():
    """The old 8k cap dropped ordinary file writes; default is now 200k."""
    tools = [{"type": "function", "function": {"name": "write"}}]
    payload = json.dumps({"filePath": "big.html", "content": "x" * 12_000})
    text = json.dumps({"tool_calls": [{"id": "call_001", "type": "function",
                                       "function": {"name": "write",
                                                    "arguments": payload}}]})
    calls, _ = T.parse_tool_calls(text, tools)
    assert calls is not None
    args = json.loads(calls[0]["function"]["arguments"])
    assert len(args["content"]) == 12_000


def test_parse_tool_calls_repairs_raw_newlines_inside_arguments():
    tools = [{"type": "function", "function": {"name": "write"}}]
    raw_inner = ('{"filePath": "a.txt", "content": "line1' + chr(10)
                 + 'line2"}')
    escaped_inner = raw_inner.replace(chr(92), chr(92) * 2).replace(
        '"', chr(92) + '"')
    text = ('{"tool_calls": [{"function": {"name": "write", '
            '"arguments": "' + escaped_inner + '"}}]}')
    calls, _ = T.parse_tool_calls(text, tools)
    assert calls is not None
    args = json.loads(calls[0]["function"]["arguments"])
    assert args["filePath"] == "a.txt"
    assert args["content"] == "line1" + chr(10) + "line2"


def test_parse_dsml_generic_close_tags():
    lt, gt, ds = chr(60), chr(62), "||DSML||"
    close = lt + "/" + gt
    text = (lt + ds + "tool_calls" + gt
            + lt + ds + 'invoke name="read"' + gt
            + lt + ds + 'parameter name="filePath" string="true"' + gt
            + "x.md" + close
            + lt + ds + 'parameter name="limit" string="false"' + gt
            + "3" + close
            + close
            + lt + "/" + ds + "tool_calls" + gt)
    tools = [{"type": "function", "function": {"name": "read"}}]
    calls, rest = T.parse_tool_calls(text, tools)
    assert calls is not None
    assert json.loads(calls[0]["function"]["arguments"]) == {
        "filePath": "x.md", "limit": 3}
    assert rest == ""


def test_parse_plain_invoke_parameter_without_dsml_prefix():
    lt, gt = chr(60), chr(62)
    text = (lt + 'invoke name="calc"' + gt
            + lt + 'parameter name="e" string="true"' + gt
            + "1+1" + lt + "/" + gt
            + lt + "/" + gt)
    tools = [{"type": "function", "function": {"name": "calc"}}]
    calls, _ = T.parse_tool_calls(text, tools)
    assert calls is not None
    assert json.loads(calls[0]["function"]["arguments"])["e"] == "1+1"


# --- forgiving bucket (Settings.forgiving_toolcalls) ---

def _bash_tool():
    return [{"type": "function",
             "function": {"name": "bash", "description": "x",
                          "parameters": {
                              "type": "object",
                              "properties": {"command": {"type": "string"}},
                              "required": ["command"]}}}]


def test_forgiving_single_pipe_strict_rejects():
    tools = _bash_tool()
    text = ('<|DSML|invoke name="bash">'
            '<|DSML|parameter name="command" string="true">echo hi</|DSML|parameter>'
            '</|DSML|invoke>')
    assert T.parse_tool_calls(text, tools)[0] is None
    calls, rest = T.parse_tool_calls(text, tools, forgiving=True)
    assert calls is not None
    assert json.loads(calls[0]["function"]["arguments"]) == {"command": "echo hi"}
    assert rest == ""


def test_forgiving_single_pipe_fullwidth():
    fw = chr(0xFF5C)
    tools = _bash_tool()
    text = (f"<{fw}DSML{fw}invoke name=\"bash\">"
            f"<{fw}DSML{fw}parameter name=\"command\" string=\"true\">echo hi</{fw}DSML{fw}parameter>"
            f"</{fw}DSML{fw}invoke>")
    assert T.parse_tool_calls(text, tools)[0] is None
    calls, _ = T.parse_tool_calls(text, tools, forgiving=True)
    assert calls is not None
    assert json.loads(calls[0]["function"]["arguments"]) == {"command": "echo hi"}


def test_forgiving_zero_pipe_and_spaced_markers():
    tools = _bash_tool()
    zero = ('<DSML invoke name="bash">'
            '<DSML parameter name="command" string="true">echo hi</DSML parameter>'
            '</DSML invoke>')
    calls, _ = T.parse_tool_calls(zero, tools, forgiving=True)
    assert calls is not None
    assert json.loads(calls[0]["function"]["arguments"]) == {"command": "echo hi"}
    assert T.parse_tool_calls(zero, tools)[0] is None
    spaced = ('<|| DSML ||invoke name="bash">'
              '<||DSML||parameter name="command" string="true">echo hi</||DSML||parameter>'
              '</||DSML||invoke>')
    calls, _ = T.parse_tool_calls(spaced, tools, forgiving=True)
    assert calls is not None


def test_forgiving_smart_quotes_and_fullwidth_equals_in_attrs():
    tools = _bash_tool()
    text = ('<||DSML||invoke name=\u201cbash\u201d>'
            '<||DSML||parameter name=\u201ccommand\u201d string=\u201ctrue\u201d>'
            'echo hi</||DSML||parameter></||DSML||invoke>')
    assert T.parse_tool_calls(text, tools)[0] is None
    calls, _ = T.parse_tool_calls(text, tools, forgiving=True)
    assert calls is not None
    assert json.loads(calls[0]["function"]["arguments"]) == {"command": "echo hi"}
    eq = ('<||DSML||invoke name\uFF1D"bash">'
          '<||DSML||parameter name="command" string="true">echo hi</||DSML||parameter>'
          '</||DSML||invoke>')
    calls, _ = T.parse_tool_calls(eq, tools, forgiving=True)
    assert calls is not None


def test_forgiving_values_stay_verbatim():
    tools = _bash_tool()
    text = ('<||DSML||invoke name="bash">'
            '<||DSML||parameter name="command" string="true">echo \u201chi\u201d'
            '</||DSML||parameter></||DSML||invoke>')
    calls, _ = T.parse_tool_calls(text, tools, forgiving=True)
    assert calls is not None
    assert json.loads(calls[0]["function"]["arguments"]) == {
        "command": "echo \u201chi\u201d"}


def test_forgiving_html_escaped_blocks():
    tools = _bash_tool()
    text = ('&lt;||DSML||invoke name=&quot;bash&quot;&gt;'
            '&lt;||DSML||parameter name=&quot;command&quot; string=&quot;true&quot;&gt;'
            'echo hi&lt;/||DSML||parameter&gt;&lt;/||DSML||invoke&gt;')
    assert T.parse_tool_calls(text, tools)[0] is None
    calls, rest = T.parse_tool_calls(text, tools, forgiving=True)
    assert calls is not None
    assert json.loads(calls[0]["function"]["arguments"]) == {"command": "echo hi"}
    assert rest == ""


def test_forgiving_html_escaped_value_unescaped():
    tools = _bash_tool()
    text = ('<||DSML||invoke name="bash">'
            '<||DSML||parameter name="command" string="true">a &gt; b'
            '</||DSML||parameter></||DSML||invoke>')
    calls, _ = T.parse_tool_calls(text, tools, forgiving=True)
    assert calls is not None
    assert json.loads(calls[0]["function"]["arguments"]) == {"command": "a > b"}


def test_forgiving_missing_name_single_tool_only():
    tools = _bash_tool()
    text = ('<||DSML||invoke>'
            '<||DSML||parameter name="command" string="true">echo hi</||DSML||parameter>'
            '</||DSML||invoke>')
    assert T.parse_tool_calls(text, tools)[0] is None
    calls, _ = T.parse_tool_calls(text, tools, forgiving=True)
    assert calls is not None
    assert calls[0]["function"]["name"] == "bash"
    # Multi-tool: guessing the tool would be unsafe, still dropped.
    multi = tools + [{"type": "function",
                      "function": {"name": "read", "description": "y",
                                   "parameters": {"type": "object",
                                                  "properties": {"path": {"type": "string"}}}}}]
    assert T.parse_tool_calls(text, multi, forgiving=True)[0] is None
    # Bare invoke with no parameters is prose noise, still skipped.
    assert T.parse_tool_calls(
        "<||DSML||invoke></||DSML||invoke>", tools, forgiving=True)[0] is None


def test_forgiving_unnamed_param_fills_lone_required():
    tools = _bash_tool()
    text = ('<||DSML||invoke name="bash">'
            '<||DSML||parameter string="true">echo hi</||DSML||parameter>'
            '</||DSML||invoke>')
    calls, _ = T.parse_tool_calls(text, tools, forgiving=True)
    assert calls is not None
    assert json.loads(calls[0]["function"]["arguments"]) == {"command": "echo hi"}
    # Ambiguous schemas stay untouched.
    two = [{"type": "function",
            "function": {"name": "cp", "description": "z",
                         "parameters": {"type": "object",
                                        "properties": {"src": {"type": "string"},
                                                       "dst": {"type": "string"}},
                                        "required": ["src", "dst"]}}}]
    text2 = ('<||DSML||invoke name="cp">'
             '<||DSML||parameter string="true">a</||DSML||parameter>'
             '</||DSML||invoke>')
    calls2, _ = T.parse_tool_calls(text2, two, forgiving=True)
    assert calls2 is not None
    assert json.loads(calls2[0]["function"]["arguments"]) == {}


def test_forgiving_json_invoke_body_used_as_args():
    tools = _bash_tool()
    text = '<||DSML||invoke name="bash">{"command": "echo hi"}</||DSML||invoke>'
    calls, rest = T.parse_tool_calls(text, tools, forgiving=True)
    assert calls is not None
    assert json.loads(calls[0]["function"]["arguments"]) == {"command": "echo hi"}
    assert rest == ""
    # Non-JSON bodies still yield empty args (never invented).
    text2 = '<||DSML||invoke name="bash">please run it</||DSML||invoke>'
    calls2, _ = T.parse_tool_calls(text2, tools, forgiving=True)
    assert calls2 is not None
    assert json.loads(calls2[0]["function"]["arguments"]) == {}


def test_forgiving_truncated_counts_new_marker_forms():
    tools = _bash_tool()
    assert T.looks_truncated_tool_attempt(
        'x <|DSML|tool_calls> <|DSML|invoke name="bash"> hi',
        tools, forgiving=True) is True
    assert T.looks_truncated_tool_attempt(
        'x <DSML invoke name="bash"> hi', tools, forgiving=True) is True
    # Strict keeps the old exact behaviour for these slips.
    assert T.looks_truncated_tool_attempt(
        'x <|DSML|tool_calls> <|DSML|invoke name="bash"> hi', tools) is False


def test_looks_truncated_tool_attempt():
    assert T.looks_truncated_tool_attempt(
        '{"tool_calls": [{"function": {"name": "write"}]') is True
    assert T.looks_truncated_tool_attempt('{"tool_calls": []}') is False
    assert T.looks_truncated_tool_attempt("normal answer") is False


def test_parse_tool_calls_recovers_unescaped_openai_arguments():
    tools = [{"type": "function", "function": {"name": "write"}}]
    text = ('{"tool_calls": [{"id": "call_001", "type": "function", '
            '"function": {"name": "write", "arguments": "{"filePath": '
            '"x.html", "content": "hello"}"}}]}')
    calls, rest = T.parse_tool_calls(text, tools)
    assert calls is not None
    assert json.loads(calls[0]["function"]["arguments"]) == {
        "filePath": "x.html", "content": "hello"}
    assert rest == ""
