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
