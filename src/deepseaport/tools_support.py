"""Tool support for harnesses: prompt injection + robust call parsing.

The web model has no native function calling, so we describe tools in a
system prompt and parse the model's structured reply back into OpenAI
`tool_calls`. DSML raw-value blocks are preferred for large/file arguments;
JSON forms are accepted for short calls. Parsing tries (first match wins):

1. {"tool_calls": [{"id":..,"type":"function","function":{"name":..,"arguments":..}}]}
2. <tool_call>{"name":..,"arguments":{..}}</tool_call>  (aliases: function_call, invoke)
3. DSML invoke/parameter blocks, ASCII + fullwidth, named or generic close tags.
4. ```json fenced object with tool_calls / name+arguments keys.
5. Bare unfenced {"name":..,"arguments":..} (valid-names gated).
6. Last-resort recovery of unescaped OpenAI-style `arguments`.

Small model-output damage is repaired (raw control chars in strings, bare
keys, trailing commas, Python literals). Individual argument size defaults
to 200k chars (Settings.tool_args_max_chars / DEEPSEAPORT_MAX_ARGS_CHARS).
Opt-in Settings.forgiving_toolcalls / DEEPSEAPORT_FORGIVING_TOOLCALLS enables
the forgiving bucket (markup repairs: pipe/space/HTML-escaped markers, smart
attribute delimiters, missing-name/param inference, JSON invoke bodies);
strict is default.
"""

from __future__ import annotations

import json
import logging
import os
import re

logger = logging.getLogger("deepseaport.tools")


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        val = int(raw)
    except (ValueError, TypeError):
        return default
    return val if val > 0 else default


MAX_ARGS_CHARS = _int_env("DEEPSEAPORT_MAX_ARGS_CHARS", 200_000)
MAX_BARE_OBJECTS = _int_env("DEEPSEAPORT_MAX_BARE_OBJECTS", 40)


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _resolve_forgiving(explicit: bool | None) -> bool:
    """One toggle for all lenient tool-call repairs.

    Explicit per-call value wins; otherwise env
    DEEPSEAPORT_FORGIVING_TOOLCALLS (or Settings.forgiving_toolcalls plumbed
    through ``parse_tool_calls(forgiving=...)``). Strict by default so
    historical behaviour is unchanged.
    """
    if explicit is not None:
        return bool(explicit)
    return _bool_env("DEEPSEAPORT_FORGIVING_TOOLCALLS", False)


def _valid_names(tools: list[dict] | None) -> set[str]:
    valid: set[str] = set()
    for t in tools or []:
        if isinstance(t, dict):
            name = t.get("function", {}).get("name") if isinstance(t.get("function"), dict) else None
            if name:
                valid.add(name)
    return valid


def _example_param_value(pinfo) -> str:
    if not isinstance(pinfo, dict):
        return "value"
    ptype = str(pinfo.get("type", "string") or "string").lower()
    if ptype in ("integer", "number"):
        return "0"
    if ptype == "boolean":
        return "true"
    if ptype == "array":
        return "[]"
    if ptype == "object":
        return "{}"
    return "value"


def tool_system_prompt(tools: list[dict], max_args_chars: int | None = None) -> str:
    lt = chr(60)
    gt = chr(62)
    ds = "||DSML||"
    op = lt + ds
    cl = lt + "/" + ds
    lines = ["You have access to the following tools. Call one when it helps.",
             ""]
    first_fn: dict = {}
    for tool in tools:
        fn = tool.get("function", {}) if isinstance(tool, dict) else {}
        if not first_fn and isinstance(fn, dict) and fn.get("name"):
            first_fn = fn
        name = fn.get("name", "unknown")
        desc = fn.get("description", "")
        params = fn.get("parameters", {}) or {}
        lines.append(f"Tool: {name}")
        if desc:
            lines.append(f"Description: {desc}")
        props = params.get("properties", {}) or {}
        required = set(params.get("required", []) or [])
        if props:
            lines.append("Parameters:")
            for pname, pinfo in props.items():
                ptype = pinfo.get("type", "string") if isinstance(pinfo, dict) else "string"
                pdesc = pinfo.get("description", "") if isinstance(pinfo, dict) else ""
                mark = " (required)" if pname in required else ""
                lines.append(f"  - {pname}: {ptype}{mark} - {pdesc}")
        lines.append("")

    names = sorted(_valid_names(tools))
    example_name = first_fn.get("name", "tool_name")
    example_props = (first_fn.get("parameters", {}) or {}).get("properties", {}) or {}
    example_required = list((first_fn.get("parameters", {}) or {}).get("required", []) or [])
    example_params = example_required or list(example_props)[:2] or ["param"]
    example_values = {
        pname: _example_param_value(example_props.get(pname, {}))
        for pname in example_params
    }
    call_lines = [op + "tool_calls" + gt,
                  op + f'invoke name="{example_name}"' + gt]
    for pname in example_params:
        call_lines.append(
            op + f'parameter name="{pname}" string="true"' + gt
            + example_values[pname] + cl + "parameter" + gt)
    call_lines += [cl + "invoke" + gt, cl + "tool_calls" + gt]

    arg_limit = max_args_chars if (isinstance(max_args_chars, int)
                                 and max_args_chars > 0) else MAX_ARGS_CHARS
    lines += [
        "PREFERRED call format (raw values; quotes, backslashes, newlines and HTML do NOT need escaping):",
        *call_lines,
        "",
        "Rules:",
        "- Prefer the DSML block format above. Use it for any call with file content, code or a shell command.",
        "- Use the exact tool name and parameter names shown in the schema above; never rename parameters.",
        "- Include every parameter marked (required).",
        "- Output ONLY the tool call. No prose, no Thought:, no Markdown fences, no extra wrapper tags.",
        "- Put parallel calls in one DSML tool_calls block (multiple invoke blocks).",
        "- Keep each call's arguments below " + f"{arg_limit:,}" + " characters. If a file or command is larger, split it into sequential smaller calls: write one chunk, then append the next chunk in later calls. The upstream model output is truncated on very long replies, so one giant call will fail.",
        "- If no tool is needed, answer normally without any tool markup.",
        "- After tool results arrive, use them and answer the user.",
        "",
        "A short JSON form is also accepted for simple calls (arguments may be an object or a JSON string):",
        '{"tool_calls": [{"id": "call_001", "type": "function", "function": {"name": "tool_name", "arguments": {"param": "value"}}}]}',
    ]
    if names:
        lines.append(f"Valid tool names: {', '.join(names)}. Do not invent other names.")
    return "\n".join(lines)


def tool_reminder_prompt(tools: list[dict] | None = None) -> str:
    """Short recency reminder appended after the user prompt (tool calls only).

    The main schema lives in tool_system_prompt() at the start; long histories
    bury it, so this repeats only the output contract at the end.
    """
    names = sorted(_valid_names(tools))
    base = ("Reminder: to call a tool, output ONLY a DSML tool_calls block "
            "(preferred; JSON tool_calls also accepted). Use the exact parameter "
            "names from the schema. Raw DSML parameter values need no escaping. "
            "If file content or a command is large, send several smaller calls "
            "instead of one oversized call. If no tool is needed, answer normally.")
    if names:
        base += f" Valid names: {', '.join(names)}."
    return base


def _message_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(p.get("text", "")) for p in content if isinstance(p, dict) and p.get("type") == "text")
    return str(content or "")


def render_prompt(messages: list[dict]) -> str:
    """Flatten OpenAI messages into the single web `prompt` string.

    - assistant tool_call messages become explicit "I called ..." records,
    - tool messages become "<Tool>Tool <name> returned: ..." records (kept
      distinct from user turns so tool boundaries survive),
    - same-role neighbours merge with a labeled separator so no boundary is
      lost; roles map to <System>/<User>/<Assistant>/<Tool> markers.
    """
    blocks: list[tuple[str, str]] = []
    for msg in messages:
        role = msg.get("role", "user")
        text = _message_text(msg.get("content", ""))
        if role == "assistant" and msg.get("tool_calls"):
            calls = "; ".join(
                f"{c.get('function', {}).get('name', '?')}({c.get('function', {}).get('arguments', '{}')})"
                for c in msg["tool_calls"] if isinstance(c, dict)
            )
            text = (text + "\n" if text else "") + f"[I called tools: {calls}]"
            role = "assistant"
        elif role == "tool":
            name = msg.get("name", "tool")
            text = f"Tool {name} returned: {text}"
            role = "tool"  # distinct role: tool output is new info, not a user turn
        blocks.append((role, text))
    # Accumulate chunks per role instead of concatenating the growing string
    # (O(n^2) on long same-role runs produced by tool loops).
    merged: list[tuple[str, list[str]]] = []
    for role, text in blocks:
        if merged and merged[-1][0] == role:
            merged[-1][1].append(text)
        else:
            merged.append((role, [text]))
    parts = []
    for idx, (role, chunks) in enumerate(merged):
        # Labeled separator preserves tool/message boundaries when same-role
        # turns merge (previously plain "\n\n" lost them).
        if len(chunks) > 1:
            text = f"\n\n--- {role} ---\n\n".join(chunks)
        else:
            text = chunks[0]
        if role == "assistant":
            parts.append(f"<Assistant>{text}<endofsentence>")
        elif role == "system":
            parts.append(f"<System>{text}")
        elif role == "tool":
            parts.append(f"<Tool>{text}")
        elif role == "user":
            parts.append(text if idx == 0 else f"<User>{text}")
        else:
            parts.append(text)
    return "".join(parts)


def _balanced_json(text: str, start: int) -> str | None:
    """Extract a balanced {...} object starting at index `start`."""
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
    return None


_LOOSE_JSON_MISS = object()


def _escape_control_chars_in_strings(text: str) -> str:
    """Escape raw control characters that models commonly put inside JSON strings.

    ``json.loads`` rejects a literal newline/tab inside a quoted value. Models
    frequently emit file content and shell commands that way, which is one of
    the main reasons a structurally valid tool call turns into "no tool call
    parsed". This repair keeps already-escaped sequences untouched.
    """
    out: list[str] = []
    in_str = False
    esc = False
    for ch in text:
        if in_str:
            if esc:
                out.append(ch)
                esc = False
            elif ch == "\\":
                out.append(ch)
                esc = True
            elif ch == '"':
                out.append(ch)
                in_str = False
            elif ch == "\n":
                out.append("\\n")
            elif ch == "\r":
                out.append("\\r")
            elif ch == "\t":
                out.append("\\t")
            elif ord(ch) < 0x20:
                out.append("\\u%04x" % ord(ch))
            else:
                out.append(ch)
        else:
            out.append(ch)
            if ch == '"':
                in_str = True
    return "".join(out)


def _remove_trailing_commas(text: str) -> str:
    """Drop trailing commas before ``}``/``]`` outside string literals."""
    out: list[str] = []
    in_str = False
    esc = False
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if in_str:
            out.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            i += 1
            continue
        if ch == '"':
            in_str = True
            out.append(ch)
            i += 1
            continue
        if ch == ",":
            j = i + 1
            while j < n and text[j] in " \t\r\n":
                j += 1
            if j < n and text[j] in "}]":
                i += 1  # skip comma, whitespace is handled on the next loop
                continue
        out.append(ch)
        i += 1
    return "".join(out)


def _quote_unquoted_keys(text: str) -> str:
    """Quote bare object keys (``{foo: 1}``) outside strings."""
    out: list[str] = []
    i = 0
    n = len(text)
    in_str = False
    esc = False
    while i < n:
        ch = text[i]
        if in_str:
            out.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            i += 1
            continue
        if ch == '"':
            in_str = True
            out.append(ch)
            i += 1
            continue
        if ch in "{,":
            out.append(ch)
            i += 1
            while i < n and text[i].isspace():
                out.append(text[i])
                i += 1
            if i < n and (text[i].isalpha() or text[i] == "_"):
                j = i
                while j < n and (text[j].isalnum() or text[j] in "_.-"):
                    j += 1
                k = j
                while k < n and text[k].isspace():
                    k += 1
                if k < n and text[k] == ":":
                    out.append('"' + text[i:j] + '"')
                    i = j
                    continue
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _replace_python_literals(text: str) -> str:
    """Replace ``True``/``False``/``None`` outside strings with JSON values."""
    out: list[str] = []
    i = 0
    n = len(text)
    in_str = False
    esc = False
    while i < n:
        ch = text[i]
        if in_str:
            out.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            i += 1
            continue
        if ch == '"':
            in_str = True
            out.append(ch)
            i += 1
            continue
        if ch.isalpha() or ch == "_":
            j = i
            while j < n and (text[j].isalnum() or text[j] == "_"):
                j += 1
            word = text[i:j]
            repl = {"True": "true", "False": "false", "None": "null"}.get(word, word)
            out.append(repl)
            i = j
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _loads_json_lenient(text: str):
    """Parse JSON with small, safe repairs for common model-output damage.

    Returns ``_LOOSE_JSON_MISS`` when nothing usable could be produced. Only
    used after strict ``json.loads`` failed, so valid JSON is never changed.
    """
    if not isinstance(text, str):
        return _LOOSE_JSON_MISS
    candidate = text.strip()
    if not candidate:
        return _LOOSE_JSON_MISS
    try:
        return json.loads(candidate)
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    repaired = _replace_python_literals(
        _quote_unquoted_keys(
            _remove_trailing_commas(
                _escape_control_chars_in_strings(candidate))))
    try:
        return json.loads(repaired)
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    # Python-style dicts (single quotes / True / None) still occur when a
    # model is asked for an "arguments" object.
    try:
        import ast
        value = ast.literal_eval(candidate)
        if isinstance(value, (dict, list)):
            return value
    except Exception:
        pass
    return _LOOSE_JSON_MISS


class _OversizeArgs(ValueError):
    """Internal signal: arguments exceed the configured limit; caller drops."""


def _normalize_args(args, call_idx: int, max_args_chars: int | None = None) -> str:
    """Normalize arguments to a JSON string.

    Oversize payloads are rejected (raise _OversizeArgs) instead of being
    truncated: slicing JSON mid-object then wrapping the fragment in
    {"_raw": ...} silently corrupts data and produces invalid arguments.
    """
    limit = max_args_chars if max_args_chars is not None else MAX_ARGS_CHARS
    if isinstance(args, dict):
        dumped = json.dumps(args, ensure_ascii=False)
        if len(dumped) > limit:
            logger.warning("rejecting oversize tool arguments (%d chars > %d)", len(dumped), limit)
            raise _OversizeArgs(f"arguments exceed {limit} chars")
        return dumped
    if isinstance(args, str):
        candidate = args.strip()
        if len(candidate) > limit:
            logger.warning("rejecting oversize tool arguments (%d chars > %d)", len(candidate), limit)
            raise _OversizeArgs(f"arguments exceed {limit} chars")
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            repaired = _loads_json_lenient(candidate)
            if repaired is not _LOOSE_JSON_MISS:
                dumped = json.dumps(repaired, ensure_ascii=False)
                if len(dumped) > limit:
                    logger.warning("rejecting oversize tool arguments (%d chars > %d)",
                                   len(dumped), limit)
                    raise _OversizeArgs(f"arguments exceed {limit} chars")
                logger.info("tool call %d arguments repaired by lenient JSON parser", call_idx)
                return dumped
            logger.warning("tool call %d has non-JSON arguments; wrapping", call_idx)
            return json.dumps({"_raw": candidate}, ensure_ascii=False)
    dumped = json.dumps(args, ensure_ascii=False)
    if len(dumped) > limit:
        logger.warning("rejecting oversize tool arguments (%d chars > %d)", len(dumped), limit)
        raise _OversizeArgs(f"arguments exceed {limit} chars")
    return dumped


def _normalize_calls(
    raw_calls: list,
    valid_names: set[str] | None,
    max_args_chars: int | None = None,
) -> tuple[list[dict], list[int]]:
    """Validate + normalize; returns (calls, dropped_indexes).

    Exact name match wins; case-insensitive fallback maps to the canonical
    name so harnesses don't reject `Read` vs `read`. Unknown names and
    oversize arguments are dropped.
    """
    lower_map: dict[str, str] = {}
    if valid_names:
        for n in valid_names:
            if isinstance(n, str) and n.lower() not in lower_map:
                lower_map[n.lower()] = n
    calls: list[dict] = []
    dropped: list[int] = []
    for i, call in enumerate(raw_calls):
        if not isinstance(call, dict):
            dropped.append(i)
            continue
        fn = call.get("function", call if "name" in call else {})
        name = fn.get("name") if isinstance(fn, dict) else None
        if not isinstance(name, str) or not name:
            dropped.append(i)
            continue
        canonical = name
        if valid_names:
            if name in valid_names:
                canonical = name
            elif name.lower() in lower_map:
                canonical = lower_map[name.lower()]
            else:
                dropped.append(i)
                continue
        args = fn.get("arguments", "{}") if isinstance(fn, dict) else "{}"
        try:
            norm_args = _normalize_args(args, i, max_args_chars)
        except _OversizeArgs:
            dropped.append(i)
            continue
        calls.append({
            "id": call.get("id", f"call_{len(calls) + 1:03d}") if isinstance(call.get("id"), str) else f"call_{len(calls) + 1:03d}",
            "type": "function",
            "function": {"name": canonical, "arguments": norm_args},
        })
    return calls, dropped


_THOUGHT_LINE_RE = re.compile(r"^\s*(thought|thinking|reasoning)\s*:.*$", re.IGNORECASE)


def _clean_rest(rest: str, forgiving: bool = False) -> str:
    """Strip Thought preamble + leftover tool markers when a call was found.

    Only called on the success path; plain answers return untouched.
    Forgiving also strips single/zero-pipe, spaced and HTML-escaped
    ``<|DSML|>`` leftovers.
    """
    cleaned = rest.strip()
    # Drop leading Thought:/Thinking: preamble lines (model narration, not answer).
    lines = cleaned.splitlines()
    while lines and (_THOUGHT_LINE_RE.match(lines[0]) or lines[0].strip() == ""):
        lines.pop(0)
    cleaned = "\n".join(lines).strip()
    # Remove stray DSML marker lines left after extraction.
    if forgiving:
        cleaned = re.sub(r"(?:<|&[lL][tT];)/?\|{0,2}\s*DSML\s*\|{0,2}[^>]*?(?:>|&[gG][tT];)",
                         "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"(?:<|&[lL][tT];)/?\|{1,2}[^>]*?(?:>|&[gG][tT];)",
                         "", cleaned).strip()
    else:
        cleaned = re.sub(r"<\/?\|\|DSML\|\|[^>]*>", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"<\/?\|\|[^>]*>", "", cleaned).strip()
    return cleaned


def _normalize_dsml_text(text: str) -> str:
    # Models emit fullwidth variants (｜＜＞); normalize to ASCII for parsing.
    return (text.replace("\uFF5C", "|").replace("\uFF1C", "<").replace("\uFF1E", ">"))


def _coerce_dsml_value(value: str, string_attr: str | None):
    v = value.strip()
    if string_attr is not None and string_attr.lower() == "true":
        return v
    low = v.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "none", "~"):
        return None
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    if len(v) >= 2 and ((v[0] == '"' and v[-1] == '"') or (v[0] == "'" and v[-1] == "'")):
        try:
            return json.loads(v if v[0] == '"' else f'"{v[1:-1]}"')
        except json.JSONDecodeError:
            return v[1:-1]
    return v


_LT = chr(60)
_GT = chr(62)
_DS = "||DSML||"
_SP = "[ ]*"
# Forgiving bucket (Settings.forgiving_toolcalls). Strict patterns stay byte
# compatible; forgiving patterns accept common model slips in place (no text
# rewrite, so match spans stay valid):
# - 0-2 pipes + inner spaces: <|DSML|>, <DSML invoke>, <|| DSML ||invoke>
#   (fullwidth ｜ is already normalized to ASCII above).
# - HTML-escaped brackets: &lt; &gt; (models escaping "XML").
# - Any whitespace (incl. newlines, NBSP) between marker parts.
_LT_STRICT = re.escape(_LT)
_LT_FORGIVING = r"(?:<|&[lL][tT];)"
_GT_STRICT = re.escape(_GT)
_GT_FORGIVING = r"(?:>|&[gG][tT];)"
_SP_STRICT = _SP
_SP_FORGIVING = r"\s*"
_DS_CORE_STRICT = r"\|\|DSML\|\|"
_DS_CORE_FORGIVING = r"\|{0,2}\s*DSML\s*\|{0,2}"
# Strict attrs stop at the first ">"; forgiving must not stop at an "&gt;"
# inside the tag (HTML-escaped form has no literal ">" until the real end).
_ATTR_STRICT = r"([^>]*)"
_ATTR_FORGIVING = r"((?:(?!&[gG][tT];)[^>])*)"


def _tag_re(name: str, *, closing: bool = False, forgiving: bool = False) -> re.Pattern:
    lt = _LT_FORGIVING if forgiving else _LT_STRICT
    sp = _SP_FORGIVING if forgiving else _SP_STRICT
    gt = _GT_FORGIVING if forgiving else _GT_STRICT
    ds_core = _DS_CORE_FORGIVING if forgiving else _DS_CORE_STRICT
    prefix = lt + ("/" if closing else "")
    return re.compile(
        prefix + sp + "(?:" + ds_core + ")?" + sp
        + name + sp + gt,
        re.IGNORECASE,
    )


def _invoke_open_re(forgiving: bool = False) -> re.Pattern:
    lt = _LT_FORGIVING if forgiving else _LT_STRICT
    sp = _SP_FORGIVING if forgiving else _SP_STRICT
    gt = _GT_FORGIVING if forgiving else _GT_STRICT
    ds_core = _DS_CORE_FORGIVING if forgiving else _DS_CORE_STRICT
    attr = _ATTR_FORGIVING if forgiving else _ATTR_STRICT
    return re.compile(
        lt + sp + "(?:" + ds_core + ")?" + sp
        + "invoke" + attr + gt,
        re.IGNORECASE,
    )


def _param_open_re(forgiving: bool = False) -> re.Pattern:
    lt = _LT_FORGIVING if forgiving else _LT_STRICT
    sp = _SP_FORGIVING if forgiving else _SP_STRICT
    gt = _GT_FORGIVING if forgiving else _GT_STRICT
    ds_core = _DS_CORE_FORGIVING if forgiving else _DS_CORE_STRICT
    attr = _ATTR_FORGIVING if forgiving else _ATTR_STRICT
    return re.compile(
        lt + sp + "(?:" + ds_core + ")?" + sp
        + "parameter" + attr + gt,
        re.IGNORECASE,
    )


_INVOKE_OPEN_RE = _invoke_open_re(False)
_INVOKE_OPEN_FORGIVING_RE = _invoke_open_re(True)
_PARAM_OPEN_RE = _param_open_re(False)
_PARAM_OPEN_FORGIVING_RE = _param_open_re(True)
_INVOKE_CLOSE_RE = _tag_re("invoke", closing=True)
_INVOKE_CLOSE_FORGIVING_RE = _tag_re("invoke", closing=True, forgiving=True)
_PARAM_CLOSE_RE = _tag_re("parameter", closing=True)
_PARAM_CLOSE_FORGIVING_RE = _tag_re("parameter", closing=True, forgiving=True)
_GENERIC_CLOSE_RE = re.compile(re.escape(_LT + "/>"))
_GENERIC_CLOSE_FORGIVING_RE = re.compile(
    _LT_FORGIVING + "/" + _SP_FORGIVING + _GT_FORGIVING)


# Forgiving fix 1+2 (markup side only; parameter *values* stay verbatim so
# intentional file content is never rewritten): smart quotes and fullwidth
# "=" as attribute delimiters, HTML entities inside tags and values.
_SMART_QUOTES = {"\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'"}
_HTML_ATTRS = (("&nbsp;", " "), ("&NBSP;", " "), ("&quot;", '"'),
               ("&QUOT;", '"'), ("&#39;", "'"), ("&#x27;", "'"),
               ("&apos;", "'"), ("&APOS;", "'"))
_HTML_VALUES = (("&lt;", "<"), ("&LT;", "<"), ("&gt;", ">"), ("&GT;", ">"),
                ("&quot;", '"'), ("&QUOT;", '"'), ("&#39;", "'"),
                ("&#x27;", "'"), ("&apos;", "'"), ("&APOS;", "'"))


def _preprocess_dsml_attrs(attr_text: str) -> str:
    """Normalize attribute text for forgiving parsing (spans unaffected)."""
    for smart, plain in _SMART_QUOTES.items():
        attr_text = attr_text.replace(smart, plain)
    attr_text = attr_text.replace("\uFF1D", "=")
    for ent, ch in _HTML_ATTRS:
        attr_text = attr_text.replace(ent, ch)
    # "&amp;" last so "&amp;quot;" stays "&quot;" (single-unescape rule).
    return attr_text.replace("&amp;", "&").replace("&AMP;", "&")


def _unescape_dsml_value(value: str) -> str:
    """Unescape HTML entities inside an extracted argument value."""
    for ent, ch in _HTML_VALUES:
        value = value.replace(ent, ch)
    return value.replace("&amp;", "&").replace("&AMP;", "&")


def _tool_param_info(tools: list[dict] | None) -> tuple[dict, dict]:
    """Schema lookup for forgiving inference: {name: {required, props}}.

    Second item maps lowercased names back to canonical ones.
    """
    info: dict[str, dict] = {}
    lowers: dict[str, str] = {}
    for t in tools or []:
        fn = t.get("function", {}) if isinstance(t, dict) else {}
        if not isinstance(fn, dict):
            continue
        nm = fn.get("name")
        if not isinstance(nm, str) or not nm or nm in info:
            continue
        params = fn.get("parameters", {}) or {}
        if not isinstance(params, dict):
            params = {}
        props = params.get("properties", {}) or {}
        req = params.get("required", []) or []
        info[nm] = {
            "required": [r for r in req if isinstance(r, str)],
            "props": [p for p in props] if isinstance(props, dict) else [],
        }
        lowers.setdefault(nm.lower(), nm)
    return info, lowers


def _assign_unnamed_param(schema: dict | None, args: dict) -> str | None:
    """Target for a nameless <parameter>: single required (else single prop).

    Returns None when ambiguous so nothing is guessed into the wrong slot.
    """
    if not schema:
        return None
    required = [r for r in schema.get("required", []) if r not in args]
    if len(schema.get("required", [])) == 1 and required:
        return required[0]
    props = [p for p in schema.get("props", []) if p not in args]
    if len(schema.get("props", [])) == 1 and props:
        return props[0]
    return None


def _parse_dsml_attrs(attr_text: str, forgiving: bool = False) -> dict[str, str]:
    """Parse ``name="value"`` / ``name='value'`` / ``name=value`` attributes.

    Forgiving also accepts smart quotes (“ ” ‘ ’), fullwidth ＝ and HTML
    entities as delimiters (attribute text only; values stay verbatim).
    """
    if forgiving:
        attr_text = _preprocess_dsml_attrs(attr_text)
    attrs: dict[str, str] = {}
    i = 0
    n = len(attr_text)
    while i < n:
        while i < n and attr_text[i].isspace():
            i += 1
        if i >= n:
            break
        j = i
        while j < n and (attr_text[j].isalnum() or attr_text[j] in "_-:."):
            j += 1
        name = attr_text[i:j].lower()
        while j < n and attr_text[j].isspace():
            j += 1
        if name and j < n and attr_text[j] == "=":
            j += 1
            while j < n and attr_text[j].isspace():
                j += 1
            if j < n and attr_text[j] in "'\"":
                quote = attr_text[j]
                j += 1
                k = attr_text.find(quote, j)
                if k < 0:
                    k = n
                value = attr_text[j:k]
                j = min(k + 1, n)
            else:
                k = j
                while k < n and not attr_text[k].isspace():
                    k += 1
                value = attr_text[j:k]
                j = k
            attrs[name] = value
        i = j + (0 if j > i else 1)
    return attrs


def _find_close(text: str, start: int, end: int,
                specific: re.Pattern,
                generic: re.Pattern | None = None) -> re.Match | None:
    m = specific.search(text, start, end)
    g = (generic or _GENERIC_CLOSE_RE).search(text, start, end)
    if m and g:
        return m if m.start() <= g.start() else g
    return m or g


def _scan_dsml_calls(text: str, forgiving: bool = False,
                     tools: list[dict] | None = None) -> tuple[list[dict], list[tuple[int, int]]]:
    """Tolerant DSML scanner (canonical, plain tags, generic close tags).

    Forgiving repairs (Settings.forgiving_toolcalls):
    - single/zero-pipe, spaced and HTML-escaped ``<|DSML|>`` markers;
    - missing invoke name, but only when exactly one tool is registered
      (multi-tool stays strict: guessing the tool would be unsafe);
    - nameless ``<parameter>`` assigned to a lone required/single property;
    - JSON-object invoke body with no parameter tags used as arguments.
    Nameless invokes with no parameters and no JSON body are still skipped
    (likely prose mentioning markup, not a call).
    """
    norm = _normalize_dsml_text(text)
    invoke_open = _INVOKE_OPEN_FORGIVING_RE if forgiving else _INVOKE_OPEN_RE
    param_open = _PARAM_OPEN_FORGIVING_RE if forgiving else _PARAM_OPEN_RE
    invoke_close = _INVOKE_CLOSE_FORGIVING_RE if forgiving else _INVOKE_CLOSE_RE
    param_close = _PARAM_CLOSE_FORGIVING_RE if forgiving else _PARAM_CLOSE_RE
    generic_close = _GENERIC_CLOSE_FORGIVING_RE if forgiving else _GENERIC_CLOSE_RE
    info, lowers = _tool_param_info(tools) if forgiving else ({}, {})
    single_tool = next(iter(info)) if len(info) == 1 else None
    opens = list(invoke_open.finditer(norm))
    out: list[dict] = []
    spans: list[tuple[int, int]] = []
    for idx, inv in enumerate(opens):
        body_start = inv.end()
        body_end = opens[idx + 1].start() if idx + 1 < len(opens) else len(norm)
        attrs = _parse_dsml_attrs(inv.group(1) or "", forgiving=forgiving)
        raw_name = attrs.get("name")
        # Fix 4: missing name. Tentative single-tool assumption, confirmed
        # below by parameter/JSON intent signals.
        eff_name = raw_name or single_tool
        if not eff_name:
            continue
        schema = info.get(eff_name)
        if schema is None and isinstance(eff_name, str):
            schema = info.get(lowers.get(eff_name.lower(), eff_name))
        named_close = invoke_close.search(norm, body_start, body_end)
        param_limit = named_close.start() if named_close else body_end
        args: dict = {}
        last_param_end = body_start
        params = list(param_open.finditer(norm, body_start, param_limit))
        for pidx, pm in enumerate(params):
            value_start = pm.end()
            value_limit = (params[pidx + 1].start()
                           if pidx + 1 < len(params) else param_limit)
            pclose = _find_close(norm, value_start, value_limit, param_close,
                                 generic_close)
            value_stop = pclose.start() if pclose else value_limit
            pattrs = _parse_dsml_attrs(pm.group(1) or "", forgiving=forgiving)
            pname = pattrs.get("name")
            raw_value = norm[value_start:value_stop]
            if forgiving:
                raw_value = _unescape_dsml_value(raw_value)
            if pname:
                args[pname] = _coerce_dsml_value(
                    raw_value, pattrs.get("string"))
            elif forgiving:
                # Fix 5: nameless parameter with an unambiguous target.
                target = _assign_unnamed_param(schema, args)
                if target:
                    args[target] = _coerce_dsml_value(
                        raw_value, pattrs.get("string"))
            last_param_end = max(last_param_end,
                                 pclose.end() if pclose else value_stop)
        body_json_found = False
        if forgiving and not params:
            # Fix 6: JSON-object body, no parameter tags (model mixed JSON
            # into a DSML invoke). Dict only; anything else keeps {}.
            body = _unescape_dsml_value(norm[body_start:param_limit]).strip()
            if body:
                parsed = _LOOSE_JSON_MISS
                if body[:1] == "{":
                    obj_text = _balanced_json(body, 0) or ""
                    if obj_text:
                        parsed = _loads_json_lenient(obj_text)
                    else:
                        parsed = _loads_json_lenient(body)
                if parsed is not _LOOSE_JSON_MISS and isinstance(parsed, dict):
                    args = dict(parsed)
                    body_json_found = True
        if not raw_name and not params and not body_json_found:
            continue
        if named_close:
            inv_close = named_close
        else:
            # Generic closers pair in order: each parameter consumes the
            # closer after its value; the next one closes the invoke.
            generic = generic_close.search(norm, last_param_end, body_end)
            inv_close = generic
        out.append({"function": {"name": eff_name, "arguments": args}})
        span_end = inv_close.end() if inv_close else body_end
        spans.append((inv.start(), span_end))
    return out, spans


def _parse_dsml_calls(text: str, forgiving: bool = False,
                      tools: list[dict] | None = None) -> list[dict]:
    """Parse DSML / plain invoke+parameter blocks into raw tool calls."""
    calls, _ = _scan_dsml_calls(text, forgiving=forgiving, tools=tools)
    return calls


def _dsml_spans(text: str, forgiving: bool = False,
                tools: list[dict] | None = None) -> list[tuple[int, int]]:
    """Spans to strip on DSML success (length-preserving normalization)."""
    norm = _normalize_dsml_text(text)
    spans: list[tuple[int, int]] = []
    for tag in ("calls", "tool_calls"):
        for rx in (_tag_re(tag, forgiving=forgiving),
                   _tag_re(tag, closing=True, forgiving=forgiving)):
            for m in rx.finditer(norm):
                spans.append((m.start(), m.end()))
    # The tolerant scanner owns invoke/parameter/plain/generic blocks. Its
    # normalized copy is length-preserving, so positions map to `text`.
    _, invoke_spans = _scan_dsml_calls(text, forgiving=forgiving, tools=tools)
    spans.extend(invoke_spans)
    return spans


def _parse_bare_name_objects(
    text: str,
    valid: set[str] | None,
    max_objects: int | None = None,
) -> tuple[list[dict], list[tuple[int, int]]]:
    """Unfenced {"name":..., "arguments":...} objects (parallel-safe).

    Generic scan, so gated by valid-names match to avoid false positives on
    normal code samples. Returns (raw_calls, spans_to_strip).
    """
    limit = max_objects if max_objects is not None else MAX_BARE_OBJECTS
    lowers = {n.lower() for n in valid} if valid else set()
    raw: list[dict] = []
    spans: list[tuple[int, int]] = []
    count = 0
    pos = 0
    n = len(text)
    while pos < n:
        if count >= limit:  # bound scan on long replies (tunable)
            break
        start = text.find("{", pos)
        if start < 0:
            break
        obj_text = _balanced_json(text, start)
        if not obj_text:
            pos = start + 1
            continue
        count += 1
        pos = start + len(obj_text)
        obj = _loads_json_lenient(obj_text)
        if obj is _LOOSE_JSON_MISS:
            continue
        if not isinstance(obj, dict):
            continue
        candidate = None
        if isinstance(obj.get("name"), str) and ("arguments" in obj or "parameters" in obj):
            args = obj.get("arguments", obj.get("parameters", {}))
            candidate = {"function": {"name": obj["name"], "arguments": args if args is not None else "{}"}}
            if isinstance(obj.get("id"), str):
                candidate["id"] = obj["id"]
        elif isinstance(obj.get("function"), dict) and isinstance(obj["function"].get("name"), str):
            candidate = obj
        if candidate is None:
            continue
        nm = candidate["function"]["name"] if isinstance(candidate.get("function"), dict) else None
        if not isinstance(nm, str):
            continue
        if valid and nm not in valid and nm.lower() not in lowers:
            continue
        raw.append(candidate)
        spans.append((start, start + len(obj_text)))
    return raw, spans


def _strip_spans(text: str, spans: list[tuple[int, int]]) -> str:
    if not spans:
        return text
    spans = sorted(spans)
    parts: list[str] = []
    last = 0
    for s, e in spans:
        if s < last:  # overlapping bare objects; skip inner
            continue
        parts.append(text[last:s])
        last = max(last, e)
    parts.append(text[last:])
    return "".join(parts)


def _looks_like_tool_attempt(text: str, valid: set[str] | None) -> bool:
    low = text.lower()
    if '"tool_calls"' in low or "<tool_call" in low or "<function_call" in low or "dsml" in low or "<invoke" in low:
        return True
    if valid:
        for n in valid:
            if isinstance(n, str) and n:
                # Word match: substring "read" must not fire on "already"/"bread".
                if re.search(r"(?<!\w)" + re.escape(n) + r"(?!\w)", text, re.IGNORECASE):
                    return True
    return False


def looks_truncated_tool_attempt(text: str, tools: list[dict] | None = None,
                                 forgiving: bool | None = None) -> bool:
    """Best-effort signal that a tool attempt was cut off mid-output.

    DeepSeek web has no finish_reason, so a model reply truncated by the
    upstream token limit arrives exactly like a normal stop. If the reply
    clearly looks like a tool call and JSON/DSML delimiters are unbalanced,
    callers can report ``finish_reason: "length"`` so harnesses retry or
    split the request instead of treating raw markup as the final answer.
    """
    if not isinstance(text, str) or not text.strip():
        return False
    low = text.lower()
    strong = ('"tool_calls"' in low or "<tool_calls" in low or "<tool_call"
              in low or "dsml" in low or "<invoke" in low
              or "<function_call" in low)
    if not strong:
        return False
    stack: list[str] = []
    in_str = False
    esc = False
    closes = {"{": "}", "[": "]", "(": ")"}
    for ch in text:
        if in_str:
            if esc:
                esc = False
            elif ord(ch) == 92:  # backslash
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in closes:
            stack.append(closes[ch])
        elif ch in "}])":
            if not stack or stack.pop() != ch:
                return True
    if stack or in_str:
        return True
    # DSML/XML-style blocks can be truncated without unbalanced braces.
    if "dsml" in low or "<invoke" in low:
        forgiving_on = _resolve_forgiving(forgiving)
        # Normalize fullwidth first so ｜ variants count the same as ASCII.
        norm_low = _normalize_dsml_text(text).lower()
        lt = chr(60)
        generic_close = norm_low.count(lt + "/" + chr(62))
        if forgiving_on:
            for tag in ("tool_calls", "calls", "invoke", "parameter"):
                opened = len(re.findall(
                    r"(?:<|&[lL][tT];)\|{0,2}\s*dsml\s*\|{0,2}" + tag, norm_low))
                closed = len(re.findall(
                    r"(?:<|&[lL][tT];)/\|{0,2}\s*dsml\s*\|{0,2}" + tag, norm_low))
                if opened > closed + generic_close:
                    return True
        else:
            ds = "||dsml||"
            for tag in ("tool_calls", "calls", "invoke", "parameter"):
                opened = norm_low.count(lt + ds + tag)
                closed = norm_low.count(lt + "/" + ds + tag)
                if opened > closed + generic_close:
                    return True
    return False


_FENCED_BLOCK_RE = re.compile(r"```(?:\w+)?\s*(.*?)```", re.DOTALL)


def _recover_loose_openai_call(text: str, valid: set[str] | None,
                                limit_args: int):
    """Last-resort recovery for an unescaped OpenAI tool-call JSON attempt.

    Models often write ``"arguments": "{"path": ..., "content": ...}"`` (the
    inner JSON was not escaped).  Strict parsing rejects it, but the intended
    payload is usually recoverable by taking the outer quote before the final
    call delimiters and parsing the inner object leniently.  Only attempted
    after every normal format failed, so valid replies are unchanged.
    """
    if not valid or text.count('"arguments"') != 1:
        return None
    low = text.lower()
    if '"tool_calls"' not in low and '"function"' not in low:
        return None
    m = re.search(r'"arguments"[ ]*:[ ]*', text)
    if m is None:
        return None
    value_start = m.end()
    if value_start >= len(text):
        return None
    if text[value_start] == '"':
        end = text.rfind('"')
        if end <= value_start:
            return None
        raw = text[value_start + 1:end]
    elif text[value_start] in "{[":
        raw = _balanced_json(text, value_start)
        if not raw:
            return None
    else:
        return None
    parsed = _loads_json_lenient(raw)
    if parsed is _LOOSE_JSON_MISS and isinstance(raw, str):
        parsed = _loads_json_lenient(raw.strip())
    if parsed is _LOOSE_JSON_MISS:
        return None
    if isinstance(parsed, str):
        parsed = _loads_json_lenient(parsed)
    if parsed is _LOOSE_JSON_MISS or not isinstance(parsed, (dict, list)):
        return None
    lower_map = {n.lower(): n for n in valid if isinstance(n, str)}
    name = None
    for candidate in reversed(re.findall(r'"name"[ ]*:[ ]*"([^"]+)"',
                                         text[:value_start])):
        if candidate in valid:
            name = candidate
            break
        mapped = lower_map.get(candidate.lower())
        if mapped:
            name = mapped
            break
    if not name:
        return None
    calls, _ = _normalize_calls([{"function": {"name": name,
                                               "arguments": parsed}}],
                                valid, limit_args)
    return calls or None


def parse_tool_calls(
    text: str,
    tools: list[dict] | None = None,
    *,
    max_args_chars: int | None = None,
    max_bare_objects: int | None = None,
    forgiving: bool | None = None,
) -> tuple[list[dict] | None, str]:
    """Return (tool_calls or None, remaining_text).

    Accepted formats (first match wins):
    1. {"tool_calls": [...]} — balanced scan, Thought prefix tolerated.
    2. <tool_call>/<function_call>/<invoke> JSON tags.
    3. <||DSML|| invoke/parameter blocks (ASCII + fullwidth variants).
    4. fenced ```json block describing a call.
    5. bare unfenced {"name":..., "arguments":...} (valid-names gated).

    Plain answers return (None, original_text) unchanged.

    Limits are tunable: `max_args_chars` defaults to MAX_ARGS_CHARS
    (env DEEPSEAPORT_MAX_ARGS_CHARS), `max_bare_objects` defaults to
    MAX_BARE_OBJECTS (env DEEPSEAPORT_MAX_BARE_OBJECTS).

    Forgiving bucket (`forgiving=True`, env DEEPSEAPORT_FORGIVING_TOOLCALLS,
    or Settings.forgiving_toolcalls): extra repairs for model slips.
    Strict (default) keeps historical behaviour. Forgiving repairs:
    1. single/zero-pipe, spaced and HTML-escaped DSML markers;
    2. smart quotes / fullwidth ＝ / entities as attribute delimiters
       (parameter values stay verbatim);
    3. missing invoke name when exactly one tool is registered;
    4. nameless parameter assigned to a lone required/single property;
    5. JSON-object invoke body with no parameter tags used as arguments.
    Markup only: values are never invented, truncated calls never executed.
    """
    forgiving_on = _resolve_forgiving(forgiving)
    valid = _valid_names(tools) if tools else set()
    valid_or_none = valid if tools else None
    limit_args = max_args_chars if max_args_chars is not None else MAX_ARGS_CHARS
    limit_bare = max_bare_objects if max_bare_objects is not None else MAX_BARE_OBJECTS

    # Format 1: {"tool_calls": [...]} — single left-to-right balanced scan.
    # Each balanced {...} is visited once and skipped past, so total work is
    # linear in the reply size (no backtrack per marker).
    pos = 0
    n = len(text)
    while pos < n:
        start = text.find("{", pos)
        if start < 0:
            break
        obj_text = _balanced_json(text, start)
        if not obj_text:
            pos = start + 1
            continue
        if '"tool_calls"' not in obj_text:
            pos = start + len(obj_text)
            continue
        obj = _loads_json_lenient(obj_text)
        if obj is _LOOSE_JSON_MISS:
            pos = start + 1
            continue
        raw = obj.get("tool_calls") if isinstance(obj, dict) else None
        if isinstance(raw, list) and raw:
            calls, _ = _normalize_calls(raw, valid_or_none, limit_args)
            if calls:
                rest = _clean_rest(text[:start] + text[start + len(obj_text):], forgiving_on)
                return calls, rest
        pos = start + len(obj_text)

    # Format 2: <tool_call>/<function_call>/<invoke> tags.
    tag_pat = re.compile(
        r"<(?:tool_call|function_call|invoke)>(.*?)</(?:tool_call|function_call|invoke)>",
        re.DOTALL | re.IGNORECASE,
    )
    tagged = []
    for m in tag_pat.finditer(text):
        obj = _loads_json_lenient(m.group(1).strip())
        if obj is _LOOSE_JSON_MISS:
            continue
        if isinstance(obj, dict) and obj.get("name"):
            tagged.append({"function": obj})
        elif isinstance(obj, dict) and isinstance(obj.get("function"), dict):
            tagged.append(obj)
    if tagged:
        calls, _ = _normalize_calls(tagged, valid_or_none, limit_args)
        if calls:
            rest = _clean_rest(tag_pat.sub("", text), forgiving_on)
            return calls, rest

    # Format 3: DSML / plain invoke+parameter blocks. The scanner is gated
    # on an invoke tag or DSML marker so ordinary prose with angle brackets
    # is not scanned.
    low = text.lower()
    if ("dsml" in low or "<invoke" in low
            or (forgiving_on and "&lt;invoke" in low)):
        dsml_raw = _parse_dsml_calls(text, forgiving=forgiving_on, tools=tools)
        if dsml_raw:
            calls, dropped = _normalize_calls(dsml_raw, valid_or_none, limit_args)
            if calls:
                # Strip on normalized text then map back by length-stable replace:
                # fullwidth->ASCII is length-stable (1 char -> 1 char), so spans align.
                spans = _dsml_spans(text, forgiving=forgiving_on, tools=tools)
                rest = _clean_rest(_strip_spans(text, spans), forgiving_on)
                # If spans came from normalized copy with same length, safe.
                return calls, rest
            # DSML present but names unknown: don't fall through to bare scan
            # producing junk; log and keep scanning fenced (explicit user intent).

    # Format 4: fenced ```json block describing a call.
    # Balanced scan inside each fence so nested arguments survive
    # (non-greedy \{.*?\} stopped at the first inner "}").
    for m in _FENCED_BLOCK_RE.finditer(text):
        inner = m.group(1)
        inner_pos = 0
        while inner_pos < len(inner):
            obj_start = inner.find("{", inner_pos)
            if obj_start < 0:
                break
            obj_text = _balanced_json(inner, obj_start)
            if not obj_text:
                inner_pos = obj_start + 1
                continue
            obj = _loads_json_lenient(obj_text)
            if obj is _LOOSE_JSON_MISS:
                inner_pos = obj_start + 1
                continue
            raw = None
            if isinstance(obj, dict) and isinstance(obj.get("tool_calls"), list):
                raw = obj["tool_calls"]
            elif isinstance(obj, dict) and obj.get("name"):
                raw = [{"function": obj}]
            elif isinstance(obj, dict) and isinstance(obj.get("function"), dict):
                raw = [obj]
            if raw:
                calls, _ = _normalize_calls(raw, valid_or_none, limit_args)
                if calls:
                    rest = _clean_rest(text[:m.start()] + text[m.end():], forgiving_on)
                    return calls, rest
            inner_pos = obj_start + len(obj_text)

    # Format 5: bare unfenced name/arguments object.
    if valid:
        bare_raw, bare_spans = _parse_bare_name_objects(text, valid, limit_bare)
        if bare_raw:
            calls, _ = _normalize_calls(bare_raw, valid_or_none, limit_args)
            if calls:
                # Strip only the spans that survived normalization.
                rest = _clean_rest(_strip_spans(text, bare_spans), forgiving_on)
                return calls, rest

    # Last resort: recover an unescaped OpenAI-style JSON attempt, e.g.
    # "arguments": "{"path": ...}" where the inner object was not escaped.
    recovered = _recover_loose_openai_call(text, valid_or_none, limit_args)
    if recovered:
        logger.info("recovered loose OpenAI tool call (%d call(s))", len(recovered))
        return recovered, ""

    if tools and _looks_like_tool_attempt(text, valid):
        logger.debug("tool parse miss preview=%.300s", text[:300])
    return None, text
