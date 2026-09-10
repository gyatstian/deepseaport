"""Tool support for harnesses: prompt injection + robust call parsing.

The web model has no native function calling, so we describe tools in a
system prompt and parse the model's structured reply back into OpenAI
`tool_calls`. Five reply formats are accepted (first match wins):

1. {"tool_calls": [{"id":..,"type":"function","function":{"name":..,"arguments":..}}]}
2. <tool_call>{"name":..,"arguments":{..}}</tool_call>  (aliases: function_call, invoke)
3. DSML <||DSML|| invoke name=../parameter blocks> (ASCII + fullwidth)
4. ```json fenced object with tool_calls / name+arguments keys.
5. Bare unfenced {"name":..,"arguments":..} (valid-names gated).
"""

from __future__ import annotations

import json
import logging
import re

logger = logging.getLogger("deepseaport.tools")

MAX_ARGS_CHARS = 8000


def _valid_names(tools: list[dict] | None) -> set[str]:
    valid: set[str] = set()
    for t in tools or []:
        if isinstance(t, dict):
            name = t.get("function", {}).get("name") if isinstance(t.get("function"), dict) else None
            if name:
                valid.add(name)
    return valid


def tool_system_prompt(tools: list[dict]) -> str:
    lines = ["You have access to the following tools. Use them when they help answer.",
             ""]
    for tool in tools:
        fn = tool.get("function", {}) if isinstance(tool, dict) else {}
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
    lines += [
        "To call tools, reply with ONLY this JSON object and nothing else:",
        '{"tool_calls": [{"id": "call_001", "type": "function", '
        '"function": {"name": "tool_name", "arguments": "{\\"param\\": \\"value\\"}"}}]}',
        "",
        "Rules:",
        '- "arguments" must be a JSON-encoded string, never a nested object.',
        "- Put parallel calls in one array with ids call_001, call_002, ...",
        "- If no tool is needed, answer normally without any JSON.",
        "- After tool results arrive, use them and answer the user.",
        "- When calling tools: no prose, no Thought:, no Thinking:, no DSML tags,",
        "  no <||...|> markers before or after the JSON object.",
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
    base = ("Reminder: if you need a tool, reply with ONLY "
            '{"tool_calls": [...]} and nothing else (no Thought:, no DSML). '
            "If no tool is needed, answer normally.")
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
    - tool messages become "Tool <name> returned: ..." records,
    - same-role neighbours merge; roles map to <User>/<Assistant> markers.
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
            role = "user"  # tool output is new information for the model
        blocks.append((role, text))
    merged: list[tuple[str, str]] = []
    for role, text in blocks:
        if merged and merged[-1][0] == role:
            merged[-1] = (role, merged[-1][1] + "\n\n" + text)
        else:
            merged.append((role, text))
    parts = []
    for idx, (role, text) in enumerate(merged):
        if role == "assistant":
            parts.append(f"<Assistant>{text}<endofsentence>")
        elif role in ("user", "system"):
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


def _normalize_args(args, call_idx: int) -> str:
    if isinstance(args, dict):
        return json.dumps(args, ensure_ascii=False)
    if isinstance(args, str):
        candidate = args.strip()
        if len(candidate) > MAX_ARGS_CHARS:
            logger.warning("truncating oversized tool arguments (%d chars)", len(candidate))
            candidate = candidate[:MAX_ARGS_CHARS]
        try:
            json.loads(candidate)
        except json.JSONDecodeError:
            logger.warning("tool call %d has non-JSON arguments; wrapping", call_idx)
            return json.dumps({"_raw": candidate}, ensure_ascii=False)
        return candidate
    return json.dumps(args, ensure_ascii=False)


def _normalize_calls(raw_calls: list, valid_names: set[str] | None) -> tuple[list[dict], list[int]]:
    """Validate + normalize; returns (calls, dropped_indexes).

    Exact name match wins; case-insensitive fallback maps to the canonical
    name so harnesses don't reject `Read` vs `read`. Unknown names dropped.
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
        calls.append({
            "id": call.get("id", f"call_{len(calls) + 1:03d}") if isinstance(call.get("id"), str) else f"call_{len(calls) + 1:03d}",
            "type": "function",
            "function": {"name": canonical, "arguments": _normalize_args(args, i)},
        })
    return calls, dropped


_THOUGHT_PREFIX_RE = re.compile(r"^\s*(thought|thinking|reasoning)\s*:.{0,500}?\n", re.IGNORECASE | re.DOTALL)
_THOUGHT_LINE_RE = re.compile(r"^\s*(thought|thinking|reasoning)\s*:.*$", re.IGNORECASE)


def _clean_rest(rest: str) -> str:
    """Strip Thought preamble + leftover tool markers when a call was found.

    Only called on the success path; plain answers return untouched.
    """
    cleaned = rest.strip()
    # Drop leading Thought:/Thinking: preamble lines (model narration, not answer).
    lines = cleaned.splitlines()
    while lines and (_THOUGHT_LINE_RE.match(lines[0]) or lines[0].strip() == ""):
        lines.pop(0)
    cleaned = "\n".join(lines).strip()
    # Remove stray DSML marker lines left after extraction.
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


def _parse_dsml_calls(text: str) -> list[dict]:
    """Parse <||DSML|| invoke name=...><||DSML|| parameter ...>...</> blocks."""
    norm = _normalize_dsml_text(text)
    invoke_pat = re.compile(
        r"<\|\|DSML\|\|\s*invoke\b([^>]*)>(.*?)</\|\|DSML\|\|\s*invoke\s*>",
        re.DOTALL | re.IGNORECASE,
    )
    param_pat = re.compile(
        r"<\|\|DSML\|\|\s*parameter\b([^>]*)>(.*?)</\|\|DSML\|\|\s*parameter\s*>",
        re.DOTALL | re.IGNORECASE,
    )
    attr_pat = re.compile(r'(\w+)\s*=\s*(".*?"|\'.*?\'|\S+)')
    out: list[dict] = []
    for inv in invoke_pat.finditer(norm):
        attrs = dict((k.lower(), v.strip('"\''))
                     for k, v in attr_pat.findall(inv.group(1) or ""))
        name = attrs.get("name")
        if not name:
            continue
        args: dict = {}
        for pm in param_pat.finditer(inv.group(2) or ""):
            pattrs = dict((k.lower(), v.strip('"\''))
                          for k, v in attr_pat.findall(pm.group(1) or ""))
            pname = pattrs.get("name")
            if not pname:
                continue
            args[pname] = _coerce_dsml_value(pm.group(2) or "", pattrs.get("string"))
        out.append({"function": {"name": name, "arguments": args}})
    return out


def _dsml_spans(text: str) -> list[tuple[int, int]]:
    """Spans to strip on DSML success (computed on normalized text)."""
    norm = _normalize_dsml_text(text)
    spans: list[tuple[int, int]] = []
    for pat in (r"<\|\|DSML\|\|\s*calls\s*>", r"</\|\|DSML\|\|\s*calls\s*>",
                r"<\|\|DSML\|\|\s*invoke\b[^>]*>.*?</\|\|DSML\|\|\s*invoke\s*>"):
        for m in re.finditer(pat, norm, re.DOTALL | re.IGNORECASE):
            spans.append((m.start(), m.end()))
    return spans


def _parse_bare_name_objects(text: str, valid: set[str] | None) -> tuple[list[dict], list[tuple[int, int]]]:
    """Unfenced {"name":..., "arguments":...} objects (parallel-safe).

    Generic scan, so gated by valid-names match to avoid false positives on
    normal code samples. Returns (raw_calls, spans_to_strip).
    """
    lowers = {n.lower() for n in valid} if valid else set()
    raw: list[dict] = []
    spans: list[tuple[int, int]] = []
    count = 0
    for m in re.finditer(r"\{", text):
        if count >= 40:  # bound O(n^2) scan on long replies
            break
        obj_text = _balanced_json(text, m.start())
        if not obj_text:
            continue
        count += 1
        try:
            obj = json.loads(obj_text)
        except json.JSONDecodeError:
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
        spans.append((m.start(), m.start() + len(obj_text)))
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
            if isinstance(n, str) and n and n.lower() in low:
                return True
    return False


def parse_tool_calls(text: str, tools: list[dict] | None = None) -> tuple[list[dict] | None, str]:
    """Return (tool_calls or None, remaining_text).

    Accepted formats (first match wins):
    1. {"tool_calls": [...]} — balanced scan, Thought prefix tolerated.
    2. <tool_call>/<function_call>/<invoke> JSON tags.
    3. <||DSML|| invoke/parameter blocks (ASCII + fullwidth variants).
    4. fenced ```json block describing a call.
    5. bare unfenced {"name":..., "arguments":...} (valid-names gated).

    Plain answers return (None, original_text) unchanged.
    """
    valid = _valid_names(tools) if tools else set()
    valid_or_none = valid if tools else None

    # Format 1: {"tool_calls": [...]} — balanced scan for robustness.
    for match in re.finditer(r'"tool_calls"\s*:\s*\[', text):
        arr_start = match.group(0).rfind("[") + match.start()
        depth = 0
        in_str = False
        esc = False
        end = -1
        for i in range(arr_start, len(text)):
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
                elif ch == "[":
                    depth += 1
                elif ch == "]":
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
        if end < 0:
            continue
        obj_start = text.rfind("{", 0, match.start())
        obj_text = _balanced_json(text, obj_start) if obj_start >= 0 else None
        if not obj_text:
            continue
        try:
            raw = json.loads(obj_text).get("tool_calls")
        except json.JSONDecodeError:
            continue
        if isinstance(raw, list) and raw:
            calls, _ = _normalize_calls(raw, valid_or_none)
            if calls:
                rest = _clean_rest(text[:obj_start] + text[obj_start + len(obj_text):])
                return calls, rest

    # Format 2: <tool_call>/<function_call>/<invoke> tags.
    tag_pat = re.compile(
        r"<(?:tool_call|function_call|invoke)>(.*?)</(?:tool_call|function_call|invoke)>",
        re.DOTALL | re.IGNORECASE,
    )
    tagged = []
    for m in tag_pat.finditer(text):
        try:
            obj = json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("name"):
            tagged.append({"function": obj})
        elif isinstance(obj, dict) and isinstance(obj.get("function"), dict):
            tagged.append(obj)
    if tagged:
        calls, _ = _normalize_calls(tagged, valid_or_none)
        if calls:
            rest = _clean_rest(tag_pat.sub("", text))
            return calls, rest

    # Format 3: DSML invoke/parameter blocks.
    if "dsml" in text.lower():
        dsml_raw = _parse_dsml_calls(text)
        if dsml_raw:
            calls, dropped = _normalize_calls(dsml_raw, valid_or_none)
            if calls:
                # Strip on normalized text then map back by length-stable replace:
                # fullwidth->ASCII is length-stable (1 char -> 1 char), so spans align.
                spans = _dsml_spans(text)
                rest = _clean_rest(_strip_spans(text, spans))
                # If spans came from normalized copy with same length, safe.
                return calls, rest
            # DSML present but names unknown: don't fall through to bare scan
            # producing junk; log and keep scanning fenced (explicit user intent).

    # Format 4: fenced ```json block describing a call.
    for m in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE):
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        raw = None
        if isinstance(obj, dict) and isinstance(obj.get("tool_calls"), list):
            raw = obj["tool_calls"]
        elif isinstance(obj, dict) and obj.get("name"):
            raw = [{"function": obj}]
        elif isinstance(obj, dict) and isinstance(obj.get("function"), dict):
            raw = [obj]
        if raw:
            calls, _ = _normalize_calls(raw, valid_or_none)
            if calls:
                rest = _clean_rest(text[:m.start()] + text[m.end():])
                return calls, rest

    # Format 5: bare unfenced name/arguments object.
    if valid:
        bare_raw, bare_spans = _parse_bare_name_objects(text, valid)
        if bare_raw:
            calls, _ = _normalize_calls(bare_raw, valid_or_none)
            if calls:
                # Strip only the spans that survived normalization.
                rest = _clean_rest(_strip_spans(text, bare_spans))
                return calls, rest

    if tools and _looks_like_tool_attempt(text, valid):
        logger.debug("tool parse miss preview=%.300s", text[:300])
    return None, text
