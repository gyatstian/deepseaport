"""userToken parsing/normalization shared by config, CLI, and the account pool.

DeepSeek stores the token in localStorage as a JSON object:
``{"value":"<token>","__version":"0"}``. Depending on how it was copied it can
arrive as raw text, JSON, double-encoded JSON, or the literal JSON-null shape
(``{"value":null}``).  ``extract_token`` accepts every one of those shapes and
returns either the usable value or an empty string.  Keeping this in one small
module avoids subtle differences between the config loader and login flows.
"""

from __future__ import annotations

import json


def extract_token(raw: object) -> str:
    """Return a usable userToken value from any common representation.

    Empty/whitespace, JSON with a null/missing ``value``, and malformed JSON
    all normalize to ``""``. Raw tokens pass through unchanged (outer quotes
    are trimmed because browser evaluations often wrap strings in quotes).
    """
    text = str(raw or "").strip()
    if not text:
        return ""
    # Browser evaluate sometimes returns a JSON string literal.
    if len(text) >= 2 and text[0] == text[-1] == '"':
        try:
            text = json.loads(text)
        except Exception:
            text = text[1:-1]
    text = str(text).strip()
    if not text:
        return ""
    if text.startswith("{"):
        try:
            data = json.loads(text)
        except Exception:
            return ""
        if isinstance(data, str):
            # Double-encoded: {"...": "{\"value\":\"...\"}"} after one parse.
            try:
                data = json.loads(data)
            except Exception:
                return data.strip()
        if isinstance(data, dict):
            value = data.get("value")
            if value is None:
                return ""
            return str(value).strip().strip("\"'")
        return ""
    return text.strip("\"'")
