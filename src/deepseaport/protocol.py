"""DeepSeek web protocol: headers, payloads, SSE parsing, error mapping."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Iterator, Literal

logger = logging.getLogger("deepseaport.protocol")

HOST = "chat.deepseek.com"
BASE = f"https://{HOST}"
LOGIN_URL = f"{BASE}/api/v0/users/login"
SESSION_URL = f"{BASE}/api/v0/chat_session/create"
DELETE_SESSION_URL = f"{BASE}/api/v0/chat_session/delete"
POW_URL = f"{BASE}/api/v0/chat/create_pow_challenge"
COMPLETION_PATH = "/api/v0/chat/completion"
COMPLETION_URL = f"{BASE}{COMPLETION_PATH}"

CHROME_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

# Error families worth a single automatic retry.
RETRYABLE_BIZ = {"POW_HEADER_ERROR", "INVALID_POW_RESPONSE", "INVALID_SESSION_ID"}


@dataclass
class StreamEvent:
    kind: Literal["thinking", "content", "finished", "usage", "error"]
    text: str = ""
    code: str = ""
    message: str = ""


class StreamParser:
    """Stateful SSE parser: continuation chunks arrive as bare {"v": "..."}
    with no "p" path, so the last content path is remembered."""

    CONTENT = "response/content"
    THINKING = "response/thinking_content"

    def __init__(self) -> None:
        self.last_path = ""

    def feed(self, line: str) -> Iterator[StreamEvent]:
        line = line.strip()
        if not line or not line.startswith("data:"):
            return
        data = line[5:].strip()
        if data == "[DONE]":
            yield StreamEvent(kind="finished")
            return
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            logger.debug("skip non-JSON SSE line: %.80s", data)
            return
        if not isinstance(chunk, dict):
            return
        if "v" not in chunk:
            # Toast errors arrive as data: {"type":"error","content":"...","finish_reason":"..."}
            # with no "v"/"code" keys (e.g. rate_limit_reached). Surface them.
            if chunk.get("type") == "error" or "content" in chunk and "finish_reason" in chunk:
                code = str(chunk.get("finish_reason") or chunk.get("type") or "UPSTREAM_ERROR")
                yield StreamEvent(kind="error", code=code,
                                  message=str(chunk.get("content") or chunk.get("msg") or ""))
                return
            biz = chunk.get("data") if isinstance(chunk.get("data"), dict) else {}
            code = chunk.get("code", (biz or {}).get("biz_code", ""))
            if code not in (0, "0", None, ""):
                yield StreamEvent(kind="error", code=str(code),
                                  message=str(chunk.get("msg") or (biz or {}).get("biz_msg") or ""))
            return
        path = chunk.get("p", "") or self.last_path
        value = chunk.get("v")
        if path == "response/status" and value == "FINISHED":
            self.last_path = ""
            yield StreamEvent(kind="finished")
            return
        if path == "response/search_status":
            return
        if path == "response/accumulated_token_usage" and isinstance(value, int):
            yield StreamEvent(kind="usage", text=str(value))
            return
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict) and item.get("p") == "status" and item.get("v") == "FINISHED":
                    self.last_path = ""
                    yield StreamEvent(kind="finished")
                    return
            return
        if not isinstance(value, str) or not value:
            return
        if path == self.THINKING:
            self.last_path = path
            yield StreamEvent(kind="thinking", text=value)
        elif path == self.CONTENT:
            self.last_path = path
            yield StreamEvent(kind="content", text=value)


def base_headers(user_agent: str = "", waf_cookies: str = "", bearer: str = "") -> dict[str, str]:
    headers = {
        "Accept": "application/json, text/event-stream",
        "Accept-Language": "en-US,en;q=0.9",
        "Content-Type": "application/json",
        "Origin": BASE,
        "Referer": f"{BASE}/",
        "User-Agent": user_agent or CHROME_UA,
    }
    if waf_cookies:
        headers["Cookie"] = waf_cookies
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"
    return headers


def login_payload(email: str = "", mobile: str = "", password: str = "") -> dict:
    if email:
        return {"email": email, "password": password, "device_id": "deepseaport-web", "os": "web"}
    return {"mobile": mobile, "area_code": None, "password": password,
            "device_id": "deepseaport-web", "os": "web"}


def completion_payload(session_id: str, prompt: str, thinking: bool, search: bool,
                       model_type: str = "default") -> dict:
    return {
        "chat_session_id": session_id,
        "parent_message_id": None,
        "model_type": model_type,
        "prompt": prompt,
        "ref_file_ids": [],
        "thinking_enabled": thinking,
        "search_enabled": search,
        "source": "web",
        "action": None,
        "preempt": False,
    }


def parse_sse_line(line: str) -> Iterator[StreamEvent]:
    """Stateless one-line parse (no continuation tracking)."""
    yield from StreamParser().feed(line)
