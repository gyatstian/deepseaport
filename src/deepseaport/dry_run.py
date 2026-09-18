"""Dry run mode: inspect the formatted prompt without touching DeepSeek.

Starts the OpenAI-compatible HTTP surface (`/health`, `/v1/models`,
`/v1/chat/completions`) with no Obscura/browser, no WAF/PoW cookies, no
accounts, and no upstream calls. `POST /v1/chat/completions` runs the normal
formatting pipeline (`_prepare`: message validation, tool system prompt +
reminder, `enable_append`-gated `append_top`/`append_bottom`) and then:

- prints the raw incoming JSON plus the formatted result to the terminal,
- replies with a normal OpenAI-shaped response whose content is that same
  dump, so any frontend renders it instead of showing empty
  (Settings "Send dry run to frontend", default ON; OFF replies with a
  short note while the terminal still gets the full dump).

`stream: true` gets the same content as SSE chunks. Nothing is persisted
to disk — inspection happens on the terminal and in the reply content.
"""

from __future__ import annotations

import datetime
import itertools
import json
import threading
import time

from fastapi import FastAPI, Header, Request
from fastapi.responses import StreamingResponse

from .config import Settings
from .server_routes import _check_auth, _prepare

# Serializes the multi-line request dumps so concurrent requests never
# interleave mid-block. Counter stays monotonic across requests.
_PRINT_LOCK = threading.Lock()
_REQUEST_IDS = itertools.count(1)


def _tool_names(tools: list) -> list[str]:
    names: list[str] = []
    for t in tools or []:
        if isinstance(t, dict):
            fn = t.get("function", {})
            name = fn.get("name") if isinstance(fn, dict) else None
            if name:
                names.append(str(name))
    return names


def _format_dry_run(num: int, raw_body: dict, prep: dict) -> str:
    """The dump text: raw incoming JSON + formatted pipeline output."""
    try:
        raw_text = json.dumps(raw_body, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        raw_text = str(raw_body)
    prompt = str(prep.get("prompt", ""))
    tools = prep.get("tools") or []
    names = _tool_names(tools)
    stamp = datetime.datetime.now().strftime("%H:%M:%S")
    lines = [
        f"--- dry run #{num} {stamp} "
        f"model={prep.get('model')} stream={bool(prep.get('stream'))} "
        f"tools={len(tools)}{(' [' + ', '.join(names) + ']') if names else ''} "
        f"thinking={bool(prep.get('thinking'))} search={bool(prep.get('search'))} ---",
        "--- raw body: what your client sent us (NOT sent to DeepSeek) ---",
        raw_text,
        f"=== formatted prompt (this gets sent to deepseek chat) "
        f"({len(prompt)} chars, {len(prompt.splitlines())} lines, "
        f"~{max(1, len(prompt) // 4)} tokens) ===",
        prompt,
        f"--- end #{num} ---",
    ]
    return "\n".join(lines)


def _print_dry_run(num: int, raw_body: dict, prep: dict) -> None:
    """Print the dump as one locked block (no interleaving under load)."""
    with _PRINT_LOCK:
        print(_format_dry_run(num, raw_body, prep), flush=True)


def _dry_content(num: int, raw_body: dict, prep: dict, send: bool) -> str:
    """Reply content: the full dump, or a short note when sending is OFF."""
    if send:
        return _format_dry_run(num, raw_body, prep)
    prompt = str(prep.get("prompt", ""))
    tools = prep.get("tools") or []
    return (
        f"[dry run #{num}] model={prep.get('model')} "
        f"prompt={len(prompt)} chars (~{max(1, len(prompt) // 4)} tokens), "
        f"tools={len(tools)}. Full raw + formatted output printed on the "
        f"server terminal only (Send dry run to frontend is OFF)."
    )


def _openai_dry_response(cid: str, created: int, prep: dict, content: str) -> dict:
    """Standard chat.completion shape so any frontend renders the content."""
    prompt = str(prep.get("prompt", ""))
    pt = max(1, len(prompt) // 4)
    ct = max(1, len(content) // 4)
    return {
        "id": cid, "object": "chat.completion", "created": created,
        "model": prep.get("model"), "dry_run": True,
        "choices": [{"index": 0,
                     "message": {"role": "assistant", "content": content},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": pt, "completion_tokens": ct,
                  "total_tokens": pt + ct},
    }


def _openai_dry_sse(cid: str, created: int, model: str, content: str):
    """SSE chunk stream carrying the same content for stream:true clients."""
    def _chunk(delta: dict, finish: str | None = None) -> bytes:
        return ("data: " + json.dumps(
            {"id": cid, "object": "chat.completion.chunk", "created": created,
             "model": model, "dry_run": True,
             "choices": [{"index": 0, "delta": delta,
                          "finish_reason": finish}]},
            ensure_ascii=False) + "\n\n").encode("utf-8")

    def _gen():
        yield _chunk({"role": "assistant"})
        for i in range(0, len(content), 2000):
            yield _chunk({"content": content[i:i + 2000]})
        yield _chunk({}, "stop")
        yield b"data: [DONE]\n\n"

    return StreamingResponse(_gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


def create_dry_app(settings: Settings | None = None) -> FastAPI:
    """FastAPI app with the dry-run surface. No lifespan, no bridge, no pool."""
    app = FastAPI(title="deepseaport-dry-run", version="0.1.0")
    app.state.settings = settings or Settings(config_path="")

    @app.get("/health")
    async def health():
        return {"ok": True, "dry_run": True}

    @app.get("/v1/models")
    async def list_models():
        import time

        try:
            from .server import MODELS as _MODELS
        except Exception:
            _MODELS = {"deepseek-flash": (False, False, "default")}
        now = int(time.time())
        return {
            "object": "list",
            "data": [
                {"id": m, "object": "model", "created": now, "owned_by": "deepseek"}
                for m in _MODELS
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(body: dict, request: Request,
                               authorization: str = Header(default="")):
        settings_: Settings = app.state.settings
        _check_auth(settings_, authorization)
        # _prepare raises 400/404 fast on bad model/messages (same as live).
        prep = _prepare(body, settings_)
        num = next(_REQUEST_IDS)
        # Terminal always gets the full dump; the flag only gates how much
        # of it is sent back to the requesting client.
        _print_dry_run(num, body, prep)
        send = bool(getattr(settings_, "send_dry_run_to_frontend", True))
        content = _dry_content(num, body, prep, send)
        cid = f"dryrun-{num:04d}"
        created = int(time.time())
        if prep.get("stream"):
            return _openai_dry_sse(cid, created, str(prep.get("model")), content)
        return _openai_dry_response(cid, created, prep, content)

    return app


def run_dry_run(settings: Settings, args=None) -> int:
    """Blocking dry-run server. Esc/q/Ctrl+C stops and returns to the menu."""
    import threading

    import uvicorn

    from .server_runner import (
        _resolve_bind_host,
        _watch_stop_keys,
    )

    try:
        from .cli import _ensure_free_port
    except Exception:
        from .server_runner import _ensure_free_port as _ensure_free_port  # type: ignore

    app = create_dry_app(settings)
    host = _resolve_bind_host(settings, args)
    port = _ensure_free_port(settings, host, int(getattr(settings, "port", 5001)))
    log_level = str(getattr(settings, "log_level", "info")).lower()
    print()
    print(f"  Dry run live at http://127.0.0.1:{port} (bind {host}:{port})")
    print("  No browser, no accounts, no DeepSeek. POST /v1/chat/completions to")
    print("  see the raw body + formatted prompt here and as the reply content.")
    print("  Press Ctrl+C or Esc to stop and return to menu.")
    print()
    config = uvicorn.Config(app, host=host, port=int(port),
                            log_level=log_level, access_log=False)
    server = uvicorn.Server(config)
    threading.Thread(target=_watch_stop_keys, args=(server,), daemon=True).start()
    try:
        server.run()
    except KeyboardInterrupt:
        print("\nDry run stopped (Ctrl+C), back to menu.")
    return 0
