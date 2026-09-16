"""Shared chat-REPL helpers for `serve + chat` and round-robin multi-server.

Extracted from deepseaport.chat_ui so the single-target and multi-target REPLs
share one command loop (`run_repl`). Pure helper moves only: payload building,
completion POST, model/history semantics, log silencing, instance settings.
deepseaport.chat_ui keeps thin re-exports for backward compatibility.
"""

from __future__ import annotations

import copy
import json
import logging
import urllib.error
import urllib.request

DEFAULT_MODEL = "deepseek-flash"
# Server's upstream stream timeout is 300s; chat needs headroom for
# session+PoW (~1s) so the client doesn't time out just as the server finishes.
COMPLETION_TIMEOUT = 330
READY_TIMEOUT = 20

# All loggers that can write to the console while chatting. Background
# INFO/WARNING records interleave with the input() line and corrupt it,
# so chat mode raises every one to ERROR (real errors still surface).
_CHAT_QUIET_LOGGERS = (
    "deepseaport",
    "deepseaport.server",
    "deepseaport.client",
    "deepseaport.tools",
    "deepseaport.pow",
    "deepseaport.accounts",
    "deepseaport.obscura",
    "deepseaport.protocol",
    "uvicorn",
    "uvicorn.error",
    "uvicorn.access",
)


def _silence_chat_logs() -> dict:
    """Raise console loggers to ERROR so chat input stays clean.

    Returns a snapshot for _restore_chat_logs(). Only levels change;
    handlers stay so real errors still print instead of vanishing.
    """
    saved: dict = {}
    for name in _CHAT_QUIET_LOGGERS:
        try:
            lg = logging.getLogger(name)
            saved[name] = lg.level
            lg.setLevel(logging.ERROR)
        except Exception:
            pass
    try:
        root = logging.getLogger()
        saved[""] = root.level
        root.setLevel(logging.ERROR)
    except Exception:
        pass
    return saved


def _restore_chat_logs(saved: dict) -> None:
    """Restore logger levels saved by _silence_chat_logs()."""
    for name, level in (saved or {}).items():
        try:
            if name == "":
                logging.getLogger().setLevel(level)
            else:
                logging.getLogger(name).setLevel(level)
        except Exception:
            pass


def _first_api_key(settings) -> str:
    """First configured API key, or "" when none/unreadable."""
    try:
        keys = list(getattr(settings, "keys", []) or [])
        return str(keys[0]) if keys else ""
    except Exception:
        return ""


def _model_list(models_map) -> list[str]:
    """Sorted model ids from the server MODELS mapping (or default)."""
    return sorted(models_map) if isinstance(models_map, dict) else [DEFAULT_MODEL]


def _build_payload(model: str, messages: list[dict]) -> dict:
    """Pure helper: OpenAI chat body for the local server (non-stream)."""
    return {"model": model, "messages": messages, "stream": False}


def _post_completion(base_url: str, api_key: str, model: str,
                     messages: list[dict],
                     timeout: int = COMPLETION_TIMEOUT) -> str:
    """POST one chat completion to the local server, return assistant text.

    Raises RuntimeError with server detail on failure (kept stdlib-only
    so chat mode adds no dependencies).
    """
    body = json.dumps(_build_payload(model, messages)).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions", data=body, method="POST",
        headers={"Content-Type": "application/json"})
    if (api_key or "").strip():
        req.add_header("Authorization", "Bearer " + api_key.strip())
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as exc:
        try:
            raw = exc.read().decode("utf-8") or ""
        except Exception:
            raw = ""
        try:
            detail = json.loads(raw or "{}")
            detail = detail.get("detail", detail) if isinstance(detail, dict) else detail
        except Exception:
            detail = raw or str(exc)
        raise RuntimeError(f"HTTP {exc.code}: {str(detail)[:300]}")
    except Exception as exc:
        raise RuntimeError(f"request failed: {exc}")
    try:
        msg = data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        if isinstance(data, dict) and data.get("error"):
            raise RuntimeError(f"server error: {str(data['error'])[:300]}")
        raise RuntimeError(f"bad response: {str(data)[:300]}")
    content = (msg.get("content") or "").strip()
    if content:
        return content
    thinking = (msg.get("reasoning_content") or "").strip()
    if thinking:
        return "[thinking]\n" + thinking
    return "(empty reply)"


def _initial_chat_model(settings, models: list[str]) -> str:
    """Last /model pick wins; unknown/empty stored value falls back."""
    if not models:
        return DEFAULT_MODEL
    want = str(getattr(settings, "chat_model", "") or "").strip().lower()
    for m in models:
        if m.lower() == want:
            return m
    if DEFAULT_MODEL in models:
        return DEFAULT_MODEL
    return models[0]


def _remember_chat_model(settings, model: str) -> None:
    """Persist /model pick for next launch (in-memory when unsaved)."""
    try:
        settings.chat_model = model
        settings.save()
    except Exception:
        try:
            settings.chat_model = model
        except Exception:
            pass


def _print_help() -> None:
    print("  Commands: /model [name] (list/switch), /clear (reset history),")
    print("            /quit or /exit (stop server, back to menu), /help (this)")


def _print_multi_help() -> None:
    print("  Commands: /model [name] (list/switch), /clear (reset history),")
    print("            /servers (list instances),")
    print("            /quit or /exit (stop all servers, back to menu), /help (this)")


def _select_usable_accounts_live(settings) -> tuple[list, int, int]:
    """Split pool into usable vs banned vs tokenless via live ban probes.

    Usable = has a token AND live check says not banned. Offline/probe
    failures count as usable (best-effort, never blocks startup on network
    errors). Probes run in parallel so N accounts cost ~1 request, not N.

    Reuses accounts.collect_ban_labels (parallel + TTL-cached) and derives
    the usable set from its banned map.

    Returns (usable_cfgs, banned_count, no_token_count).
    """
    from .accounts import ban_label_for, collect_ban_labels

    all_accs = list(getattr(settings, "accounts", []) or [])
    cands = [a for a in all_accs if (getattr(a, "token", "") or "").strip()]
    no_token = len(all_accs) - len(cands)
    if not cands:
        return [], 0, no_token
    try:
        ban_map = collect_ban_labels(cands, force_refresh=True) or {}
    except Exception:
        ban_map = {}
    usable: list = []
    banned = 0
    for acc in cands:
        found, _ = ban_label_for(ban_map, acc.identifier)
        if found:
            banned += 1
        else:
            usable.append(acc)
    return usable, banned, no_token


def _prompt_instance_count(cap: int) -> int | None:
    """Ask how many instances. None on cancel/empty/EOF. Retries on bad input."""
    while True:
        try:
            raw = input(f"How many instances? (1-{cap}, Enter=cancel): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if not raw:
            return None
        try:
            n = int(raw)
        except ValueError:
            print(f"  ! Not a number (1-{cap}).")
            continue
        if 1 <= n <= cap:
            return n
        print(f"  ! Out of range 1-{cap}.")


def _build_instance_settings(settings, assigned, spares, port):
    """Per-instance Settings copy: pinned account + unused-only failover pool.

    `accounts` = [assigned, *spares] where spares are usable accounts not
    picked by any instance, so mid-request ban failover can only land on
    accounts no other instance uses as primary. `active_account` pins the
    pool to try `assigned` first. `config_path` is blanked so instance
    endpoint saves (select/add/token) stay in-memory and can never truncate
    the real config.json down to the subset.
    """
    inst = copy.copy(settings)
    inst.accounts = [assigned, *[s for s in spares if s is not assigned]]
    inst.active_account = assigned.identifier
    inst.port = int(port)
    inst.log_level = "ERROR"  # quiet chat, in-memory on the copy only
    inst.config_path = ""  # in-memory: never persist the subset to disk
    return inst


def run_repl(*, model: str, models: list[str], api_key: str, history: list[dict],
             settings, post, can_continue, print_help, next_target,
             servers_command=None) -> str:
    """Shared chat REPL loop; returns the (possibly switched) model.

    Callers inject target-specific seams so this stays single vs round-robin:
    - `post(base_url, api_key, model, history)` -> assistant text.
    - `can_continue()` -> False stops the loop (server exited).
    - `print_help()` -> /help text for this mode.
    - `next_target()` -> (base_url, tag) where tag is "" for single-target
      and "[i/n :port ident]" for multi-target (drives reply/error labels).
    - `servers_command()` -> optional /servers handler (multi only).
    """
    while True:
        if not can_continue():
            break
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        low = text.lower()
        if low in ("/quit", "/exit", "/q"):
            break
        if low in ("/help", "/?", "/h"):
            print_help()
            continue
        if low in ("/clear", "/reset"):
            history.clear()
            print("  ok: history cleared")
            continue
        if servers_command is not None and low == "/servers":
            servers_command()
            continue
        if low == "/model" or low.startswith("/model "):
            parts = text.split(None, 1)
            if len(parts) == 1:
                print(f"  model: {model}  |  available: {', '.join(models)}")
                continue
            want = parts[1].strip().lower()
            match = next((m for m in models if m.lower() == want), None)
            if match is None:
                print(f"  ! unknown model: {parts[1].strip()} "
                      f"(available: {', '.join(models)})")
                continue
            model = match
            _remember_chat_model(settings, model)
            print(f"  ok: model -> {model} (remembered)")
            continue
        target, tag = next_target()
        history.append({"role": "user", "content": text})
        try:
            reply = post(target, api_key, model, history)
        except RuntimeError as exc:
            history.pop()
            print(f"  ! {tag} {exc}" if tag else f"  ! {exc}")
            continue
        history.append({"role": "assistant", "content": reply})
        print()
        print(f"Assistant {tag}: {reply}" if tag else f"Assistant: {reply}")
        print()
    return model
