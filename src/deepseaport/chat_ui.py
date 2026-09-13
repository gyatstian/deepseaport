"""Server-with-chat mode: live API server + simple local chat REPL.

Stdlib only, Windows-safe (plain input(), no curses). The uvicorn server
runs in a background thread with quiet logs so the screen stays a chat
window; external clients can still use the OpenAI-compatible API at the
printed URL while you chat. Exit with /quit (or Ctrl+C) to stop the
server and return to the main menu.
"""

from __future__ import annotations

import copy
import json
import logging
import random
import threading
import time
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


def _resolve_bind_host(settings, args=None) -> str:
    """--host flag wins; else listen=true -> 0.0.0.0, false -> 127.0.0.1."""
    from .cli import _resolve_bind_host as _resolve
    return _resolve(settings, args)


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


def _wait_for_server(base_url: str, timeout: int = READY_TIMEOUT) -> bool:
    """Poll GET /health until 200 or timeout. True when ready."""
    deadline = time.monotonic() + max(1, int(timeout))
    url = base_url.rstrip("/") + "/health"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


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


def _allocate_sequential_ports(host: str, base: int, count: int,
                               is_free=None) -> list[int]:
    """Allocate `count` distinct free ports scanning upward from `base`.

    Mirrors the single-server auto-fix: each candidate is socket-checked;
    busy ones are skipped (5001 taken -> 5002 -> 5003 ...). Never mutates
    settings. Returns fewer than `count` when the range is exhausted.
    """
    from .cli import _scan_free_ports

    return _scan_free_ports(host, base, count, check=is_free)


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


def _stop_all_servers(servers, threads, timeout: float = 5.0) -> None:
    for srv in servers:
        try:
            srv.should_exit = True
        except Exception:
            pass
    for th in threads:
        try:
            th.join(timeout=timeout)
        except Exception:
            pass


def run_multi_chat_servers(settings, args=None, count: int | None = None) -> int:
    """Start N chat servers (distinct ports/accounts), one round-robin REPL.

    Each instance gets a random distinct usable (tokened, live-checked
    non-banned) account and a distinct free port scanning up from
    settings.port. The original settings object is never mutated
    (no port save, no account rewrite). Returns 0 on clean stop/cancel,
    1 when startup is blocked or a server never became ready.
    """
    if args is not None and int(getattr(args, "workers", 1) or 1) != 1:
        print("  note: chat mode uses 1 worker (ignoring --workers).")
    if not getattr(settings, "accounts", None):
        print("  ! No accounts yet. Add one first.")
        return 1
    if not any((getattr(a, "token", "") or "").strip()
               for a in (settings.accounts or [])):
        print("  ! Accounts have no token. Add one (y = auto-token) first.")
        return 1

    print("  Checking account bans (live, parallel)...")
    usable, banned, no_token = _select_usable_accounts_live(settings)
    cap = len(usable)
    if cap == 0:
        print("  ! No usable accounts (all banned or missing token). "
              "Add/unban accounts first.")
        return 1
    if count is None:
        print(f"  Usable accounts: {cap} ({banned} banned, "
              f"{no_token} without token skipped)")
        n = _prompt_instance_count(cap)
        if n is None:
            print("  Cancelled.")
            return 0
    else:
        try:
            n = int(count)
        except (TypeError, ValueError):
            print(f"  ! Bad instance count: {count!r}.")
            return 1
        if not 1 <= n <= cap:
            print(f"  ! Need 1-{cap} instance(s) for {cap} usable account(s). "
                  "Add/unban accounts first.")
            return 1

    picked = random.sample(usable, n)
    bind_host = _resolve_bind_host(settings, args)
    try:
        base = int(getattr(settings, "port", 5001) or 5001)
    except (TypeError, ValueError):
        base = 5001
    ports = _allocate_sequential_ports(bind_host, base, n)
    if len(ports) < n:
        print(f"  ! Only {len(ports)} free port(s) from {base}; need {n}. "
              "Free some ports and retry.")
        return 1

    spares = [a for a in usable if not any(a is p for p in picked)]
    instances = [_build_instance_settings(settings, acc, spares, port)
                 for acc, port in zip(picked, ports)]

    import uvicorn

    from .server import MODELS, create_app

    saved_logs = _silence_chat_logs()
    servers: list = []
    threads: list = []
    try:
        for inst in instances:
            app = create_app(inst)
            config = uvicorn.Config(app, host=bind_host, port=int(inst.port),
                                    log_level="error", access_log=False)
            srv = uvicorn.Server(config)
            th = threading.Thread(target=srv.run, daemon=True)
            th.start()
            servers.append(srv)
            threads.append(th)
    except Exception:
        _stop_all_servers(servers, threads)
        _restore_chat_logs(saved_logs)
        raise

    bases = [f"http://127.0.0.1:{p}" for p in ports]
    for i, base_url in enumerate(bases):
        if not _wait_for_server(base_url):
            alive = threads[i].is_alive() if i < len(threads) else False
            if not alive:
                print(f"  ! server {i + 1}/{n} failed to start "
                      f"on {bind_host}:{ports[i]}")
                _stop_all_servers(servers, threads)
                _restore_chat_logs(saved_logs)
                return 1
            print(f"  ! server {i + 1}/{n} slow to respond, "
                  f"continuing anyway ({base_url}).")

    try:
        api_key = _first_api_key(settings)
    except Exception:
        api_key = ""
    models = _model_list(MODELS)
    model = _initial_chat_model(settings, models)
    history: list[dict] = []

    print()
    for i, (port, acc) in enumerate(zip(ports, picked), 1):
        print(f"  [{i}/{n}] API http://127.0.0.1:{port} "
              f"(bind {bind_host}:{port}) account {acc.identifier}")
    print(f"  Chat round-robin across {n} server(s). Model: {model}")
    _print_multi_help()
    print()
    next_idx = 0
    try:
        while True:
            if any(getattr(s, "should_exit", False) for s in servers):
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
                _print_multi_help()
                continue
            if low in ("/clear", "/reset"):
                history.clear()
                print("  ok: history cleared")
                continue
            if low == "/servers":
                for i, (port, acc) in enumerate(zip(ports, picked), 1):
                    alive = (threads[i - 1].is_alive()
                             if i - 1 < len(threads) else False)
                    print(f"  [{i}/{n}] :{port} {acc.identifier} "
                          f"({'up' if alive else 'down'})")
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
            target = next_idx % n
            next_idx += 1
            tag = f"[{target + 1}/{n} :{ports[target]} {picked[target].identifier}]"
            history.append({"role": "user", "content": text})
            try:
                reply = _post_completion(bases[target], api_key, model, history)
            except RuntimeError as exc:
                history.pop()
                print(f"  ! {tag} {exc}")
                continue
            history.append({"role": "assistant", "content": reply})
            print()
            print(f"Assistant {tag}: {reply}")
            print()
    finally:
        print()
        print("  Stopping servers, back to menu...")
        _stop_all_servers(servers, threads)
        _restore_chat_logs(saved_logs)
    return 0


def run_chat_server(settings, args=None) -> int:
    """Start server in background thread, run chat REPL in foreground.

    Returns 0 on clean stop (back to menu), 1 when the server never
    became ready. Ctrl+C also stops and returns.
    """
    import uvicorn

    from .cli import _ensure_free_port
    from .server import MODELS, create_app

    if args is not None and int(getattr(args, "workers", 1) or 1) != 1:
        print("  note: chat mode uses 1 worker (ignoring --workers).")
    bind_host = _resolve_bind_host(settings, args)
    port = _ensure_free_port(settings, bind_host, int(getattr(settings, "port", 5001)))
    client_base = f"http://127.0.0.1:{port}"

    # Silence BEFORE the server starts: lifespan warmup (obscura cookies,
    # PoW wasm) logs INFO straight onto the chat input line otherwise.
    # settings.log_level is overridden in-memory only (never saved) so the
    # lifespan's apply_log_level() also stays quiet; restored on exit.
    orig_log_level = getattr(settings, "log_level", "INFO")
    settings.log_level = "ERROR"
    saved_logs = _silence_chat_logs()
    try:
        app = create_app(settings)
        config = uvicorn.Config(app, host=bind_host, port=port,
                                log_level="error", access_log=False)
        server = uvicorn.Server(config)
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
    except Exception:
        try:
            settings.log_level = orig_log_level
        except Exception:
            pass
        _restore_chat_logs(saved_logs)
        raise

    if not _wait_for_server(client_base):
        if not thread.is_alive():
            print(f"  ! server failed to start on {bind_host}:{port}")
            try:
                server.should_exit = True
            except Exception:
                pass
            try:
                settings.log_level = orig_log_level
            except Exception:
                pass
            _restore_chat_logs(saved_logs)
            return 1
        print(f"  ! server slow to respond, continuing anyway ({client_base}).")

    api_key = _first_api_key(settings)
    models = _model_list(MODELS)
    model = _initial_chat_model(settings, models)
    history: list[dict] = []

    print()
    print(f"  Chat + API live at {client_base} (bind {bind_host}:{port})")
    print(f"  Model: {model}  |  Account: {(settings.active_account or 'ALL').strip() or 'ALL'}")
    _print_help()
    print()
    try:
        while True:
            if getattr(server, "should_exit", False):
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
                _print_help()
                continue
            if low in ("/clear", "/reset"):
                history.clear()
                print("  ok: history cleared")
                continue
            if low == "/model" or low.startswith("/model "):
                parts = text.split(None, 1)
                if len(parts) == 1:
                    print(f"  model: {model}  |  available: {', '.join(models)}")
                    continue
                want = parts[1].strip().lower()
                match = next((m for m in models if m.lower() == want), None)
                if match is None:
                    print(f"  ! unknown model: {parts[1].strip()} (available: {', '.join(models)})")
                    continue
                model = match
                _remember_chat_model(settings, model)
                print(f"  ok: model -> {model} (remembered)")
                continue
            history.append({"role": "user", "content": text})
            try:
                reply = _post_completion(client_base, api_key, model, history)
            except RuntimeError as exc:
                history.pop()
                print(f"  ! {exc}")
                continue
            history.append({"role": "assistant", "content": reply})
            print()
            print(f"Assistant: {reply}")
            print()
    finally:
        print()
        print("  Stopping server, back to menu...")
        try:
            server.should_exit = True
        except Exception:
            pass
        thread.join(timeout=10)
        try:
            settings.log_level = orig_log_level
        except Exception:
            pass
        _restore_chat_logs(saved_logs)
    return 0
