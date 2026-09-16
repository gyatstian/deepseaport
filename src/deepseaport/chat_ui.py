"""Server-with-chat mode: live API server + simple local chat REPL.

Stdlib only, Windows-safe (plain input(), no curses). The uvicorn server
runs in a background thread with quiet logs so the screen stays a chat
window; external clients can still use the OpenAI-compatible API at the
printed URL while you chat. Exit with /quit (or Ctrl+C) to stop the
server and return to the main menu.

Shared helpers live in deepseaport.chat_repl (REPL/commands) and
deepseaport.server_runner (ports/bootstrap); the names are re-exported here
so existing importers (tests, tui, cli) keep working.
"""

from __future__ import annotations

import random

from . import chat_repl
from .chat_repl import (
    COMPLETION_TIMEOUT,
    DEFAULT_MODEL,
    READY_TIMEOUT,
    _build_instance_settings,
    _build_payload,
    _first_api_key,
    _initial_chat_model,
    _model_list,
    _post_completion,
    _print_help,
    _print_multi_help,
    _prompt_instance_count,
    _remember_chat_model,
    _restore_chat_logs,
    _select_usable_accounts_live,
    _silence_chat_logs,
)
from .server_runner import (
    _allocate_sequential_ports,
    _resolve_bind_host,
    _start_background_server,
    _stop_all_servers,
    _wait_for_server,
)

__all__ = [
    "DEFAULT_MODEL",
    "COMPLETION_TIMEOUT",
    "READY_TIMEOUT",
    "run_chat_server",
    "run_multi_chat_servers",
]


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

    from .server import MODELS, create_app

    saved_logs = _silence_chat_logs()
    servers: list = []
    threads: list = []
    try:
        for inst in instances:
            app = create_app(inst)
            srv, th = _start_background_server(app, bind_host, int(inst.port))
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

    state = {"idx": 0}

    def _next_target():
        target = state["idx"] % n
        state["idx"] += 1
        tag = (f"[{target + 1}/{n} :{ports[target]} "
               f"{picked[target].identifier}]")
        return bases[target], tag

    def _servers_command():
        for i, (port, acc) in enumerate(zip(ports, picked), 1):
            alive = (threads[i - 1].is_alive()
                     if i - 1 < len(threads) else False)
            print(f"  [{i}/{n}] :{port} {acc.identifier} "
                  f"({'up' if alive else 'down'})")

    try:
        chat_repl.run_repl(
            model=model,
            models=models,
            api_key=api_key,
            history=history,
            settings=settings,
            post=_post_completion,
            can_continue=lambda: not any(
                getattr(s, "should_exit", False) for s in servers),
            print_help=_print_multi_help,
            next_target=_next_target,
            servers_command=_servers_command,
        )
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
        server, thread = _start_background_server(app, bind_host, port)
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
        chat_repl.run_repl(
            model=model,
            models=models,
            api_key=api_key,
            history=history,
            settings=settings,
            post=_post_completion,
            can_continue=lambda: not getattr(server, "should_exit", False),
            print_help=_print_help,
            next_target=lambda: (client_base, ""),
        )
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
