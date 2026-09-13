"""CLI: serve | waf | models | chat."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

import uvicorn


def _watch_stop_keys(server) -> None:
    """Daemon: Esc/q stops a running server so TUI returns to menu.

    Best-effort, never raises. Ctrl+C is handled by uvicorn/SIGINT;
    this adds Esc (and q) without Enter on Windows (msvcrt) and POSIX.
    """
    try:
        try:
            import msvcrt  # type: ignore
        except ImportError:
            msvcrt = None  # type: ignore
        if msvcrt is not None:
            import time as _time

            while not getattr(server, "should_exit", False):
                try:
                    if msvcrt.kbhit():
                        ch = msvcrt.getch()
                        if ch in (b"\x1b", b"q", b"Q", b"\x03"):
                            print("\nStopping server (key pressed), returning to menu...")
                            server.should_exit = True
                            break
                    else:
                        _time.sleep(0.1)
                except Exception:
                    return
            return
        # POSIX fallback: cbreak stdin + select, restore on exit.
        import select as _select
        import sys as _sys

        try:
            import termios as _termios
            import tty as _tty
        except ImportError:
            return
        try:
            _fd = _sys.stdin.fileno()
        except Exception:
            return
        try:
            _old = _termios.tcgetattr(_fd)
        except Exception:
            return
        try:
            _tty.setcbreak(_fd)
            while not getattr(server, "should_exit", False):
                try:
                    _r, _, _ = _select.select([_sys.stdin], [], [], 0.2)
                except Exception:
                    return
                if _r:
                    try:
                        _ch = _sys.stdin.read(1)
                    except Exception:
                        return
                    if _ch in ("\x1b", "q", "Q"):
                        print("\nStopping server (key pressed), returning to menu...")
                        server.should_exit = True
                        break
        except Exception:
            pass
        finally:
            try:
                _termios.tcsetattr(_fd, _termios.TCSADRAIN, _old)
            except Exception:
                pass
    except Exception:
        return


def _is_port_free(host: str, port: int) -> bool:
    """True when host:port can be bound right now (pre-flight check)."""
    import socket

    family = socket.AF_INET6 if ":" in (host or "") else socket.AF_INET
    sock = socket.socket(family, socket.SOCK_STREAM)
    try:
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        except OSError:
            pass
        sock.bind((host, int(port)))
        return True
    except OSError:
        return False
    finally:
        try:
            sock.close()
        except OSError:
            pass


def _resolve_bind_host(settings, args=None) -> str:
    """--host flag wins; else listen=true -> 0.0.0.0, false -> 127.0.0.1."""
    override = ""
    if args is not None:
        override = (getattr(args, "host", "") or "").strip()
    if override:
        return override
    return "0.0.0.0" if bool(getattr(settings, "listen", False)) else "127.0.0.1"


def _scan_free_ports(host: str, start: int, count: int,
                     limit: int | None = None, check=None) -> list[int]:
    """Collect up to `count` free ports scanning upward from `start`.

    Stops at `start+limit-1` when limit is given, else 65535. `check` defaults
    to _is_port_free. Returns fewer than `count` when the range is exhausted.
    """
    free_fn = check if check is not None else _is_port_free
    try:
        candidate = int(start)
    except (TypeError, ValueError):
        candidate = 5001
    if candidate < 1:
        candidate = 1
    end = 65535 if limit is None else min(65535, candidate + int(limit) - 1)
    ports: list[int] = []
    while len(ports) < count and candidate <= end:
        try:
            free = bool(free_fn(host, candidate))
        except Exception:
            free = False
        if free:
            ports.append(candidate)
        candidate += 1
    return ports


def _next_free_port(host: str, port: int, limit: int = 50) -> int | None:
    """Scan port+1..port+limit for a free port. None when all busy."""
    found = _scan_free_ports(host, int(port) + 1, 1, limit=limit)
    return found[0] if found else None


def _ensure_free_port(settings, host: str, port: int) -> int:
    """Port auto-fix: busy port offers next free one and saves it.

    Interactive (tty stdin): asks Y/n. Non-interactive: auto-picks next
    free when available. Returns the port to use (original when user
    declines or no free port found).
    """
    import sys as _sys

    port = int(port)
    if _is_port_free(host, port):
        return port
    nxt = _next_free_port(host, port)
    if nxt is None:
        print(f"Port {port} busy and no free port found nearby.")
        return port
    try:
        interactive = bool(_sys.stdin.isatty())
    except Exception:
        interactive = False
    if interactive:
        try:
            ans = input(f"Port {port} busy. Use {nxt} instead? [Y/n]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return port
        if ans not in ("", "y", "yes"):
            return port
    else:
        print(f"Port {port} busy, auto-using {nxt}.")
    settings.port = int(nxt)
    try:
        settings.save()
        print(f"Port -> {nxt} (saved)")
    except Exception:
        print(f"Port -> {nxt} (unsaved)")
    return int(nxt)


def _run_server(settings, args) -> int:
    from .server import apply_log_level, create_app

    import threading

    apply_log_level(getattr(settings, "log_level", "INFO"))
    app = create_app(settings)
    # --host flag overrides config; else listen=true -> 0.0.0.0, false -> 127.0.0.1.
    host = _resolve_bind_host(settings, args)
    port = int(getattr(settings, "port", 5001))
    log_level = str(getattr(settings, "log_level", "info")).lower()
    workers = int(getattr(args, "workers", 1) or 1)
    port = _ensure_free_port(settings, host, port)
    if workers != 1:
        # Multi-worker spawns subprocesses; Esc listener can't reach them.
        # Ctrl+C still stops; TUI main_menu loop returns to menu.
        try:
            uvicorn.run(app, host=host, port=port,
                        log_level=log_level, workers=workers)
        except KeyboardInterrupt:
            print("\nServer stopped, back to menu.")
        return 0
    config = uvicorn.Config(app, host=host, port=port, log_level=log_level)
    server = uvicorn.Server(config)
    threading.Thread(target=_watch_stop_keys, args=(server,), daemon=True).start()
    try:
        server.run()
    except KeyboardInterrupt:
        print("\nServer stopped (Ctrl+C), back to menu.")
    return 0


def cmd_serve(args) -> int:
    from .config import load_settings

    settings = load_settings(args.config)
    if args.port:
        settings.port = args.port

    no_tui = bool(getattr(args, "no_tui", False))
    if not no_tui and sys.stdin.isatty():
        from .tui import main_menu

        def serve_fn(current_settings) -> int:
            return _run_server(current_settings, args)

        def chat_fn(current_settings) -> int:
            from .chat_ui import run_chat_server
            return run_chat_server(current_settings, args)

        def multi_fn(current_settings) -> int:
            from .chat_ui import run_multi_chat_servers
            return run_multi_chat_servers(current_settings, args)

        return main_menu(settings, serve_fn, chat_fn, multi_fn)
    return _run_server(settings, args)


def cmd_waf(args) -> int:
    from .config import load_settings
    from .obscura_bridge import ObscuraBridge

    settings = load_settings(args.config)
    bridge = ObscuraBridge(settings.obscura_bin, settings.obscura_profile)
    print("binary:", bridge.binary)
    print("version:", bridge.version())
    cookies = bridge.warmup()
    print("cookies:", sorted(cookies))
    print("has_waf_token:", bridge.has_waf_token())
    print("ua:", bridge.state.user_agent)
    return 0 if bridge.has_waf_token() else 2


def cmd_models(args) -> int:
    from .server import MODELS

    print(json.dumps(sorted(MODELS), indent=2, ensure_ascii=False))
    return 0


def cmd_accounts(args) -> int:
    from .accounts import TOKEN_HELP, AccountPool, extract_token, sync_current_from_settings
    from .config import AccountConfig, load_settings

    settings = load_settings(args.config)
    # Sync stale CURRENT selection (e.g. manually edited config).
    _tmp = AccountPool(settings.accounts, current=settings.active_account)
    if sync_current_from_settings(_tmp, settings):
        settings.active_account = _tmp.current
        settings.save()
    action = getattr(args, "accounts_action", "list")

    if action == "list":
        pool = AccountPool(settings.accounts, current=settings.active_account)
        rows = pool.status()
        print(f"CURRENT: [{pool.current or 'ALL (auto-failover)'}]")
        if not rows:
            print("pool empty")
            print(TOKEN_HELP)
            return 0
        try:
            from .accounts import ban_label_for as _ban_lookup, collect_ban_labels as _ban_map
            from .protocol import format_ban_label as _ban_label
            ban_map = _ban_map(settings.accounts) or {}
        except Exception:
            ban_map = {}
            _ban_lookup = None  # type: ignore
            _ban_label = None  # type: ignore
        for i, r in enumerate(rows, 1):
            mark = " *CURRENT*" if r.get("current") else ""
            token_flag = "yes" if _account_has_token(settings, str(r["identifier"])) else "MISSING"
            ban_flag = ""
            try:
                found, until = (_ban_lookup(ban_map, str(r["identifier"]))
                                if _ban_lookup else (False, None))
                if found:
                    ban_flag = f"  {_ban_label(until)}" if _ban_label else "  (BANNED)"
            except Exception:
                ban_flag = ""
            print(f"{i}. {r['identifier']}{mark}  uses={r['uses']}  "
                  f"busy={r['busy']}  cooldown={r['cooldown_remaining']}s  token={token_flag}{ban_flag}")
        missing = [r["identifier"] for r in rows if not _account_has_token(settings, str(r["identifier"]))]
        if missing:
            print(f"missing token: {', '.join(missing)}")
            print(TOKEN_HELP)
        return 0

    if action == "select":
        ident = (getattr(args, "identifier", "") or "").strip()
        if getattr(args, "clear", False) or not ident or ident.lower() in ("none", "all", "auto"):
            settings.active_account = ""
            settings.save()
            print("Selection cleared -> ALL (auto-failover)")
            return 0
        pool = AccountPool(settings.accounts, current=settings.active_account)
        if not pool.set_current(ident):
            print(f"not found: {ident}")
            return 1
        settings.active_account = pool.current
        settings.save()
        print(f"CURRENT -> [{pool.current}] (saved)")
        return 0

    if action == "add":
        raw_token = args.token or ""
        cfg = AccountConfig(email=args.email or "", mobile=args.mobile or "",
                            password=args.password or "", token=extract_token(raw_token))
        if not (cfg.email or cfg.mobile or cfg.token):
            print("need --email, --mobile, or --token")
            print(TOKEN_HELP)
            return 2
        pool = AccountPool(settings.accounts, current=settings.active_account)
        try:
            from .accounts import add_account as _add
            _add(settings, pool, cfg)
        except ValueError as exc:
            print(f"add failed: {exc}")
            return 1
        settings.save()
        print(f"added {cfg.identifier} -> {settings.config_path}")
        if not cfg.token:
            print("WARNING: no token. Password-only fails (RISK_DEVICE_DETECTED).")
            print("TIP: python -m deepseaport login --email "
                  f"{cfg.email or cfg.mobile or ''} (auto-capture)")
            print(TOKEN_HELP)
        return 0

    if action == "set-token":
        from .accounts import set_account_token as _set
        token = extract_token(args.token or "")
        if not args.identifier or not token:
            print("usage: accounts set-token <email> --token <userToken value>")
            print(TOKEN_HELP)
            return 2
        matched = _set(settings, args.identifier, token)
        if matched is not None:
            settings.save()
            print(f"token updated for {matched.identifier}")
            return 0
        print(f"not found: {args.identifier}")
        return 1

    if action == "remove":
        identifier = args.identifier or ""
        if not identifier:
            print("need identifier (email/mobile/token)")
            return 2
        pool = AccountPool(settings.accounts, current=settings.active_account)
        if not pool.remove(identifier):
            print(f"not found: {identifier}")
            return 1
        # Keep config file in sync with pool.
        from .accounts import remove_account as _remove
        _remove(settings, identifier)
        settings.save()
        print(f"removed {identifier}")
        return 0

    if action == "unblock":
        # Cooldown lives in-memory; offline CLI can only confirm config.
        # Running server: use POST /v1/accounts/unblock instead.
        print("cooldown is in-memory. Use running server: POST /v1/accounts/unblock")
        return 0

    print(f"unknown accounts action: {action}")
    return 2


def _account_has_token(settings, identifier: str) -> bool:
    from .accounts import account_has_token
    return account_has_token(settings, identifier)


def _upsert_account(settings, email: str, password: str, token: str) -> None:
    """Append or update account by email. Preserves pool, fixes wipe bug."""
    from .accounts import _matches as _m, extract_token
    from .config import AccountConfig

    token = extract_token(token)
    for a in settings.accounts:
        if a.email and _m(a, email):
            a.token = token
            if password:
                a.password = password
            return
    cfg = AccountConfig(email=email, password=password, token=token)
    settings.accounts.append(cfg)
    # New account becomes CURRENT (default) immediately.
    settings.active_account = cfg.identifier


def _obscura_login_token(email: str, password: str, settings) -> str | None:
    """Run Obscura browser login, return userToken value or None."""
    import json as _json
    import time

    from .accounts import extract_token
    from .mcp_client import McpClient
    from .obscura_bridge import ObscuraBridge, profile_lock

    bridge = ObscuraBridge(settings.obscura_bin, settings.obscura_profile)
    # Login drives a long-lived MCP browser on the SAME --storage-dir the app
    # bridge uses for WAF warmups. Without the shared profile lock, a
    # concurrent 403 warmup/token refresh could spawn a second obscura on that
    # dir and corrupt the cookie jar (=> every subsequent request 403s). Hold
    # the lock for the whole MCP subprocess lifetime.
    with profile_lock(bridge.profile):
        client = McpClient(bridge.binary, bridge.profile)
        try:
            client.call("browser_navigate", {"url": "https://chat.deepseek.com/sign_in", "waitUntil": "load"})
            time.sleep(3)
            client.call("browser_interactive_elements", {"limit": 40})
            client.call("browser_fill", {"ref": "e1", "value": email})
            client.call("browser_fill", {"ref": "e2", "value": password})
            client.call("browser_click", {"ref": "e8"})
            for _ in range(24):
                time.sleep(5)
                res = client.call("browser_evaluate", {
                    "expression": "JSON.stringify({url:location.href, tok:localStorage.getItem('userToken')})"})
                try:
                    raw = _json.loads(res["content"][0]["text"]).get("tok") or "null"
                    token = extract_token(raw)
                    if token:
                        return token
                except Exception:
                    pass
            print("login failed: no token captured (captcha or bad credentials?)")
            print("MANUAL fallback: login in Chrome, F12 > Application > Local Storage >")
            print("  https://chat.deepseek.com > userToken > copy `value`, then:")
            print(f"  python -m deepseaport accounts set-token {email} --token <value>")
            return None
        finally:
            client.close()


def cmd_login(args) -> int:
    """Interactive login inside Obscura (real device fingerprint) -> saves userToken."""
    import getpass

    from .config import load_settings

    settings = load_settings(args.config)
    email = args.email or input("DeepSeek email: ").strip()
    password = args.password or getpass.getpass("DeepSeek password: ")
    token = _obscura_login_token(email, password, settings)
    if not token:
        return 3
    _upsert_account(settings, email, password, token)
    settings.save()
    print(f"login ok, token saved for {email} -> {settings.config_path}")
    return 0


def cmd_chat(args) -> int:
    """Single non-stream chat against the local server logic (needs account)."""
    from .accounts import AccountPool
    from .config import load_settings
    from .obscura_bridge import ObscuraBridge
    from .server import _complete_with_failover_sync, _prepare, create_app

    settings = load_settings(args.config)
    app = create_app(settings)
    app.state.bridge = ObscuraBridge(settings.obscura_bin, settings.obscura_profile)
    app.state.bridge.warmup()
    prep = _prepare({"model": args.model, "messages": [{"role": "user", "content": args.prompt}]}, settings)
    pool = AccountPool(settings.accounts, current=settings.active_account)
    item = pool.acquire(timeout=30,
                        allow_failover=bool(getattr(settings, "use_multiple_accounts", True)))
    # Same-request ban failover as the server endpoint: a banned CURRENT
    # transparently retries on the next unbanned account.
    result = _complete_with_failover_sync(app, pool, prep, item)
    print(("THINKING:\n" + result["thinking"] + "\n\n" if result["thinking"] else "") + result["content"])
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="deepseaport")
    parser.add_argument("--config", default=None)
    sub = parser.add_subparsers(dest="cmd")
    serve = sub.add_parser("serve")
    serve.add_argument("--host", default="",
                       help="bind address; overrides config listen=true/false")
    serve.add_argument("--port", type=int, default=0)
    serve.add_argument("--workers", type=int, default=1)
    serve.add_argument("--no-tui", action="store_true",
                       help="skip the main-page TUI and start the server directly")
    sub.add_parser("waf")
    sub.add_parser("models")
    acc = sub.add_parser("accounts", help="list/add/remove pooled accounts")
    acc_sub = acc.add_subparsers(dest="accounts_action", required=True)
    acc_sub.add_parser("list", help="show pooled accounts + token status")
    acc_sel = acc_sub.add_parser("select", help="set CURRENT account (preferred first)")
    acc_sel.add_argument("identifier", nargs="?", default="",
                         help="email to select; empty/ALL/NONE clears")
    acc_sel.add_argument("--clear", action="store_true",
                         help="clear selection -> ALL (auto-failover)")
    acc_add = acc_sub.add_parser("add", help="add account (needs --token, see TOKEN_HELP)")
    acc_add.add_argument("--email", default="", help="account email")
    acc_add.add_argument("--mobile", default="", help="account mobile")
    acc_add.add_argument("--password", default="", help="stored for reference, direct login fails")
    acc_add.add_argument("--token", default="",
                         help="userToken value from chat.deepseek.com localStorage (raw or JSON)")
    acc_st = acc_sub.add_parser("set-token", help="paste/refresh userToken for existing account")
    acc_st.add_argument("identifier", help="email/mobile/identifier")
    acc_st.add_argument("--token", default="", required=True,
                        help="userToken value (raw 64-char or full JSON)")
    acc_rm = acc_sub.add_parser("remove", help="delete account from pool")
    acc_rm.add_argument("identifier", help="email/mobile/identifier/token")
    acc_sub.add_parser("unblock", help="hint for clearing in-memory cooldown")
    login = sub.add_parser("login")
    login.add_argument("--email", default="")
    login.add_argument("--password", default="")
    chat = sub.add_parser("chat")
    chat.add_argument("--prompt", required=True)
    chat.add_argument("--model", default="deepseek-flash")
    args = parser.parse_args(argv)
    if args.cmd is None:
        # No subcommand (e.g. double-clicked exe) -> start the server.
        args = argparse.Namespace(config=args.config, host="", port=0,
                                  workers=1, no_tui=False, cmd="serve")
    if args.cmd == "serve":
        return cmd_serve(args)
    if args.cmd == "waf":
        return cmd_waf(args)
    if args.cmd == "models":
        return cmd_models(args)
    if args.cmd == "accounts":
        return cmd_accounts(args)
    if args.cmd == "login":
        return cmd_login(args)
    if args.cmd == "chat":
        return cmd_chat(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
