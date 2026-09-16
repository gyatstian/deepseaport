"""CLI: serve | waf | models | chat."""

from __future__ import annotations

import argparse
import json
import sys

from . import server_runner as _runner
from .auth import browser_login as _browser_login, unpack_login_result as _unpack_login_result
from .server_runner import _resolve_bind_host, _watch_stop_keys

# Shared server bootstrap lives in deepseaport.server_runner (single source of
# truth for cli + chat_ui). Thin wrappers below keep the historical cli-level
# names importable and monkeypatchable (tests/tui import these from cli).


def _is_port_free(host: str, port: int) -> bool:
    return _runner._is_port_free(host, port)


def _scan_free_ports(host: str, start: int, count: int,
                     limit: int | None = None, check=None) -> list[int]:
    return _runner._scan_free_ports(host, start, count, limit=limit, check=check)


def _next_free_port(host: str, port: int, limit: int = 50) -> int | None:
    return _runner._next_free_port(host, port, limit=limit, check=_is_port_free)


def _ensure_free_port(settings, host: str, port: int) -> int:
    return _runner._ensure_free_port(settings, host, port,
                                     is_free=_is_port_free,
                                     next_free=_next_free_port)


def _run_server(settings, args) -> int:
    return _runner._run_server(settings, args, ensure_free=_ensure_free_port)


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
        missing = [r["identifier"] for r in rows
                   if not _account_has_token(settings, str(r["identifier"]))
                   and not r.get("banned")]
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
        # Clear persisted ban markers in config. A running server also needs
        # POST /v1/accounts/unblock to drop its in-memory cooldown singleton.
        from .accounts import _matches as _m
        ident = (getattr(args, "identifier", "") or "").strip()
        n = 0
        for a in settings.accounts:
            if ident and not _m(a, ident):
                continue
            if getattr(a, "banned", False) or getattr(a, "banned_until", 0.0):
                n += 1
            a.banned = False
            a.banned_until = 0.0
        settings.save()
        print(f"persisted ban state cleared for {n} account(s)")
        print("If a server is running, also POST /v1/accounts/unblock "
              "(or restart it) to clear its in-memory cooldown.")
        return 0

    print(f"unknown accounts action: {action}")
    return 2


def _account_has_token(settings, identifier: str) -> bool:
    from .accounts import account_has_token
    return account_has_token(settings, identifier)


def _upsert_account(settings, email: str, password: str, token: str):
    """Append or update account by email. Preserves pool, fixes wipe bug."""
    from .accounts import _matches as _m, extract_token
    from .config import AccountConfig

    token = extract_token(token)
    for a in settings.accounts:
        if a.email and _m(a, email):
            a.token = token
            if password:
                a.password = password
            return a
    cfg = AccountConfig(email=email, password=password, token=token)
    settings.accounts.append(cfg)
    # New account becomes CURRENT (default) immediately.
    settings.active_account = cfg.identifier
    return cfg


# Login implementation lives in deepseaport.auth (selector-driven, visible
# text only). These wrappers keep the historical CLI/TUI/server import points
# stable and are intentionally tiny.

def _unpack_login_token_result(res) -> tuple[str | None, bool, str]:
    return _unpack_login_result(res)


def _obscura_login_token(email: str, password: str, settings) -> tuple[str | None, bool, str]:
    """Run the shared Obscura browser login and return legacy tuple shape."""
    return _browser_login(email, password, settings).as_tuple()


def cmd_login(args) -> int:
    """Interactive login inside Obscura (real device fingerprint) -> saves userToken."""
    import getpass

    from .config import load_settings

    settings = load_settings(args.config)
    email = args.email or input("DeepSeek email: ").strip()
    password = args.password or getpass.getpass("DeepSeek password: ")
    token, login_banned, ban_detail = _unpack_login_token_result(
        _obscura_login_token(email, password, settings))
    if not token:
        if login_banned:
            print(f"login refused for {email}: account BANNED per login page.")
            if ban_detail:
                print(f"Evidence: {ban_detail[:300]}")
        return 3

    acc = _upsert_account(settings, email, password, token)
    if login_banned:
        # A suspended account can still complete login and expose a token.
        # Keep that token (so the Accounts list can probe/display the ban) and
        # persist the best human-readable expiry we can recover.
        try:
            from .auth import parse_ban_until
            until = parse_ban_until(ban_detail)
        except Exception:
            until = None
        try:
            from .accounts import check_ban_for_token
            _banned, _until = check_ban_for_token(token)
            until = _until or until
        except Exception:
            pass
        acc.banned = True
        acc.banned_until = float(until or 0.0)
        settings.save()
        print(f"login ok, token saved for {email}, but account is BANNED.")
        if acc.banned_until:
            from .protocol import format_ban_datetime
            print("Expires:", format_ban_datetime(acc.banned_until))
        if ban_detail:
            print(f"Evidence: {ban_detail[:300]}")
        print(f"Token -> {settings.config_path} (accounts list will show BANNED)")
        return 3

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
    from .obscura_bridge import register_default_bridge

    app.state.bridge = ObscuraBridge(settings.obscura_bin, settings.obscura_profile)
    # Publish for protocol.bridge_headers ban probes; without this the probe
    # falls back to bare headers (no WAF cookie/UA) and can false-negative.
    register_default_bridge(app.state.bridge)
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
    acc_add.add_argument("--password", default="", help="required for browser auto-refresh on 40003 (direct API login fails)")
    acc_add.add_argument("--token", default="",
                         help="userToken value from chat.deepseek.com localStorage (raw or JSON)")
    acc_st = acc_sub.add_parser("set-token", help="paste/refresh userToken for existing account")
    acc_st.add_argument("identifier", help="email/mobile/identifier")
    acc_st.add_argument("--token", default="", required=True,
                        help="userToken value (raw 64-char or full JSON)")
    acc_rm = acc_sub.add_parser("remove", help="delete account from pool")
    acc_rm.add_argument("identifier", help="email/mobile/identifier/token")
    acc_unblock = acc_sub.add_parser("unblock", help="clear persisted + in-memory ban/cooldown")
    acc_unblock.add_argument("identifier", nargs="?", default="",
                             help="email/mobile/identifier; empty = all accounts")
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
