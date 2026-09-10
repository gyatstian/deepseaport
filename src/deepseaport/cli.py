"""CLI: serve | waf | models | chat."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

import uvicorn


def _run_server(settings, args) -> int:
    from .server import apply_log_level, create_app

    apply_log_level(getattr(settings, "log_level", "INFO"))
    app = create_app(settings)
    # --host flag overrides config; else listen=true -> 0.0.0.0, false -> 127.0.0.1.
    host = (getattr(args, "host", "") or "").strip() or (
        "0.0.0.0" if bool(getattr(settings, "listen", False)) else "127.0.0.1")
    uvicorn.run(app, host=host, port=settings.port,
                log_level=str(getattr(settings, "log_level", "info")).lower(),
                workers=getattr(args, "workers", 1) or 1)
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

        return main_menu(settings, serve_fn)
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
        for i, r in enumerate(rows, 1):
            mark = " *CURRENT*" if r.get("current") else ""
            token_flag = "yes" if _account_has_token(settings, str(r["identifier"])) else "MISSING"
            print(f"{i}. {r['identifier']}{mark}  uses={r['uses']}  "
                  f"busy={r['busy']}  cooldown={r['cooldown_remaining']}s  token={token_flag}")
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
            pool.add(cfg)
        except ValueError as exc:
            print(f"add failed: {exc}")
            return 1
        settings.accounts.append(cfg)
        if len(settings.accounts) == 1:
            settings.active_account = cfg.identifier
        settings.save()
        print(f"added {cfg.identifier} -> {settings.config_path}")
        if not cfg.token:
            print("WARNING: no token. Password-only fails (RISK_DEVICE_DETECTED).")
            print("TIP: python -m deepseaport accounts login --email "
                  f"{cfg.email or cfg.mobile or ''} (auto-capture)")
            print(TOKEN_HELP)
        return 0

    if action == "set-token":
        from .accounts import _matches as _m
        token = extract_token(args.token or "")
        if not args.identifier or not token:
            print("usage: accounts set-token <email> --token <userToken value>")
            print(TOKEN_HELP)
            return 2
        for a in settings.accounts:
            if _m(a, args.identifier):
                a.token = token
                settings.save()
                print(f"token updated for {a.identifier}")
                return 0
        print(f"not found: {args.identifier}")
        return 1

    if action == "login":
        # Auto-capture token via Obscura, upsert into pool.
        email = args.email or ""
        if not email:
            print("need --email")
            return 2
        token = _obscura_login_token(email, args.password or "", settings)
        if not token:
            return 3
        _upsert_account(settings, email, args.password or "", token)
        settings.save()
        print(f"login ok, token saved for {email} -> {settings.config_path}")
        return 0

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
        kept = []
        removed = False
        for a in settings.accounts:
            from .accounts import _matches as _m
            if not removed and _m(a, identifier):
                removed = True
                continue
            kept.append(a)
        settings.accounts = kept
        if settings.active_account and not any(
                _m(a, settings.active_account) for a in settings.accounts):
            settings.active_account = ""
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
    from .accounts import _matches as _m
    for a in settings.accounts:
        if _m(a, identifier):
            return bool(a.token)
    return False


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
    if len(settings.accounts) == 1 and not getattr(settings, "active_account", ""):
        settings.active_account = cfg.identifier


def _obscura_login_token(email: str, password: str, settings) -> str | None:
    """Run Obscura browser login, return userToken value or None."""
    import json as _json
    import time

    from .accounts import extract_token
    from .mcp_client import McpClient
    from .obscura_bridge import ObscuraBridge

    _ = ObscuraBridge(settings.obscura_bin, settings.obscura_profile)
    bridge_probe = ObscuraBridge(settings.obscura_bin, settings.obscura_profile)
    client = McpClient(bridge_probe.binary, bridge_probe.profile)
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
    from .server import _prepare, _run_completion, create_app

    settings = load_settings(args.config)
    app = create_app(settings)
    app.state.bridge = ObscuraBridge(settings.obscura_bin, settings.obscura_profile)
    app.state.bridge.warmup()
    prep = _prepare({"model": args.model, "messages": [{"role": "user", "content": args.prompt}]}, settings)
    pool = AccountPool(settings.accounts, current=settings.active_account)
    item = pool.acquire(timeout=30)
    try:
        result = _run_completion(app, item, prep)
    finally:
        AccountPool.release(item)
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
    acc_li = acc_sub.add_parser("login", help="auto-capture token via Obscura browser")
    acc_li.add_argument("--email", default="", required=True)
    acc_li.add_argument("--password", default="")
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
