"""Stdlib-only TUI: main page (Start server / Settings) + settings editor.

No third-party deps (Windows-safe: plain input(), no curses) so `serve`
works everywhere. All edits mutate the passed Settings in place and persist
via Settings.save().
"""

from __future__ import annotations

from typing import Callable

from .config import VALID_LOG_LEVELS, Settings


def _on_off(value: bool) -> str:
    return "ON" if value else "OFF"


_DIV = "-" * 44


def _header(title: str) -> None:
    """Blank line + title + blank line. Single visual anchor per screen."""
    print()
    print(f"=== {title} ===")
    print()


def _divider() -> None:
    print(f"  {_DIV}")


def _ok(msg: str) -> None:
    print(f"  ok: {msg}")


def _info(msg: str) -> None:
    print(f"  {msg}")


def _err(msg: str) -> None:
    print(f"  ! {msg}")


def _ask(prompt: str, default: str = "") -> str:
    try:
        suffix = f" [{default}]" if default else ""
        return input(f"{prompt}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        raise


def _has_token(settings: Settings, identifier: str) -> bool:
    from .accounts import _matches as _m
    for a in settings.accounts:
        if _m(a, identifier):
            return bool(a.token)
    return False


def _toggle_bool(settings: Settings, attr: str, label: str) -> None:
    """Instant ON<->OFF flip, saved immediately (no y/n prompt)."""
    setattr(settings, attr, not bool(getattr(settings, attr, True)))
    settings.save()
    _ok(f"{label} -> {_on_off(getattr(settings, attr))} (saved)")


def _toggle_stream(settings: Settings) -> None:
    """Instant buffered<->live flip, saved immediately."""
    settings.stream_mode = "live" if settings.stream_mode != "live" else "buffered"
    settings.save()
    _ok(f"Stream mode -> {settings.stream_mode} (saved)")


def _edit_choice(settings: Settings, attr: str, label: str, options: tuple[str, ...]) -> None:
    current = str(getattr(settings, attr))
    print()
    _info(f"{label} (current: {current})")
    for i, opt in enumerate(options, 1):
        mark = " *" if opt == current else ""
        print(f"    {i}. {opt}{mark}")
    print()
    ans = _ask("Pick number (Enter=cancel)", default="")
    if not ans:
        _info("Cancelled.")
        return
    try:
        idx = int(ans) - 1
    except ValueError:
        # Allow typing the value directly.
        lowered = {o.lower(): o for o in options}
        if ans.lower() in lowered:
            setattr(settings, attr, lowered[ans.lower()])
            settings.save()
            _ok(f"{label} -> {getattr(settings, attr)} (saved)")
        else:
            _err("Invalid choice, cancelled.")
        return
    if 0 <= idx < len(options):
        setattr(settings, attr, options[idx])
        settings.save()
        _ok(f"{label} -> {getattr(settings, attr)} (saved)")
    else:
        _err("Invalid choice, cancelled.")


def _edit_retries(settings: Settings) -> None:
    print()
    _info(f"Retry count per failure class (pow/session/WAF), current: {settings.max_retries}.")
    _info("0 disables retries; 1 matches the original one-retry behaviour.")
    print()
    ans = _ask("Enter 0-5 (Enter=cancel)", default="")
    if not ans:
        _info("Cancelled.")
        return
    try:
        value = int(ans)
    except ValueError:
        _err("Not a number, cancelled.")
        return
    if 0 <= value <= 5:
        settings.max_retries = value
        settings.save()
        _ok(f"Retry count -> {value} (saved)")
    else:
        _err("Out of range 0-5, cancelled.")


def settings_menu(settings: Settings) -> None:
    """Blocking settings editor; returns on Back.

    Two-state settings (bools + buffered/live stream) flip instantly on
    number press. Multi-value settings (retry count, log level) prompt.
    """
    descriptions = {
        1: "Tool calling on/off",
        2: "WAF warmup at startup",
        3: "Auto-delete session after use",
        4: "Retries per failure class (0 disables)",
        5: "Fetch session + challenge in parallel",
        6: "Log level for deepseaport loggers",
        7: "buffered (one reply) vs live (token deltas)",
        8: "Serve on 0.0.0.0 (LAN) vs 127.0.0.1 (localhost only)",
    }
    while True:
        _header(f"Settings ({settings.config_path or 'unsaved'})")
        print(f"  1. Tool calling .......... {_on_off(settings.enable_tools)}")
        _info(f"     {descriptions[1]}")
        print(f"  2. Startup WAF warmup .... {_on_off(settings.warmup_on_startup)}")
        _info(f"     {descriptions[2]}")
        print(f"  3. Auto session delete ... {_on_off(settings.auto_delete_session)}")
        _info(f"     {descriptions[3]}")
        print(f"  4. Retry count ........... {settings.max_retries}")
        _info(f"     {descriptions[4]}")
        print(f"  5. Parallel fetch ........ {_on_off(settings.parallel_challenge_fetch)}")
        _info(f"     {descriptions[5]}")
        print(f"  6. Log level ............. {settings.log_level}")
        _info(f"     {descriptions[6]}")
        print(f"  7. Stream mode ........... {settings.stream_mode}")
        _info(f"     {descriptions[7]}")
        print(f"  8. Listen all interfaces . {_on_off(settings.listen)}")
        _info(f"     {descriptions[8]}")
        print()
        print("  9. Back (q)")
        print()
        try:
            choice = input("Select [1-9/q]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if choice in ("9", "b", "back", "q", "quit", ""):
            return
        if choice not in {str(i) for i in range(1, 9)}:
            _err("Unknown option.")
            continue
        print()
        _divider()
        try:
            if choice == "1":
                _toggle_bool(settings, "enable_tools", "Tool calling")
            elif choice == "2":
                _toggle_bool(settings, "warmup_on_startup", "Startup WAF warmup")
            elif choice == "3":
                _toggle_bool(settings, "auto_delete_session", "Auto session delete")
            elif choice == "4":
                _edit_retries(settings)
            elif choice == "5":
                _toggle_bool(settings, "parallel_challenge_fetch", "Parallel challenge fetch")
            elif choice == "6":
                _edit_choice(settings, "log_level", "Log level", VALID_LOG_LEVELS)
            elif choice == "7":
                _toggle_stream(settings)
            elif choice == "8":
                _toggle_bool(settings, "listen", "Listen all interfaces (0.0.0.0)")
        except (EOFError, KeyboardInterrupt):
            print()
            return


def accounts_menu(settings: Settings) -> None:
    """List/select/add/remove pooled accounts. Mutates settings + saves.

    Number NEVER deletes: it SELECTS current account. Delete is explicit
    via `d` + confirmation. Missing token offers in-TUI auto-login.
    """
    from .accounts import (TOKEN_HELP, AccountPool, extract_token,
                           sync_current_from_settings)
    from .config import AccountConfig

    def _pool() -> AccountPool:
        pool = AccountPool(settings.accounts, current=settings.active_account)
        if sync_current_from_settings(pool, settings):
            settings.save()
        return pool

    def _offer_auto_login(email: str, password: str) -> str:
        """Token skipped: offer to run Obscura login inside TUI."""
        if not email:
            return ""
        try:
            ans = input("No token. Run auto-login via Obscura now? [Y/n]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return ""
        if ans not in ("", "y", "yes"):
            print()
            print(TOKEN_HELP)
            return ""
        if not password:
            try:
                import getpass
                password = getpass.getpass("DeepSeek password (Enter=skip): ")
            except (EOFError, KeyboardInterrupt):
                print()
                return ""
        if not password:
            _err("Auto-login needs password. Cancelled.")
            print()
            print(TOKEN_HELP)
            return ""
        from .cli import _obscura_login_token
        _info("Running Obscura login (browser, ~30s)...")
        try:
            token = _obscura_login_token(email, password, settings)
        except Exception as exc:
            _err(f"Auto-login failed: {exc}")
            print()
            print(TOKEN_HELP)
            return ""
        if not token:
            return ""
        _ok(f"Auto-login ok for {email}")
        return token

    while True:
        pool = _pool()
        rows = pool.status()
        current = pool.current or ""
        _header(f"Accounts ({len(settings.accounts)})")
        _info(f"Current: [{current or 'ALL (auto-failover)'}]")
        print()
        if rows:
            id_width = max([len("Identifier")] + [len(r["identifier"]) for r in rows])
            id_width = min(id_width, 32)
            print(f"    {'#':<2} {'Identifier':<{id_width}}  {'Uses':<4} {'Token':<7} {'Cur'}")
            print(f"    {'--':<2} {'-' * id_width}  {'----':<4} {'-----':<7} {'---'}")
            for i, row in enumerate(rows, 1):
                mark = "*" if row.get("current") else ""
                has = _has_token(settings, row["identifier"])
                ident = row["identifier"]
                if len(ident) > 32:
                    ident = ident[:29] + "..."
                print(f"    {i:<2} {ident:<{id_width}}  {row['uses']:<4} "
                      f"{'yes' if has else 'missing':<7} {mark}")
        else:
            _info("(no accounts yet -- use 'a' to add one)")
        print()
        _info("number = select account (never deletes)")
        print()
        print("  a. Add account")
        print("  t. Set/refresh token")
        print("  d. Delete account (asks confirmation)")
        print("  c. Clear selection (use ALL)")
        print("  h. How to get token")
        print()
        print("  b. Back (q)")
        print()
        try:
            choice = input("Select [number/a/t/d/c/h/b]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if choice in ("b", "back", "q", "quit", ""):
            return
        if choice == "h":
            print()
            print(TOKEN_HELP)
            continue
        if choice == "c":
            settings.active_account = ""
            settings.save()
            _ok("Selection cleared -> ALL (auto-failover)")
            continue
        if choice == "a":
            print()
            try:
                email = _ask("Email (Enter=skip)", default="")
                password = _ask("Password (Enter=skip)", default="")
                token = _ask("Token (Enter=skip = auto-login offer)", default="")
            except (EOFError, KeyboardInterrupt):
                print()
                return
            token = extract_token(token)
            if not token and (email or password):
                token = _offer_auto_login(email, password)
                if token and not password:
                    pass  # password already asked inside offer
            cfg = AccountConfig(email=email, password=password, token=token)
            try:
                pool.add(cfg)
            except ValueError as exc:
                _err(f"Add failed: {exc}")
                continue
            settings.accounts.append(cfg)
            # First account becomes CURRENT for clarity.
            if len(settings.accounts) == 1:
                settings.active_account = cfg.identifier
            settings.save()
            _ok(f"Added {cfg.identifier} (saved)")
            if not cfg.token:
                _err("No token. Password-only fails.")
                print()
                print(TOKEN_HELP)
            continue
        if choice == "t":
            print()
            try:
                ident = _ask("Account email", default="")
                token = _ask("New token value (Enter=skip = auto-login offer)", default="")
            except (EOFError, KeyboardInterrupt):
                print()
                return
            token = extract_token(token)
            if not ident:
                _info("Cancelled.")
                continue
            if not token:
                from .accounts import _matches as _m
                existing = next((a for a in settings.accounts if _m(a, ident)), None)
                if existing is None:
                    _err(f"Not found: {ident}")
                    continue
                token = _offer_auto_login(existing.email, existing.password)
                if not token:
                    continue
            from .accounts import _matches as _m
            for a in settings.accounts:
                if _m(a, ident):
                    a.token = token
                    settings.save()
                    _ok(f"Token updated for {a.identifier} (saved)")
                    break
            else:
                _err(f"Not found: {ident}")
            continue
        if choice == "d":
            print()
            try:
                ident = _ask("Identifier to DELETE (email or number)", default="")
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if not ident:
                _info("Cancelled (nothing deleted).")
                continue
            # Allow number for delete, but ONLY inside explicit d flow.
            rows_now = pool.status()
            if ident.isdigit():
                idx = int(ident) - 1
                if 0 <= idx < len(rows_now):
                    ident = rows_now[idx]["identifier"]
                else:
                    _err("Out of range, nothing deleted.")
                    continue
            try:
                confirm = input(f"Delete '{ident}'? Type y to confirm: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                continue
            if confirm not in ("y", "yes"):
                _info("Cancelled (nothing deleted).")
                continue
            if not pool.remove(ident):
                _err(f"Not found: {ident} (nothing deleted)")
                continue
            from .accounts import _matches as _m
            dropped = False
            kept = []
            for a in settings.accounts:
                if not dropped and _m(a, ident):
                    dropped = True
                    continue
                kept.append(a)
            settings.accounts = kept
            if settings.active_account and not any(
                    _m(a, settings.active_account) for a in settings.accounts):
                settings.active_account = ""
            settings.save()
            _ok(f"Removed {ident} (saved)")
            continue
        # Number = SELECT, never delete.
        try:
            idx = int(choice) - 1
        except ValueError:
            _err("Unknown option. Number selects, d deletes.")
            continue
        rows_now = pool.status()
        if 0 <= idx < len(rows_now):
            ident = rows_now[idx]["identifier"]
            if pool.set_current(ident):
                settings.active_account = pool.current
                settings.save()
                _ok(f"Current -> [{ident}] (saved). Failover to others on cooldown.")
            else:
                _err(f"Select failed: {ident}")
        else:
            _err("Out of range.")


def main_menu(settings: Settings, serve_fn: Callable[[Settings], int]) -> int:
    """Blocking main page. Returns serve exit code (0 = quit without serving)."""
    while True:
        _header("deepseaport")
        _info(f"Config   : {settings.config_path or '(unsaved)'}")
        _info(f"Host     : {'0.0.0.0 (all interfaces)' if settings.listen else '127.0.0.1 (localhost)'}")
        _info(f"Port     : {settings.port}")
        _info(f"Stream   : {settings.stream_mode}  |  Tools: {_on_off(settings.enable_tools)}")
        cur = (settings.active_account or "").strip() or "ALL"
        _info(f"Accounts : {len(settings.accounts)}  |  Current: [{cur}]")
        print()
        print("  1. Start server")
        print("  2. Settings")
        print("  3. Accounts")
        print("  q. Quit")
        print()
        try:
            choice = input("Select [1-3/q]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if choice == "1":
            return serve_fn(settings)
        if choice == "2":
            settings_menu(settings)
            continue
        if choice == "3":
            accounts_menu(settings)
            continue
        if choice in ("q", "quit", "0", "exit"):
            return 0
        _err("Unknown option: type 1, 2, 3, or q.")
