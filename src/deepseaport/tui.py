"""Stdlib-only TUI: main page (Start server / Settings) + settings editor.

No third-party deps (Windows-safe: plain input(), no curses) so `serve`
works everywhere. All edits mutate the passed Settings in place and persist
via Settings.save().
"""

from __future__ import annotations

import os
import sys
from typing import Callable

from .config import VALID_LOG_LEVELS, Settings


def _on_off(value: bool) -> str:
    return "ON" if value else "OFF"


def _use_color() -> bool:
    """False when NO_COLOR is set or stdout is not a tty (logs/tests stay clean)."""
    if os.environ.get("NO_COLOR") is not None:
        return False
    try:
        return bool(sys.stdout.isatty())
    except Exception:
        return False


def _c(text: str, code: str) -> str:
    if not _use_color():
        return text
    return f"\x1b[{code}m{text}\x1b[0m"


def _clear_screen() -> None:
    """Clear screen on menu enter. No-op when not a tty (tests/pipes)."""
    try:
        if not sys.stdout.isatty():
            return
    except Exception:
        return
    try:
        if os.name == "nt":
            # Enable ANSI VT processing on Windows 10+, best-effort.
            try:
                import ctypes

                kernel32 = ctypes.windll.kernel32
                handle = kernel32.GetStdHandle(-11)
                mode = ctypes.c_ulong()
                if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                    kernel32.SetConsoleMode(handle, mode.value | 0x0004)
            except Exception:
                pass
        print("\x1b[2J\x1b[H", end="")
    except Exception:
        pass


def _header(title: str) -> None:
    """Blank line + title + blank line. Single visual anchor per screen."""
    print()
    print(_c(f"=== {title} ===", "1;36"))
    print()


def _section(title: str) -> None:
    """Group related settings without adding a nested menu."""
    print()
    print(_c(f"  {title}", "1"))


def _setting_line(number: int, label: str, value: str,
                  width: int = 28) -> None:
    """One settings row: number, label, current value."""
    print(f"  {number:>2}. {label:<{width}} {value}")


def _ok(msg: str) -> None:
    print(_c(f"  ok: {msg}", "32"))


def _info(msg: str) -> None:
    print(f"  {msg}")


def _err(msg: str) -> None:
    print(_c(f"  ! {msg}", "31"))


def _ask(prompt: str, default: str = "") -> str:
    try:
        suffix = f" [{default}]" if default else ""
        return input(f"{prompt}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        raise


def _ask_password(prompt: str = "Password (Enter=skip)") -> str:
    """Hidden password prompt (no echo). Preserves inner/edge spaces.

    Falls back to visible _ask when getpass cannot control the terminal
    (e.g. no tty in tests) — getpass itself falls back to input() there,
    so mocked builtins.input still works.
    """
    try:
        import getpass as _gp

        try:
            # getpass strips only the trailing newline, keeps spaces intact.
            return _gp.getpass(f"{prompt}: ")
        except (EOFError, KeyboardInterrupt):
            print()
            raise
        except Exception:
            return _ask(prompt)
    except (EOFError, KeyboardInterrupt):
        print()
        raise


def _has_token(settings: Settings, identifier: str) -> bool:
    from .accounts import account_has_token
    return account_has_token(settings, identifier)


def _toggle_bool(settings: Settings, attr: str, label: str) -> None:
    """Instant ON<->OFF flip, saved immediately (no y/n prompt)."""
    setattr(settings, attr, not bool(getattr(settings, attr, True)))
    settings.save()
    _ok(f"{label} -> {_on_off(getattr(settings, attr))} (saved)")


def _toggle_stream(settings: Settings) -> None:
    """Instant buffered<->live flip, saved immediately."""
    settings.stream_mode = "live" if settings.stream_mode != "live" else "buffered"
    settings.save()
    _ok(f"Reply streaming -> {settings.stream_mode} (saved)")


def _edit_choice(settings: Settings, attr: str, label: str, options: tuple[str, ...]) -> None:
    current = str(getattr(settings, attr))
    print()
    _info(f"{label} (current: {current})")
    for i, opt in enumerate(options, 1):
        mark = " *" if opt == current else ""
        print(f"    {i}. {opt}{mark}")
    print()
    ans = _ask("Number or value (Enter=cancel)", default="")
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
    _info(f"Retries per failed request (current: {settings.max_retries}).")
    _info("0 disables retries; the original default is 1.")
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


def _mask_key(key: str) -> str:
    k = (key or "").strip()
    if len(k) <= 8:
        return "***"
    return f"{k[:4]}***{k[-2:]}"


def _edit_port(settings: Settings) -> None:
    """Prompt for port, free-check via cli helpers, save."""
    from .cli import _ensure_free_port, _resolve_bind_host

    print()
    _info(f"Port (current: {settings.port}). 1-65535.")
    print()
    ans = _ask("Enter port (Enter=cancel)", default="")
    if not ans:
        _info("Cancelled.")
        return
    try:
        candidate = int(ans)
    except ValueError:
        _err("Not a number, cancelled.")
        return
    if not 1 <= candidate <= 65535:
        _err("Out of range 1-65535, cancelled.")
        return
    host = _resolve_bind_host(settings)
    try:
        picked = _ensure_free_port(settings, host, candidate)
    except (EOFError, KeyboardInterrupt):
        print()
        return
    # _ensure_free_port saves the winner when it auto-picks; when the
    # candidate was free it returns it unsaved — persist either way.
    try:
        settings.port = int(picked)
        settings.save()
    except (EOFError, KeyboardInterrupt):
        print()
        return
    _ok(f"Port -> {settings.port} (saved)")


def _edit_keys(settings: Settings) -> None:
    """Add/remove API keys. Values masked, never printed in full."""
    while True:
        print()
        keys = list(getattr(settings, "keys", []) or [])
        if keys:
            _info(f"API keys ({len(keys)}):")
            for i, k in enumerate(keys, 1):
                print(f"    {i}. {_mask_key(k)}")
        else:
            _info("API keys (0): empty = local API needs no Bearer.")
        print()
        print("    a. Add key")
        print("    d. Delete key by number")
        print("    b. Back")
        print()
        try:
            ans = input("Keys (a/d <n>/b): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not ans or ans in ("b", "back", "q", "quit"):
            return
        if ans in ("a", "add"):
            try:
                import getpass as _gp

                try:
                    raw = _gp.getpass("New API key (Enter=cancel): ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    return
                except Exception:
                    raw = _ask("New API key (Enter=cancel)", default="")
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if not raw:
                _info("Cancelled.")
                continue
            if raw in keys:
                _err("Key already present, cancelled.")
                continue
            keys.append(raw)
            settings.keys = keys
            settings.save()
            _ok(f"Key added ({len(keys)} total, saved)")
            continue
        if ans.startswith("d"):
            parts = ans.split()
            num = parts[1] if len(parts) > 1 else ""
            if not num:
                try:
                    num = _ask("Number to DELETE", default="")
                except (EOFError, KeyboardInterrupt):
                    print()
                    return
            try:
                idx = int(num) - 1
            except ValueError:
                _err("Not a number, cancelled.")
                continue
            if 0 <= idx < len(keys):
                try:
                    confirm = input(f"Delete key {idx + 1} ({_mask_key(keys[idx])})? Type y: ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    print()
                    continue
                if confirm not in ("y", "yes"):
                    _info("Cancelled (nothing deleted).")
                    continue
                del keys[idx]
                settings.keys = keys
                settings.save()
                _ok(f"Key removed ({len(keys)} left, saved)")
            else:
                _err("Out of range, nothing deleted.")
            continue
        _err("Unknown option. Use a, d <n>, or b.")


def _edit_obscura_bin(settings: Settings) -> None:
    """Set obscura binary path. Empty/'auto'/'clear' resets to auto-discover."""
    from pathlib import Path

    print()
    cur = (getattr(settings, "obscura_bin", "") or "").strip()
    _info(f"Obscura binary (current: {cur or '(auto-discover)'}).")
    _info("Drop obscura.exe next to config.json, or set explicit path.")
    print()
    ans = _ask("Enter path (Enter=cancel, 'clear'=auto)", default="")
    if not ans:
        _info("Cancelled.")
        return
    low = ans.strip().lower()
    if low in ("clear", "auto", "none"):
        settings.obscura_bin = ""
        settings.save()
        _ok("Obscura binary -> auto-discover (saved)")
        return
    p = Path(ans.strip()).expanduser()
    if not p.exists():
        _err(f"Path not found: {p}. Not saved.")
        _info("TIP: place obscura.exe next to config.json, bin/, tools/, vendor/.")
        return
    settings.obscura_bin = str(p)
    settings.save()
    _ok(f"Obscura binary -> {settings.obscura_bin} (saved)")


def _edit_chat_model(settings: Settings) -> None:
    """Pick chat_model from server MODELS (fallback to known four)."""
    try:
        from .server import MODELS as _MODELS

        options = tuple(sorted(_MODELS)) if isinstance(_MODELS, dict) else ()
    except Exception:
        options = ()
    if not options:
        options = ("deepseek-flash", "deepseek-flash-reasoner",
                   "deepseek-flash-search", "deepseek-flash-reasoner-search")
    _edit_choice(settings, "chat_model", "Chat model", options)


def settings_menu(settings: Settings) -> None:
    """Blocking settings editor; returns on Back.

    Settings are grouped by purpose instead of shown as one flat list. The
    numbering is kept stable so existing users do not lose muscle memory;
    toggles save immediately and multi-value options prompt for a new value.
    """
    _clear_screen()
    while True:
        _header(f"Settings ({settings.config_path or 'unsaved'})")

        _section("Request handling")
        _setting_line(1, "Tool calling", _on_off(settings.enable_tools))
        _setting_line(2, "Warm up WAF at startup",
                      _on_off(settings.warmup_on_startup))
        _setting_line(3, "Delete session after reply",
                      _on_off(settings.auto_delete_session))
        _setting_line(4, "Retries per failed request", str(settings.max_retries))
        _setting_line(5, "Parallel setup requests",
                      _on_off(settings.parallel_challenge_fetch))

        _section("Chat & logs")
        _setting_line(6, "Log level", str(settings.log_level))
        _setting_line(7, "Reply streaming", str(settings.stream_mode))

        _section("Server access")
        _setting_line(8, "Allow LAN access", _on_off(settings.listen))
        _setting_line(9, "Port", str(settings.port))
        nkeys = len(getattr(settings, "keys", []) or [])
        keys_disp = f"{nkeys} set" if nkeys else "none (open)"
        _setting_line(10, "API keys", keys_disp)

        _section("Setup & defaults")
        obsc = (getattr(settings, "obscura_bin", "") or "").strip() or "(auto)"
        if len(obsc) > 30:
            obsc = "..." + obsc[-27:]
        _setting_line(11, "Obscura binary", obsc)
        _setting_line(12, "Chat model", str(settings.chat_model))

        _section("Accounts")
        _setting_line(13, "Account failover",
                      _on_off(getattr(settings, "use_multiple_accounts", True)))

        print()
        print("  14. Back (q)")
        print()
        try:
            choice = input("Select: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if choice in ("14", "b", "back", "q", "quit", ""):
            return
        if choice not in {str(i) for i in range(1, 14)}:
            _err("Unknown option: type 1-13, or q.")
            continue
        # Clear the menu before prompting so repeated edits do not stack
        # multiple copies of the settings screen in the terminal.
        _clear_screen()
        try:
            if choice == "1":
                _toggle_bool(settings, "enable_tools", "Tool calling")
            elif choice == "2":
                _toggle_bool(settings, "warmup_on_startup",
                             "WAF warmup at startup")
            elif choice == "3":
                _toggle_bool(settings, "auto_delete_session",
                             "Delete session after reply")
            elif choice == "4":
                _edit_retries(settings)
            elif choice == "5":
                _toggle_bool(settings, "parallel_challenge_fetch",
                             "Parallel setup requests")
            elif choice == "6":
                _edit_choice(settings, "log_level", "Log level", VALID_LOG_LEVELS)
            elif choice == "7":
                _toggle_stream(settings)
            elif choice == "8":
                _toggle_bool(settings, "listen", "Allow LAN access")
            elif choice == "9":
                _edit_port(settings)
            elif choice == "10":
                _edit_keys(settings)
            elif choice == "11":
                _edit_obscura_bin(settings)
            elif choice == "12":
                _edit_chat_model(settings)
            elif choice == "13":
                _toggle_bool(settings, "use_multiple_accounts",
                             "Account failover")
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

    def _offer_auto_login(email: str, password: str, confirmed: bool = False) -> str:
        """Run Obscura login inside TUI. confirmed=True skips the Y/n prompt."""
        if not email:
            return ""
        if not confirmed:
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
        from .cli import _obscura_login_token, _unpack_login_token_result
        _info("Running Obscura login (browser, ~30s)...")
        try:
            token, login_banned, ban_detail = _unpack_login_token_result(
                _obscura_login_token(email, password, settings))
        except Exception as exc:
            _err(f"Auto-login failed: {exc}")
            print()
            print(TOKEN_HELP)
            return ""
        if login_banned:
            if token:
                try:
                    from .auth import parse_ban_until
                    from .accounts import _matches as _m
                    until = parse_ban_until(ban_detail)
                    for a in settings.accounts:
                        if _m(a, email):
                            a.banned = True
                            a.banned_until = float(until or 0.0)
                except Exception:
                    pass
                _err(f"Auto-login captured a token for {email}, "
                     "but the account is BANNED. Token saved; Account "
                     "list will show the ban.")
                if ban_detail:
                    print(f"Evidence: {ban_detail[:300]}")
                return token
            _err(f"Auto-login refused for {email}: account BANNED per login page.")
            if ban_detail:
                print(f"Evidence: {ban_detail[:300]}")
            return ""
        if not token:
            return ""
        _ok(f"Auto-login ok for {email}")
        return token

    _clear_screen()
    while True:
        pool = _pool()
        rows = pool.status()
        current = pool.current or ""
        # Best-effort ban labels without blocking menu render: instant TTL
        # cache read + background refresh (network never on the draw path).
        # Skips missing/short (test) tokens without network; never raises.
        try:
            from .accounts import (ban_label_for as _ban_lookup,
                                   get_cached_ban_labels as _ban_cached,
                                   refresh_ban_labels_background as _ban_refresh)
            from . import protocol as _P
            ban_map = _ban_cached(settings.accounts) or {}
            try:
                _ban_refresh(settings.accounts)
            except Exception:
                pass
        except Exception:
            ban_map = {}
            _ban_lookup = None  # type: ignore
            _P = None  # type: ignore
        _header(f"Accounts ({len(settings.accounts)})")
        _info(f"Current: [{current or 'ALL (auto-failover)'}]")
        print()
        if rows:
            id_width = max([len("Identifier")] + [len(r["identifier"]) for r in rows])
            id_width = min(id_width, 32)
            print(f"    {'#':<2} {'Identifier':<{id_width}}  {'Uses':<4} {'Token':<7} {'Busy':<4} {'Cool':<6} {'Cur'}")
            print(f"    {'--':<2} {'-' * id_width}  {'----':<4} {'-----':<7} {'----':<4} {'----':<6} {'---'}")
            for i, row in enumerate(rows, 1):
                mark = "*" if row.get("current") else ""
                has = _has_token(settings, row["identifier"])
                ident = row["identifier"]
                if len(ident) > 32:
                    ident = ident[:29] + "..."
                busy = "busy" if row.get("busy") else "-"
                try:
                    cool = float(row.get("cooldown_remaining", 0) or 0)
                except (TypeError, ValueError):
                    cool = 0
                cool_s = f"{cool:.0f}s" if cool > 0 else "-"
                # Red ban suffix: "(BANNED: 16 September)" from mute_until.
                ban_suffix = ""
                try:
                    _found, _until = (_ban_lookup(ban_map, row["identifier"])
                                      if _ban_lookup else (False, None))
                    if _found and _P is not None:
                        ban_suffix = " " + _c(_P.format_ban_label(_until), "31")
                    elif _found:
                        ban_suffix = " (BANNED)"
                except Exception:
                    ban_suffix = ""
                print(f"    {i:<2} {ident:<{id_width}}  {row['uses']:<4} "
                      f"{'yes' if has else 'missing':<7} {busy:<4} {cool_s:<6} {mark}{ban_suffix}")
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
            choice = input("Select: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if choice in ("b", "back", "q", "quit", ""):
            return
        if choice == "h":
            _clear_screen()
            print()
            print(TOKEN_HELP)
            continue
        if choice == "c":
            _clear_screen()
            settings.active_account = ""
            settings.save()
            _ok("Selection cleared -> ALL (auto-failover)")
            continue
        if choice == "a":
            _clear_screen()
            print()
            try:
                email = _ask("Email (Enter=skip)", default="")
                password = _ask_password("Password (Enter=skip)")
                raw_token = _ask("Token (y = auto-token)", default="")
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if raw_token.strip().lower() in ("y", "yes"):
                token = _offer_auto_login(email, password, confirmed=True)
            else:
                token = extract_token(raw_token)
            cfg = AccountConfig(email=email, password=password, token=token)
            try:
                from .accounts import add_account as _add
                _add(settings, pool, cfg)
            except ValueError as exc:
                _err(f"Add failed: {exc}")
                continue
            settings.save()
            _ok(f"Added {cfg.identifier} (saved)")
            if not cfg.token:
                _err("No token. Password-only fails.")
                print()
                print(TOKEN_HELP)
            continue
        if choice == "t":
            _clear_screen()
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
            from .accounts import set_account_token as _set
            matched = _set(settings, ident, token)
            if matched is not None:
                settings.save()
                _ok(f"Token updated for {matched.identifier} (saved)")
            else:
                _err(f"Not found: {ident}")
            continue
        if choice == "d":
            _clear_screen()
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
            from .accounts import remove_account as _remove
            _remove(settings, ident)
            settings.save()
            _ok(f"Removed {ident} (saved)")
            continue
        # Number = SELECT, never delete.
        try:
            idx = int(choice) - 1
        except ValueError:
            _err("Unknown option. Number selects, d deletes.")
            continue
        _clear_screen()
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


def _preflight_reason(settings: Settings) -> str:
    """Block Start when serving would fail fast. "" means ready.

    Empty pool -> 503 on first request; accounts without any token ->
    401/failover on first request. Caller jumps to the Accounts menu so
    the fix is one step away.
    """
    if not settings.accounts:
        return "No accounts yet. Add one first."
    for a in settings.accounts:
        if (a.token or "").strip():
            return ""
    return "Accounts have no token. Add one (y = auto-token) first."


def _refresh_from_disk(settings: Settings) -> None:
    """Pick up account/config changes made via API while serving."""
    try:
        from .config import load_settings as _reload
        if settings.config_path:
            _fresh = _reload(settings.config_path)
            settings.accounts = _fresh.accounts
            settings.active_account = _fresh.active_account
    except Exception:
        pass


def main_menu(settings: Settings, serve_fn: Callable[[Settings], int],
              chat_fn: Callable[[Settings], int] | None = None,
              multi_fn: Callable[[Settings], int] | None = None) -> int:
    """Blocking main page. Server stop returns to menu (0 = quit)."""
    _clear_screen()
    accounts_num = "5" if chat_fn is not None else "3"
    while True:
        _header("deepseaport")
        _info(f"Config   : {settings.config_path or '(unsaved)'}")
        _info(f"Host     : {'0.0.0.0 (all interfaces)' if settings.listen else '127.0.0.1 (localhost only)'}")
        _info(f"Port     : {settings.port}")
        _info(f"Reply    : {settings.stream_mode}  |  Tools: {_on_off(settings.enable_tools)}"
              f"  |  Failover: {_on_off(getattr(settings, 'use_multiple_accounts', True))}")
        cur = (settings.active_account or "").strip() or "ALL"
        _info(f"Accounts : {len(settings.accounts)}  |  Current: [{cur}]")
        has_token = any((getattr(a, "token", "") or "").strip()
                        for a in settings.accounts)
        if not settings.accounts:
            _info(f"Setup    : no accounts yet -> Accounts ({accounts_num}) to add one")
        elif not has_token:
            _info(f"Setup    : no account token -> Accounts ({accounts_num}) to refresh one")
        print()
        print("  1. Start server")
        if chat_fn is not None:
            print("  2. Start server with chat")
            print("  3. Run multiple servers")
            print("  4. Settings")
            print("  5. Accounts")
        else:
            print("  2. Settings")
            print("  3. Accounts")
        print("  q. Quit")
        print()
        try:
            choice = input("Select: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if choice == "1":
            reason = _preflight_reason(settings)
            if reason:
                print()
                _err(reason)
                if "token" in reason.lower():
                    from .accounts import TOKEN_HELP as _TH
                    print()
                    print(_TH)
                accounts_menu(settings)
                continue
            print()
            _info("Server running. Press Ctrl+C or Esc to stop and return to menu.")
            print()
            try:
                serve_fn(settings)
            except (EOFError, KeyboardInterrupt):
                print()
                _info("Server stopped, back to menu.")
            _refresh_from_disk(settings)
            continue
        if chat_fn is not None and choice == "2":
            reason = _preflight_reason(settings)
            if reason:
                print()
                _err(reason)
                if "token" in reason.lower():
                    from .accounts import TOKEN_HELP as _TH
                    print()
                    print(_TH)
                accounts_menu(settings)
                continue
            try:
                chat_fn(settings)
            except (EOFError, KeyboardInterrupt):
                print()
                _info("Chat stopped, back to menu.")
            _refresh_from_disk(settings)
            continue
        if chat_fn is not None and choice == "3":
            reason = _preflight_reason(settings)
            if reason:
                print()
                _err(reason)
                if "token" in reason.lower():
                    from .accounts import TOKEN_HELP as _TH
                    print()
                    print(_TH)
                accounts_menu(settings)
                continue
            try:
                if multi_fn is not None:
                    multi_fn(settings)
                else:
                    from .chat_ui import run_multi_chat_servers as _multi
                    _multi(settings)
            except (EOFError, KeyboardInterrupt):
                print()
                _info("Multi-server stopped, back to menu.")
            _refresh_from_disk(settings)
            continue
        if choice == ("4" if chat_fn is not None else "2"):
            settings_menu(settings)
            _clear_screen()
            continue
        if choice == ("5" if chat_fn is not None else "3"):
            accounts_menu(settings)
            _clear_screen()
            continue
        if choice in ("q", "quit", "0", "exit"):
            return 0
        _err("Unknown option: type 1, 2, 3, 4, 5, or q." if chat_fn is not None
             else "Unknown option: type 1, 2, 3, or q.")
