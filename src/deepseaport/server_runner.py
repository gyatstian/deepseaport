"""Shared server bootstrap: bind-host resolution, port scanning/auto-fix,
uvicorn Config/Server wiring, readiness polling, and shutdown helpers.

Single source of truth for the serve (cli) and chat (chat_ui) entry points;
previously these helpers were duplicated across deepseaport.cli and
deepseaport.chat_ui. cli.py / chat_ui.py keep thin re-exports for backward
compatibility (tests and tui import the historical names from there).
"""

from __future__ import annotations

import time
import urllib.request

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


def _next_free_port(host: str, port: int, limit: int = 50,
                    check=None) -> int | None:
    """Scan port+1..port+limit for a free port. None when all busy."""
    found = _scan_free_ports(host, int(port) + 1, 1, limit=limit, check=check)
    return found[0] if found else None


def _ensure_free_port(settings, host: str, port: int,
                      is_free=None, next_free=None) -> int:
    """Port auto-fix: busy port offers next free one and saves it.

    Interactive (tty stdin): asks Y/n. Non-interactive (headless): auto-picks
    the next free port when available. Returns the port to use (original when
    user declines or no free port found).

    `is_free` / `next_free` are injection seams used by deepseaport.cli so
    tests can monkeypatch the historical cli-level names; default to the
    canonical helpers in this module.
    """
    import sys as _sys

    free = is_free if is_free is not None else _is_port_free
    nxt_fn = next_free if next_free is not None else _next_free_port

    port = int(port)
    if free(host, port):
        return port
    nxt = nxt_fn(host, port)
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


def _allocate_sequential_ports(host: str, base: int, count: int,
                               is_free=None) -> list[int]:
    """Allocate `count` distinct free ports scanning upward from `base`.

    Mirrors the single-server auto-fix: each candidate is socket-checked;
    busy ones are skipped (5001 taken -> 5002 -> 5003 ...). Never mutates
    settings. Returns fewer than `count` when the range is exhausted.
    """
    return _scan_free_ports(host, base, count, check=is_free)


def _wait_for_server(base_url: str, timeout: int = 20) -> bool:
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


def _start_background_server(app, host: str, port: int,
                             log_level: str = "error",
                             access_log: bool = False):
    """Build a uvicorn Config/Server and run it in a daemon thread.

    Shared by `serve + chat` (one server) and multi-chat (N servers).
    Returns (server, thread) so callers can join/stop them.
    """
    import threading

    config = uvicorn.Config(app, host=host, port=int(port),
                            log_level=log_level, access_log=access_log)
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    return server, thread


def _run_server(settings, args, ensure_free=None) -> int:
    """Blocking serve loop (single worker) or uvicorn.run (multi-worker).

    `ensure_free` is an injection seam for deepseaport.cli so the historical
    cli-level `_ensure_free_port` monkeypatch keeps working.
    """
    from .server import apply_log_level, create_app

    import threading

    apply_log_level(getattr(settings, "log_level", "INFO"))
    app = create_app(settings)
    # --host flag overrides config; else listen=true -> 0.0.0.0, false -> 127.0.0.1.
    host = _resolve_bind_host(settings, args)
    port = int(getattr(settings, "port", 5001))
    log_level = str(getattr(settings, "log_level", "info")).lower()
    workers = int(getattr(args, "workers", 1) or 1)
    if ensure_free is None:
        ensure_free = _ensure_free_port
    port = ensure_free(settings, host, port)
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
