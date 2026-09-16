"""Config: config.json + environment, no secrets committed."""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from .tokens import extract_token

logger = logging.getLogger("deepseaport.config")

DEFAULT_PORT = 5001
PACKAGE_DIR = Path(__file__).resolve().parent

# Serializes config.json writes within the process. Callers span the event
# loop (endpoint saves) and worker threads (_ensure_token / account pool);
# without this a concurrent Path.write_text can truncate/interleave the file.
# NOTE: process-local only. Separate OS processes (uvicorn --workers N) each
# have their own lock; atomic os.replace keeps the file valid JSON but the
# last writer wins (a ban/token saved in one worker can be overwritten by a
# concurrent save from another). No file lock: it would add a Windows
# dependency + stale-lock risk without fixing last-writer-wins.
_CONFIG_SAVE_LOCK = threading.Lock()


def _app_dir() -> Path:
    # Frozen exe: exe location; source: repo root.
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return PACKAGE_DIR.parent.parent


REPO_DIR = _app_dir()
DATA_DIR = Path(os.environ.get("DEEPSEAPORT_DATA", str(REPO_DIR / "data")))


@dataclass
class AccountConfig:
    email: str = ""
    mobile: str = ""
    password: str = ""
    token: str = ""
    # Persisted ban marker so CLI/TUI labels survive process restarts and
    # accounts whose token is missing. ``banned_until`` is 0 when unknown.
    banned: bool = False
    banned_until: float = 0.0

    def __setattr__(self, name, value):
        # Normalize at every assignment, not just construction. Runtime token
        # refresh, TUI set-token, and API endpoints all mutate this field and
        # must never leak the full JSON wrapper into an Authorization header.
        if name == "token":
            value = extract_token(value)
        super().__setattr__(name, value)

    def __post_init__(self) -> None:
        # Normalize pasted JSON shapes (including {"value":null}) at the
        # boundary so the rest of the app only ever sees a raw token or "".
        self.token = extract_token(self.token)
        try:
            self.banned_until = max(0.0, float(self.banned_until or 0.0))
        except (TypeError, ValueError):
            self.banned_until = 0.0
        # Expired persisted bans shouldn't hide a usable account forever.
        if self.banned and self.banned_until and self.banned_until <= time.time():
            self.banned = False
            self.banned_until = 0.0

    @property
    def identifier(self) -> str:
        return (self.email or self.mobile or (self.token[:10] + "..." if self.token else "?")).strip()


VALID_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")
VALID_STREAM_MODES = ("buffered", "live")


@dataclass
class Settings:
    keys: list[str] = field(default_factory=list)
    accounts: list[AccountConfig] = field(default_factory=list)
    active_account: str = ""
    obscura_bin: str = ""
    obscura_profile: str = ""
    # Real-browser login: "" disables it; load_settings() defaults a fresh
    # config to "auto" so Chrome/Helium discovery works out of the box.
    # The browser is used only for interactive login/token refresh, never the
    # hot path; Obscura still owns WAF cookies.
    browser_bin: str = ""
    browser_headless: bool = False
    port: int = DEFAULT_PORT
    # true -> bind 0.0.0.0 (all interfaces); false -> 127.0.0.1 only.
    # --host CLI flag overrides both.
    listen: bool = False
    config_path: str = ""
    # TUI-tunable behaviour. Defaults preserve the historical behaviour.
    enable_tools: bool = True
    warmup_on_startup: bool = True
    auto_delete_session: bool = True
    max_retries: int = 1
    parallel_challenge_fetch: bool = True
    # True -> busy CURRENT fails over to another healthy account (parallel
    # subagents / multitasking). False -> stick to CURRENT, queue on it.
    use_multiple_accounts: bool = True
    log_level: str = "INFO"
    stream_mode: str = "live"
    # Last /model picked in server-with-chat mode. Validated in chat_ui
    # (MODELS lives in server.py; importing it here would cycle).
    chat_model: str = "deepseek-flash"

    def save(self) -> None:
        if not self.config_path:
            return
        payload = {
            "keys": self.keys,
            "accounts": [
                {"email": a.email, "mobile": a.mobile, "password": a.password,
                 "token": extract_token(a.token),
                 "banned": bool(getattr(a, "banned", False)),
                 "banned_until": float(getattr(a, "banned_until", 0.0) or 0.0)}
                for a in self.accounts
            ],
            "active_account": self.active_account,
            "obscura_bin": self.obscura_bin,
            "obscura_profile": self.obscura_profile,
            "browser_bin": self.browser_bin,
            "browser_headless": self.browser_headless,
            "port": self.port,
            "listen": self.listen,
            "enable_tools": self.enable_tools,
            "warmup_on_startup": self.warmup_on_startup,
            "auto_delete_session": self.auto_delete_session,
            "max_retries": self.max_retries,
            "parallel_challenge_fetch": self.parallel_challenge_fetch,
            "use_multiple_accounts": self.use_multiple_accounts,
            "log_level": self.log_level,
            "stream_mode": self.stream_mode,
            "chat_model": self.chat_model,
        }
        data = json.dumps(payload, ensure_ascii=False, indent=2)
        target = Path(self.config_path)
        # Atomic write: unique temp file in the same dir, then os.replace.
        # Concurrent writers (event loop + worker threads) can't interleave
        # or observe a half-written config. If the rename is blocked (see
        # below) we degrade to an in-place rewrite.
        with _CONFIG_SAVE_LOCK:
            target.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(
                prefix=target.name + ".", suffix=".tmp", dir=str(target.parent))
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    fh.write(data)
                    fh.flush()
                    os.fsync(fh.fileno())
                # Transient AV/indexer locks block rename for ~ms: retry so the
                # atomic path still wins instead of degrading to in-place.
                replaced = False
                last_err: OSError | None = None
                for attempt in range(4):
                    try:
                        os.replace(tmp_name, target)
                        tmp_name = None
                        replaced = True
                        break
                    except PermissionError as exc:
                        last_err = exc
                        time.sleep(0.05 * (attempt + 1))
                    except OSError as exc:
                        last_err = exc
                        break
                if not replaced:
                    # Windows: an external handle (editor, antivirus, another
                    # process) can block the rename even though in-place writes
                    # are allowed, because rename needs FILE_SHARE_DELETE on
                    # every open handle. Fall back to a direct overwrite (with
                    # fsync to shrink the crash half-write window); the save
                    # lock still serializes our own writers. Loud warning: the
                    # non-atomic path risks a truncated config on crash, but
                    # raising here would 500 endpoints whose in-memory state
                    # already succeeded.
                    logger.warning(
                        "config replace blocked (%s), overwriting in place: %s",
                        last_err, target)
                    with open(target, "w", encoding="utf-8") as fh:
                        fh.write(data)
                        fh.flush()
                        try:
                            os.fsync(fh.fileno())
                        except OSError:
                            pass
            finally:
                if tmp_name:
                    try:
                        os.unlink(tmp_name)
                    except OSError:
                        pass


def _default_config_path() -> Path:
    # Resolution order is intentional: explicit DEEPSEAPORT_CONFIG env wins,
    # then a config.json in the current working directory (per-deployment
    # override), then the app/exe directory. Keeping cwd support (not a single
    # path) preserves portable multi-instance setups; callers log the chosen
    # file so a different-cwd silent 401/503 is diagnosable.
    env = os.environ.get("DEEPSEAPORT_CONFIG")
    if env:
        return Path(env)
    here = Path.cwd() / "config.json"
    if here.exists():
        return here
    return _app_dir() / "config.json"


def load_settings(path: str | None = None) -> Settings:
    cfg_path = Path(path) if path else _default_config_path()
    try:
        logger.info("using config: %s (cwd=%s)", cfg_path, Path.cwd())
    except Exception:
        pass
    raw: dict = {}
    if cfg_path.exists():
        try:
            raw = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            raw = {}
    keys = raw.get("keys", [])
    env_keys = os.environ.get("DEEPSEAPORT_KEYS")
    if env_keys:
        keys = [k.strip() for k in env_keys.split(",") if k.strip()]

    accounts: list[AccountConfig] = []
    for a in raw.get("accounts", []):
        if isinstance(a, dict) and (a.get("email") or a.get("mobile") or a.get("token")):
            accounts.append(AccountConfig(
                email=a.get("email", ""), mobile=a.get("mobile", ""),
                password=a.get("password", ""), token=a.get("token", ""),
                banned=bool(a.get("banned", False)),
                banned_until=a.get("banned_until", 0.0),
            ))
    # Single-account env: replaces placeholder config, otherwise appends as an
    # extra failover account (DEEPSEAPORT_KEYS replaces keys; TOKEN appends).
    env_token = os.environ.get("DEEPSEAPORT_TOKEN") or os.environ.get("DEEPSEEK_TOKEN")
    env_email = os.environ.get("DEEPSEAPORT_EMAIL") or os.environ.get("DEEPSEEK_EMAIL")
    env_password = os.environ.get("DEEPSEAPORT_PASSWORD") or os.environ.get("DEEPSEEK_PASSWORD")
    if env_token or (env_email and env_password):
        env_acc = AccountConfig(email=env_email or "", password=env_password or "", token=env_token or "")
        if not accounts or not (accounts[0].token or accounts[0].email):
            accounts = [env_acc]
        else:
            accounts.append(env_acc)

    def _bool(env_key: str, raw_key: str, default: bool) -> bool:
        if env_key in os.environ:
            return os.environ[env_key].strip().lower() in ("1", "true", "yes", "on")
        val = raw.get(raw_key, default)
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.strip().lower() in ("1", "true", "yes", "on")
        return bool(val) if val is not None else default

    def _int(env_key: str, raw_key: str, default: int) -> int:
        if env_key in os.environ:
            try:
                return max(0, int(os.environ[env_key]))
            except ValueError:
                return default
        try:
            return max(0, int(raw.get(raw_key, default)))
        except (ValueError, TypeError):
            return default

    def _str(env_key: str, raw_key: str, default: str,
             valid: tuple[str, ...] | None = None) -> str:
        raw_val = raw.get(raw_key, default)
        val = str(os.environ.get(env_key, raw_val)).strip()
        if valid and val not in valid:
            # Case-insensitive match for convenience, else fall back.
            lowered = {v.lower(): v for v in valid}
            return lowered.get(val.lower(), default)
        return val or default

    port = _int("DEEPSEAPORT_PORT", "port", DEFAULT_PORT)
    return Settings(
        keys=keys,
        accounts=accounts,
        active_account=str(os.environ.get("DEEPSEAPORT_ACTIVE_ACCOUNT",
                                           raw.get("active_account", ""))).strip(),
        obscura_bin=os.environ.get("OBSCURA_BIN", raw.get("obscura_bin", "")),
        obscura_profile=os.environ.get("OBSCURA_PROFILE", raw.get("obscura_profile", "")),
        browser_bin=os.environ.get(
            "DEEPSEAPORT_BROWSER_BIN", raw.get("browser_bin", "auto")),
        browser_headless=_bool("DEEPSEAPORT_BROWSER_HEADLESS", "browser_headless", False),
        port=port,
        listen=_bool("DEEPSEAPORT_LISTEN", "listen", False),
        config_path=str(cfg_path),
        enable_tools=_bool("DEEPSEAPORT_ENABLE_TOOLS", "enable_tools", True),
        warmup_on_startup=_bool("DEEPSEAPORT_WARMUP_ON_STARTUP", "warmup_on_startup", True),
        auto_delete_session=_bool("DEEPSEAPORT_AUTO_DELETE_SESSION", "auto_delete_session", True),
        max_retries=_int("DEEPSEAPORT_MAX_RETRIES", "max_retries", 1),
        parallel_challenge_fetch=_bool(
            "DEEPSEAPORT_PARALLEL_FETCH", "parallel_challenge_fetch", True),
        use_multiple_accounts=_bool(
            "DEEPSEAPORT_USE_MULTIPLE_ACCOUNTS", "use_multiple_accounts", True),
        log_level=_str("DEEPSEAPORT_LOG_LEVEL", "log_level", "INFO", VALID_LOG_LEVELS),
        stream_mode=_str("DEEPSEAPORT_STREAM_MODE", "stream_mode", "live", VALID_STREAM_MODES),
        chat_model=_str("DEEPSEAPORT_CHAT_MODEL", "chat_model", "deepseek-flash"),
    )
