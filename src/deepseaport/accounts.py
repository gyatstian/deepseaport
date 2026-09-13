"""Account pool: rotation + one in-flight stream per account + login.

Thread-safe. Supports runtime add/remove. Cooldown accounts auto-skipped,
acquire() fails over to next healthy account.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

from curl_cffi import requests as crequests

from . import protocol as P
from .config import AccountConfig

logger = logging.getLogger("deepseaport.accounts")

# Backwards-compatible alias: single source in protocol (shared with client).
IMPERSONATE = P.IMPERSONATE
LOGIN_TIMEOUT = 30
COOLDOWN_SECONDS = 120
_POLL_INTERVAL = 0.05
# Ban check skips obviously-fake tokens (unit tests use "t1", "tok-...").
# Real userToken values are 64-char; short ones would only waste a request.
MIN_TOKEN_LEN_FOR_BAN_CHECK = 20
BAN_CHECK_TIMEOUT = 12

TOKEN_HELP = (
    "userToken needed per account. Two ways:\n"
    "  AUTO (recommended): python -m deepseaport login --email you@x.com\n"
    "    Obscura browser logs in, captures token, appends to pool.\n"
    "  MANUAL: login at https://chat.deepseek.com in Chrome, then F12 >\n"
    "    Application > Local Storage > https://chat.deepseek.com > userToken.\n"
    "    Copy the `value` field (64-char). Or Console:\n"
    "      JSON.parse(localStorage.getItem('userToken')).value\n"
    "    Then: python -m deepseaport accounts add --email you@x.com --token <value>\n"
    "    Paste raw value OR full JSON; both accepted. "
    "Password-only accounts fail: direct POST login hits RISK_DEVICE_DETECTED."
)


def extract_token(raw: str) -> str:
    """Accept raw 64-char value, {"value": ...} JSON, or double-encoded JSON."""
    import json as _json

    s = (raw or "").strip().strip("\"'")
    if not s:
        return ""
    if s.startswith("{"):
        try:
            data = _json.loads(s)
            if isinstance(data, str):  # double-encoded from browser_evaluate
                data = _json.loads(data)
            if isinstance(data, dict) and data.get("value"):
                return str(data["value"]).strip()
        except Exception:
            pass
    # localStorage.getItem returns JSON string; bare value passes through.
    return s


def _norm(identifier: str) -> str:
    return (identifier or "").strip().lower()


def _matches(cfg: AccountConfig, identifier: str) -> bool:
    key = _norm(identifier)
    if not key:
        return False
    if _norm(cfg.email) and _norm(cfg.email) == key:
        return True
    if cfg.mobile and cfg.mobile.strip() == identifier.strip():
        return True
    if _norm(cfg.identifier) == key:
        return True
    # Allow full token match (not prefix) for delete by token.
    if cfg.token and cfg.token.strip() == identifier.strip():
        return True
    return False


def _is_duplicate(a: AccountConfig, b: AccountConfig) -> bool:
    if a.email and b.email and _norm(a.email) == _norm(b.email):
        return True
    if a.mobile and b.mobile and a.mobile.strip() == b.mobile.strip():
        return True
    if a.token and b.token and a.token.strip() == b.token.strip():
        return True
    return False


@dataclass
class PooledAccount:
    cfg: AccountConfig
    lock: threading.Lock = field(default_factory=threading.Lock)
    bad_until: float = 0.0
    uses: int = 0

    @property
    def identifier(self) -> str:
        return self.cfg.identifier

    @property
    def busy(self) -> bool:
        return self.lock.locked()

    @property
    def cooldown_remaining(self) -> float:
        return max(0.0, self.bad_until - time.time())

    @property
    def available(self) -> bool:
        return not self.busy and self.cooldown_remaining <= 0

    def status(self) -> dict:
        return {
            "identifier": self.identifier,
            "email": self.cfg.email,
            "busy": self.busy,
            "cooldown_remaining": round(self.cooldown_remaining, 1),
            "uses": self.uses,
        }


def _current_matches(cfg: AccountConfig, current: str) -> bool:
    return bool(current) and _matches(cfg, current)


def sync_current_from_settings(pool: AccountPool, settings) -> bool:
    """Copy settings.active_account into pool. Clears stale selection.

    Returns True when selection changed (caller should settings.save()).
    """
    want = (getattr(settings, "active_account", "") or "").strip()
    if not want:
        if pool.current:
            pool.set_current(None)
            return True
        return False
    cur_item = pool.get_current()
    if cur_item is not None and _matches(cur_item.cfg, want):
        return False
    if pool.set_current(want):
        # Normalize stored id to canonical identifier.
        settings.active_account = pool.current
        return True
    # Stale id (deleted from config): clear.
    settings.active_account = ""
    pool.set_current(None)
    return True


def account_has_token(settings, identifier: str) -> bool:
    """True when settings holds a matching account that carries a token."""
    for a in (getattr(settings, "accounts", None) or []):
        if _matches(a, identifier):
            return bool(a.token)
    return False


def add_account(settings, pool: "AccountPool", cfg: AccountConfig) -> PooledAccount:
    """Add cfg to pool + settings and make it CURRENT. Raises ValueError on dup."""
    item = pool.add(cfg)
    settings.accounts.append(cfg)
    # New account becomes CURRENT (default) immediately.
    settings.active_account = cfg.identifier
    try:
        pool.set_current(cfg.identifier)
    except Exception:
        pass
    return item


def set_account_token(settings, identifier: str, token: str) -> AccountConfig | None:
    """Set token on the first matching settings account; return it or None."""
    for a in (getattr(settings, "accounts", None) or []):
        if _matches(a, identifier):
            a.token = token
            return a
    return None


def remove_account(settings, identifier: str) -> bool:
    """Drop the first matching account from settings; clear stale CURRENT.

    Mirrors AccountPool.remove() ordering. Returns True when one was removed.
    """
    kept, dropped = [], False
    for a in (getattr(settings, "accounts", None) or []):
        if not dropped and _matches(a, identifier):
            dropped = True
            continue
        kept.append(a)
    if not dropped:
        return False
    settings.accounts = kept
    if settings.active_account and not any(
            _matches(a, settings.active_account) for a in settings.accounts):
        settings.active_account = ""
    return True


def ban_label_for(ban_map: dict, identifier: str) -> tuple[bool, float | None]:
    """Look up a ban expiry: exact key, then case-insensitive fallback.

    Returns (found, mute_until). found is True when the identifier is banned
    even if the expiry is unknown (None).
    """
    if not ban_map:
        return False, None
    if identifier in ban_map:
        return True, ban_map[identifier]
    key = str(identifier).strip().lower()
    for bid, until in ban_map.items():
        if str(bid).strip().lower() == key:
            return True, until
    return False, None


class AccountPool:
    def __init__(self, accounts: list[AccountConfig] | None = None,
                 current: str | None = None) -> None:
        self._items: list[PooledAccount] = [PooledAccount(cfg=a) for a in (accounts or [])]
        self._cursor = 0
        self._guard = threading.Lock()
        self._current: str = (current or "").strip()

    @property
    def current(self) -> str:
        with self._guard:
            return self._current

    def set_current(self, identifier: str | None) -> bool:
        """Select CURRENT account. None/empty clears to auto-failover mode.

        Returns False when identifier not found (selection unchanged).
        """
        key = (identifier or "").strip()
        with self._guard:
            if not key:
                self._current = ""
                return True
            for item in self._items:
                if _matches(item.cfg, key):
                    self._current = item.identifier
                    logger.info("current account -> %s", item.identifier)
                    return True
            return False

    def get_current(self) -> PooledAccount | None:
        with self._guard:
            cur = self._current
            items = list(self._items)
        if not cur:
            return None
        for item in items:
            if _matches(item.cfg, cur):
                return item
        return None

    def __len__(self) -> int:
        with self._guard:
            return len(self._items)

    def add(self, cfg: AccountConfig) -> PooledAccount:
        """Add account. Raises ValueError on empty/duplicate."""
        if not (cfg.email or cfg.mobile or cfg.token):
            raise ValueError("account needs email, mobile, or token")
        with self._guard:
            for item in self._items:
                if _is_duplicate(item.cfg, cfg):
                    raise ValueError(f"account already in pool: {item.identifier}")
            item = PooledAccount(cfg=cfg)
            self._items.append(item)
            logger.info("account added: %s (pool=%d)", item.identifier, len(self._items))
            return item

    def remove(self, identifier: str) -> bool:
        """Delete account by email/mobile/identifier/token. True if removed.

        Safe while account in-flight: holder keeps its ref, release() still works.
        Clears CURRENT selection when deleted account was selected.
        """
        with self._guard:
            for i, item in enumerate(self._items):
                if _matches(item.cfg, identifier):
                    del self._items[i]
                    if self._current and _matches(item.cfg, self._current):
                        self._current = ""
                    logger.info("account removed: %s (pool=%d)",
                                item.identifier, len(self._items))
                    return True
        return False

    def get(self, identifier: str) -> PooledAccount | None:
        with self._guard:
            for item in self._items:
                if _matches(item.cfg, identifier):
                    return item
        return None

    def status(self) -> list[dict]:
        with self._guard:
            items = list(self._items)
            cur = self._current
        rows = [item.status() for item in items]
        for row, item in zip(rows, items):
            row["current"] = bool(cur) and _matches(item.cfg, cur)
        return rows

    def clear_cooldown(self, identifier: str | None = None) -> int:
        """Clear cooldown for one account (or all when None). Returns count."""
        n = 0
        with self._guard:
            for item in self._items:
                if identifier is None or _matches(item.cfg, identifier):
                    if item.bad_until > 0:
                        n += 1
                    item.bad_until = 0.0
        if n:
            logger.info("cooldown cleared for %d account(s)", n)
        return n

    def acquire(self, timeout: float = 60, allow_failover: bool = True) -> PooledAccount:
        """Return account with lock held. Skips busy + cooldown, round-robin.

        CURRENT account (when set) tried first, others fail over after it.
        When allow_failover=False and CURRENT is in the pool, only CURRENT
        is considered (sticky: concurrent requests queue on it instead of
        spreading to other accounts). Busy is per-account lock state.
        Raises ValueError when pool empty, TimeoutError when all busy/cooldown
        for longer than timeout.
        """
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            with self._guard:
                if not self._items:
                    raise ValueError("no DeepSeek accounts in pool")
                snapshot = list(self._items)
                cur = self._current
                start = self._cursor % len(snapshot)
                self._cursor += 1
            now = time.time()
            ordered = snapshot[start:] + snapshot[:start]
            if cur:
                preferred = [it for it in snapshot if _matches(it.cfg, cur)]
                if preferred and not allow_failover:
                    ordered = preferred
                else:
                    rest = [it for it in ordered if not _matches(it.cfg, cur)]
                    ordered = preferred + rest
            for item in ordered:
                with self._guard:
                    if item not in self._items:
                        continue  # removed concurrently
                if item.bad_until > now:
                    continue  # on cool -> fail over to next
                if item.lock.acquire(blocking=False):
                    with self._guard:
                        if item not in self._items:
                            item.lock.release()
                            continue
                    if item.bad_until > time.time():
                        item.lock.release()  # marked bad while racing
                        continue
                    return item
            if time.monotonic() >= deadline:
                raise TimeoutError("no DeepSeek account free (busy or cooldown)")
            time.sleep(min(_POLL_INTERVAL, max(0.001, deadline - time.monotonic())))

    async def aacquire(self, timeout: float = 60, allow_failover: bool = True) -> PooledAccount:
        """Async acquire: same rotation/skip semantics, no worker thread held.

        Polls with asyncio.sleep so the default executor stays free for real
        blocking work (challenge fetch, parse, status, save). Used by the
        async FastAPI endpoint; sync worker threads keep using acquire().
        allow_failover=False pins to CURRENT (see acquire()).
        """
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            with self._guard:
                if not self._items:
                    raise ValueError("no DeepSeek accounts in pool")
                snapshot = list(self._items)
                cur = self._current
                start = self._cursor % len(snapshot)
                self._cursor += 1
            now = time.time()
            ordered = snapshot[start:] + snapshot[:start]
            if cur:
                preferred = [it for it in snapshot if _matches(it.cfg, cur)]
                if preferred and not allow_failover:
                    ordered = preferred
                else:
                    rest = [it for it in ordered if not _matches(it.cfg, cur)]
                    ordered = preferred + rest
            for item in ordered:
                with self._guard:
                    if item not in self._items:
                        continue  # removed concurrently
                if item.bad_until > now:
                    continue  # on cool -> fail over to next
                if item.lock.acquire(blocking=False):
                    with self._guard:
                        if item not in self._items:
                            item.lock.release()
                            continue
                    if item.bad_until > time.time():
                        item.lock.release()  # marked bad while racing
                        continue
                    return item
            if time.monotonic() >= deadline:
                raise TimeoutError("no DeepSeek account free (busy or cooldown)")
            await asyncio.sleep(min(_POLL_INTERVAL, max(0.001, deadline - time.monotonic())))

    @contextmanager
    def slot(self, timeout: float = 60, allow_failover: bool = True) -> Iterator[PooledAccount]:
        """`with pool.slot() as acc:` auto-release. Simple use path."""
        item = self.acquire(timeout=timeout, allow_failover=allow_failover)
        try:
            yield item
        finally:
            self.release(item)

    @staticmethod
    def release(item: PooledAccount) -> None:
        item.uses += 1
        try:
            item.lock.release()
        except RuntimeError:
            pass  # already released / never acquired

    @staticmethod
    def mark_bad(item: PooledAccount, seconds: float = COOLDOWN_SECONDS) -> None:
        item.bad_until = time.time() + max(0.0, seconds)
        logger.warning("account %s cooling down for %ds", item.cfg.identifier, int(seconds))


def login(cfg: AccountConfig, headers: dict[str, str]) -> str:
    """Password login (WAF cookies in headers); returns fresh user token."""
    payload = P.login_payload(email=cfg.email, mobile=cfg.mobile, password=cfg.password)
    resp = crequests.post(P.LOGIN_URL, headers=headers, json=payload,
                          impersonate=IMPERSONATE, timeout=LOGIN_TIMEOUT)
    data = _json(resp)
    token = _dig(data, ["data", "biz_data", "user", "token"])
    if not token:
        raise RuntimeError(f"login failed: HTTP {resp.status_code} body={resp.text[:200]}")
    cfg.token = token
    logger.info("login ok for %s", cfg.identifier)
    return token


def _json(resp) -> dict:
    return P.parse_json(resp)


def _dig(data: dict, path: list[str]):
    cur = data
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def check_ban_for_token(token: str, timeout: int = BAN_CHECK_TIMEOUT) -> tuple[bool, float | None]:
    """Best-effort ban probe for one token. Never raises; (False, None) on skip/fail.

    Skips empty/short (test) tokens without network. Otherwise GETs
    users/current and returns (is_banned, mute_until).
    """
    tok = (token or "").strip()
    if len(tok) < MIN_TOKEN_LEN_FOR_BAN_CHECK:
        return False, None
    try:
        # Local import: client pulls curl session, not needed at module import.
        from . import client as _DS

        headers = P.bridge_headers(bearer=tok)
        return _DS.check_ban(headers, timeout=timeout)
    except Exception as exc:
        logger.debug("ban check ignored: %s", exc)
        return False, None


# TTL cache for ban labels: the TUI menu redraws on every keypress, and a
# network probe per redraw blocks the menu (2 accounts ~1s, offline up to
# 12s). Cache for BAN_LABEL_TTL seconds; TUI reads the cache instantly and
# refreshes in background so menu render never blocks on network.
BAN_LABEL_TTL = 60.0
_BAN_CACHE_LOCK = threading.Lock()
_BAN_CACHE: dict = {"ts": 0.0, "key": None, "result": {}}
_BAN_REFRESH_INFLIGHT = False


def _ban_targets(accounts: list[AccountConfig] | None) -> tuple[tuple[str, str], ...]:
    """Sorted (identifier, token) pairs worth probing; shared key/normalizer."""
    try:
        targets = [((a.identifier or ""), ((a.token or "").strip()))
                   for a in (accounts or [])]
        targets = [(i, t) for i, t in targets if len(t) >= MIN_TOKEN_LEN_FOR_BAN_CHECK]
        return tuple(sorted(targets))
    except Exception:
        return ()


def _ban_cache_key(accounts: list[AccountConfig] | None) -> tuple:
    return _ban_targets(accounts)


def get_cached_ban_labels(accounts: list[AccountConfig] | None) -> dict[str, float | None]:
    """Instant, never-network ban labels (may be empty/stale). For menu render."""
    try:
        key = _ban_cache_key(accounts)
        if not key:
            return {}
        with _BAN_CACHE_LOCK:
            if _BAN_CACHE.get("key") != key:
                return {}
            return dict(_BAN_CACHE.get("result") or {})
    except Exception:
        return {}


def refresh_ban_labels_background(accounts: list[AccountConfig] | None,
                                  timeout: int = BAN_CHECK_TIMEOUT) -> None:
    """Refresh ban-label cache in a daemon thread unless fresh/refreshing."""
    global _BAN_REFRESH_INFLIGHT
    try:
        key = _ban_cache_key(accounts)
        if not key:
            return
        with _BAN_CACHE_LOCK:
            if _BAN_REFRESH_INFLIGHT:
                return
            try:
                fresh = ((time.time() - float(_BAN_CACHE.get("ts") or 0)) < BAN_LABEL_TTL
                         and _BAN_CACHE.get("key") == key)
            except Exception:
                fresh = False
            if fresh:
                return
            _BAN_REFRESH_INFLIGHT = True
        snapshot = list(accounts or [])

        def _run() -> None:
            global _BAN_REFRESH_INFLIGHT
            try:
                collect_ban_labels(snapshot, timeout=timeout, force_refresh=True)
            except Exception:
                pass
            finally:
                with _BAN_CACHE_LOCK:
                    _BAN_REFRESH_INFLIGHT = False

        threading.Thread(target=_run, daemon=True).start()
    except Exception:
        pass


def collect_ban_labels(accounts: list[AccountConfig],
                       timeout: int = BAN_CHECK_TIMEOUT,
                       ttl: float = BAN_LABEL_TTL,
                       force_refresh: bool = False) -> dict[str, float | None]:
    """Map identifier -> mute_until for banned accounts only (parallel, best-effort).

    Empty dict when none banned, offline, or tokens missing/short. Never raises.
    Used by TUI/CLI account managers to render red '(BANNED: D Month)'.
    Results are TTL-cached (default 60s) so menu redraws don't probe network.
    """
    import concurrent.futures as _fut

    key = _ban_targets(accounts)
    targets = list(key)
    if not targets:
        return {}
    if not force_refresh and ttl and ttl > 0:
        try:
            with _BAN_CACHE_LOCK:
                if (_BAN_CACHE.get("key") == key
                        and (time.time() - float(_BAN_CACHE.get("ts") or 0)) < float(ttl)):
                    return dict(_BAN_CACHE.get("result") or {})
        except Exception:
            pass
    out: dict[str, float | None] = {}

    def _one(pair: tuple[str, str]) -> tuple[str, bool, float | None]:
        ident, tok = pair
        try:
            banned, until = check_ban_for_token(tok, timeout=timeout)
            return ident, banned, until
        except Exception:
            return ident, False, None

    try:
        with _fut.ThreadPoolExecutor(max_workers=min(8, len(targets))) as ex:
            for ident, banned, until in ex.map(_one, targets):
                if banned:
                    out[ident] = until
    except Exception as exc:
        logger.debug("collect_ban_labels ignored: %s", exc)
    try:
        with _BAN_CACHE_LOCK:
            _BAN_CACHE["key"] = key
            _BAN_CACHE["ts"] = time.time()
            _BAN_CACHE["result"] = dict(out)
    except Exception:
        pass
    return out
