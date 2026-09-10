"""Account pool: rotation + one in-flight stream per account + login.

Thread-safe. Supports runtime add/remove. Cooldown accounts auto-skipped,
acquire() fails over to next healthy account.
"""

from __future__ import annotations

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

IMPERSONATE = "chrome"
LOGIN_TIMEOUT = 30
COOLDOWN_SECONDS = 120
_POLL_INTERVAL = 0.05

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

    def acquire(self, timeout: float = 60) -> PooledAccount:
        """Return account with lock held. Skips busy + cooldown, round-robin.

        CURRENT account (when set) tried first, others fail over after it.
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

    @contextmanager
    def slot(self, timeout: float = 60) -> Iterator[PooledAccount]:
        """`with pool.slot() as acc:` auto-release. Simple use path."""
        item = self.acquire(timeout=timeout)
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
    try:
        return resp.json()
    except Exception:
        return {}


def _dig(data: dict, path: list[str]):
    cur = data
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur
