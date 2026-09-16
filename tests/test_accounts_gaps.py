"""Offline gap tests for deepseaport.accounts (no network, <1s each)."""

import asyncio
import threading
import time

import pytest

import deepseaport.accounts as A
from deepseaport.accounts import (
    AccountPool,
    PooledAccount,
    _ban_cache_key,
    _ban_targets,
    _current_matches,
    _is_duplicate,
    _matches,
    _norm,
    _persisted_ban_labels,
    account_has_token,
    add_account,
    ban_label_for,
    check_ban_for_token,
    collect_ban_labels,
    get_cached_ban_labels,
    refresh_ban_labels_background,
    remove_account,
    set_account_token,
    sync_current_from_settings,
)
from deepseaport.config import AccountConfig, Settings


LONG = "x" * 32


def _cfg(email="", token="", mobile="", **kw):
    return AccountConfig(email=email, token=token, mobile=mobile, **kw)


def _settings(*cfgs, active=""):
    return Settings(accounts=list(cfgs), active_account=active, config_path="")


@pytest.fixture(autouse=True)
def _reset_ban_cache():
    with A._BAN_CACHE_LOCK:
        A._BAN_CACHE.update({"ts": 0.0, "key": None, "result": {}})
        A._BAN_REFRESH_INFLIGHT = False
    yield
    with A._BAN_CACHE_LOCK:
        A._BAN_CACHE.update({"ts": 0.0, "key": None, "result": {}})
        A._BAN_REFRESH_INFLIGHT = False


# --- _norm / _matches / _is_duplicate ---

def test_norm_case_and_strip():
    assert _norm("  A@X.com ") == "a@x.com"
    assert _norm("") == ""
    assert _norm(None) == ""  # type: ignore[arg-type]


def test_matches_email_case_insensitive():
    cfg = _cfg(email="A@X.com")
    assert _matches(cfg, "a@x.com") is True
    assert _matches(cfg, "  A@X.COM ") is True
    assert _matches(cfg, "other@x.com") is False


def test_matches_mobile():
    cfg = _cfg(mobile="12345")
    assert _matches(cfg, "12345") is True
    assert _matches(cfg, " 12345 ") is True
    assert _matches(cfg, "99999") is False


def test_matches_identifier_and_token():
    cfg = _cfg(email="a@x.com", token="tokval-abc")
    assert _matches(cfg, cfg.identifier) is True
    assert _matches(cfg, "tokval-abc") is True
    assert _matches(cfg, "tokval") is False  # prefix not enough
    assert _matches(cfg, "") is False


def test_is_duplicate_variants():
    assert _is_duplicate(_cfg(email="a@x.com"), _cfg(email="A@X.COM")) is True
    assert _is_duplicate(_cfg(mobile="111"), _cfg(mobile="111")) is True
    assert _is_duplicate(_cfg(token="t1"), _cfg(token="t1")) is True
    assert _is_duplicate(_cfg(email="a@x.com"), _cfg(email="b@x.com")) is False
    # only one side has email -> no dup
    assert _is_duplicate(_cfg(email="a@x.com"), _cfg(mobile="111")) is False
    assert _is_duplicate(_cfg(), _cfg()) is False


# --- PooledAccount ---

def test_identifier_fallback():
    assert PooledAccount(cfg=_cfg(email="e@x.com")).identifier == "e@x.com"
    assert PooledAccount(cfg=_cfg(mobile="555")).identifier == "555"
    tok = "1234567890ABCDEFGH"
    item = PooledAccount(cfg=_cfg(token=tok))
    assert item.identifier == tok[:10] + "..."
    assert PooledAccount(cfg=_cfg()).identifier == "?"


def test_busy_true_when_lock_held():
    item = PooledAccount(cfg=_cfg(email="a@x.com"))
    assert item.busy is False
    item.lock.acquire()
    try:
        assert item.busy is True
        assert item.available is False
    finally:
        item.lock.release()
    assert item.busy is False


def test_banned_flag_and_expiry_clear():
    cfg = _cfg(email="a@x.com")
    cfg.banned = True
    cfg.banned_until = 0.0
    item = PooledAccount(cfg=cfg)
    assert item.banned is True
    assert item.available is False
    # expired known-expiry clears lazily
    cfg2 = _cfg(email="b@x.com")
    cfg2.banned = True
    cfg2.banned_until = time.time() - 10
    item2 = PooledAccount(cfg=cfg2)
    assert item2.banned is False
    assert cfg2.banned is False
    assert cfg2.banned_until == 0.0


def test_mark_banned_future_until_cooldown():
    item = PooledAccount(cfg=_cfg(email="a@x.com"))
    until = time.time() + 300
    AccountPool.mark_banned(item, until=until)
    assert item.cfg.banned is True
    assert item.cfg.banned_until == pytest.approx(until)
    assert item.banned is True
    assert 290 < item.cooldown_remaining <= 300
    assert item.available is False


def test_mark_banned_unknown_expiry_uses_default():
    item = PooledAccount(cfg=_cfg(email="a@x.com"))
    AccountPool.mark_banned(item, until=None, default_seconds=50)
    assert item.cfg.banned is True
    assert item.cfg.banned_until == 0.0
    assert 40 < item.cooldown_remaining <= 50


def test_cooldown_remaining_and_available():
    item = PooledAccount(cfg=_cfg(email="a@x.com"))
    assert item.cooldown_remaining == 0.0
    assert item.available is True
    AccountPool.mark_bad(item, seconds=60)
    assert 50 < item.cooldown_remaining <= 60
    assert item.available is False


def test_status_keys():
    item = PooledAccount(cfg=_cfg(email="a@x.com"))
    s = item.status()
    assert {"identifier", "email", "busy", "cooldown_remaining",
            "uses", "banned", "banned_until"} <= set(s)
    assert s["identifier"] == "a@x.com"
    assert s["busy"] is False


# --- module helpers ---

def test_current_matches():
    cfg = _cfg(email="a@x.com")
    assert _current_matches(cfg, "A@X.com") is True
    assert _current_matches(cfg, "") is False
    assert _current_matches(cfg, "b@x.com") is False


def test_sync_current_stale_clears():
    s = _settings(_cfg(email="a@x.com", token="t"), active="stale@x.com")
    pool = AccountPool(s.accounts, current=s.active_account)
    assert sync_current_from_settings(pool, s) is True
    assert s.active_account == ""
    assert pool.current == ""


def test_sync_current_empty_want_clears_current():
    s = _settings(_cfg(email="a@x.com", token="t"), active="")
    pool = AccountPool(s.accounts, current="a@x.com")
    assert sync_current_from_settings(pool, s) is True
    assert pool.current == ""


def test_account_has_token():
    s = _settings(_cfg(email="a@x.com", token="tok123"))
    assert account_has_token(s, "a@x.com") is True
    assert account_has_token(s, "A@X.COM") is True
    assert account_has_token(s, "missing@x.com") is False
    s2 = _settings(_cfg(email="b@x.com"))
    assert account_has_token(s2, "b@x.com") is False


def test_add_account_sets_current_and_dup_raises():
    s = _settings(active="")
    pool = AccountPool([])
    item = add_account(s, pool, _cfg(email="a@x.com", token="t1"))
    assert item.identifier == "a@x.com"
    assert s.active_account == "a@x.com"
    assert pool.current == "a@x.com"
    assert len(s.accounts) == 1
    with pytest.raises(ValueError):
        add_account(s, pool, _cfg(email="A@X.com", token="t-other"))


def test_set_account_token_missing_none():
    s = _settings(_cfg(email="a@x.com", token="t1"))
    assert set_account_token(s, "missing@x.com", "zzz") is None
    got = set_account_token(s, "a@x.com", "newtok")
    assert got is not None and got.token == "newtok"


def test_remove_account_clears_current_and_missing():
    s = _settings(_cfg(email="a@x.com", token="t1"),
                  _cfg(email="b@x.com", token="t2"), active="a@x.com")
    assert remove_account(s, "a@x.com") is True
    assert [a.email for a in s.accounts] == ["b@x.com"]
    assert s.active_account == ""
    assert remove_account(s, "missing@x.com") is False


def test_ban_label_for():
    assert ban_label_for({}, "a") == (False, None)
    assert ban_label_for({"a@x.com": None}, "missing") == (False, None)
    assert ban_label_for({"a@x.com": 123.0}, "a@x.com") == (True, 123.0)
    assert ban_label_for({"A@X.com": None}, "a@x.com") == (True, None)


# --- AccountPool behaviour ---

def test_slot_releases_lock():
    pool = AccountPool([_cfg(email="a@x.com", token="t")])
    with pool.slot(timeout=1) as item:
        assert item.busy is True
    assert item.busy is False
    # reusable right away
    with pool.slot(timeout=1):
        pass


def test_release_unknown_no_raise():
    item = PooledAccount(cfg=_cfg(email="z@x.com"))
    AccountPool.release(item)  # never acquired -> swallowed RuntimeError
    assert item.uses == 1


def test_mark_bad_sets_cooldown():
    pool = AccountPool([_cfg(email="a@x.com", token="t")])
    item = pool.get("a@x.com")
    assert item is not None
    AccountPool.mark_bad(item, seconds=45)
    assert 30 < item.cooldown_remaining <= 45
    assert item.available is False


def test_mark_banned_persists_on_cfg():
    pool = AccountPool([_cfg(email="a@x.com", token="t")])
    item = pool.get("a@x.com")
    assert item is not None
    until = time.time() + 200
    AccountPool.mark_banned(item, until=until)
    assert item.cfg.banned is True
    assert item.cfg.banned_until == pytest.approx(until)


def test_acquire_allow_failover_pins_vs_spreads():
    pool = AccountPool([_cfg(email="a@x.com", token="t1"),
                        _cfg(email="b@x.com", token="t2")])
    assert pool.set_current("a@x.com") is True
    held = pool.acquire(timeout=1, allow_failover=False)
    assert held.cfg.email == "a@x.com"
    try:
        # pinned: only current considered -> busy current times out fast
        with pytest.raises(TimeoutError):
            pool.acquire(timeout=0.05, allow_failover=False)
        # failover allowed: spreads to b
        other = pool.acquire(timeout=1, allow_failover=True)
        try:
            assert other.cfg.email == "b@x.com"
        finally:
            AccountPool.release(other)
    finally:
        AccountPool.release(held)


def test_aacquire_prefers_current():
    pool = AccountPool([_cfg(email="a@x.com", token="t1"),
                        _cfg(email="b@x.com", token="t2")])
    assert pool.set_current("b@x.com") is True

    async def _go():
        item = await pool.aacquire(timeout=1)
        try:
            return item.cfg.email
        finally:
            AccountPool.release(item)

    assert asyncio.run(_go()) == "b@x.com"


def test_aacquire_no_failover_pins_current():
    pool = AccountPool([_cfg(email="a@x.com", token="t1"),
                        _cfg(email="b@x.com", token="t2")])
    assert pool.set_current("a@x.com") is True

    async def _go():
        item = await pool.aacquire(timeout=1, allow_failover=False)
        try:
            return item.cfg.email
        finally:
            AccountPool.release(item)

    assert asyncio.run(_go()) == "a@x.com"


# --- check_ban_for_token ---

def test_check_ban_short_token_no_network(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("network must not be called")
    monkeypatch.setattr("deepseaport.client.check_ban", _boom)
    assert check_ban_for_token("t1") == (False, None)
    assert check_ban_for_token("") == (False, None)


def test_check_ban_long_token_calls_client(monkeypatch):
    seen = {}

    def _fake(headers, timeout=12):
        seen["timeout"] = timeout
        assert "x" * 32 in str(headers.get("Authorization", ""))
        return True, 999.0

    monkeypatch.setattr("deepseaport.client.check_ban", _fake)
    assert check_ban_for_token(LONG) == (True, 999.0)
    assert seen["timeout"] == 12


def test_check_ban_exception_returns_false_none(monkeypatch):
    def _boom(headers, timeout=12):
        raise RuntimeError("offline")

    monkeypatch.setattr("deepseaport.client.check_ban", _boom)
    assert check_ban_for_token(LONG) == (False, None)


# --- ban label cache ---

def test_persisted_ban_labels_expired_clears():
    cfg = _cfg(email="a@x.com", token=LONG)
    cfg.banned = True
    cfg.banned_until = time.time() - 5  # set post-init to bypass auto-clear
    out = _persisted_ban_labels([cfg])
    assert out == {}
    assert cfg.banned is False
    assert cfg.banned_until == 0.0


def test_persisted_ban_labels_future_and_unknown():
    future = time.time() + 500
    c1 = _cfg(email="a@x.com", token=LONG)
    c1.banned = True
    c1.banned_until = future
    c2 = _cfg(email="b@x.com", token=LONG)
    c2.banned = True
    c2.banned_until = 0.0
    out = _persisted_ban_labels([c1, c2])
    assert out == {"a@x.com": future, "b@x.com": None}


def test_ban_targets_skips_banned_and_short():
    banned = _cfg(email="ban@x.com", token=LONG)
    banned.banned = True
    short = _cfg(email="short@x.com", token="t1")
    good = _cfg(email="good@x.com", token=LONG)
    assert _ban_targets([banned, short, good]) == (("good@x.com", LONG),)
    assert _ban_targets([]) == ()


def test_ban_cache_key_changes_on_token_change():
    k1 = _ban_cache_key([_cfg(email="a@x.com", token=LONG)])
    k2 = _ban_cache_key([_cfg(email="a@x.com", token=LONG + "y")])
    assert k1 != k2
    assert k1 == _ban_cache_key([_cfg(email="a@x.com", token=LONG)])


def test_get_cached_returns_persisted_plus_cached_no_network(monkeypatch):
    persisted = _cfg(email="p@x.com", token="t1")
    persisted.banned = True
    live = _cfg(email="live@x.com", token=LONG)
    key = _ban_cache_key([persisted, live])
    with A._BAN_CACHE_LOCK:
        A._BAN_CACHE.update({"ts": time.time(), "key": key,
                             "result": {"live@x.com": 42.0}})

    def _boom(*a, **k):
        raise AssertionError("no network allowed")

    monkeypatch.setattr("deepseaport.client.check_ban", _boom)
    out = get_cached_ban_labels([persisted, live])
    assert out["p@x.com"] is None
    assert out["live@x.com"] == 42.0


def test_refresh_singleflight_second_call_no_new_thread(monkeypatch):
    accts = [_cfg(email="a@x.com", token=LONG)]
    calls = []
    real_thread = threading.Thread

    class _T:
        def __init__(self, *a, **k):
            calls.append(1)

        def start(self):
            pass

    monkeypatch.setattr(threading, "Thread", _T)
    try:
        with A._BAN_CACHE_LOCK:
            A._BAN_REFRESH_INFLIGHT = True
        refresh_ban_labels_background(accts)
        assert calls == []
    finally:
        monkeypatch.setattr(threading, "Thread", real_thread)
        with A._BAN_CACHE_LOCK:
            A._BAN_REFRESH_INFLIGHT = False


def test_refresh_ttl_fresh_skip_no_thread(monkeypatch):
    accts = [_cfg(email="a@x.com", token=LONG)]
    key = _ban_cache_key(accts)
    with A._BAN_CACHE_LOCK:
        A._BAN_CACHE.update({"ts": time.time(), "key": key, "result": {}})
    started = []
    real_thread = threading.Thread

    class _T:
        def __init__(self, *a, **k):
            started.append(1)

        def start(self):
            pass

    monkeypatch.setattr(threading, "Thread", _T)
    try:
        refresh_ban_labels_background(accts)
        assert started == []
    finally:
        monkeypatch.setattr(threading, "Thread", real_thread)


def test_collect_parallel_banned_mapping(monkeypatch):
    a = _cfg(email="a@x.com", token=LONG + "a")
    b = _cfg(email="b@x.com", token=LONG + "b")

    def _fake(tok, timeout=12):
        assert timeout == 12
        if tok == LONG + "a":
            return True, 77.0
        return False, None

    monkeypatch.setattr(A, "check_ban_for_token", _fake)
    out = collect_ban_labels([a, b], force_refresh=True)
    assert out == {"a@x.com": 77.0}


def test_collect_ttl_hit_avoids_reprobe(monkeypatch):
    a = _cfg(email="a@x.com", token=LONG)
    calls = []

    def _fake(tok, timeout=12):
        calls.append(tok)
        return True, 5.0

    monkeypatch.setattr(A, "check_ban_for_token", _fake)
    first = collect_ban_labels([a], ttl=60.0, force_refresh=True)
    assert first == {"a@x.com": 5.0}
    assert len(calls) == 1
    second = collect_ban_labels([a], ttl=60.0, force_refresh=False)
    assert second == {"a@x.com": 5.0}
    assert len(calls) == 1  # cache hit, no re-probe


def test_collect_offline_empty_returns_empty(monkeypatch):
    assert collect_ban_labels([], force_refresh=True) == {}
    assert collect_ban_labels([_cfg(email="s@x.com", token="t1")],
                              force_refresh=True) == {}

    def _boom(tok, timeout=12):
        raise RuntimeError("offline")

    monkeypatch.setattr(A, "check_ban_for_token", _boom)
    assert collect_ban_labels([_cfg(email="o@x.com", token=LONG)],
                              force_refresh=True) == {}
