"""Gaps for chat_ui: helpers + multi/single server REPL (all faked, offline)."""

import json
import logging
import types


def _mk_acc(email, token="t1"):
    from deepseaport.config import AccountConfig
    return AccountConfig(email=email, token=token)


class _FakeCfg:
    def __init__(self, *a, **k):
        pass


class _FakeSrv:
    def __init__(self, cfg=None):
        self.should_exit = False
        self.config = cfg

    def run(self):
        pass


class _AliveThread:
    def __init__(self, target=None, daemon=None):
        self._target = target
        self.daemon = daemon
        self.started = False
        self._alive = True

    def start(self):
        self.started = True

    def is_alive(self):
        return True

    def join(self, timeout=None):
        self._alive = False


class _DeadThread(_AliveThread):
    def is_alive(self):
        return False


def _install_single_fakes(monkeypatch, chat_ui, wait_ret=True, post_ret="hi", dead=False):
    monkeypatch.setattr("uvicorn.Config", _FakeCfg)
    monkeypatch.setattr("uvicorn.Server", _FakeSrv)
    monkeypatch.setattr("threading.Thread", _DeadThread if dead else _AliveThread)
    monkeypatch.setattr("deepseaport.server.create_app", lambda st: object())
    monkeypatch.setattr(chat_ui, "_wait_for_server", lambda *a, **k: wait_ret)
    if isinstance(post_ret, Exception):
        def _boom(*a, **k):
            raise post_ret
        monkeypatch.setattr(chat_ui, "_post_completion", _boom)
    elif callable(post_ret):
        monkeypatch.setattr(chat_ui, "_post_completion", post_ret)
    else:
        monkeypatch.setattr(chat_ui, "_post_completion", lambda *a, **k: post_ret)


# _first_api_key

def test_first_api_key_empty():
    from deepseaport import chat_ui
    from deepseaport.config import Settings
    assert chat_ui._first_api_key(Settings(keys=[], config_path="")) == ""


def test_first_api_key_unreadable():
    from deepseaport import chat_ui
    assert chat_ui._first_api_key(object()) == ""


# _model_list

def test_model_list_non_dict():
    from deepseaport import chat_ui
    assert chat_ui._model_list(None) == [chat_ui.DEFAULT_MODEL]
    assert chat_ui._model_list(["a"]) == [chat_ui.DEFAULT_MODEL]
    assert chat_ui._model_list("x") == [chat_ui.DEFAULT_MODEL]


def test_model_list_sorted():
    from deepseaport import chat_ui
    assert chat_ui._model_list({"z": 1, "a": 1, "m": 1}) == ["a", "m", "z"]


# _remember_chat_model

def test_remember_chat_model_saves_tmp(tmp_path):
    from deepseaport import chat_ui
    from deepseaport.config import load_settings
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({}), encoding="utf-8")
    s = load_settings(str(cfg))
    chat_ui._remember_chat_model(s, "deepseek-flash-search")
    assert s.chat_model == "deepseek-flash-search"
    assert load_settings(str(cfg)).chat_model == "deepseek-flash-search"


def test_remember_chat_model_fallback_on_save_error(monkeypatch):
    from deepseaport import chat_ui
    from deepseaport.config import Settings

    def _boom(self):
        raise OSError("disk fail")

    monkeypatch.setattr(Settings, "save", _boom)
    s = Settings(chat_model="a", config_path="")
    chat_ui._remember_chat_model(s, "b")
    assert s.chat_model == "b"


# _initial_chat_model

def test_initial_chat_model_case_insensitive():
    from deepseaport import chat_ui
    from deepseaport.config import Settings
    models = ["deepseek-flash", "deepseek-flash-search"]
    s = Settings(chat_model="DEEPSEEK-FLASH-SEARCH", config_path="")
    assert chat_ui._initial_chat_model(s, models) == "deepseek-flash-search"


# _select_usable_accounts_live

def test_select_usable_split(monkeypatch):
    from deepseaport import chat_ui
    from deepseaport.config import Settings
    a_ok = _mk_acc("ok@x.com", "tok-ok")
    a_ban = _mk_acc("ban@x.com", "tok-ban")
    a_empty = _mk_acc("empty@x.com", "")
    s = Settings(accounts=[a_ok, a_ban, a_empty], config_path="")
    monkeypatch.setattr(
        "deepseaport.accounts.collect_ban_labels",
        lambda accs, force_refresh=False, **k: {a_ban.identifier: 9999999999.0},
    )
    usable, banned, no_token = chat_ui._select_usable_accounts_live(s)
    assert [a.identifier for a in usable] == [a_ok.identifier]
    assert banned == 1
    assert no_token == 1


def test_select_usable_probe_exception_all_usable(monkeypatch):
    from deepseaport import chat_ui
    from deepseaport.config import Settings
    a1 = _mk_acc("a1@x.com", "tok-1")
    a2 = _mk_acc("a2@x.com", "tok-2")
    a3 = _mk_acc("a3@x.com", "")
    s = Settings(accounts=[a1, a2, a3], config_path="")

    def _boom(*a, **k):
        raise RuntimeError("net down")

    monkeypatch.setattr("deepseaport.accounts.collect_ban_labels", _boom)
    usable, banned, no_token = chat_ui._select_usable_accounts_live(s)
    assert sorted(a.identifier for a in usable) == sorted([a1.identifier, a2.identifier])
    assert banned == 0
    assert no_token == 1


def test_select_usable_empty():
    from deepseaport import chat_ui
    from deepseaport.config import Settings
    assert chat_ui._select_usable_accounts_live(Settings(accounts=[], config_path="")) == ([], 0, 0)


# _prompt_instance_count

def test_prompt_cancel_empty(monkeypatch):
    from deepseaport import chat_ui
    monkeypatch.setattr("builtins.input", lambda *a, **k: "")
    assert chat_ui._prompt_instance_count(3) is None


def test_prompt_eof_none(monkeypatch):
    from deepseaport import chat_ui

    def _eof(*a, **k):
        raise EOFError("eof")

    monkeypatch.setattr("builtins.input", _eof)
    assert chat_ui._prompt_instance_count(2) is None


def test_prompt_nan_retry_then_valid(monkeypatch):
    from deepseaport import chat_ui
    seq = iter(["nan", "2"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(seq))
    assert chat_ui._prompt_instance_count(3) == 2


def test_prompt_out_of_range_retry(monkeypatch):
    from deepseaport import chat_ui
    seq = iter(["99", "1"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(seq))
    assert chat_ui._prompt_instance_count(2) == 1


# _allocate_sequential_ports

def test_allocate_exhausted_short():
    from deepseaport import chat_ui
    assert chat_ui._allocate_sequential_ports("127.0.0.1", 5001, 2, is_free=lambda h, p: False) == []


def test_allocate_skips_busy():
    from deepseaport import chat_ui
    got = chat_ui._allocate_sequential_ports("127.0.0.1", 5001, 2, is_free=lambda h, p: p != 5001)
    assert got == [5002, 5003]


# _build_instance_settings

def test_build_instance_settings_pinned():
    from deepseaport import chat_ui
    from deepseaport.config import Settings
    a1 = _mk_acc("a1@x.com", "tok-1")
    a2 = _mk_acc("a2@x.com", "tok-2")
    a3 = _mk_acc("a3@x.com", "tok-3")
    s = Settings(accounts=[a1, a2, a3], active_account="old",
                 config_path="/tmp/real.json", log_level="INFO", port=5001)
    inst = chat_ui._build_instance_settings(s, a1, [a1, a2], 5009)
    assert inst.active_account == a1.identifier
    assert inst.config_path == ""
    assert inst.log_level == "ERROR"
    assert inst.port == 5009
    assert inst.accounts == [a1, a2]
    # original not mutated
    assert s.port == 5001
    assert s.config_path == "/tmp/real.json"
    assert s.log_level == "INFO"
    assert s.active_account == "old"
    assert s.accounts == [a1, a2, a3]


# _stop_all_servers

def test_stop_all_servers_joins_swallows():
    from deepseaport import chat_ui

    class BoomSrv:
        def __setattr__(self, name, val):
            if name == "should_exit":
                raise RuntimeError("boom")
            super().__setattr__(name, val)

    class GoodSrv:
        def __init__(self):
            self.should_exit = False

    class GoodThread:
        def __init__(self):
            self.joined = False

        def join(self, timeout=None):
            self.joined = True

    class BoomThread:
        def join(self, timeout=None):
            raise RuntimeError("join boom")

    good, gt = GoodSrv(), GoodThread()
    chat_ui._stop_all_servers([good, BoomSrv()], [gt, BoomThread()], timeout=0.01)
    assert good.should_exit is True
    assert gt.joined is True


# run_multi_chat_servers blocked paths

def test_run_multi_no_accounts():
    from deepseaport import chat_ui
    from deepseaport.config import Settings
    assert chat_ui.run_multi_chat_servers(Settings(accounts=[], config_path="")) == 1


def test_run_multi_tokenless():
    from deepseaport import chat_ui
    from deepseaport.config import Settings
    s = Settings(accounts=[_mk_acc("a@x.com", "")], config_path="")
    assert chat_ui.run_multi_chat_servers(s) == 1


def test_run_multi_all_banned(monkeypatch):
    from deepseaport import chat_ui
    from deepseaport.config import Settings
    s = Settings(accounts=[_mk_acc("a@x.com", "tok-1")], config_path="")
    monkeypatch.setattr(chat_ui, "_select_usable_accounts_live", lambda st: ([], 1, 0))
    assert chat_ui.run_multi_chat_servers(s, count=1) == 1


def test_run_multi_prompt_cancel(monkeypatch):
    from deepseaport import chat_ui
    from deepseaport.config import Settings
    a1, a2 = _mk_acc("a1@x.com", "t1"), _mk_acc("a2@x.com", "t2")
    s = Settings(accounts=[a1, a2], config_path="")
    monkeypatch.setattr(chat_ui, "_select_usable_accounts_live", lambda st: ([a1, a2], 0, 0))
    monkeypatch.setattr(chat_ui, "_prompt_instance_count", lambda cap: None)
    assert chat_ui.run_multi_chat_servers(s) == 0


def test_run_multi_bad_count_string(monkeypatch):
    from deepseaport import chat_ui
    from deepseaport.config import Settings
    a1 = _mk_acc("a1@x.com", "t1")
    s = Settings(accounts=[a1], config_path="")
    monkeypatch.setattr(chat_ui, "_select_usable_accounts_live", lambda st: ([a1], 0, 0))
    assert chat_ui.run_multi_chat_servers(s, count="bad") == 1


def test_run_multi_out_of_range_count(monkeypatch):
    from deepseaport import chat_ui
    from deepseaport.config import Settings
    a1 = _mk_acc("a1@x.com", "t1")
    s = Settings(accounts=[a1], config_path="")
    monkeypatch.setattr(chat_ui, "_select_usable_accounts_live", lambda st: ([a1], 0, 0))
    assert chat_ui.run_multi_chat_servers(s, count=99) == 1


def test_run_multi_ports_exhausted(monkeypatch):
    from deepseaport import chat_ui
    from deepseaport.config import Settings
    a1, a2 = _mk_acc("a1@x.com", "t1"), _mk_acc("a2@x.com", "t2")
    s = Settings(accounts=[a1, a2], port=5011, config_path="")
    monkeypatch.setattr(chat_ui, "_select_usable_accounts_live", lambda st: ([a1, a2], 0, 0))
    monkeypatch.setattr(chat_ui, "_allocate_sequential_ports", lambda h, b, n, is_free=None: [5011])
    assert chat_ui.run_multi_chat_servers(s, count=2) == 1


def test_run_multi_repl_roundrobin_commands(monkeypatch, capsys):
    from deepseaport import chat_ui
    from deepseaport.config import Settings
    a1 = _mk_acc("a1@x.com", "tok-a1")
    a2 = _mk_acc("a2@x.com", "tok-a2")
    s = Settings(accounts=[a1, a2], keys=["k"], port=5011, config_path="")
    monkeypatch.setattr(chat_ui, "_select_usable_accounts_live", lambda st: ([a1, a2], 0, 0))
    monkeypatch.setattr(chat_ui, "_allocate_sequential_ports",
                        lambda h, b, n, is_free=None: [5011, 5012][:n])
    created = []

    class FakeSrv(_FakeSrv):
        def __init__(self, cfg=None):
            super().__init__(cfg)
            created.append(self)

    monkeypatch.setattr("uvicorn.Config", _FakeCfg)
    monkeypatch.setattr("uvicorn.Server", FakeSrv)
    monkeypatch.setattr("threading.Thread", _AliveThread)
    monkeypatch.setattr("deepseaport.server.create_app", lambda st: object())
    monkeypatch.setattr(chat_ui, "_wait_for_server", lambda *a, **k: True)
    calls = []

    def _post(base, key, model, hist):
        calls.append((base, list(hist)))
        return "reply-" + base[-4:]

    monkeypatch.setattr(chat_ui, "_post_completion", _post)
    seq = iter(["/model nope-xyz", "/model", "/servers", "/clear", "hi1", "hi2", "/quit"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(seq))
    assert chat_ui.run_multi_chat_servers(s, count=2) == 0
    assert len(calls) == 2
    assert calls[0][0].endswith("5011")
    assert calls[1][0].endswith("5012")
    out = capsys.readouterr().out
    assert "unknown model" in out
    assert "history cleared" in out
    assert "5011" in out
    assert all(getattr(x, "should_exit", False) for x in created)


# run_chat_server

def test_run_chat_workers_note_and_ensure_called(monkeypatch, capsys):
    from deepseaport import chat_ui
    from deepseaport.config import Settings
    s = Settings(accounts=[_mk_acc("a@x.com", "t1")], port=5001, config_path="", log_level="INFO")
    called = {}
    monkeypatch.setattr("deepseaport.cli._ensure_free_port",
                        lambda st, host, port: called.setdefault("port", port) or 5099)
    _install_single_fakes(monkeypatch, chat_ui, wait_ret=True, post_ret="hi")
    monkeypatch.setattr("builtins.input", lambda *a, **k: "/quit")
    rc = chat_ui.run_chat_server(s, types.SimpleNamespace(workers=2))
    assert rc == 0
    assert called.get("port") == 5001
    assert "1 worker" in capsys.readouterr().out
    assert s.log_level == "INFO"


def test_run_chat_server_fail_dead(monkeypatch):
    from deepseaport import chat_ui
    from deepseaport.config import Settings
    s = Settings(port=5001, config_path="", log_level="INFO")
    monkeypatch.setattr("deepseaport.cli._ensure_free_port", lambda st, h, p: 5098)
    _install_single_fakes(monkeypatch, chat_ui, wait_ret=False, dead=True)
    monkeypatch.setattr("builtins.input", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no input")))
    assert chat_ui.run_chat_server(s) == 1
    assert s.log_level == "INFO"


def test_run_chat_slow_alive_continues(monkeypatch, capsys):
    from deepseaport import chat_ui
    from deepseaport.config import Settings
    s = Settings(port=5001, config_path="", log_level="INFO")
    monkeypatch.setattr("deepseaport.cli._ensure_free_port", lambda st, h, p: 5098)
    _install_single_fakes(monkeypatch, chat_ui, wait_ret=False, post_ret="hi", dead=False)
    monkeypatch.setattr("builtins.input", lambda *a, **k: "/quit")
    assert chat_ui.run_chat_server(s) == 0
    assert "slow to respond" in capsys.readouterr().out
    assert s.log_level == "INFO"


def test_run_chat_repl_unknown_history_pop_quit_logs(monkeypatch, capsys):
    import threading  # noqa: F401  (ensures threading patch target exists)
    from deepseaport import chat_ui
    from deepseaport.config import Settings
    s = Settings(port=5001, config_path="", log_level="INFO")
    lg = logging.getLogger("deepseaport.obscura")
    root = logging.getLogger()
    old_lg, old_root = lg.level, root.level
    lg.setLevel(logging.INFO)
    root.setLevel(logging.INFO)
    try:
        monkeypatch.setattr("deepseaport.cli._ensure_free_port", lambda st, h, p: 5098)
        monkeypatch.setattr("uvicorn.Config", _FakeCfg)
        monkeypatch.setattr("uvicorn.Server", _FakeSrv)
        monkeypatch.setattr("threading.Thread", _AliveThread)
        monkeypatch.setattr("deepseaport.server.create_app", lambda st: object())
        monkeypatch.setattr(chat_ui, "_wait_for_server", lambda *a, **k: True)
        hists = []
        n = {"c": 0}

        def _post(base, key, model, hist):
            hists.append([dict(m) for m in hist])
            n["c"] += 1
            if n["c"] == 1:
                raise RuntimeError("boom-fail")
            return "ok-reply"

        monkeypatch.setattr(chat_ui, "_post_completion", _post)
        seq = iter(["/model nope-xyz", "hello-fail", "hello-ok", "/quit"])
        monkeypatch.setattr("builtins.input", lambda *a, **k: next(seq))
        assert chat_ui.run_chat_server(s) == 0
        assert n["c"] == 2
        assert hists[0] == [{"role": "user", "content": "hello-fail"}]
        assert hists[1] == [{"role": "user", "content": "hello-ok"}]
        out = capsys.readouterr().out
        assert "unknown model" in out
        assert "boom-fail" in out
        assert s.log_level == "INFO"
        assert logging.getLogger("deepseaport.obscura").level == logging.INFO
        assert logging.getLogger().level == logging.INFO
    finally:
        lg.setLevel(old_lg)
        root.setLevel(old_root)
