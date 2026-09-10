"""Speed/responsiveness test: deepseek-flash-reasoner round trip.

Default tier is LIVE: a real request through the running pipeline
(Obscura WAF -> session -> PoW -> SSE completion), measuring wall time,
time-to-first-byte, time-to-first-reasoning-token, time-to-first-content
and per-phase breakdown. Needs an account token (config.json or env) and
the obscura binary on PATH — skips with a clear reason otherwise.

A offline stubbed tier (no network) runs too, so the phase-ordering
assertions still hold on machines without credentials.

Run:
    pytest tests/test_speed.py -s        # live + offline
    pytest tests/test_speed.py -s -k offline   # no network
"""

from __future__ import annotations

import inspect
import time

import pytest

from deepseaport import client as DS
from deepseaport import pow as PoW
from deepseaport.config import AccountConfig, Settings, load_settings
from deepseaport.server import create_app

REASONER = "deepseek-flash-reasoner"
QUESTION = "Solve 6*7 step by step, keep it very short."


def _body(stream: bool = False) -> dict:
    return {
        "model": REASONER,
        "stream": stream,
        "messages": [{"role": "user", "content": QUESTION}],
    }


class _Timed:
    """Records (label, ms) per instrumented call; prints a report."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, float]] = []

    def record(self, label: str, t0: float) -> None:
        self.rows.append((label, (time.perf_counter() - t0) * 1000))

    def install(self, monkeypatch, fakes: dict | None = None) -> None:
        """Wrap DS/PoW entry points with timing. fakes replaces the impl."""
        fakes = fakes or {}
        for label, owner, name in (
            ("create_session", DS, "create_session"),
            ("delete_session", DS, "delete_session"),
            ("fetch_pow", DS, "fetch_pow"),
            ("solve_pow", PoW, "solve"),
            ("stream_completion", DS, "stream_completion"),
        ):
            fn = fakes.get(name, getattr(owner, name))
            monkeypatch.setattr(owner, name, self._timed(label, fn))

    def _timed(self, label: str, fn):
        if inspect.isgeneratorfunction(fn):
            def wrap_gen(*args, **kwargs):
                t0 = time.perf_counter()
                gen = fn(*args, **kwargs)
                first = next(gen)
                self.record(f"{label} (ttfb)", t0)
                yield first
                try:
                    for item in gen:
                        yield item
                finally:
                    self.record(label, t0)
            return wrap_gen

        def wrap_call(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                self.record(label, t0)
        return wrap_call

    def report(self, title: str, extra: list[tuple[str, float]] = ()) -> None:
        print(f"\n--- {title} ---")
        for label, ms in [*self.rows, *extra]:
            print(f"  {label:<28s} {ms:8.1f} ms")
        print()


# ---------------------------------------------------------------- live


def _live_ready() -> tuple[bool, str]:
    try:
        settings = load_settings()
    except Exception as exc:  # pragma: no cover
        return False, f"settings unreadable: {exc}"
    if not settings.accounts:
        return False, "no accounts in config"
    if not any(a.token for a in settings.accounts):
        return False, "no account token (run: deepseaport login)"
    from deepseaport.obscura_bridge import ObscuraBridge
    bridge = ObscuraBridge(settings.obscura_bin, settings.obscura_profile)
    if shutil_which(bridge.binary) is None and not _exists(bridge.binary):
        return False, f"obscura binary not found: {bridge.binary}"
    return True, ""


def _exists(path: str) -> bool:
    import os
    return bool(path) and os.path.isfile(path)


def shutil_which(cmd: str):
    import shutil
    return shutil.which(cmd) if cmd else None


def test_live_reasoner_latency(monkeypatch):
    """Real reasoner round trip: wall, TTFB, first-reasoning, per-phase."""
    from fastapi.testclient import TestClient

    ok, why = _live_ready()
    if not ok:
        pytest.skip(why)

    settings = load_settings()
    from deepseaport.obscura_bridge import ObscuraBridge

    t_warm0 = time.perf_counter()
    bridge = ObscuraBridge(settings.obscura_bin, settings.obscura_profile)
    bridge.warmup()
    warm_ms = (time.perf_counter() - t_warm0) * 1000
    if not bridge.has_waf_token():
        pytest.skip("no aws-waf-token after warmup (network/WAF issue?)")

    app = create_app(settings)
    app.state.bridge = bridge

    timed = _Timed()
    timed.install(monkeypatch)  # time the REAL client calls

    key = settings.keys[0] if settings.keys else ""
    headers = {"Authorization": f"Bearer {key}"} if key else {}

    t0 = time.perf_counter()
    first_line = first_reason = first_content = None
    with TestClient(app).stream(
            "POST", "/v1/chat/completions", json=_body(stream=True),
            headers=headers) as resp:
        assert resp.status_code == 200, resp.read().decode(errors="replace")
        for ln in resp.iter_lines():
            now = (time.perf_counter() - t0) * 1000
            if first_line is None and ln.strip():
                first_line = now
            if '"reasoning_content"' in ln and first_reason is None:
                first_reason = now
            if '"content"' in ln and '"reasoning_content"' not in ln and first_content is None:
                first_content = now
            if ln.strip() == "data: [DONE]":
                break
    wall_ms = (time.perf_counter() - t0) * 1000

    assert first_line is not None, "no SSE data at all"
    assert first_reason is not None, "reasoner produced no reasoning_content"
    assert first_content is not None, "reasoner produced no content"
    timed.report(
        f"LIVE reasoner (stream) model={REASONER}",
        [
            ("waf warmup", warm_ms),
            ("ttfb (first byte)", first_line),
            ("first reasoning token", first_reason),
            ("first content token", first_content),
            ("wall (request done)", wall_ms),
        ])


def test_live_reasoner_nonstream_wall(monkeypatch):
    """Real reasoner, buffered: one number — full request wall time."""
    from fastapi.testclient import TestClient

    ok, why = _live_ready()
    if not ok:
        pytest.skip(why)

    settings = load_settings()
    from deepseaport.obscura_bridge import ObscuraBridge
    bridge = ObscuraBridge(settings.obscura_bin, settings.obscura_profile)
    bridge.warmup()
    if not bridge.has_waf_token():
        pytest.skip("no aws-waf-token after warmup")

    app = create_app(settings)
    app.state.bridge = bridge
    timed = _Timed()
    timed.install(monkeypatch)

    key = settings.keys[0] if settings.keys else ""
    headers = {"Authorization": f"Bearer {key}"} if key else {}

    t0 = time.perf_counter()
    resp = TestClient(app).post(
        "/v1/chat/completions", json=_body(stream=False), headers=headers)
    wall_ms = (time.perf_counter() - t0) * 1000

    assert resp.status_code == 200, resp.text
    msg = resp.json()["choices"][0]["message"]
    assert msg.get("content"), "empty reply"
    assert msg.get("reasoning_content"), "reasoner returned no reasoning_content"
    timed.report(f"LIVE reasoner (non-stream) model={REASONER}",
                 [("wall (end to end)", wall_ms)])


# ------------------------------------------------------------- offline


def _offline_settings(stream_mode: str = "buffered") -> Settings:
    return Settings(
        keys=["sk-bench"],
        accounts=[AccountConfig(email="bench@x.com", token="bench-token")],
        config_path="",
        warmup_on_startup=False,
        max_retries=0,
        log_level="WARNING",
        stream_mode=stream_mode,
    )


class _FakeBridge:
    def __init__(self) -> None:
        self.state = _FakeState()

    def cookie_header(self) -> str:
        return ""

    def has_waf_token(self) -> bool:
        return True

    def warmup(self) -> dict:
        return {}


class _FakeState:
    user_agent = "bench-ua"
    cookies = {}


def _make_fakes(phases: dict[str, float]) -> dict:
    delay = phases.get

    def fake_create_session(headers):
        time.sleep(delay("create_session", 0.03))
        return "bench-session"

    def fake_delete_session(headers, session_id):
        pass

    def fake_fetch_pow(headers):
        time.sleep(delay("fetch_pow", 0.02))
        return {
            "algorithm": PoW.ALGORITHM,
            "challenge": "bench-challenge",
            "salt": "bench-salt",
            "difficulty": 1000,
            "expire_at": 2**31,
            "signature": "bench-sig",
            "target_path": "/api/v0/chat/completion",
        }

    def fake_solve(algorithm, challenge, salt, difficulty, expire_at):
        time.sleep(delay("solve_pow", 0.05))
        return 42

    def fake_stream_completion(headers, payload, timeout=120):
        time.sleep(delay("ttfb", 0.20))
        yield StreamEvent(kind="thinking", text="Step 1: six groups of seven.")
        time.sleep(delay("think_gap", 0.05))
        yield StreamEvent(kind="content", text="42.")
        yield StreamEvent(kind="usage", text="47")
        yield StreamEvent(kind="finished")

    return {
        "create_session": fake_create_session,
        "delete_session": fake_delete_session,
        "fetch_pow": fake_fetch_pow,
        "solve": fake_solve,
        "stream_completion": fake_stream_completion,
    }


# imported late so module import never fails without fastapi/httpx
from deepseaport.protocol import StreamEvent  # noqa: E402


def test_offline_reasoner_phase_timing(monkeypatch):
    """Stubbed full call: phases must run in order (session|pow parallel)."""
    from fastapi.testclient import TestClient

    app = create_app(_offline_settings())
    app.state.bridge = _FakeBridge()
    timed = _Timed()
    timed.install(monkeypatch, _make_fakes({"ttfb": 0.20}))

    t0 = time.perf_counter()
    resp = TestClient(app).post(
        "/v1/chat/completions", json=_body(),
        headers={"Authorization": "Bearer sk-bench"})
    wall_ms = (time.perf_counter() - t0) * 1000

    assert resp.status_code == 200, resp.text
    msg = resp.json()["choices"][0]["message"]
    assert msg["content"] == "42."
    assert "Step 1" in msg["reasoning_content"]
    assert resp.json()["usage"]["total_tokens"] > 0

    labels = [lbl for lbl, _ in timed.rows]
    expected = ["create_session", "fetch_pow", "solve_pow", "stream_completion"]
    assert [l for l in expected if l in labels] == expected
    idx = {lbl: i for i, lbl in enumerate(labels)}
    assert idx["solve_pow"] > max(idx["create_session"], idx["fetch_pow"])
    assert wall_ms > timed.rows[-1][1]
    timed.report("offline non-stream reasoner (stubbed)", [("wall", wall_ms)])


def test_offline_reasoner_streaming_ttfb(monkeypatch):
    """Stubbed stream=true: role chunk first, then reasoning, then content."""
    from fastapi.testclient import TestClient

    app = create_app(_offline_settings(stream_mode="live"))
    app.state.bridge = _FakeBridge()
    timed = _Timed()
    timed.install(monkeypatch, _make_fakes({"ttfb": 0.10}))

    t0 = time.perf_counter()
    resp = TestClient(app).post(
        "/v1/chat/completions", json=_body(stream=True),
        headers={"Authorization": "Bearer sk-bench"})
    wall_ms = (time.perf_counter() - t0) * 1000

    assert resp.status_code == 200, resp.text
    lines = [ln for ln in resp.text.splitlines() if ln.startswith("data: ")]
    assert lines and '{"role": "assistant"}' in lines[0]
    reasoning_seen = content_seen = done = False
    for ln in lines:
        if '"reasoning_content"' in ln:
            reasoning_seen = True
        if '"content": "42' in ln:
            content_seen = True
        if '"finish_reason": "stop"' in ln or 'data: [DONE]' in ln:
            done = True
    assert reasoning_seen, "reasoning chunk missing"
    assert content_seen, "content chunk missing"
    assert done, "stream not finished"
    timed.report("offline streaming reasoner TTFB (stubbed)", [("wall", wall_ms)])
