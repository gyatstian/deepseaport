"""DeepSeekHashV1 proof-of-work solver (wasmtime + shipped wasm)."""

from __future__ import annotations

import base64
import ctypes
import json
import logging
import os
import struct
import threading
from pathlib import Path

from .config import DATA_DIR

logger = logging.getLogger("deepseaport.pow")

# Env overrides for rotation survival without code change. Defaults preserve
# current pinned behaviour; setting DEEPSEAPORT_WASM_URL lets ops point at a
# rotated hash without waiting for a release.
WASM_NAME = os.environ.get(
    "DEEPSEAPORT_WASM_NAME", "sha3_wasm_bg.7b9ca65ddd.wasm")
WASM_URL = os.environ.get(
    "DEEPSEAPORT_WASM_URL",
    "https://fe-static.deepseek.com/chat/static/" + WASM_NAME)
ALGORITHM = "DeepSeekHashV1"
# WebAssembly modules start with the magic bytes b"\0asm" (then version 1).
_WASM_MAGIC = b"\x00asm"

# Cached compiled wasm: Engine + Module are thread-safe/shareable, while
# Store/instance stay per-call. Recompiling per request cost 100-300ms.
_ENGINE = None
_MODULE = None
_MODULE_KEY: tuple | None = None
_MODULE_LOCK = threading.Lock()
_LINKER = None
_LINKER_LOCK = threading.Lock()
_WASM_PATH: Path | None = None

# DATA_DIR is resolved once at import from the environment; it never changes
# for a process. Cache the joined path to avoid a mkdir syscall per request.
def wasm_path() -> Path:
    global _WASM_PATH
    if _WASM_PATH is None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        _WASM_PATH = DATA_DIR / WASM_NAME
    return _WASM_PATH


def ensure_wasm(cookies: str = "", user_agent: str = "") -> Path:
    """Return local wasm path, downloading it (WAF cookies help) if missing."""
    path = wasm_path()
    if path.exists() and path.stat().st_size > 1000:
        return path
    from curl_cffi import requests as crequests

    headers = {"User-Agent": user_agent or "Mozilla/5.0", "Referer": "https://chat.deepseek.com/"}
    if cookies:
        headers["Cookie"] = cookies
    resp = crequests.get(WASM_URL, headers=headers, impersonate="chrome", timeout=60)
    resp.raise_for_status()
    content = resp.content or b""
    # Wrong host / WAF interstitial returns index.html with a 200. Reject
    # anything that is not a real wasm module before it reaches disk, so a
    # bad download fails here instead of as a confusing Module parse later.
    if len(content) <= 1000:
        raise RuntimeError(f"PoW wasm download too small ({len(content)} bytes)")
    if not content.startswith(_WASM_MAGIC):
        raise RuntimeError(
            f"PoW wasm download is not a wasm module "
            f"(magic={content[:4]!r}, {len(content)} bytes)")
    path.write_bytes(content)
    logger.info("downloaded PoW wasm (%d bytes)", len(content))
    return path


def _get_engine_module():
    """Return shared (engine, module), compiling once per wasm file version."""
    global _ENGINE, _MODULE, _MODULE_KEY
    path = wasm_path()
    try:
        stat = path.stat()
    except FileNotFoundError:
        raise
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if _ENGINE is not None and _MODULE is not None and _MODULE_KEY == key:
        return _ENGINE, _MODULE
    with _MODULE_LOCK:
        if _ENGINE is not None and _MODULE is not None and _MODULE_KEY == key:
            return _ENGINE, _MODULE
        from wasmtime import Engine, Module

        wasm_bytes = path.read_bytes()
        engine = _ENGINE if _ENGINE is not None else Engine()
        module = Module(engine, wasm_bytes)
        _ENGINE, _MODULE, _MODULE_KEY = engine, module, key
        return engine, module


def prewarm() -> bool:
    """Compile Engine/Module now so first request skips 100-300ms compile.

    Best-effort: missing wasm or wasmtime errors return False, never raise.
    Store/instance stay per-solve (Store is not thread-safe to share).
    """
    try:
        _get_engine_module()
        return True
    except Exception as exc:
        logger.debug("PoW prewarm skipped: %s", exc)
        return False


def solve(algorithm: str, challenge: str, salt: str, difficulty: int | float, expire_at: int) -> int | None:
    """Search the answer int via the site's own wasm. Returns None on failure."""
    if algorithm != ALGORITHM:
        raise ValueError(f"unsupported PoW algorithm: {algorithm}")
    from wasmtime import Linker, Store

    prefix = f"{salt}_{expire_at}_"
    engine, module = _get_engine_module()
    store = Store(engine)
    # Linker is thread-safe and bound to the (cached) engine: share it instead
    # of rebuilding per solve. Store stays per-call (not thread-safe to share).
    global _LINKER
    if _LINKER is None:
        with _LINKER_LOCK:
            if _LINKER is None:
                _LINKER = Linker(engine)
    linker = _LINKER
    instance = linker.instantiate(store, module)
    exports = instance.exports(store)
    memory = exports["memory"]
    add_to_stack = exports["__wbindgen_add_to_stack_pointer"]
    alloc = exports["__wbindgen_export_0"]
    wasm_solve = exports["wasm_solve"]

    base = ctypes.cast(memory.data_ptr(store), ctypes.c_void_p).value

    def write(offset: int, data: bytes) -> None:
        ctypes.memmove(base + offset, data, len(data))

    def read(offset: int, size: int) -> bytes:
        return ctypes.string_at(base + offset, size)

    def encode(text: str) -> tuple[int, int]:
        data = text.encode("utf-8")
        ptr = alloc(store, len(data), 1)
        ptr = int(ptr.value) if hasattr(ptr, "value") else int(ptr)
        write(ptr, data)
        return ptr, len(data)

    retptr = add_to_stack(store, -16)
    ptr_c, len_c = encode(challenge)
    ptr_p, len_p = encode(prefix)
    try:
        wasm_solve(store, retptr, ptr_c, len_c, ptr_p, len_p, float(difficulty))
        status = struct.unpack("<i", read(retptr, 4))[0]
        value = struct.unpack("<d", read(retptr + 8, 8))[0]
    finally:
        add_to_stack(store, 16)
    if status == 0:
        return None
    return int(value)


def build_header(challenge: dict) -> str:
    """Encode a solved challenge as the X-DS-PoW-Response header value."""
    payload = {
        "algorithm": challenge["algorithm"],
        "challenge": challenge["challenge"],
        "salt": challenge["salt"],
        "answer": challenge["answer"],
        "signature": challenge["signature"],
        "target_path": challenge["target_path"],
    }
    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return base64.b64encode(raw).decode("utf-8")
