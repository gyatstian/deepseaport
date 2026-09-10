"""Obscura bridge: WAF-cleared cookies + browser UA for the hot path.

chat.deepseek.com sits behind an AWS WAF JavaScript challenge. Obscura
(h4ckf0r0day/obscura) executes the challenge in its V8 engine and persists
cookies in a profile dir. This module:

1. warms the profile (two sequential fetches: challenge -> real app),
2. dumps the cookie jar (`--dump cookies`, includes HttpOnly),
3. captures the engine's User-Agent for header consistency,
4. exposes the jar as a `Cookie` header for direct `curl_cffi` calls.

Direct HTTP (not DOM driving) stays the hot path; Obscura is only
re-invoked to refresh the WAF token (startup + on 403).
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("deepseaport.obscura")

CHAT_HOME = "https://chat.deepseek.com/"
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
# No absolute/host-specific paths here: the repo is published to GitHub and
# everyone checks out to a different folder. Probes repo-local dirs for a
# dropped-in binary (e.g. extracted release folder
# `obscura-x86_64-windows-no-render/obscura.exe` next to config.json).
# Glob patterns support versioned release dirs.
OBSCURA_LOCAL_PATTERNS = (
    "obscura.exe",
    "obscura",
    "bin/obscura.exe",
    "bin/obscura",
    "tools/obscura.exe",
    "tools/obscura",
    "vendor/obscura.exe",
    "vendor/obscura",
    "obscura-*/obscura.exe",
    "obscura-*/obscura",
)

log = logger


@dataclass
class WafState:
    cookies: dict[str, str] = field(default_factory=dict)
    user_agent: str = ""
    warmed: bool = False


class ObscuraBridge:
    def __init__(self, binary: str = "", profile: str = "") -> None:
        self.binary = binary or self._discover()
        default_profile = str(Path.home() / ".deepseaport" / "obscura-profile")
        self.profile = profile or default_profile
        Path(self.profile).mkdir(parents=True, exist_ok=True)
        self.state = WafState()
        # All obscura subprocesses share one --storage-dir. Parallel warmups
        # (multi-account 403 refreshes) would race on it and corrupt the
        # profile/cookie jar, so cold refreshes serialize. The hot path
        # (cookie_header/has_waf_token) never takes this lock. RLock because
        # warmup() calls dump_cookies()/fetch_ua() which also lock.
        self._proc_lock = threading.RLock()

    @staticmethod
    def _search_roots() -> list[Path]:
        """Dirs to probe for a repo-local binary: cwd first, then repo root."""
        roots: list[Path] = []
        for raw in (Path.cwd(), REPO_ROOT):
            try:
                resolved = raw.resolve()
            except Exception:
                continue
            if resolved not in roots:
                roots.append(resolved)
        return roots

    @staticmethod
    def _discover() -> str:
        found = shutil.which("obscura")
        if found:
            return found
        for base in ObscuraBridge._search_roots():
            for pat in OBSCURA_LOCAL_PATTERNS:
                try:
                    hits = sorted(base.glob(pat))
                except Exception:
                    continue
                for hit in hits:
                    if hit.is_file():
                        return str(hit)
        return "obscura"  # hope for PATH at call time

    def _run(self, *args: str, timeout: int = 60) -> subprocess.CompletedProcess[str]:
        cmd = [self.binary, "--storage-dir", self.profile, *args]
        log.debug("obscura: %s", " ".join(cmd[1:]))
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

    def version(self) -> str:
        try:
            proc = subprocess.run([self.binary, "--version"], capture_output=True, text=True, timeout=15)
            return (proc.stdout or proc.stderr).strip()
        except Exception as exc:
            return f"unavailable: {exc}"

    def _parse_cookie_dump(self, payload: str) -> dict[str, str]:
        """Extract {name: value} from obscura --dump cookies stdout."""
        payload = payload or ""
        start = payload.find("[")
        try:
            items = json.loads(payload[start:] if start >= 0 else payload or "[]")
            return {c["name"]: c["value"] for c in items if c.get("name") and c.get("value")}
        except Exception:
            return {}

    def warmup(self, timeout: int = 45) -> dict[str, str]:
        """Solve/refresh the WAF challenge; return cookie dict (may be empty)."""
        with self._proc_lock:
            # Pass 1: execute challenge.js, persist cookies to profile. Its stdout
            # already carries the cookie dump, so parsing it avoids a second full
            # browser spawn (pass1 -> dump_cookies would each launch the engine).
            cookies: dict[str, str] = {}
            try:
                proc = self._run("fetch", CHAT_HOME, "--dump", "cookies", "--timeout", "30",
                                 "--wait", "6", "--quiet", timeout=timeout)
                cookies = self._parse_cookie_dump(proc.stdout)
            except Exception as exc:
                log.warning("obscura warmup pass1 failed: %s", exc)
            # Pass 2 (fallback): read back the jar (proves persistence) + capture UA.
            # UA rarely changes: reuse cached value to save one browser spawn.
            if not cookies:
                cookies = self.dump_cookies()
            if not cookies and self.state.cookies:
                log.warning("obscura dump empty, keeping %d cached cookies", len(self.state.cookies))
                cookies = dict(self.state.cookies)
            ua = self.state.user_agent or self.fetch_ua()
            self.state = WafState(cookies=cookies, user_agent=ua, warmed=bool(cookies))
            log.info("obscura warmup: %d cookies, ua=%s", len(cookies), ua[:60])
            return cookies

    def dump_cookies(self, timeout: int = 30) -> dict[str, str]:
        with self._proc_lock:
            try:
                proc = self._run("fetch", CHAT_HOME, "--dump", "cookies",
                                 "--timeout", "20", "--quiet", timeout=timeout)
                return self._parse_cookie_dump(proc.stdout)
            except Exception as exc:
                log.warning("obscura dump_cookies failed: %s", exc)
                return {}

    def fetch_ua(self, timeout: int = 30) -> str:
        with self._proc_lock:
            try:
                proc = self._run("fetch", "https://example.com", "--eval", "navigator.userAgent",
                                 "--timeout", "20", "--quiet", timeout=timeout)
                return proc.stdout.strip().strip('"')[:300]
            except Exception:
                return ""

    def cookie_header(self) -> str:
        return "; ".join(f"{k}={v}" for k, v in self.state.cookies.items())

    def has_waf_token(self) -> bool:
        return "aws-waf-token" in self.state.cookies

    def status(self) -> dict:
        return {
            "binary": self.binary,
            "version": self.version(),
            "profile": self.profile,
            "warmed": self.state.warmed,
            "has_waf_token": self.has_waf_token(),
            "cookie_names": sorted(self.state.cookies),
            "user_agent": self.state.user_agent,
        }
