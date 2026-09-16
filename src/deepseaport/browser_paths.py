"""Locate a real Chromium-family browser for login automation.

Obscura's no-render engine cannot initialize Shumei's `fp.min.js` SDK, so the
sign-in page posts ``device_id: null`` and DeepSeek rejects the login.  A real
Chromium/Helium/Chrome/Edge browser can generate the fingerprint normally.
This module only discovers a binary; the CDP driver lives in real_browser.py.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def _candidates() -> list[str]:
    out: list[str] = []
    home = Path.home()
    local = Path(os.environ.get("LOCALAPPDATA", ""))
    pf = Path(os.environ.get("PROGRAMFILES", ""))
    pf86 = Path(os.environ.get("PROGRAMFILES(X86)", ""))
    # Helium (imput build) is what many DeepSeek users run as their main browser.
    out += [
        str(local / "imput" / "Helium" / "Application" / "chrome.exe"),
        str(local / "Helium" / "Application" / "chrome.exe"),
    ]
    # Google Chrome.
    out += [
        str(local / "Google" / "Chrome" / "Application" / "chrome.exe"),
        str(pf / "Google" / "Chrome" / "Application" / "chrome.exe"),
        str(pf86 / "Google" / "Chrome" / "Application" / "chrome.exe"),
    ]
    # Microsoft Edge.
    out += [
        str(local / "Microsoft" / "Edge" / "Application" / "msedge.exe"),
        str(pf / "Microsoft" / "Edge" / "Application" / "msedge.exe"),
        str(pf86 / "Microsoft" / "Edge" / "Application" / "msedge.exe"),
    ]
    # Chromium / Brave.
    out += [
        str(local / "Chromium" / "Application" / "chrome.exe"),
        str(local / "BraveSoftware" / "Brave-Browser" / "Application" / "brave.exe"),
        str(home / ".local" / "share" / "helium" / "chrome"),
        "/usr/bin/google-chrome",
        "/usr/bin/chromium",
        "/usr/bin/chromium-browser",
        "/usr/bin/microsoft-edge",
    ]
    return [p for p in out if p]


def discover_browser() -> str:
    """Return a usable Chromium-family browser path, or an empty string."""
    for command in ("helium", "chrome", "google-chrome", "chromium",
                    "chromium-browser", "msedge", "brave"):
        found = shutil.which(command)
        if found:
            return found
    for raw in _candidates():
        try:
            if raw and Path(raw).is_file():
                return raw
        except Exception:
            continue
    return ""


def resolve_browser(value: str) -> str:
    """Resolve a configured browser path / ``auto`` / empty value."""
    value = (value or "").strip()
    if not value:
        return ""
    if value.lower() in ("auto", "default"):
        return discover_browser()
    try:
        if Path(value).is_file():
            return value
    except Exception:
        pass
    return shutil.which(value) or ""
