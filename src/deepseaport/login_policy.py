"""Shared DeepSeek login page policy for both browser drivers.

Single source of truth for page semantics used by the Obscura MCP driver
(``auth.BrowserLogin``) and the CDP driver (``real_browser.RealBrowser``).
Transport stays driver-specific (MCP tools vs CDP websocket); only the page
policy lives here: selectors, phrase lists, visibility/snapshot JS,
poll/timeout constants, text collapse, and classify/verdict helpers.

Phrase union (deliberately more sensitive, no narrowing):
- captcha: auth.py list verbatim (real_browser had no phrase list, only the
  ``captcha_visible`` overlay/frame flag).
- credential: auth.py 12 phrases + real_browser ``fail_words`` extras
  (``incorrect``, ``nie uda``, ``niepowod``, ``bledn``, ``nieprawid``;
  ``login failed``/``invalid password`` were already in auth).
- ban: auth.py 8 phrases + real_browser ``ban_words`` extras
  (``muted``, ``zawiesz``, ``naruszen``, ``zablok``;
  ``suspend``/``banned``/``violation`` were already in auth).
  Generic entries (``incorrect``, ``muted``) subsume some specific ones but
  all are kept so neither driver narrows. Specific phrases stay first so the
  returned phrase string keeps auth's specificity when both match.

Preserved notes:
- Shumei/device_id: DeepSeek's sign-in loads Shumei ``fp.min.js`` and only
  posts a real ``device_id`` when that SDK initializes (real Chromium works;
  Obscura no-render posts ``device_id: null`` and is rejected). Discovery
  lives in ``browser_paths.py``; CDP driving lives in ``real_browser.py``.
- React commit timing: fill then wait ~0.6s before clicking submit so the SPA
  commits controlled-input state; clicking too early posts empty credentials.
- Hidden cf-overlay: the sign-in page ships a hidden ``#cf-overlay``
  containing "One more step before you proceed...". Raw ``body.innerText``
  matches that on every healthy page, so classification runs only on the
  visible-text projection (``dsVisibleText``) plus an explicit visibility
  check for the challenge overlay/frame. The CDP snapshot below adopts the
  same projection (it previously used raw ``innerText``); slice limits and
  token keys stay per-driver (MCP ``tok``/4000, CDP ``token``/3000).

Deferred: typed verdict (Enum/dataclass) and a global ban authority; the
verdict helper stays a ``(kind, phrase)`` tuple to preserve current callers.
"""

from __future__ import annotations

import json

SIGN_IN_URL = "https://chat.deepseek.com/sign_in"

# --- selectors (union; no narrowing) -------------------------------------
EMAIL_SELECTOR = 'input.ds-input__input[type="text"], input[placeholder*="Phone"]'
PASSWORD_SELECTOR = 'input.ds-input__input[type="password"], input[type="password"]'
LOGIN_FALLBACK_SELECTOR = 'div[role="button"].ds-button--primary.ds-button--xl'
# CDP click selector: auth fallback plus the plain-button variant real used.
LOGIN_BUTTON_SELECTOR = (
    'div[role="button"].ds-button--primary.ds-button--xl,'
    "button.ds-button--primary"
)
# Cookie banner: auth matched essential+all; real matched essential only.
COOKIE_SELECTOR = (
    ".cookie_banner-accept-essential-button, .cookie_banner-accept-all-button"
)
COOKIE_SELECTOR_ESSENTIAL = ".cookie_banner-accept-essential-button"

# --- timeouts / polling (values preserved per driver) --------------------
LOGIN_TIMEOUT = 90.0  # Obscura/MCP driver (auth.BrowserLogin)
POLL_START = 0.35
POLL_MAX = 1.25
POLL_BACKOFF = 1.35
_POLL_START = POLL_START
_POLL_MAX = POLL_MAX
REAL_LOGIN_TIMEOUT = 75.0  # CDP driver (real_browser.RealBrowser)
DEFAULT_LOGIN_TIMEOUT = REAL_LOGIN_TIMEOUT
REAL_POLL_INTERVAL = 0.6
REACT_COMMIT_DELAY = 0.6  # SPA onChange commit wait before clicking submit
COOKIE_DISMISS_DELAY_MCP = 0.25
COOKIE_DISMISS_DELAY_CDP = 0.3
EMAIL_WAIT_TIMEOUT = 20
NAVIGATE_RENDER_TIMEOUT = 30.0

# --- phrase lists (union; order preserved so specific phrases win) -------
LOGIN_CAPTCHA_PHRASES = (
    "captcha", "turnstile", "verify you are human", "verifying you are human",
    "verify you are", "human verification", "slide to verify", "puzzle",
    "click to verify", "one more step", "just a moment", "checking your browser",
    "attention required", "security check", "prove you are human",
    "complete the security check", "before you proceed",
)
LOGIN_CREDENTIAL_PHRASES = (
    "incorrect password", "wrong password", "invalid email or password",
    "invalid password", "account does not exist", "email not registered",
    "user not found", "password is incorrect", "login failed", "login error",
    "too many attempts", "account is locked",
    # real_browser fail_words extras (union, more sensitive):
    "incorrect", "nie uda", "niepowod", "bledn", "nieprawid",
)
LOGIN_BAN_PHRASES = (
    "suspend", "banned", "violation", "user is muted", "is muted",
    "suspended until", "account has been suspended", "due to violation",
    # real_browser ban_words extras (union, more sensitive):
    "muted", "zawiesz", "naruszen", "zablok",
)

# --- JS visibility helpers (verbatim from auth.py) -----------------------
JS_VISIBILITY = r"""
function dsVisible(el) {
  if (!el) return false;
  for (var n = el; n && n.nodeType === 1; n = n.parentElement) {
    if (n.getAttribute && n.getAttribute('aria-hidden') === 'true') return false;
    var cs = getComputedStyle(n);
    if (cs.display === 'none' || cs.visibility === 'hidden') return false;
    var op = parseFloat(cs.opacity);
    if (!isNaN(op) && op <= 0.01) return false;
  }
  return true;
}
function dsVisibleText() {
  var out = [];
  function walk(node) {
    if (!node) return;
    if (node.nodeType === 3) {
      var t = (node.nodeValue || '').trim();
      if (t) out.push(t);
      return;
    }
    if (node.nodeType !== 1) return;
    var tag = node.tagName;
    if (tag === 'SCRIPT' || tag === 'STYLE' || tag === 'NOSCRIPT' || tag === 'TEMPLATE') return;
    if (node.getAttribute && node.getAttribute('aria-hidden') === 'true') return;
    var cs = getComputedStyle(node);
    if (cs.display === 'none' || cs.visibility === 'hidden') return;
    var op = parseFloat(cs.opacity);
    if (!isNaN(op) && op <= 0.01) return;
    for (var i = 0; i < node.childNodes.length; i++) walk(node.childNodes[i]);
  }
  walk(document.body || document.documentElement);
  return out.join(' ');
}
"""
_JS_VISIBILITY = JS_VISIBILITY

PAGE_PROBE = ("(function(){" + JS_VISIBILITY + r"""
  var overlay = document.getElementById('cf-overlay');
  var frame = document.querySelector('iframe[src*="challenges.cloudflare.com"], iframe[src*="turnstile"]');
  var captcha = dsVisible(overlay) || dsVisible(frame);
  var token = null;
  try { token = localStorage.getItem('userToken'); } catch (e) { token = null; }
  return JSON.stringify({
    url: location.href,
    tok: token,
    body: dsVisibleText().slice(0, 4000),
    captcha_visible: captcha
  });
})()""")
_PAGE_PROBE = PAGE_PROBE

MARK_LOGIN = ("(function(){" + JS_VISIBILITY + r"""
  var candidates = document.querySelectorAll('div[role="button"], button');
  for (var i = 0; i < candidates.length; i++) {
    var el = candidates[i];
    var text = (el.innerText || el.textContent || '').trim();
    var disabled = el.disabled || el.getAttribute('aria-disabled') === 'true';
    if (text === 'Log in' && !disabled && dsVisible(el)) {
      el.setAttribute('data-deepseaport-login', '1');
      return 'ok';
    }
  }
  return 'not-found';
})()""")
_MARK_LOGIN = MARK_LOGIN

MARK_COOKIE = ("(function(){" + JS_VISIBILITY + r"""
  var el = document.querySelector('.cookie_banner-accept-essential-button, .cookie_banner-accept-all-button');
  if (el && dsVisible(el)) {
    el.setAttribute('data-deepseaport-cookie', '1');
    return 'ok';
  }
  return 'not-found';
})()""")
_MARK_COOKIE = MARK_COOKIE

# CDP snapshot: same visibility projection as PAGE_PROBE, but keeps the
# real_browser wire shape (``token`` key, 3000-char body slice).
CDP_SNAPSHOT_JS = ("(function(){" + JS_VISIBILITY + r"""
  var overlay = document.getElementById('cf-overlay');
  var frame = document.querySelector('iframe[src*="challenges.cloudflare.com"], iframe[src*="turnstile"]');
  var token = null;
  try { token = localStorage.getItem('userToken'); } catch (e) { token = null; }
  return JSON.stringify({
    url: location.href,
    token: token,
    body: dsVisibleText().slice(0, 3000),
    captcha_visible: dsVisible(overlay) || dsVisible(frame)
  });
})()""")

# CDP cookie dismiss: union selector + visibility gate (was essential-only,
# no visibility check). Same 'clicked'/'none' contract as before.
CDP_DISMISS_COOKIE_JS = ("(function(){" + JS_VISIBILITY + r"""
  var el = document.querySelector('.cookie_banner-accept-essential-button, .cookie_banner-accept-all-button');
  if (el && dsVisible(el)) { el.click(); return 'clicked'; }
  return 'none';
})()""")

# CDP login click JS, built from LOGIN_BUTTON_SELECTOR so the two stay in
# sync. Same 'CLICKED'/'NOBTN' contract as before.
CDP_CLICK_LOGIN_JS = (
    "(function(){"
    "var b=document.querySelector(" + json.dumps(LOGIN_BUTTON_SELECTOR) + ");"
    "if(!b) return 'NOBTN'; b.click(); return 'CLICKED';"
    "})()"
)


# --- text helpers --------------------------------------------------------
def collapse_text(text: object, limit: int = 500) -> str:
    try:
        return " ".join(str(text or "").split())[:limit]
    except Exception:
        return ""


def clean_text(text: object) -> str:
    try:
        return " ".join(str(text or "").split())
    except Exception:
        return ""


def classify_page_state(visible_text: str, captcha_visible: bool = False) -> tuple[str, str]:
    """Return (kind, phrase) where kind is captcha/credential/ban or empty.

    Priority (preserved): visible challenge flag first, then captcha
    phrases, then credential phrases, then ban phrases.
    """
    try:
        if captcha_visible:
            return "captcha", "visible challenge overlay"
        low = (visible_text or "").lower()
        for phrase in LOGIN_CAPTCHA_PHRASES:
            if phrase and phrase in low:
                return "captcha", phrase
        for phrase in LOGIN_CREDENTIAL_PHRASES:
            if phrase and phrase in low:
                return "credential", phrase
        for phrase in LOGIN_BAN_PHRASES:
            if phrase and phrase in low:
                return "ban", phrase
    except Exception:
        return "", ""
    return "", ""


def snapshot_verdict(snapshot: dict | None) -> tuple[str, str]:
    """Classify either driver's snapshot dict ({body, captcha_visible}).

    Thin verdict helper over :func:`classify_page_state`; loop policy
    (auth's ban-streak vs real's ban-hint flag) stays driver-specific.
    """
    try:
        if not isinstance(snapshot, dict):
            return "", ""
        return classify_page_state(
            str(snapshot.get("body") or ""),
            bool(snapshot.get("captcha_visible")),
        )
    except Exception:
        return "", ""


class LoginPagePolicy:
    """Discoverability namespace for the shared page policy.

    Module-level names are the single source of truth (so ``auth`` /
    ``real_browser`` re-exports and test monkeypatching keep working);
    these attributes alias them. RHS names resolve to module globals.
    """

    SIGN_IN_URL = SIGN_IN_URL  # noqa: F821
    EMAIL_SELECTOR = EMAIL_SELECTOR  # noqa: F821
    PASSWORD_SELECTOR = PASSWORD_SELECTOR  # noqa: F821
    LOGIN_FALLBACK_SELECTOR = LOGIN_FALLBACK_SELECTOR  # noqa: F821
    LOGIN_BUTTON_SELECTOR = LOGIN_BUTTON_SELECTOR  # noqa: F821
    COOKIE_SELECTOR = COOKIE_SELECTOR  # noqa: F821
    LOGIN_TIMEOUT = LOGIN_TIMEOUT  # noqa: F821
    POLL_START = POLL_START  # noqa: F821
    POLL_MAX = POLL_MAX  # noqa: F821
    POLL_BACKOFF = POLL_BACKOFF  # noqa: F821
    REAL_LOGIN_TIMEOUT = REAL_LOGIN_TIMEOUT  # noqa: F821
    DEFAULT_LOGIN_TIMEOUT = DEFAULT_LOGIN_TIMEOUT  # noqa: F821
    REAL_POLL_INTERVAL = REAL_POLL_INTERVAL  # noqa: F821
    REACT_COMMIT_DELAY = REACT_COMMIT_DELAY  # noqa: F821
    LOGIN_CAPTCHA_PHRASES = LOGIN_CAPTCHA_PHRASES  # noqa: F821
    LOGIN_CREDENTIAL_PHRASES = LOGIN_CREDENTIAL_PHRASES  # noqa: F821
    LOGIN_BAN_PHRASES = LOGIN_BAN_PHRASES  # noqa: F821
    JS_VISIBILITY = JS_VISIBILITY  # noqa: F821
    PAGE_PROBE = PAGE_PROBE  # noqa: F821
    MARK_LOGIN = MARK_LOGIN  # noqa: F821
    MARK_COOKIE = MARK_COOKIE  # noqa: F821
    CDP_SNAPSHOT_JS = CDP_SNAPSHOT_JS  # noqa: F821
    CDP_DISMISS_COOKIE_JS = CDP_DISMISS_COOKIE_JS  # noqa: F821
    CDP_CLICK_LOGIN_JS = CDP_CLICK_LOGIN_JS  # noqa: F821

    classify = staticmethod(classify_page_state)
    verdict = staticmethod(snapshot_verdict)
    collapse = staticmethod(collapse_text)
    clean = staticmethod(clean_text)
