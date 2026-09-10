## Setup in simple terms
- install [obscura](https://github.com/h4ckf0r0day/obscura/releases) (no-render stealth is the one i use)
- install the newest [release](https://github.com/gyatstian/deepseaport/releases) (TIP: CLICK THE BLUE TEXT)
- place obscura and deepseaport next to eachother or put obscura in PATH
- open deepseaport, add account(s) and then start server.
- *optionally play around with settings or/and config.json!*
- base url: http://127.0.0.1:5001/v1

# notice
- deepseek on the website feels lobotomized/lazy, though it's not that bad, and you can definitely make it much better with skills or good prompting. it seems the longer the prompt is, the more he thinks<br>
- tool calling seems ok. sometimes fails on long/complicated ones but he finds a way around<br>
- app needs https://aka.ms/vs/17/release/vc_redist.x64.exe beacuse of obscura

# deepseaport — DeepSeek web2api (Obscura-backed)

Unofficial, research-only bridge: exposes `chat.deepseek.com` web chat as an
OpenAI-compatible API (`/v1/chat/completions`, streaming + `tools`).

How it works:

- **Obscura** (`h4ckf0r0day/obscura` headless browser) owns a persistent
  profile that solves the AWS WAF JavaScript challenge and holds the
  `aws-waf-token` cookie. All DeepSeek HTTP calls replay that cookie jar,
  with a browser User-Agent captured from the same engine.
- Hot path is direct HTTP (`curl_cffi`, Chrome TLS impersonation):
  login → create session → PoW challenge (`DeepSeekHashV1` wasm) →
  `POST /api/v0/chat/completion` (SSE `p`/`v` events) → delete session.
- No DOM driving per token: fast enough for agentic loops.

Agentic-harness support:

- Full OpenAI `tools` schema in, `tool_calls` out (stream + non-stream),
  parallel calls, stable `call_xxx` ids, `finish_reason: "tool_calls"`.
- `role: tool` results are folded back into the web prompt, so multi-step
  tool loops work against a model with no native function calling.
- One in-flight stream per account (DeepSeek limitation) enforced with
  per-account locks + rotation; auto session cleanup; PoW/WAF retry once.

## Quickstart
**For both:**
Obscura binary (`h4ckf0r0day/obscura`, i personally use no-render stealth): drop
`obscura.exe` next to `config.json` (or under `bin/`, `tools/`,
`vendor/`, `obscura-*/`), or set `obscura_bin` / `OBSCURA_BIN`. path also works.

**A. exe:**
- just run the .exe from the releases

**B. python:**
```powershell
pip install -e .
python -m deepseaport login --email you@x.com  # browser login via Obscura, saves token
python -m deepseaport waf                      # verify Obscura + WAF cookie
python -m deepseaport serve                    # :5001
```



```powershell
curl http://127.0.0.1:5001/v1/models
curl -N http://127.0.0.1:5001/v1/chat/completions `
  -H "Content-Type: application/json" `
  -d '{"model":"deepseek-flash","messages":[{"role":"user","content":"hi"}],"stream":true}'
```

## Config

`config.json` (or env, see `.env.example`):

```json
{
  "keys": ["sk-local-dev"],
  "accounts": [{ "email": "you@x.com", "password": "...", "token": "" }],
  "obscura_bin": "",
  "obscura_profile": "",
  "port": 5001,
  "listen": false
}
```

`"listen": true` serves on `0.0.0.0` (LAN-visible); default `false` binds
`127.0.0.1` only. `--host` flag overrides both.

You can skip email/password by pasting a `userToken` from a logged-in
browser (localStorage) into `accounts[].token`.

## Endpoints

- `GET /health`, `GET /v1/models`, `POST /v1/chat/completions`
- `GET /v1/waf/status` — Obscura binary, profile, WAF cookie state

Models: `deepseek-flash`, `deepseek-flash-reasoner`,
`deepseek-flash-search`, `deepseek-flash-reasoner-search`.
Reasoner models return `reasoning_content` alongside `content`.

See [ARCHITECTURE.md](ARCHITECTURE.md) for protocol details, auth/PoW/SSE
notes, and the tool-calling design.

## Disclaimer

This is an independent educational/experimental project. 
Use of this project is at your own risk, and users are responsible for complying with all relevant terms and laws. 
It is provided as-is, without warranties.
Unofficial reverse engineering for personal research. Respect DeepSeek's
Terms of Service; web sessions can break or be rate-limited at any time.
Prefer the official `platform.deepseek.com` API for production.
