# HCL reviewer proxy (optional)

A reviewer can run the whole lamp demo on my OpenAI quota without ever
receiving my OpenAI API key.

```
reviewer's Ubuntu machine
        |  Authorization: Bearer <temporary reviewer token>
        v
  this function on Vercel        <- the real key lives here, in Vercel's
        |  Authorization: Bearer <HCL-evaluation project key>      env store
        v
  api.openai.com
```

This directory is **entirely optional and entirely separate** from the robot.
It has no dependencies, no database, no accounts, no Docker, and no knowledge of
the robot at all: it cannot reach `LampController`, a joint, PyBullet, or the
orchestrator. It sees three opaque request bodies and forwards them.

`src/character.py` is **unchanged**. The official OpenAI Python SDK already
reads `OPENAI_BASE_URL` from the environment (`openai/_client.py`), so reviewer
mode is two environment variables and nothing else.

---

## The allowed surface

Three paths, POST only. Everything else — every other path, every other method,
every other model, an oversized body, a malformed `Authorization` header — is
refused before a byte reaches OpenAI.

| Path                       | Model(s) allowed                        | Max body     |
| -------------------------- | --------------------------------------- | ------------ |
| `/v1/responses`            | `gpt-5.6-luna` (`HCL_ALLOWED_RESPONSES_MODELS`) | 1,200,000 B |
| `/v1/audio/transcriptions` | `gpt-4o-mini-transcribe`                | 1,000,000 B  |
| `/v1/audio/speech`         | `tts-1`, voice `alloy`, format `wav`    | 8,192 B      |

Every ceiling is derived from a constant in the robot application, with
headroom:

- `attention.OBSERVE_MAX_BYTES` is 400 kB of JPEG → ~533 kB base64 + prompt and
  schema, so 1.2 MB.
- `audio_io` records 4.0 s of mono int16, worst case 48 kHz → 384 kB of PCM plus
  WAV and multipart framing, so 1 MB.
- `character.MAX_REPLY_CHARS` is 400, so TTS gets 8 kB and a 500-character
  `input` cap.
- `max_output_tokens` is required and capped at 1000 (the app's largest is 300),
  which bounds the cost of any single call.

Only the top-level body keys `character.py` actually sends are accepted:
`model`, `instructions`, `input`, `text`, `max_output_tokens` for `/v1/responses`.
That is what stops this being a general relay — `tools`, `stream`, `store`,
`background`, `previous_response_id` and friends are all refused as unsupported
parameters. The *contents* of `text` and `input` are forwarded byte-for-byte:
the proxy does not know or rewrite the robot's structured-output schemas, and
the local application still validates every answer itself.

---

## Environment variables (server side only, set in Vercel)

| Name | Required | Purpose |
| ---- | -------- | ------- |
| `OPENAI_EVAL_API_KEY` | yes | The key from a **separate OpenAI project** created for HCL evaluation. Never leaves the server. |
| `HCL_REVIEWER_TOKENS` | yes | Comma-separated list of accepted reviewer tokens. Plural so a new token can be added before the old one is removed. |
| `HCL_REVIEWER_TOKEN_EXPIRES_AT` | no | ISO-8601 instant, e.g. `2026-09-30T00:00:00Z`. Everything 401s after it. |
| `HCL_PROXY_RATE_LIMIT` | no | `requests/windowSeconds`, default `60/60`. |
| `HCL_ALLOWED_RESPONSES_MODELS` | no | Comma-separated override, for when `LAMP_VISION_MODEL` points the vision call elsewhere. |

If `OPENAI_EVAL_API_KEY` or `HCL_REVIEWER_TOKENS` is missing the proxy answers
`503` to everything. It never falls back to something permissive.

---

## Running the demo: the two modes

**Normal / direct (unchanged, and what I use):**

```bash
export OPENAI_API_KEY=sk-...        # my real key
unset OPENAI_BASE_URL
python src/engagement_demo.py
```

**Reviewer / proxy:**

```bash
export OPENAI_API_KEY='<the temporary reviewer token I send you>'
export OPENAI_BASE_URL='https://<the-vercel-host>/v1'
python src/engagement_demo.py
```

That is the entire difference. No code path, no flag, no import changes.

---

## Deploying

From this directory:

```bash
cd reviewer_proxy
npx vercel login
npx vercel link                     # create a NEW project, e.g. hcl-reviewer-proxy

# the server-side secrets (paste at the prompt; they are never written to disk here)
npx vercel env add OPENAI_EVAL_API_KEY production
npx vercel env add HCL_REVIEWER_TOKENS production
npx vercel env add HCL_REVIEWER_TOKEN_EXPIRES_AT production   # optional

npx vercel deploy --prod
```

Generate the reviewer token with something you will not have to think about:

```bash
python3 -c "import secrets; print('hclrev_' + secrets.token_urlsafe(32))"
```

Send that token and the deployment URL to the reviewer. The URL does not need to
be secret; the token is the authentication.

### Vercel settings that must be configured by hand

1. **Deployment Protection.** Project → Settings → Deployment Protection.
   Vercel Authentication is enabled for *Preview* deployments by default, which
   would make the proxy unreachable for a reviewer. Deploy to **production**
   (`--prod`) and confirm Vercel Authentication is **off for Production**, or
   turn it off for Preview if you hand over a preview URL instead.

2. **WAF rate-limit rule** — the actual abuse boundary. Project → **Firewall**
   → Configure → **+ New Rule**:

   - Name: `reviewer-proxy-rate-limit`
   - If: `Request Path` `starts with` `/v1`
   - Then: **Rate Limit**, Fixed Window
   - Time Window: `60s`, Request Limit: `60`
   - Key: `IP Address`
   - Action: `Deny` (429)
   - Save Rule → **Review Changes** → **Publish**

   Available on Hobby (one rate-limit rule per project, IP / JA4 keys, fixed
   window, 10 s–10 min). The in-function limiter in `lib/proxy.js` is a cheap
   per-instance brake and is *not* a substitute: a serverless instance is one of
   many and its counter dies with it.

3. **A spend limit on the evaluation OpenAI project** — a backstop for the
   bill, not a request-stopping control. The controls that actually stop
   requests are the token, the path/model/parameter allowlists, the body
   ceilings, the `max_output_tokens` cap and the WAF rule.

---

## Revoking

**A reviewer token** (takes effect on the next deployment, seconds):

```bash
cd reviewer_proxy
npx vercel env rm HCL_REVIEWER_TOKENS production
npx vercel env add HCL_REVIEWER_TOKENS production    # paste the remaining tokens, or a new one
npx vercel deploy --prod                             # env changes apply to new deployments
```

To kill access outright without redeploying, delete the Vercel project or pause
it (Project → Settings → General → Pause / Delete). Setting
`HCL_REVIEWER_TOKEN_EXPIRES_AT` at deploy time makes the token self-revoking.

**The OpenAI evaluation key** (immediate, and touches nothing else I own,
because it is its own project):

1. platform.openai.com → the **HCL evaluation** project → API keys → revoke the key.
2. Create a replacement in the same project.
3. `npx vercel env rm OPENAI_EVAL_API_KEY production`, `npx vercel env add …`,
   `npx vercel deploy --prod`.

To retire the whole evaluation path: archive the OpenAI project. My personal
key and my other projects are unaffected either way.

---

## Tests

```bash
cd reviewer_proxy
node --test          # 32 tests, no network, no Vercel, no API quota
```

`lib/proxy.js` takes a Web `Request` plus an env object and returns a Web
`Response`, so the tests drive it directly with a stub `fetch` that records
exactly what would have been sent upstream. That is how the "the reviewer can
never supply or override the upstream key" cases are asserted.

The Python side is covered by `tests/test_reviewer_proxy_mode.py`
(`python -m pytest tests -q`), which pins the SDK's `OPENAI_BASE_URL` behaviour
in both modes so a future SDK bump cannot silently send a reviewer token to the
real API.

---

## Live smoke test

After deploying, with `HOST` and `TOKEN` set:

```bash
HOST=https://<the-vercel-host>
TOKEN='<the reviewer token>'

# 1. the allowed path succeeds (expect 200 and a JSON response object)
curl -sS -o /dev/null -w '%{http_code}\n' "$HOST/v1/responses" \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"model":"gpt-5.6-luna","instructions":"Reply with the word ok.","input":"ping","max_output_tokens":16}'

# 2. no token          -> 401
curl -sS -o /dev/null -w '%{http_code}\n' "$HOST/v1/responses" \
  -H 'Content-Type: application/json' -d '{}'

# 3. wrong token       -> 401
curl -sS -o /dev/null -w '%{http_code}\n' "$HOST/v1/responses" \
  -H 'Authorization: Bearer wrong' -H 'Content-Type: application/json' -d '{}'

# 4. other endpoint    -> 404
curl -sS -o /dev/null -w '%{http_code}\n' "$HOST/v1/chat/completions" \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' -d '{}'

# 5. other method      -> 405
curl -sS -o /dev/null -w '%{http_code}\n' -X GET "$HOST/v1/responses" \
  -H "Authorization: Bearer $TOKEN"

# 6. other model       -> 400
curl -sS -o /dev/null -w '%{http_code}\n' "$HOST/v1/responses" \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"model":"gpt-4o","input":"ping","max_output_tokens":16}'

# 7. oversized body    -> 413
python3 -c "print('{\"model\":\"gpt-5.6-luna\",\"max_output_tokens\":16,\"input\":\"' + 'A'*1300000 + '\"}')" \
  | curl -sS -o /dev/null -w '%{http_code}\n' "$HOST/v1/responses" \
      -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' --data-binary @-

# 8. the reviewer cannot bring their own upstream key -> still 401
curl -sS -o /dev/null -w '%{http_code}\n' "$HOST/v1/responses" \
  -H 'Authorization: Bearer sk-anything' -H 'Content-Type: application/json' -d '{}'
```

Expected: `200 401 401 404 405 400 413 401`.

Then the real check — the demo itself, end to end, on the reviewer's machine:

```bash
export OPENAI_API_KEY="$TOKEN"
export OPENAI_BASE_URL="$HOST/v1"
python src/engagement_demo.py
```
