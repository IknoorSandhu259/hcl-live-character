# HCL reviewer proxy (optional)

Lets a reviewer run the demo on my OpenAI quota without receiving my API key.

    reviewer  --Bearer <reviewer token>-->  this function  --Bearer <eval key>-->  api.openai.com

Not a general relay: POST only, to `/v1/responses`, `/v1/audio/transcriptions`
and `/v1/audio/speech`, restricted to `gpt-5.6-luna`, `gpt-4o-mini-transcribe`
and `tts-1`, with body ceilings derived from the demo (1.2 MB / 1 MB / 8 kB) and
only the parameters `src/character.py` actually sends. Everything else fails
closed. `src/character.py` is unchanged — the OpenAI SDK reads `OPENAI_BASE_URL`
itself, so reviewer mode is two environment variables:

```bash
export OPENAI_API_KEY='<reviewer token>'          # instead of my real key
export OPENAI_BASE_URL='https://<vercel-host>/v1' # unset for normal direct mode
```

## Environment variables (Vercel, server-side only)

| Name | |
| --- | --- |
| `OPENAI_EVAL_API_KEY` | A **dedicated restricted API key created inside my existing OpenAI "HCL Demo" project** — not my personal key, so it can be revoked on its own. |
| `HCL_REVIEWER_TOKEN` | The temporary token I hand to the reviewer. |

Missing either → the proxy answers 503 to everything.

## Layout

One static function file per allowed operation — `api/v1/responses.js`,
`api/v1/audio/transcriptions.js`, `api/v1/audio/speech.js` — all three
delegating to `lib/proxy.js`. Do **not** replace these with a catch-all
`api/[...path].js`: under the zero-config `/api` builder that compiles to a
single-segment matcher, so `/api/v1/responses` falls through to the platform's
blanket 404 (`X-Vercel-Error: NOT_FOUND`). The `vercel.json` rewrite maps
`/v1/*` onto them; `https://<host>/api/v1` also works directly if the rewrite is
ever removed.

## Deploy

```bash
cd reviewer_proxy
npx vercel link                                   # a NEW project, e.g. hcl-reviewer-proxy
npx vercel env add OPENAI_EVAL_API_KEY production
npx vercel env add HCL_REVIEWER_TOKEN production  # python3 -c "import secrets; print('hclrev_'+secrets.token_urlsafe(32))"
npx vercel deploy --prod
```

Two dashboard settings are required and are not in `vercel.json`:

1. **Deployment Protection** — Vercel Authentication is on by default for
   *Preview* and would lock the reviewer out. Deploy `--prod` and confirm it is
   off for Production.
2. **WAF rate limit** — this is the rate-limit control; there is none in the
   function. Project → Firewall → Configure → New Rule: if `Request Path`
   starts with `/v1`, then **Rate Limit**, Fixed Window, 60s / 60 requests, key
   `IP Address`, action Deny → Review Changes → Publish. (Available on Hobby:
   one rate-limit rule per project.)

## Smoke test

```bash
HOST=https://<vercel-host>; TOKEN='<reviewer token>'
c() { curl -sS -o /dev/null -w '%{http_code} ' "$@"; }
J='-H Content-Type:application/json'
c "$HOST/v1/responses" -H "Authorization: Bearer $TOKEN" $J \
  -d '{"model":"gpt-5.6-luna","instructions":"Reply ok.","input":"ping","max_output_tokens":16}'
c "$HOST/v1/responses" $J -d '{}'                                        # no token
c "$HOST/v1/responses" -H 'Authorization: Bearer wrong' $J -d '{}'       # wrong token
c "$HOST/v1/chat/completions" -H "Authorization: Bearer $TOKEN" $J -d '{}' # platform 404
c -X GET "$HOST/v1/responses" -H "Authorization: Bearer $TOKEN"
c "$HOST/v1/responses" -H "Authorization: Bearer $TOKEN" $J \
  -d '{"model":"gpt-4o","input":"ping","max_output_tokens":16}'; echo
# expect: 200 401 401 404 405 400
```

Unsupported paths are refused by Vercel's own 404 before reaching the function;
`lib/proxy.js` refuses them too, as defence in depth.

Offline tests: `node --test` here, `python -m pytest tests -q` in the repo root.
Routing itself: `npx vercel build`, then check `.vercel/output/functions/` holds
`api/v1/responses.func`, `api/v1/audio/speech.func` and
`api/v1/audio/transcriptions.func`.

## Revoke

- **Reviewer token**: `npx vercel env rm HCL_REVIEWER_TOKEN production`, add a
  new value, `npx vercel deploy --prod`. To cut access instantly instead, pause
  or delete the Vercel project (Settings → General).
- **Eval key**: revoke that one key in the OpenAI **HCL Demo** project, issue a
  replacement, then update `OPENAI_EVAL_API_KEY` and redeploy. My personal key
  and the project's other keys are unaffected.
