/**
 * The whole reviewer proxy, as one pure-ish function.
 *
 *   reviewer's machine --[ reviewer token ]--> this function --[ eval key ]--> api.openai.com
 *
 * It is deliberately NOT a general OpenAI relay. Three exact paths, three exact
 * shapes, an allowlist of top-level body keys per path, and a model allowlist.
 * Anything else is refused before a byte leaves Vercel.
 *
 * This file imports nothing from Vercel and nothing from npm: it takes a Web
 * `Request` and an env object and returns a Web `Response`, so the test suite
 * drives it directly with a stub `fetch` and never touches the network.
 *
 * What it does NOT know about: the robot, joints, LampController, PyBullet, the
 * orchestrator. It cannot reach them; it only sees opaque request bodies.
 */

import { createHash, timingSafeEqual } from 'node:crypto';

// --------------------------------------------------------------------------
// The allowed surface
// --------------------------------------------------------------------------

export const UPSTREAM_ORIGIN = 'https://api.openai.com';

/** Speech-to-text. `character.STT_MODEL`. */
const STT_MODEL = 'gpt-4o-mini-transcribe';
/** Text-to-speech. `character.TTS_MODEL`. */
const TTS_MODEL = 'tts-1';
/**
 * Reasoning + vision, both on the Responses API. `character.CHARACTER_MODEL`
 * and `character.VISION_MODEL` -- the latter is overridable locally by
 * `LAMP_VISION_MODEL`, which is why this one list is env-configurable.
 */
const DEFAULT_RESPONSES_MODELS = ['gpt-5.6-luna'];

/** The only voice `character.TTS_VOICE` asks for. */
const ALLOWED_TTS_VOICES = ['alloy'];
/** The only audio container the local player decodes (`audio_io.decode_wav`). */
const ALLOWED_TTS_FORMATS = ['wav'];
/** `character.STT_RESPONSE_FORMAT`; the gpt-4o-*-transcribe family allows only this. */
const ALLOWED_STT_FORMATS = ['json'];

/**
 * Top-level body keys `character.py` actually sends. Everything else is
 * refused, which is what keeps this from becoming a general relay: no `tools`,
 * no `stream`, no `store`, no `previous_response_id`, no `background`, no
 * `reasoning`, no `metadata`. The *contents* of `text`/`input` are passed
 * through untouched -- the local app owns those schemas and validates the
 * answer itself.
 */
const RESPONSES_KEYS = ['model', 'instructions', 'input', 'text', 'max_output_tokens'];
const SPEECH_KEYS = ['model', 'voice', 'input', 'response_format'];
/** Multipart field names the SDK sends for a transcription. */
const TRANSCRIPTION_FIELDS = ['file', 'model', 'response_format'];

// --------------------------------------------------------------------------
// Size and cost ceilings
// --------------------------------------------------------------------------
//
// Every number here is derived from a constant in the robot application, with
// headroom, so the proxy is comfortably big enough for the demo and far too
// small to be worth stealing as a bulk relay.

/**
 * `attention.OBSERVE_MAX_BYTES` is 400_000 bytes of JPEG; base64 inflates that
 * to ~533_400 chars, plus the prompt, the JSON schema and framing.
 */
const MAX_RESPONSES_BYTES = 1_200_000;
/**
 * `audio_io` records `RECORD_SECONDS` (4.0) of mono int16. At the worst
 * fallback rate (48 kHz) that is 384_000 bytes of PCM plus a WAV header plus
 * multipart framing.
 */
const MAX_TRANSCRIPTION_BYTES = 1_000_000;
/** `character.MAX_REPLY_CHARS` is 400. A JSON envelope around it is tiny. */
const MAX_SPEECH_BYTES = 8_192;

/** Backstop over `character.MAX_REPLY_CHARS` (400). */
const MAX_SPEECH_INPUT_CHARS = 500;
/** Backstop over the largest of MAX_OUTPUT_TOKENS (300). Bounds cost per call. */
const MAX_OUTPUT_TOKENS_CEILING = 1_000;

/** How long we will wait upstream. `character`'s longest budget is 25s. */
const UPSTREAM_TIMEOUT_MS = 28_000;

/** Best-effort per-instance limiter defaults: requests / window seconds. */
const DEFAULT_RATE_LIMIT = { limit: 60, windowSeconds: 60 };

// --------------------------------------------------------------------------
// Routing table
// --------------------------------------------------------------------------

const ROUTES = {
  '/v1/responses': { maxBytes: MAX_RESPONSES_BYTES, check: checkResponses, json: true },
  '/v1/audio/speech': { maxBytes: MAX_SPEECH_BYTES, check: checkSpeech, json: true },
  '/v1/audio/transcriptions': {
    maxBytes: MAX_TRANSCRIPTION_BYTES,
    check: checkTranscription,
    json: false,
  },
};

// --------------------------------------------------------------------------
// Entry point
// --------------------------------------------------------------------------

/**
 * Handle one reviewer request.
 *
 * @param {Request} request
 * @param {Record<string, string|undefined>} env  process.env, or a stub
 * @param {{fetch?: typeof fetch, now?: () => number}} [deps]
 * @returns {Promise<Response>}
 */
export async function handleRequest(request, env, deps = {}) {
  const doFetch = deps.fetch ?? globalThis.fetch;
  const now = deps.now ?? Date.now;

  // -- the endpoint -------------------------------------------------------
  // Fail closed on the path before anything else, so an unknown path never
  // even reaches the token comparison.
  const endpoint = normalizePath(request.url);
  const route = Object.prototype.hasOwnProperty.call(ROUTES, endpoint)
    ? ROUTES[endpoint]
    : null;
  if (!route) {
    return fail(404, 'unsupported_endpoint', `this proxy does not serve ${endpoint}`);
  }
  if (request.method !== 'POST') {
    return fail(405, 'unsupported_method', 'only POST is accepted', { Allow: 'POST' });
  }

  // -- server configuration -----------------------------------------------
  // A misconfigured proxy refuses service rather than falling back to
  // something permissive.
  const upstreamKey = (env.OPENAI_EVAL_API_KEY ?? '').trim();
  const tokens = splitList(env.HCL_REVIEWER_TOKENS);
  if (!upstreamKey || tokens.length === 0) {
    return fail(503, 'proxy_not_configured', 'the proxy is not configured to serve requests');
  }

  // -- reviewer authentication --------------------------------------------
  const presented = bearerToken(request.headers.get('authorization'));
  if (presented === null) {
    return fail(401, 'invalid_authorization', 'expected an "Authorization: Bearer <token>" header');
  }
  if (!matchesAny(presented, tokens)) {
    return fail(401, 'invalid_token', 'the reviewer token was not recognised');
  }
  const expiry = (env.HCL_REVIEWER_TOKEN_EXPIRES_AT ?? '').trim();
  if (expiry) {
    const deadline = Date.parse(expiry);
    if (Number.isNaN(deadline)) {
      return fail(503, 'proxy_not_configured', 'the proxy is not configured to serve requests');
    }
    if (now() > deadline) {
      return fail(401, 'token_expired', 'the reviewer token has expired');
    }
  }

  // -- rate limit ---------------------------------------------------------
  const limited = rateLimit(fingerprint(presented, request), parseRateLimit(env), now);
  if (limited) {
    return fail(429, 'rate_limited', `too many requests; retry in ${limited} seconds`, {
      'Retry-After': String(limited),
    });
  }

  // -- size ---------------------------------------------------------------
  // Checked twice: the declared length first (cheap, rejects before we buffer)
  // and then the bytes we actually received.
  const declared = Number(request.headers.get('content-length') ?? '');
  if (Number.isFinite(declared) && declared > route.maxBytes) {
    return tooLarge(route.maxBytes);
  }
  const body = Buffer.from(await request.arrayBuffer());
  if (body.length > route.maxBytes) {
    return tooLarge(route.maxBytes);
  }
  if (body.length === 0) {
    return fail(400, 'empty_body', 'the request had no body');
  }

  // -- shape and model ----------------------------------------------------
  const contentType = request.headers.get('content-type') ?? '';
  const verdict = route.check(body, contentType, env);
  if (verdict.error) {
    return fail(400, verdict.code, verdict.error);
  }

  // -- forward ------------------------------------------------------------
  // Headers are built from scratch. Nothing the client sent -- least of all
  // its Authorization, or any OpenAI-Organization/OpenAI-Project header --
  // is carried through, and the host is a constant.
  const upstreamHeaders = {
    Authorization: `Bearer ${upstreamKey}`,
    'Content-Type': route.json ? 'application/json' : contentType,
    Accept: request.headers.get('accept') === 'application/json' ? 'application/json' : '*/*',
    'User-Agent': 'hcl-reviewer-proxy',
  };

  const abort = new AbortController();
  const timer = setTimeout(() => abort.abort(), UPSTREAM_TIMEOUT_MS);
  let upstream;
  try {
    upstream = await doFetch(`${UPSTREAM_ORIGIN}${endpoint}`, {
      method: 'POST',
      headers: upstreamHeaders,
      body,
      signal: abort.signal,
    });
  } catch (exc) {
    // The message may quote the request we just built, headers included.
    console.error('[proxy] upstream request failed', redact(describe(exc), upstreamKey));
    return fail(502, 'upstream_unreachable', 'the upstream request failed');
  } finally {
    clearTimeout(timer);
  }

  console.log(
    `[proxy] POST ${endpoint} model=${verdict.model ?? '-'} ` +
      `in=${body.length}B status=${upstream.status}`,
  );
  return await relayResponse(upstream, upstreamKey);
}

// --------------------------------------------------------------------------
// Path
// --------------------------------------------------------------------------

/**
 * Reduce a request URL to a canonical `/v1/...` endpoint.
 *
 * Vercel's rewrite turns `/v1/responses` into `/api/v1/responses` before the
 * function sees it, so both spellings normalise to the same thing and the
 * routing table stays readable. Percent-escapes and `..` are resolved by
 * `new URL` first, so `/v1/audio/../../v1/responses` cannot smuggle a path.
 */
function normalizePath(rawUrl) {
  let pathname;
  try {
    pathname = new URL(rawUrl, 'http://proxy.invalid').pathname;
  } catch {
    return '';
  }
  if (pathname.startsWith('/api/')) {
    pathname = pathname.slice(4);
  }
  if (pathname.length > 1 && pathname.endsWith('/')) {
    pathname = pathname.slice(0, -1);
  }
  return pathname;
}

// --------------------------------------------------------------------------
// Per-endpoint validation
// --------------------------------------------------------------------------
//
// Each returns `{model}` on success or `{code, error}` on refusal. None of them
// inspects or rewrites the robot's structured-output schemas: `text` and
// `input` are forwarded byte-for-byte, because the local application owns
// those contracts and validates the answers itself.

function checkResponses(body, contentType, env) {
  const parsed = parseJson(body, contentType);
  if (parsed.error) return parsed;
  const payload = parsed.value;

  const extra = Object.keys(payload).filter((key) => !RESPONSES_KEYS.includes(key));
  if (extra.length) {
    return refuse('unsupported_parameter', `unsupported parameters: ${extra.sort().join(', ')}`);
  }
  const allowed = splitList(env.HCL_ALLOWED_RESPONSES_MODELS);
  const models = allowed.length ? allowed : DEFAULT_RESPONSES_MODELS;
  if (typeof payload.model !== 'string' || !models.includes(payload.model)) {
    return refuse('unsupported_model', 'that model is not enabled on this proxy');
  }
  const cap = payload.max_output_tokens;
  if (cap !== undefined && (!Number.isInteger(cap) || cap < 1 || cap > MAX_OUTPUT_TOKENS_CEILING)) {
    return refuse(
      'unsupported_parameter',
      `max_output_tokens must be an integer in 1..${MAX_OUTPUT_TOKENS_CEILING}`,
    );
  }
  if (cap === undefined) {
    return refuse('unsupported_parameter', 'max_output_tokens is required by this proxy');
  }
  return { model: payload.model };
}

function checkSpeech(body, contentType) {
  const parsed = parseJson(body, contentType);
  if (parsed.error) return parsed;
  const payload = parsed.value;

  const extra = Object.keys(payload).filter((key) => !SPEECH_KEYS.includes(key));
  if (extra.length) {
    return refuse('unsupported_parameter', `unsupported parameters: ${extra.sort().join(', ')}`);
  }
  if (payload.model !== TTS_MODEL) {
    return refuse('unsupported_model', 'that model is not enabled on this proxy');
  }
  if (!ALLOWED_TTS_VOICES.includes(payload.voice)) {
    return refuse('unsupported_parameter', 'that voice is not enabled on this proxy');
  }
  if (payload.response_format !== undefined && !ALLOWED_TTS_FORMATS.includes(payload.response_format)) {
    return refuse('unsupported_parameter', 'that response_format is not enabled on this proxy');
  }
  if (typeof payload.input !== 'string' || !payload.input.trim()) {
    return refuse('unsupported_parameter', 'input must be a non-empty string');
  }
  if (payload.input.length > MAX_SPEECH_INPUT_CHARS) {
    return refuse(
      'unsupported_parameter',
      `input is longer than the ${MAX_SPEECH_INPUT_CHARS} character limit`,
    );
  }
  return { model: payload.model };
}

function checkTranscription(body, contentType) {
  if (!/^multipart\/form-data\s*;/i.test(contentType)) {
    return refuse('unsupported_content_type', 'expected multipart/form-data');
  }
  const boundary = /boundary=(?:"([^"]+)"|([^;]+))/i.exec(contentType);
  if (!boundary) {
    return refuse('unsupported_content_type', 'the multipart boundary is missing');
  }
  const fields = readTextFields(body, (boundary[1] ?? boundary[2]).trim());
  if (fields === null) {
    return refuse('malformed_body', 'the multipart body could not be read');
  }
  const extra = fields.names.filter((name) => !TRANSCRIPTION_FIELDS.includes(name));
  if (extra.length) {
    return refuse('unsupported_parameter', `unsupported fields: ${extra.sort().join(', ')}`);
  }
  if (fields.values.model !== STT_MODEL) {
    return refuse('unsupported_model', 'that model is not enabled on this proxy');
  }
  const format = fields.values.response_format;
  if (format !== undefined && !ALLOWED_STT_FORMATS.includes(format)) {
    return refuse('unsupported_parameter', 'that response_format is not enabled on this proxy');
  }
  return { model: fields.values.model };
}

function parseJson(body, contentType) {
  if (!/^application\/json\b/i.test(contentType)) {
    return refuse('unsupported_content_type', 'expected application/json');
  }
  let value;
  try {
    value = JSON.parse(body.toString('utf8'));
  } catch {
    return refuse('malformed_body', 'the request body was not valid JSON');
  }
  if (value === null || typeof value !== 'object' || Array.isArray(value)) {
    return refuse('malformed_body', 'the request body was not a JSON object');
  }
  return { value };
}

function refuse(code, error) {
  return { code, error };
}

/**
 * Pull the *text* fields out of a multipart body without a dependency.
 *
 * Parts carrying a filename (the audio itself) are recorded by name and their
 * bytes are ignored; the body is forwarded upstream unmodified either way, so
 * this is a read, never a rewrite. Returns null if the framing is unreadable.
 */
function readTextFields(body, boundary) {
  const delimiter = Buffer.from(`--${boundary}`);
  const names = [];
  const values = Object.create(null);

  let cursor = body.indexOf(delimiter);
  if (cursor === -1) return null;

  while (cursor !== -1) {
    let start = cursor + delimiter.length;
    if (body.slice(start, start + 2).toString('latin1') === '--') break; // closing delimiter
    if (body.slice(start, start + 2).toString('latin1') === '\r\n') start += 2;

    const next = body.indexOf(delimiter, start);
    const end = next === -1 ? body.length : next;
    const headerEnd = body.indexOf('\r\n\r\n', start);
    if (headerEnd === -1 || headerEnd > end) return null;

    const headers = body.slice(start, headerEnd).toString('utf8');
    const named = /name="([^"]*)"/i.exec(headers);
    if (!named) return null;
    const name = named[1];
    names.push(name);

    if (!/filename="/i.test(headers)) {
      // A scalar form field: bounded before decoding so a hostile body cannot
      // make us materialise megabytes of "field value".
      const raw = body.slice(headerEnd + 4, Math.max(headerEnd + 4, end - 2));
      if (raw.length > 256) return null;
      values[name] = raw.toString('utf8');
    }
    cursor = next;
  }
  return names.length ? { names, values } : null;
}

// --------------------------------------------------------------------------
// Reviewer tokens
// --------------------------------------------------------------------------

/** Extract the bearer value, or null if the header is absent or malformed. */
function bearerToken(header) {
  if (typeof header !== 'string') return null;
  const match = /^Bearer[ \t]+(\S.*)$/i.exec(header.trim());
  if (!match) return null;
  const token = match[1].trim();
  return token.length ? token : null;
}

/** Compare against every configured token in constant time. */
function matchesAny(presented, tokens) {
  const offered = sha256(presented);
  let ok = false;
  for (const token of tokens) {
    // No short-circuit: every candidate is compared, so the time taken does
    // not reveal which one (or how many) matched.
    if (timingSafeEqual(offered, sha256(token))) ok = true;
  }
  return ok;
}

function sha256(text) {
  return createHash('sha256').update(text, 'utf8').digest();
}

function splitList(raw) {
  return (raw ?? '')
    .split(',')
    .map((entry) => entry.trim())
    .filter(Boolean);
}

// --------------------------------------------------------------------------
// Rate limiting
// --------------------------------------------------------------------------
//
// Best effort and deliberately so: a serverless instance is one of many, and
// the counter dies with it. It is a cheap brake on a single runaway client,
// not the abuse boundary. The boundary is the Vercel WAF rate-limit rule
// documented in README.md, plus the per-request ceilings above.

const buckets = new Map();

function parseRateLimit(env) {
  const raw = (env.HCL_PROXY_RATE_LIMIT ?? '').trim();
  const match = /^(\d+)\s*\/\s*(\d+)$/.exec(raw);
  if (!match) return DEFAULT_RATE_LIMIT;
  const limit = Number(match[1]);
  const windowSeconds = Number(match[2]);
  if (limit < 1 || windowSeconds < 1) return DEFAULT_RATE_LIMIT;
  return { limit, windowSeconds };
}

/** Returns seconds to wait when over the limit, or 0 when the request may go. */
function rateLimit(key, { limit, windowSeconds }, now) {
  const at = now();
  const windowMs = windowSeconds * 1000;
  for (const [existing, bucket] of buckets) {
    if (bucket.resetAt <= at) buckets.delete(existing);
  }
  let bucket = buckets.get(key);
  if (!bucket || bucket.resetAt <= at) {
    bucket = { count: 0, resetAt: at + windowMs };
    buckets.set(key, bucket);
  }
  bucket.count += 1;
  if (bucket.count > limit) {
    return Math.max(1, Math.ceil((bucket.resetAt - at) / 1000));
  }
  return 0;
}

/** A bucket key that identifies a client without storing their token. */
function fingerprint(token, request) {
  const ip =
    request.headers.get('x-real-ip') ??
    (request.headers.get('x-forwarded-for') ?? '').split(',')[0].trim() ??
    '';
  return `${sha256(token).toString('hex').slice(0, 16)}:${ip}`;
}

/** Test seam: the limiter is process-global, so tests need to reset it. */
export function _resetRateLimit() {
  buckets.clear();
}

// --------------------------------------------------------------------------
// Responses
// --------------------------------------------------------------------------

/**
 * Relay the upstream reply with a header allowlist.
 *
 * Only `content-type` survives. Nothing that identifies the account behind the
 * key (`openai-organization`, `openai-project`, rate-limit headers that expose
 * the project's quota) and no `set-cookie` is passed on.
 *
 * A failing upstream call can quote the offending key back inside its JSON
 * error body, so text bodies are redacted before they leave. Successful binary
 * bodies (the TTS wav) are passed through untouched.
 */
async function relayResponse(upstream, upstreamKey) {
  const contentType = upstream.headers.get('content-type') ?? 'application/octet-stream';
  const headers = { 'Content-Type': contentType, 'Cache-Control': 'no-store' };

  const textual = /json|text|xml/i.test(contentType);
  if (upstream.ok && !textual) {
    const bytes = Buffer.from(await upstream.arrayBuffer());
    return new Response(bytes, { status: upstream.status, headers });
  }
  const text = await upstream.text();
  return new Response(redact(text, upstreamKey), { status: upstream.status, headers });
}

function fail(status, code, message, extraHeaders = {}) {
  // Shaped like an OpenAI error so the SDK surfaces it usefully, and carrying
  // only text this file wrote -- never an env value, never a header.
  return new Response(
    JSON.stringify({ error: { message, type: 'hcl_reviewer_proxy', code } }),
    {
      status,
      headers: { 'Content-Type': 'application/json', 'Cache-Control': 'no-store', ...extraHeaders },
    },
  );
}

function tooLarge(maxBytes) {
  return fail(413, 'payload_too_large', `request body exceeds the ${maxBytes} byte limit`);
}

/** Strip the server key -- exact value and key-shaped tokens -- from *text*. */
export function redact(text, secret) {
  if (!text) return text;
  let out = String(text);
  if (secret && out.includes(secret)) out = out.split(secret).join('<redacted>');
  return out.replace(/\bsk-[A-Za-z0-9_-]{8,}/g, '<redacted>');
}

function describe(exc) {
  // Never `repr`/stack: a fetch error can carry the request, headers included.
  const name = exc && exc.name ? exc.name : 'Error';
  const message = exc && typeof exc.message === 'string' ? exc.message.split('\n')[0] : '';
  return message ? `${name}: ${message}` : name;
}
