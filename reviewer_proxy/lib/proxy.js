/**
 * Optional HCL reviewer proxy: a reviewer token in, my OpenAI key out.
 *
 * Not a general relay. Three POST paths, three models, an allowlist of the
 * top-level body keys `src/character.py` actually sends, and size ceilings
 * derived from the demo. Everything else fails closed before a byte reaches
 * OpenAI. It knows nothing about the robot and can reach nothing in it.
 *
 * Takes a Web Request plus an env object and returns a Web Response, so the
 * tests drive it directly with a stub fetch. No npm dependencies.
 */

import { createHash, timingSafeEqual } from 'node:crypto';

const UPSTREAM = 'https://api.openai.com';

// Models, voice and formats: exactly what src/character.py asks for.
const RESPONSES_MODEL = 'gpt-5.6-luna';
const STT_MODEL = 'gpt-4o-mini-transcribe';
const TTS_MODEL = 'tts-1';
const TTS_VOICE = 'alloy';

// Ceilings derived from the app, with headroom:
//   attention.OBSERVE_MAX_BYTES 400kB JPEG -> ~533kB base64 + prompt + schema
//   audio_io 4.0s mono int16 @ 48kHz worst case -> 384kB PCM + framing
//   character.MAX_REPLY_CHARS 400; MAX_OUTPUT_TOKENS 300
const MAX_SPEECH_INPUT_CHARS = 500;
const MAX_OUTPUT_TOKENS = 1000;

const ROUTES = {
  '/v1/responses': { maxBytes: 1_200_000, json: true, check: checkResponses },
  '/v1/audio/speech': { maxBytes: 8_192, json: true, check: checkSpeech },
  '/v1/audio/transcriptions': { maxBytes: 1_000_000, json: false, check: checkTranscription },
};

// Refusing every other top-level key is what stops this being a general
// relay: no tools, stream, store, background or previous_response_id.
const RESPONSES_KEYS = ['model', 'instructions', 'input', 'text', 'max_output_tokens'];
const SPEECH_KEYS = ['model', 'voice', 'input', 'response_format'];
const STT_FIELDS = ['file', 'model', 'response_format'];

/** @param {Request} request @param {Record<string,string|undefined>} env */
export async function handleRequest(request, env, fetchImpl = globalThis.fetch) {
  const endpoint = normalizePath(request.url);
  const route = Object.hasOwn(ROUTES, endpoint) ? ROUTES[endpoint] : null;
  if (!route) return fail(404, `this proxy does not serve ${endpoint}`);
  if (request.method !== 'POST') return fail(405, 'only POST is accepted', { Allow: 'POST' });

  const upstreamKey = (env.OPENAI_EVAL_API_KEY ?? '').trim();
  const reviewerToken = (env.HCL_REVIEWER_TOKEN ?? '').trim();
  if (!upstreamKey || !reviewerToken) return fail(503, 'the proxy is not configured');

  const presented = bearerToken(request.headers.get('authorization'));
  if (presented === null || !sameSecret(presented, reviewerToken)) {
    return fail(401, 'expected "Authorization: Bearer <reviewer token>"');
  }

  // Declared length first (cheap), then what actually arrived.
  const declared = Number(request.headers.get('content-length') ?? '');
  if (declared > route.maxBytes) return fail(413, `body exceeds ${route.maxBytes} bytes`);
  const body = Buffer.from(await request.arrayBuffer());
  if (body.length > route.maxBytes) return fail(413, `body exceeds ${route.maxBytes} bytes`);
  if (body.length === 0) return fail(400, 'the request had no body');

  const contentType = request.headers.get('content-type') ?? '';
  const refusal = route.check(body, contentType);
  if (refusal) return fail(400, refusal);

  // Headers are built from scratch: nothing the client sent is forwarded,
  // least of all its Authorization, and the host is a constant.
  let upstream;
  try {
    upstream = await fetchImpl(`${UPSTREAM}${endpoint}`, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${upstreamKey}`,
        'Content-Type': route.json ? 'application/json' : contentType,
      },
      body,
      signal: AbortSignal.timeout(28_000),
    });
  } catch (exc) {
    // Never repr/stack: a fetch error carries the request headers.
    console.error('[proxy] upstream failed:', redact(String(exc?.name), upstreamKey));
    return fail(502, 'the upstream request failed');
  }

  console.log(`[proxy] ${endpoint} in=${body.length}B status=${upstream.status}`);
  return relay(upstream, upstreamKey);
}

/** Vercel's rewrite prefixes /api; both spellings normalise to one endpoint. */
function normalizePath(rawUrl) {
  let path;
  try {
    path = new URL(rawUrl, 'http://proxy.invalid').pathname;
  } catch {
    return '';
  }
  if (path.startsWith('/api/')) path = path.slice(4);
  return path.length > 1 && path.endsWith('/') ? path.slice(0, -1) : path;
}

// -- validation: returns a refusal string, or null to allow -----------------
// The contents of `text` and `input` are never inspected or rewritten -- the
// local app owns those schemas and validates every answer itself.

function checkResponses(body, contentType) {
  const payload = parseJson(body, contentType);
  if (typeof payload === 'string') return payload;

  const extra = Object.keys(payload).filter((k) => !RESPONSES_KEYS.includes(k));
  if (extra.length) return `unsupported parameters: ${extra.sort().join(', ')}`;
  if (payload.model !== RESPONSES_MODEL) return 'that model is not enabled on this proxy';
  const cap = payload.max_output_tokens;
  if (!Number.isInteger(cap) || cap < 1 || cap > MAX_OUTPUT_TOKENS) {
    return `max_output_tokens must be an integer in 1..${MAX_OUTPUT_TOKENS}`;
  }
  return null;
}

function checkSpeech(body, contentType) {
  const payload = parseJson(body, contentType);
  if (typeof payload === 'string') return payload;

  const extra = Object.keys(payload).filter((k) => !SPEECH_KEYS.includes(k));
  if (extra.length) return `unsupported parameters: ${extra.sort().join(', ')}`;
  if (payload.model !== TTS_MODEL) return 'that model is not enabled on this proxy';
  if (payload.voice !== TTS_VOICE) return 'that voice is not enabled on this proxy';
  if (payload.response_format !== undefined && payload.response_format !== 'wav') {
    return 'that response_format is not enabled on this proxy';
  }
  if (typeof payload.input !== 'string' || !payload.input.trim()) {
    return 'input must be a non-empty string';
  }
  if (payload.input.length > MAX_SPEECH_INPUT_CHARS) {
    return `input is longer than ${MAX_SPEECH_INPUT_CHARS} characters`;
  }
  return null;
}

function checkTranscription(body, contentType) {
  const boundary = /^multipart\/form-data\s*;.*boundary=(?:"([^"]+)"|([^;]+))/i.exec(contentType);
  if (!boundary) return 'expected multipart/form-data with a boundary';
  const fields = readTextFields(body, (boundary[1] ?? boundary[2]).trim());
  if (!fields) return 'the multipart body could not be read';

  const extra = fields.names.filter((n) => !STT_FIELDS.includes(n));
  if (extra.length) return `unsupported fields: ${extra.sort().join(', ')}`;
  if (fields.values.model !== STT_MODEL) return 'that model is not enabled on this proxy';
  const format = fields.values.response_format;
  if (format !== undefined && format !== 'json') {
    return 'that response_format is not enabled on this proxy';
  }
  return null;
}

function parseJson(body, contentType) {
  if (!/^application\/json\b/i.test(contentType)) return 'expected application/json';
  let value;
  try {
    value = JSON.parse(body.toString('utf8'));
  } catch {
    return 'the request body was not valid JSON';
  }
  if (value === null || typeof value !== 'object' || Array.isArray(value)) {
    return 'the request body was not a JSON object';
  }
  return value;
}

/**
 * Read the scalar form fields out of a multipart body. Parts with a filename
 * (the audio) are recorded by name only; the body is forwarded unmodified
 * either way, so this is a read, never a rewrite.
 */
function readTextFields(body, boundary) {
  const delimiter = Buffer.from(`--${boundary}`);
  const names = [];
  const values = Object.create(null);
  let cursor = body.indexOf(delimiter);
  if (cursor === -1) return null;

  while (cursor !== -1) {
    let start = cursor + delimiter.length;
    if (body.slice(start, start + 2).toString('latin1') === '--') break;
    if (body.slice(start, start + 2).toString('latin1') === '\r\n') start += 2;

    const next = body.indexOf(delimiter, start);
    const end = next === -1 ? body.length : next;
    const headerEnd = body.indexOf('\r\n\r\n', start);
    if (headerEnd === -1 || headerEnd > end) return null;

    const headers = body.slice(start, headerEnd).toString('utf8');
    const named = /name="([^"]*)"/i.exec(headers);
    if (!named) return null;
    names.push(named[1]);

    if (!/filename="/i.test(headers)) {
      const raw = body.slice(headerEnd + 4, Math.max(headerEnd + 4, end - 2));
      if (raw.length > 256) return null; // no hostile "field value"
      values[named[1]] = raw.toString('utf8');
    }
    cursor = next;
  }
  return names.length ? { names, values } : null;
}

// -- credentials ------------------------------------------------------------

function bearerToken(header) {
  const match = typeof header === 'string' ? /^Bearer[ \t]+(\S.*)$/i.exec(header.trim()) : null;
  return match ? match[1].trim() : null;
}

/** Constant-time compare via fixed-length digests. */
function sameSecret(a, b) {
  const digest = (t) => createHash('sha256').update(t, 'utf8').digest();
  return timingSafeEqual(digest(a), digest(b));
}

/** Strip the server key -- exact value and key-shaped tokens -- from text. */
export function redact(text, secret) {
  if (!text) return text;
  const out = secret ? String(text).split(secret).join('<redacted>') : String(text);
  return out.replace(/\bsk-[A-Za-z0-9_-]{8,}/g, '<redacted>');
}

// -- responses --------------------------------------------------------------

/**
 * Relay with a header allowlist: only content-type survives, so nothing that
 * identifies the account behind the key is passed on. A failing call can quote
 * the key back in its JSON body, so text is redacted; successful binary (the
 * TTS wav) passes through untouched.
 */
async function relay(upstream, upstreamKey) {
  const contentType = upstream.headers.get('content-type') ?? 'application/octet-stream';
  const headers = { 'Content-Type': contentType, 'Cache-Control': 'no-store' };
  if (upstream.ok && !/json|text|xml/i.test(contentType)) {
    return new Response(Buffer.from(await upstream.arrayBuffer()), {
      status: upstream.status,
      headers,
    });
  }
  return new Response(redact(await upstream.text(), upstreamKey), {
    status: upstream.status,
    headers,
  });
}

/** Shaped like an OpenAI error, carrying only text written in this file. */
function fail(status, message, extraHeaders = {}) {
  return new Response(JSON.stringify({ error: { message, type: 'hcl_reviewer_proxy' } }), {
    status,
    headers: { 'Content-Type': 'application/json', 'Cache-Control': 'no-store', ...extraHeaders },
  });
}
