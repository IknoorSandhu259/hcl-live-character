/**
 * Reviewer-proxy tests. No network, no Vercel, no API quota.
 *
 *   node --test test/
 *
 * `handleRequest` takes a Web Request and an env object and returns a Web
 * Response, so every case here is a plain function call with a stub `fetch`
 * standing in for api.openai.com. The stub records exactly what it was asked
 * to send, which is how the "the reviewer cannot supply the upstream key"
 * assertions are made.
 */

import test from 'node:test';
import assert from 'node:assert/strict';

import { handleRequest, redact, _resetRateLimit } from '../lib/proxy.js';

// --------------------------------------------------------------------------
// Fixtures
// --------------------------------------------------------------------------

const SERVER_KEY = 'sk-proj-REAL-SERVER-KEY-must-never-escape-0123456789';
const TOKEN = 'hcl-reviewer-3f9a2c7e1b4d';

const ENV = {
  OPENAI_EVAL_API_KEY: SERVER_KEY,
  HCL_REVIEWER_TOKENS: TOKEN,
};

const HOST = 'https://proxy.example.com';

/** A stub upstream that records its calls and replies with `reply`. */
function stubFetch(reply = { status: 200, body: '{"ok":true}', contentType: 'application/json' }) {
  const calls = [];
  const impl = async (url, init) => {
    calls.push({ url, init });
    return new Response(reply.body, {
      status: reply.status,
      headers: {
        'content-type': reply.contentType,
        'openai-organization': 'org-secret',
        'set-cookie': 'session=nope',
      },
    });
  };
  impl.calls = calls;
  return impl;
}

function post(path, { body, contentType, token = TOKEN, method = 'POST', headers = {} } = {}) {
  const all = { ...headers };
  if (token !== null) all.Authorization = `Bearer ${token}`;
  if (contentType) all['Content-Type'] = contentType;
  return new Request(`${HOST}${path}`, { method, headers: all, body });
}

function jsonPost(path, payload, options = {}) {
  return post(path, {
    body: JSON.stringify(payload),
    contentType: 'application/json',
    ...options,
  });
}

const GOOD_RESPONSES = {
  model: 'gpt-5.6-luna',
  instructions: 'You are the voice of a small desk lamp robot.',
  input: 'hello there',
  text: { format: { type: 'json_schema', name: 'lamp_turn', strict: true, schema: {} } },
  max_output_tokens: 300,
};

const GOOD_SPEECH = {
  model: 'tts-1',
  voice: 'alloy',
  input: 'Hello, I am a lamp.',
  response_format: 'wav',
};

/** Build a multipart body shaped like the one the OpenAI SDK sends for STT. */
function multipart(fields, { boundary = 'formboundary123' } = {}) {
  const chunks = [];
  for (const [name, value] of Object.entries(fields)) {
    if (value && value.filename !== undefined) {
      chunks.push(
        Buffer.from(
          `--${boundary}\r\nContent-Disposition: form-data; name="${name}"; ` +
            `filename="${value.filename}"\r\nContent-Type: audio/wav\r\n\r\n`,
        ),
        Buffer.from(value.bytes),
        Buffer.from('\r\n'),
      );
    } else {
      chunks.push(
        Buffer.from(
          `--${boundary}\r\nContent-Disposition: form-data; name="${name}"\r\n\r\n${value}\r\n`,
        ),
      );
    }
  }
  chunks.push(Buffer.from(`--${boundary}--\r\n`));
  return {
    body: Buffer.concat(chunks),
    contentType: `multipart/form-data; boundary=${boundary}`,
  };
}

function goodTranscription(fields = {}) {
  return multipart({
    file: { filename: 'utterance.wav', bytes: Buffer.alloc(2048, 7) },
    model: 'gpt-4o-mini-transcribe',
    response_format: 'json',
    ...fields,
  });
}

async function errorOf(response) {
  return (await response.json()).error;
}

test.beforeEach(() => _resetRateLimit());

// --------------------------------------------------------------------------
// 1. The happy path, for each of the three allowed operations
// --------------------------------------------------------------------------

test('valid token + allowed endpoint reaches the upstream', async () => {
  const fetchImpl = stubFetch();
  const response = await handleRequest(
    jsonPost('/v1/responses', GOOD_RESPONSES),
    ENV,
    { fetch: fetchImpl },
  );

  assert.equal(response.status, 200);
  assert.deepEqual(await response.json(), { ok: true });
  assert.equal(fetchImpl.calls.length, 1);
  assert.equal(fetchImpl.calls[0].url, 'https://api.openai.com/v1/responses');
});

test('the Vercel rewrite spelling /api/v1/... routes identically', async () => {
  const fetchImpl = stubFetch();
  const response = await handleRequest(
    jsonPost('/api/v1/responses', GOOD_RESPONSES),
    ENV,
    { fetch: fetchImpl },
  );
  assert.equal(response.status, 200);
  assert.equal(fetchImpl.calls[0].url, 'https://api.openai.com/v1/responses');
});

test('speech-to-text is allowed as multipart', async () => {
  const fetchImpl = stubFetch();
  const { body, contentType } = goodTranscription();
  const response = await handleRequest(
    post('/v1/audio/transcriptions', { body, contentType }),
    ENV,
    { fetch: fetchImpl },
  );

  assert.equal(response.status, 200);
  assert.equal(fetchImpl.calls[0].url, 'https://api.openai.com/v1/audio/transcriptions');
  // Forwarded byte-for-byte, boundary intact.
  assert.ok(Buffer.from(fetchImpl.calls[0].init.body).equals(body));
  assert.equal(fetchImpl.calls[0].init.headers['Content-Type'], contentType);
});

test('text-to-speech is allowed and binary audio is relayed unchanged', async () => {
  const wav = Buffer.from('RIFF....WAVEfmt ');
  const fetchImpl = stubFetch({ status: 200, body: wav, contentType: 'audio/wav' });
  const response = await handleRequest(
    jsonPost('/v1/audio/speech', GOOD_SPEECH),
    ENV,
    { fetch: fetchImpl },
  );

  assert.equal(response.status, 200);
  assert.equal(response.headers.get('content-type'), 'audio/wav');
  assert.ok(Buffer.from(await response.arrayBuffer()).equals(wav));
});

// --------------------------------------------------------------------------
// 2 & 3. Authentication
// --------------------------------------------------------------------------

test('a missing Authorization header is refused', async () => {
  const fetchImpl = stubFetch();
  const response = await handleRequest(
    jsonPost('/v1/responses', GOOD_RESPONSES, { token: null }),
    ENV,
    { fetch: fetchImpl },
  );

  assert.equal(response.status, 401);
  assert.equal((await errorOf(response)).code, 'invalid_authorization');
  assert.equal(fetchImpl.calls.length, 0);
});

test('a malformed Authorization header is refused', async () => {
  const fetchImpl = stubFetch();
  for (const header of ['', 'Bearer', 'Bearer   ', TOKEN, `Basic ${TOKEN}`, 'Bearer\t']) {
    const response = await handleRequest(
      jsonPost('/v1/responses', GOOD_RESPONSES, { token: null, headers: { Authorization: header } }),
      ENV,
      { fetch: fetchImpl },
    );
    assert.equal(response.status, 401, `header ${JSON.stringify(header)} should be refused`);
  }
  assert.equal(fetchImpl.calls.length, 0);
});

test('a wrong token is refused', async () => {
  const fetchImpl = stubFetch();
  for (const wrong of ['nope', `${TOKEN}x`, TOKEN.slice(0, -1), TOKEN.toUpperCase(), SERVER_KEY]) {
    const response = await handleRequest(
      jsonPost('/v1/responses', GOOD_RESPONSES, { token: wrong }),
      ENV,
      { fetch: fetchImpl },
    );
    assert.equal(response.status, 401);
    assert.equal((await errorOf(response)).code, 'invalid_token');
  }
  assert.equal(fetchImpl.calls.length, 0);
});

test('several tokens can be live at once, so rotation never has a gap', async () => {
  const fetchImpl = stubFetch();
  const env = { ...ENV, HCL_REVIEWER_TOKENS: ` ${TOKEN} , second-token ` };
  for (const token of [TOKEN, 'second-token']) {
    const response = await handleRequest(
      jsonPost('/v1/responses', GOOD_RESPONSES, { token }),
      env,
      { fetch: fetchImpl },
    );
    assert.equal(response.status, 200);
  }
  const revoked = await handleRequest(
    jsonPost('/v1/responses', GOOD_RESPONSES, { token: 'third-token' }),
    env,
    { fetch: fetchImpl },
  );
  assert.equal(revoked.status, 401);
});

test('an expired token is refused', async () => {
  const fetchImpl = stubFetch();
  const env = { ...ENV, HCL_REVIEWER_TOKEN_EXPIRES_AT: '2026-01-01T00:00:00Z' };
  const response = await handleRequest(jsonPost('/v1/responses', GOOD_RESPONSES), env, {
    fetch: fetchImpl,
    now: () => Date.parse('2026-01-02T00:00:00Z'),
  });

  assert.equal(response.status, 401);
  assert.equal((await errorOf(response)).code, 'token_expired');
  assert.equal(fetchImpl.calls.length, 0);
});

test('an unconfigured proxy fails closed rather than open', async () => {
  const fetchImpl = stubFetch();
  const broken = [
    { HCL_REVIEWER_TOKENS: TOKEN }, // no upstream key
    { OPENAI_EVAL_API_KEY: SERVER_KEY }, // no reviewer token
    {}, // nothing at all
    { ...ENV, HCL_REVIEWER_TOKEN_EXPIRES_AT: 'not-a-date' },
  ];
  for (const env of broken) {
    const response = await handleRequest(jsonPost('/v1/responses', GOOD_RESPONSES), env, {
      fetch: fetchImpl,
    });
    assert.equal(response.status, 503);
  }
  assert.equal(fetchImpl.calls.length, 0);
});

// --------------------------------------------------------------------------
// 4 & 5. Endpoint and method
// --------------------------------------------------------------------------

test('unsupported endpoints are refused', async () => {
  const fetchImpl = stubFetch();
  const paths = [
    '/v1/chat/completions',
    '/v1/embeddings',
    '/v1/models',
    '/v1/files',
    '/v1/fine_tuning/jobs',
    '/v1/images/generations',
    '/v1/responses/resp_123',
    '/v1/audio/translations',
    '/responses',
    '/',
    '/v1/responses/../../v1/chat/completions',
  ];
  for (const path of paths) {
    const response = await handleRequest(jsonPost(path, GOOD_RESPONSES), ENV, { fetch: fetchImpl });
    assert.equal(response.status, 404, `${path} should be refused`);
    assert.equal((await errorOf(response)).code, 'unsupported_endpoint');
  }
  assert.equal(fetchImpl.calls.length, 0);
});

test('a query string cannot smuggle a second endpoint', async () => {
  const fetchImpl = stubFetch();
  const response = await handleRequest(
    jsonPost('/v1/responses?path=/v1/chat/completions', GOOD_RESPONSES),
    ENV,
    { fetch: fetchImpl },
  );
  assert.equal(response.status, 200);
  assert.equal(fetchImpl.calls[0].url, 'https://api.openai.com/v1/responses');
});

test('unsupported methods are refused', async () => {
  const fetchImpl = stubFetch();
  for (const method of ['GET', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS']) {
    const request = post('/v1/responses', { method });
    const response = await handleRequest(request, ENV, { fetch: fetchImpl });
    assert.equal(response.status, 405, `${method} should be refused`);
    assert.equal(response.headers.get('allow'), 'POST');
  }
  assert.equal(fetchImpl.calls.length, 0);
});

// --------------------------------------------------------------------------
// 6. Models and parameters
// --------------------------------------------------------------------------

test('an unsupported model on /v1/responses is refused', async () => {
  const fetchImpl = stubFetch();
  for (const model of ['gpt-4o', 'o3', 'gpt-5.6-luna-extra', '', null, undefined, 42]) {
    const response = await handleRequest(
      jsonPost('/v1/responses', { ...GOOD_RESPONSES, model }),
      ENV,
      { fetch: fetchImpl },
    );
    assert.equal(response.status, 400, `${model} should be refused`);
    assert.equal((await errorOf(response)).code, 'unsupported_model');
  }
  assert.equal(fetchImpl.calls.length, 0);
});

test('the responses model allowlist is env-configurable for LAMP_VISION_MODEL', async () => {
  const fetchImpl = stubFetch();
  const env = { ...ENV, HCL_ALLOWED_RESPONSES_MODELS: 'gpt-5.6-luna, gpt-4o-mini' };
  const allowed = await handleRequest(
    jsonPost('/v1/responses', { ...GOOD_RESPONSES, model: 'gpt-4o-mini' }),
    env,
    { fetch: fetchImpl },
  );
  assert.equal(allowed.status, 200);

  const refused = await handleRequest(
    jsonPost('/v1/responses', { ...GOOD_RESPONSES, model: 'gpt-4o' }),
    env,
    { fetch: fetchImpl },
  );
  assert.equal(refused.status, 400);
});

test('an unsupported STT model is refused', async () => {
  const fetchImpl = stubFetch();
  const { body, contentType } = goodTranscription({ model: 'whisper-1' });
  const response = await handleRequest(
    post('/v1/audio/transcriptions', { body, contentType }),
    ENV,
    { fetch: fetchImpl },
  );
  assert.equal(response.status, 400);
  assert.equal((await errorOf(response)).code, 'unsupported_model');
  assert.equal(fetchImpl.calls.length, 0);
});

test('an unsupported TTS model or voice is refused', async () => {
  const fetchImpl = stubFetch();
  for (const payload of [
    { ...GOOD_SPEECH, model: 'gpt-4o-mini-tts' },
    { ...GOOD_SPEECH, voice: 'nova' },
    { ...GOOD_SPEECH, response_format: 'mp3' },
  ]) {
    const response = await handleRequest(jsonPost('/v1/audio/speech', payload), ENV, {
      fetch: fetchImpl,
    });
    assert.equal(response.status, 400);
  }
  assert.equal(fetchImpl.calls.length, 0);
});

test('this is not a general relay: extra parameters are refused', async () => {
  const fetchImpl = stubFetch();
  const payloads = [
    { ...GOOD_RESPONSES, stream: true },
    { ...GOOD_RESPONSES, tools: [{ type: 'function', name: 'move_joint' }] },
    { ...GOOD_RESPONSES, tool_choice: 'auto' },
    { ...GOOD_RESPONSES, store: true },
    { ...GOOD_RESPONSES, background: true },
    { ...GOOD_RESPONSES, previous_response_id: 'resp_1' },
    { ...GOOD_RESPONSES, metadata: { x: 'y' } },
    { ...GOOD_RESPONSES, max_output_tokens: 100000 },
    { ...GOOD_RESPONSES, max_output_tokens: 'lots' },
  ];
  for (const payload of payloads) {
    const response = await handleRequest(jsonPost('/v1/responses', payload), ENV, {
      fetch: fetchImpl,
    });
    assert.equal(response.status, 400, `${JSON.stringify(payload).slice(0, 60)} should be refused`);
    assert.equal((await errorOf(response)).code, 'unsupported_parameter');
  }
  assert.equal(fetchImpl.calls.length, 0);
});

test('extra multipart fields are refused', async () => {
  const fetchImpl = stubFetch();
  const { body, contentType } = goodTranscription({ stream: 'true' });
  const response = await handleRequest(
    post('/v1/audio/transcriptions', { body, contentType }),
    ENV,
    { fetch: fetchImpl },
  );
  assert.equal(response.status, 400);
  assert.equal((await errorOf(response)).code, 'unsupported_parameter');
  assert.equal(fetchImpl.calls.length, 0);
});

test('a wrong or malformed content type is refused', async () => {
  const fetchImpl = stubFetch();
  const cases = [
    post('/v1/responses', { body: 'x=1', contentType: 'application/x-www-form-urlencoded' }),
    post('/v1/responses', { body: '{not json', contentType: 'application/json' }),
    post('/v1/responses', { body: '[1,2]', contentType: 'application/json' }),
    post('/v1/audio/transcriptions', { body: '{}', contentType: 'application/json' }),
    post('/v1/audio/transcriptions', { body: 'x', contentType: 'multipart/form-data' }),
    post('/v1/responses', { body: '', contentType: 'application/json' }),
  ];
  for (const request of cases) {
    const response = await handleRequest(request, ENV, { fetch: fetchImpl });
    assert.equal(response.status, 400);
  }
  assert.equal(fetchImpl.calls.length, 0);
});

// --------------------------------------------------------------------------
// 7. Size
// --------------------------------------------------------------------------

test('an oversized body is refused', async () => {
  const fetchImpl = stubFetch();

  // A frame far larger than attention.OBSERVE_MAX_BYTES allows.
  const huge = await handleRequest(
    jsonPost('/v1/responses', { ...GOOD_RESPONSES, input: 'A'.repeat(1_300_000) }),
    ENV,
    { fetch: fetchImpl },
  );
  assert.equal(huge.status, 413);
  assert.equal((await errorOf(huge)).code, 'payload_too_large');

  // A reply far longer than character.MAX_REPLY_CHARS allows.
  const chatty = await handleRequest(
    jsonPost('/v1/audio/speech', { ...GOOD_SPEECH, input: 'la '.repeat(4000) }),
    ENV,
    { fetch: fetchImpl },
  );
  assert.equal(chatty.status, 413);

  // A recording far longer than audio_io.RECORD_SECONDS allows.
  const long = goodTranscription({
    file: { filename: 'utterance.wav', bytes: Buffer.alloc(1_100_000, 1) },
  });
  const recorded = await handleRequest(
    post('/v1/audio/transcriptions', long),
    ENV,
    { fetch: fetchImpl },
  );
  assert.equal(recorded.status, 413);

  assert.equal(fetchImpl.calls.length, 0);
});

test('a lying Content-Length is rejected before the body is buffered', async () => {
  const fetchImpl = stubFetch();
  const response = await handleRequest(
    jsonPost('/v1/audio/speech', GOOD_SPEECH, { headers: { 'Content-Length': '900000' } }),
    ENV,
    { fetch: fetchImpl },
  );
  assert.equal(response.status, 413);
  assert.equal(fetchImpl.calls.length, 0);
});

test('a spoken reply just over the character limit is refused', async () => {
  const fetchImpl = stubFetch();
  const response = await handleRequest(
    jsonPost('/v1/audio/speech', { ...GOOD_SPEECH, input: 'a'.repeat(501) }),
    ENV,
    { fetch: fetchImpl },
  );
  assert.equal(response.status, 400);
  assert.equal(fetchImpl.calls.length, 0);
});

// --------------------------------------------------------------------------
// 8. The reviewer cannot reach or replace the upstream credential
// --------------------------------------------------------------------------

test('the upstream Authorization is always the server key, never the client header', async () => {
  const fetchImpl = stubFetch();
  const request = jsonPost('/v1/responses', GOOD_RESPONSES, {
    headers: {
      'X-Forwarded-Authorization': 'Bearer sk-attacker-key',
      'OpenAI-Organization': 'org-attacker',
      'OpenAI-Project': 'proj-attacker',
      Cookie: 'a=b',
      'X-Api-Key': 'sk-attacker-key',
    },
  });
  const response = await handleRequest(request, ENV, { fetch: fetchImpl });

  assert.equal(response.status, 200);
  const sent = fetchImpl.calls[0].init.headers;
  assert.equal(sent.Authorization, `Bearer ${SERVER_KEY}`);
  // Only headers this proxy authored are forwarded.
  assert.deepEqual(Object.keys(sent).sort(), [
    'Accept',
    'Authorization',
    'Content-Type',
    'User-Agent',
  ]);
});

test('the reviewer token never becomes the upstream credential', async () => {
  const fetchImpl = stubFetch();
  await handleRequest(jsonPost('/v1/responses', GOOD_RESPONSES), ENV, { fetch: fetchImpl });
  const sent = fetchImpl.calls[0].init.headers;
  assert.ok(!sent.Authorization.includes(TOKEN));
});

test('the client cannot choose the upstream host', async () => {
  const fetchImpl = stubFetch();
  const request = jsonPost('/v1/responses', GOOD_RESPONSES, {
    headers: {
      Host: 'evil.example.com',
      'X-Forwarded-Host': 'evil.example.com',
      'X-Target-Url': 'https://evil.example.com/v1/responses',
    },
  });
  await handleRequest(request, ENV, { fetch: fetchImpl });
  assert.ok(fetchImpl.calls[0].url.startsWith('https://api.openai.com/'));
});

// --------------------------------------------------------------------------
// 9. Nothing leaks the server key
// --------------------------------------------------------------------------

test('an upstream error body that quotes the key is redacted', async () => {
  const fetchImpl = stubFetch({
    status: 401,
    contentType: 'application/json',
    body: JSON.stringify({
      error: { message: `Incorrect API key provided: ${SERVER_KEY}. Check your key.` },
    }),
  });
  const response = await handleRequest(jsonPost('/v1/responses', GOOD_RESPONSES), ENV, {
    fetch: fetchImpl,
  });

  assert.equal(response.status, 401);
  const text = await response.text();
  assert.ok(!text.includes(SERVER_KEY));
  assert.ok(text.includes('<redacted>'));
});

test('no refusal or transport failure mentions the key', async () => {
  const exploding = async () => {
    throw new Error(`connect ECONNREFUSED with Authorization: Bearer ${SERVER_KEY}`);
  };
  const responses = await Promise.all([
    handleRequest(jsonPost('/v1/responses', GOOD_RESPONSES, { token: null }), ENV, {
      fetch: stubFetch(),
    }),
    handleRequest(jsonPost('/v1/responses', GOOD_RESPONSES, { token: 'wrong' }), ENV, {
      fetch: stubFetch(),
    }),
    handleRequest(jsonPost('/v1/chat/completions', {}), ENV, { fetch: stubFetch() }),
    handleRequest(jsonPost('/v1/responses', { ...GOOD_RESPONSES, model: 'gpt-4o' }), ENV, {
      fetch: stubFetch(),
    }),
    handleRequest(jsonPost('/v1/responses', GOOD_RESPONSES), ENV, { fetch: exploding }),
  ]);
  for (const response of responses) {
    const text = await response.text();
    assert.ok(!text.includes(SERVER_KEY), `leaked in: ${text}`);
    assert.ok(!text.includes('sk-'), `key-shaped text in: ${text}`);
  }
});

test('upstream response headers that identify the account are stripped', async () => {
  const fetchImpl = stubFetch();
  const response = await handleRequest(jsonPost('/v1/responses', GOOD_RESPONSES), ENV, {
    fetch: fetchImpl,
  });
  assert.equal(response.headers.get('openai-organization'), null);
  assert.equal(response.headers.get('set-cookie'), null);
  assert.equal(response.headers.get('cache-control'), 'no-store');
});

test('redact() strips both the exact key and anything key-shaped', () => {
  assert.equal(redact(`a ${SERVER_KEY} b`, SERVER_KEY), 'a <redacted> b');
  assert.equal(redact('a sk-someOtherKey123456 b', SERVER_KEY), 'a <redacted> b');
  assert.equal(redact('nothing here', SERVER_KEY), 'nothing here');
});

// --------------------------------------------------------------------------
// Rate limiting
// --------------------------------------------------------------------------

test('a runaway client is throttled', async () => {
  const fetchImpl = stubFetch();
  const env = { ...ENV, HCL_PROXY_RATE_LIMIT: '3/60' };
  const statuses = [];
  for (let i = 0; i < 5; i += 1) {
    const response = await handleRequest(jsonPost('/v1/responses', GOOD_RESPONSES), env, {
      fetch: fetchImpl,
    });
    statuses.push(response.status);
  }
  assert.deepEqual(statuses, [200, 200, 200, 429, 429]);
  assert.equal(fetchImpl.calls.length, 3);
});

test('the throttle window reopens', async () => {
  const fetchImpl = stubFetch();
  const env = { ...ENV, HCL_PROXY_RATE_LIMIT: '1/60' };
  let clock = 1_000_000;
  const deps = { fetch: fetchImpl, now: () => clock };

  assert.equal((await handleRequest(jsonPost('/v1/responses', GOOD_RESPONSES), env, deps)).status, 200);
  assert.equal((await handleRequest(jsonPost('/v1/responses', GOOD_RESPONSES), env, deps)).status, 429);
  clock += 61_000;
  assert.equal((await handleRequest(jsonPost('/v1/responses', GOOD_RESPONSES), env, deps)).status, 200);
});
