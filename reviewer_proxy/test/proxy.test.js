/**
 * Reviewer-proxy tests: no network, no Vercel, no API quota.
 *
 *   node --test
 *
 * The stub fetch records exactly what would have gone upstream, which is how
 * the "the reviewer can never supply or override the server key" cases are
 * asserted.
 */

import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

import { handleRequest, redact } from '../lib/proxy.js';

const SERVER_KEY = 'sk-proj-REAL-SERVER-KEY-must-never-escape-0123456789';
const TOKEN = 'hclrev_3f9a2c7e1b4d';
const ENV = { OPENAI_EVAL_API_KEY: SERVER_KEY, HCL_REVIEWER_TOKEN: TOKEN };

const RESPONSES = {
  model: 'gpt-5.6-luna',
  instructions: 'You are the voice of a small desk lamp robot.',
  input: 'hello there',
  text: { format: { type: 'json_schema', name: 'lamp_turn', strict: true, schema: {} } },
  max_output_tokens: 300,
};
const SPEECH = { model: 'tts-1', voice: 'alloy', input: 'Hello.', response_format: 'wav' };

/** Stub upstream that records its calls. */
function stub(reply = { status: 200, body: '{"ok":true}', contentType: 'application/json' }) {
  const calls = [];
  const impl = async (url, init) => {
    calls.push({ url, init });
    return new Response(reply.body, {
      status: reply.status,
      headers: { 'content-type': reply.contentType, 'openai-organization': 'org-secret' },
    });
  };
  impl.calls = calls;
  return impl;
}

function post(path, { body, contentType, token = TOKEN, method = 'POST', headers = {} } = {}) {
  const all = { ...headers };
  if (token !== null) all.Authorization = `Bearer ${token}`;
  if (contentType) all['Content-Type'] = contentType;
  return new Request(`https://proxy.example.com${path}`, { method, headers: all, body });
}

const jsonPost = (path, payload, options = {}) =>
  post(path, { body: JSON.stringify(payload), contentType: 'application/json', ...options });

/** A multipart body shaped like the one the OpenAI SDK sends for STT. */
function multipart(fields) {
  const boundary = 'formboundary123';
  const chunks = [];
  for (const [name, value] of Object.entries(fields)) {
    if (value?.filename !== undefined) {
      chunks.push(
        Buffer.from(
          `--${boundary}\r\nContent-Disposition: form-data; name="${name}"; ` +
            `filename="${value.filename}"\r\nContent-Type: audio/wav\r\n\r\n`,
        ),
        value.bytes,
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
  return { body: Buffer.concat(chunks), contentType: `multipart/form-data; boundary=${boundary}` };
}

const transcription = (fields = {}) =>
  multipart({
    file: { filename: 'utterance.wav', bytes: Buffer.alloc(2048, 7) },
    model: 'gpt-4o-mini-transcribe',
    response_format: 'json',
    ...fields,
  });

// -- deployment routing -----------------------------------------------------

test('each allowed endpoint has a static Vercel entrypoint file', () => {
  // A catch-all `api/[...path].js` compiles to a single-segment matcher under
  // the zero-config /api builder, so `/api/v1/responses` fell through to the
  // platform's blanket 404. Static paths are matched by the filesystem
  // handler and cannot regress that way.
  for (const endpoint of ['/v1/responses', '/v1/audio/speech', '/v1/audio/transcriptions']) {
    const entrypoint = new URL(`../api${endpoint}.js`, import.meta.url);
    assert.ok(existsSync(fileURLToPath(entrypoint)), `missing api${endpoint}.js`);
  }
});

// -- the three allowed operations ------------------------------------------

test('the three demo operations reach the upstream', async () => {
  const fetchImpl = stub();
  const stt = transcription();
  const requests = [
    jsonPost('/v1/responses', RESPONSES),
    jsonPost('/v1/audio/speech', SPEECH),
    post('/v1/audio/transcriptions', stt),
  ];
  for (const request of requests) {
    assert.equal((await handleRequest(request, ENV, fetchImpl)).status, 200);
  }
  assert.deepEqual(
    fetchImpl.calls.map((c) => c.url),
    [
      'https://api.openai.com/v1/responses',
      'https://api.openai.com/v1/audio/speech',
      'https://api.openai.com/v1/audio/transcriptions',
    ],
  );
  // Multipart is forwarded byte-for-byte, boundary intact.
  assert.ok(Buffer.from(fetchImpl.calls[2].init.body).equals(stt.body));
  assert.equal(fetchImpl.calls[2].init.headers['Content-Type'], stt.contentType);
});

test('the Vercel rewrite spelling /api/v1/... routes identically', async () => {
  const fetchImpl = stub();
  assert.equal((await handleRequest(jsonPost('/api/v1/responses', RESPONSES), ENV, fetchImpl)).status, 200);
  assert.equal(fetchImpl.calls[0].url, 'https://api.openai.com/v1/responses');
});

test('binary audio is relayed unchanged', async () => {
  const wav = Buffer.from('RIFF....WAVEfmt ');
  const fetchImpl = stub({ status: 200, body: wav, contentType: 'audio/wav' });
  const response = await handleRequest(jsonPost('/v1/audio/speech', SPEECH), ENV, fetchImpl);
  assert.ok(Buffer.from(await response.arrayBuffer()).equals(wav));
});

// -- authentication ---------------------------------------------------------

test('missing, malformed and wrong authorization are all refused', async () => {
  const fetchImpl = stub();
  const headers = ['', 'Bearer', 'Bearer   ', TOKEN, `Basic ${TOKEN}`];
  const tokens = ['nope', `${TOKEN}x`, TOKEN.slice(0, -1), TOKEN.toUpperCase(), SERVER_KEY];

  const none = await handleRequest(jsonPost('/v1/responses', RESPONSES, { token: null }), ENV, fetchImpl);
  assert.equal(none.status, 401);
  for (const header of headers) {
    const request = jsonPost('/v1/responses', RESPONSES, { token: null, headers: { Authorization: header } });
    assert.equal((await handleRequest(request, ENV, fetchImpl)).status, 401, header);
  }
  for (const token of tokens) {
    const request = jsonPost('/v1/responses', RESPONSES, { token });
    assert.equal((await handleRequest(request, ENV, fetchImpl)).status, 401, token);
  }
  assert.equal(fetchImpl.calls.length, 0);
});

test('an unconfigured proxy fails closed rather than open', async () => {
  const fetchImpl = stub();
  for (const env of [{ HCL_REVIEWER_TOKEN: TOKEN }, { OPENAI_EVAL_API_KEY: SERVER_KEY }, {}]) {
    assert.equal((await handleRequest(jsonPost('/v1/responses', RESPONSES), env, fetchImpl)).status, 503);
  }
  assert.equal(fetchImpl.calls.length, 0);
});

// -- route, method, model, size ---------------------------------------------

test('unsupported endpoints are refused', async () => {
  const fetchImpl = stub();
  const paths = [
    '/v1/chat/completions',
    '/v1/embeddings',
    '/v1/models',
    '/v1/files',
    '/v1/images/generations',
    '/v1/responses/resp_123',
    '/v1/audio/translations',
    '/responses',
    '/',
    '/v1/responses/../../v1/chat/completions',
  ];
  for (const path of paths) {
    assert.equal((await handleRequest(jsonPost(path, RESPONSES), ENV, fetchImpl)).status, 404, path);
  }
  // A query string cannot smuggle a second endpoint.
  const ok = await handleRequest(jsonPost('/v1/responses?p=/v1/chat/completions', RESPONSES), ENV, fetchImpl);
  assert.equal(ok.status, 200);
  assert.equal(fetchImpl.calls[0].url, 'https://api.openai.com/v1/responses');
});

test('unsupported methods are refused', async () => {
  const fetchImpl = stub();
  for (const method of ['GET', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS']) {
    const response = await handleRequest(post('/v1/responses', { method }), ENV, fetchImpl);
    assert.equal(response.status, 405, method);
    assert.equal(response.headers.get('allow'), 'POST');
  }
  assert.equal(fetchImpl.calls.length, 0);
});

test('only the three demo models are enabled', async () => {
  const fetchImpl = stub();
  const bad = [
    jsonPost('/v1/responses', { ...RESPONSES, model: 'gpt-4o' }),
    jsonPost('/v1/responses', { ...RESPONSES, model: 42 }),
    jsonPost('/v1/audio/speech', { ...SPEECH, model: 'gpt-4o-mini-tts' }),
    jsonPost('/v1/audio/speech', { ...SPEECH, voice: 'nova' }),
    jsonPost('/v1/audio/speech', { ...SPEECH, response_format: 'mp3' }),
    post('/v1/audio/transcriptions', transcription({ model: 'whisper-1' })),
  ];
  for (const request of bad) {
    assert.equal((await handleRequest(request, ENV, fetchImpl)).status, 400);
  }
  assert.equal(fetchImpl.calls.length, 0);
});

test('this is not a general relay: extra parameters are refused', async () => {
  const fetchImpl = stub();
  const bad = [
    { ...RESPONSES, stream: true },
    { ...RESPONSES, tools: [{ type: 'function', name: 'move_joint' }] },
    { ...RESPONSES, store: true },
    { ...RESPONSES, background: true },
    { ...RESPONSES, previous_response_id: 'resp_1' },
    { ...RESPONSES, max_output_tokens: 100000 },
    { ...RESPONSES, max_output_tokens: undefined },
  ];
  for (const payload of bad) {
    const response = await handleRequest(jsonPost('/v1/responses', payload), ENV, fetchImpl);
    assert.equal(response.status, 400, JSON.stringify(payload).slice(0, 50));
  }
  // ...and in multipart too.
  const stt = await handleRequest(
    post('/v1/audio/transcriptions', transcription({ stream: 'true' })),
    ENV,
    fetchImpl,
  );
  assert.equal(stt.status, 400);
  assert.equal(fetchImpl.calls.length, 0);
});

test('malformed bodies and wrong content types are refused', async () => {
  const fetchImpl = stub();
  const bad = [
    post('/v1/responses', { body: 'x=1', contentType: 'application/x-www-form-urlencoded' }),
    post('/v1/responses', { body: '{not json', contentType: 'application/json' }),
    post('/v1/responses', { body: '[1,2]', contentType: 'application/json' }),
    post('/v1/responses', { body: '', contentType: 'application/json' }),
    post('/v1/audio/transcriptions', { body: '{}', contentType: 'application/json' }),
    post('/v1/audio/transcriptions', { body: 'x', contentType: 'multipart/form-data' }),
  ];
  for (const request of bad) {
    assert.equal((await handleRequest(request, ENV, fetchImpl)).status, 400);
  }
  assert.equal(fetchImpl.calls.length, 0);
});

test('oversized bodies are refused', async () => {
  const fetchImpl = stub();
  const bad = [
    // A frame larger than attention.OBSERVE_MAX_BYTES allows.
    jsonPost('/v1/responses', { ...RESPONSES, input: 'A'.repeat(1_300_000) }),
    // A recording longer than audio_io.RECORD_SECONDS allows.
    post('/v1/audio/transcriptions', transcription({
      file: { filename: 'utterance.wav', bytes: Buffer.alloc(1_100_000, 1) },
    })),
    // A lying Content-Length is rejected before the body is buffered.
    jsonPost('/v1/audio/speech', SPEECH, { headers: { 'Content-Length': '900000' } }),
  ];
  for (const request of bad) {
    assert.equal((await handleRequest(request, ENV, fetchImpl)).status, 413);
  }
  // A reply longer than character.MAX_REPLY_CHARS allows.
  const chatty = await handleRequest(
    jsonPost('/v1/audio/speech', { ...SPEECH, input: 'a'.repeat(501) }),
    ENV,
    fetchImpl,
  );
  assert.equal(chatty.status, 400);
  assert.equal(fetchImpl.calls.length, 0);
});

// -- the reviewer cannot reach or replace the server key --------------------

test('the upstream credential is always the server key, never a client header', async () => {
  const fetchImpl = stub();
  const request = jsonPost('/v1/responses', RESPONSES, {
    headers: {
      'X-Forwarded-Authorization': 'Bearer sk-attacker-key',
      'OpenAI-Organization': 'org-attacker',
      'OpenAI-Project': 'proj-attacker',
      'X-Api-Key': 'sk-attacker-key',
      Cookie: 'a=b',
      'X-Forwarded-Host': 'evil.example.com',
    },
  });
  await handleRequest(request, ENV, fetchImpl);

  const sent = fetchImpl.calls[0].init.headers;
  assert.equal(sent.Authorization, `Bearer ${SERVER_KEY}`);
  assert.ok(!sent.Authorization.includes(TOKEN));
  assert.deepEqual(Object.keys(sent).sort(), ['Authorization', 'Content-Type']);
  // The client cannot choose the upstream host either.
  assert.ok(fetchImpl.calls[0].url.startsWith('https://api.openai.com/'));
});

// -- nothing leaks the server key -------------------------------------------

test('an upstream error body that quotes the key is redacted', async () => {
  const fetchImpl = stub({
    status: 401,
    contentType: 'application/json',
    body: JSON.stringify({ error: { message: `Incorrect API key provided: ${SERVER_KEY}.` } }),
  });
  const response = await handleRequest(jsonPost('/v1/responses', RESPONSES), ENV, fetchImpl);
  const text = await response.text();

  assert.equal(response.status, 401);
  assert.ok(!text.includes(SERVER_KEY));
  assert.ok(text.includes('<redacted>'));
});

test('no refusal or transport failure mentions the key', async () => {
  const exploding = async () => {
    throw new Error(`ECONNREFUSED with Authorization: Bearer ${SERVER_KEY}`);
  };
  const responses = await Promise.all([
    handleRequest(jsonPost('/v1/responses', RESPONSES, { token: null }), ENV, stub()),
    handleRequest(jsonPost('/v1/responses', RESPONSES, { token: 'wrong' }), ENV, stub()),
    handleRequest(jsonPost('/v1/chat/completions', {}), ENV, stub()),
    handleRequest(jsonPost('/v1/responses', { ...RESPONSES, model: 'gpt-4o' }), ENV, stub()),
    handleRequest(jsonPost('/v1/responses', RESPONSES), ENV, exploding),
  ]);
  for (const response of responses) {
    const text = await response.text();
    assert.ok(!text.includes(SERVER_KEY), text);
    assert.ok(!text.includes('sk-'), text);
  }
});

test('response headers that identify the account are stripped', async () => {
  const response = await handleRequest(jsonPost('/v1/responses', RESPONSES), ENV, stub());
  assert.equal(response.headers.get('openai-organization'), null);
  assert.equal(response.headers.get('cache-control'), 'no-store');
});

test('redact() strips both the exact key and anything key-shaped', () => {
  assert.equal(redact(`a ${SERVER_KEY} b`, SERVER_KEY), 'a <redacted> b');
  assert.equal(redact('a sk-someOtherKey123456 b', SERVER_KEY), 'a <redacted> b');
  assert.equal(redact('nothing here', SERVER_KEY), 'nothing here');
});
