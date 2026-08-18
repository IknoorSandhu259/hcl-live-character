/**
 * The Vercel entry point. Everything interesting is in ../lib/proxy.js, which
 * knows nothing about Vercel and is therefore testable without it.
 *
 * The `fetch` Web Standard export receives every HTTP method, so the handler
 * can answer an unsupported method with a 405 of its own rather than leaving
 * it to the platform.
 */

import { handleRequest } from '../lib/proxy.js';

export default {
  fetch(request) {
    return handleRequest(request, process.env);
  },
};
