// Vercel entry point. The `fetch` Web Standard export receives every method,
// so the handler answers an unsupported one with its own 405.
import { handleRequest } from '../lib/proxy.js';

export default {
  fetch: (request) => handleRequest(request, process.env),
};
