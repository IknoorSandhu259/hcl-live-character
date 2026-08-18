// One static entrypoint per allowed operation: /api/v1/responses.
// Static paths only -- a catch-all under /api compiles to a single-segment
// matcher and never receives a nested path like /api/v1/responses.
import { handleRequest } from '../../lib/proxy.js';

export default {
  fetch: (request) => handleRequest(request, process.env),
};
