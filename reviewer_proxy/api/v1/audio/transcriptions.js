// /api/v1/audio/transcriptions -- see api/v1/responses.js.
import { handleRequest } from '../../../lib/proxy.js';

export default {
  fetch: (request) => handleRequest(request, process.env),
};
