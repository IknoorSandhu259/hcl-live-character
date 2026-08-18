// /api/v1/audio/speech -- see api/v1/responses.js.
import { handleRequest } from '../../../lib/proxy.js';

export default {
  fetch: (request) => handleRequest(request, process.env),
};
