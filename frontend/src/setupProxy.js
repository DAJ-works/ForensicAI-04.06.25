// Configures the Create React App dev-server proxy.
//
// All browser requests to /api/* are forwarded to the backend. This runs in
// the dev server's Node process (not the browser), so:
//   - PROXY_TARGET points at the backend. In Docker Compose this must be the
//     service name (http://backend:5000); for local `npm start` it defaults to
//     http://localhost:5000.
//   - REACT_APP_API_KEY is injected as the X-API-Key header here, so the key
//     stays server-side and every /api call (including SSE) is authenticated
//     without each fetch() having to set the header itself.
const { createProxyMiddleware } = require('http-proxy-middleware');

module.exports = function (app) {
  const target = process.env.PROXY_TARGET || 'http://localhost:5000';
  const apiKey = process.env.REACT_APP_API_KEY || '';

  app.use(
    '/api',
    createProxyMiddleware({
      target,
      changeOrigin: true,
      headers: apiKey ? { 'X-API-Key': apiKey } : undefined,
    })
  );
};
