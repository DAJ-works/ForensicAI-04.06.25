"""API key authentication middleware.

Checks the ``X-API-Key`` header on incoming API requests against the key
configured in the environment (loaded from a ``.env`` file). Register it on
the Flask app once at startup with :func:`init_auth`.

Environment variables
---------------------
API_KEY
    The shared secret clients must send in the ``X-API-Key`` header. If this
    is unset the server refuses to start, so the API is never accidentally
    left unauthenticated.
"""

import hmac
import logging
import os

from flask import jsonify, request

logger = logging.getLogger(__name__)

# Header clients must supply.
API_KEY_HEADER = "X-API-Key"

# Path prefixes that are served WITHOUT authentication. The React SPA and its
# static bundle are public (the browser can't attach the header when loading
# the page itself), and CORS preflight requests never carry custom headers.
# /api/health is exempt so load balancers / orchestrators can probe it.
EXEMPT_PREFIXES = ("/static",)
EXEMPT_PATHS = ("/api/health",)


def _ensure_dotenv_loaded():
    """Load variables from a .env file into the process environment (once)."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        # python-dotenv is optional; variables may already be in the env.
        logger.warning("python-dotenv not installed; relying on process env")


def _load_api_key():
    """Read API_KEY from the environment, loading a .env file if present."""
    _ensure_dotenv_loaded()
    return os.environ.get("API_KEY", "").strip()


def get_allowed_origins():
    """Return the list of CORS-allowed origins from the CORS_ORIGINS env var.

    CORS_ORIGINS is a comma-separated list of origins, e.g.
    ``http://localhost:3000,https://app.example.com``. If it is unset or
    empty, an empty list is returned so the caller can fail closed rather
    than fall back to a wildcard.
    """
    _ensure_dotenv_loaded()
    value = os.environ.get("CORS_ORIGINS", "")
    return [o.strip() for o in value.split(",") if o.strip()]


def _is_protected(path):
    """Return True if the request path requires authentication."""
    if not path.startswith("/api/"):
        return False
    if path in EXEMPT_PATHS:
        return False
    return not path.startswith(EXEMPT_PREFIXES)


def init_auth(app):
    """Attach the X-API-Key check to ``app`` as a before_request hook.

    Raises RuntimeError if no API_KEY is configured, to fail closed rather
    than serve the API without authentication.
    """
    api_key = _load_api_key()
    if not api_key:
        raise RuntimeError(
            "API_KEY is not set. Define it in your .env file or environment "
            "before starting the server."
        )

    @app.before_request
    def require_api_key():
        # CORS preflight requests can't carry custom headers; let them through
        # (they never reach the actual handler without a follow-up request).
        if request.method == "OPTIONS":
            return None

        if not _is_protected(request.path):
            return None

        provided = request.headers.get(API_KEY_HEADER, "")
        # Constant-time comparison to avoid leaking the key via timing.
        if provided and hmac.compare_digest(provided, api_key):
            return None

        logger.warning(
            "Rejected unauthenticated request to %s from %s",
            request.path,
            request.remote_addr,
        )
        return jsonify({"error": "Unauthorized: missing or invalid API key"}), 401

    logger.info("API key authentication enabled (header: %s)", API_KEY_HEADER)
