"""Shared authentication and endpoint helpers for Vertex AI Model Garden.

All `google.*` imports are function-local (lazy) so importing this module never
pulls google-auth / the Vertex SDK during test collection or in environments
that do not use Vertex. Authentication is via Google Application Default
Credentials (ADC: `gcloud auth application-default login`) or a service-account
JSON pointed at by GOOGLE_APPLICATION_CREDENTIALS — never a vendor API key.
"""

from __future__ import annotations

import os

_CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
_DEFAULT_LOCATION = "global"


def resolve_project(explicit: str | None) -> str | None:
    """Vertex project: explicit arg, else GOOGLE_CLOUD_PROJECT / GCLOUD_PROJECT."""
    return (
        explicit
        or os.environ.get("GOOGLE_CLOUD_PROJECT")
        or os.environ.get("GCLOUD_PROJECT")
    )


def resolve_location(explicit: str | None) -> str:
    """Vertex location: explicit arg, else GOOGLE_CLOUD_LOCATION, else 'global'."""
    return explicit or os.environ.get("GOOGLE_CLOUD_LOCATION") or _DEFAULT_LOCATION


def get_access_token(credentials=None) -> str:
    """Return a fresh Google OAuth access token for the OpenAI-compatible path.

    Used only by the Grok client (Vertex MaaS via the OpenAI-compatible
    endpoint), which passes the token as the OpenAI ``api_key``. Tokens expire
    in ~1 hour; this fetches/refreshes one at client-build time.
    """
    try:
        import google.auth
        import google.auth.transport.requests
    except ImportError as exc:  # pragma: no cover - exercised via vertex clients
        raise ImportError(
            "Vertex AI Model Garden support requires the optional dependencies. "
            'Install them with:  pip install -e ".[vertex]"'
        ) from exc

    if credentials is None:
        credentials, _ = google.auth.default(scopes=[_CLOUD_PLATFORM_SCOPE])
    credentials.refresh(google.auth.transport.requests.Request())
    return credentials.token


def openapi_base_url(project: str, location: str) -> str:
    """Build the Vertex OpenAI-compatible chat-completions base URL.

    The ``global`` endpoint uses the host ``aiplatform.googleapis.com`` with no
    ``{location}-`` prefix; regional endpoints prepend the region.
    """
    host = (
        "https://aiplatform.googleapis.com"
        if location == "global"
        else f"https://{location}-aiplatform.googleapis.com"
    )
    return f"{host}/v1/projects/{project}/locations/{location}/endpoints/openapi"
