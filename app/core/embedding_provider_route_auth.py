"""Auth/header metadata helpers for embedding provider routes."""

import os
from collections.abc import Mapping
from dataclasses import dataclass

REQUEST_HEADERS_METADATA_KEY = "request_headers"
AUTH_METADATA_KEY = "auth"
AUTH_TYPE_NONE = "none"
AUTH_TYPE_BEARER = "bearer"
AUTH_TYPE_API_KEY = "api_key"
SUPPORTED_AUTH_TYPES = (AUTH_TYPE_NONE, AUTH_TYPE_BEARER, AUTH_TYPE_API_KEY)


@dataclass(frozen=True)
class EmbeddingProviderRouteRequestMetadata:
    request_headers: dict[str, str]
    auth_type: str
    auth_token_env: str | None = None
    auth_header_name: str | None = None


class InvalidEmbeddingProviderRouteAuthError(ValueError):
    """Raised when route auth/header metadata is invalid."""


def normalize_embedding_provider_route_metadata(
    runtime_metadata: Mapping[str, object] | None,
) -> dict[str, object]:
    metadata = dict(runtime_metadata or {})
    request_headers = _normalize_request_headers(metadata.get(REQUEST_HEADERS_METADATA_KEY))
    auth = _normalize_auth_metadata(metadata.get(AUTH_METADATA_KEY))

    if request_headers:
        metadata[REQUEST_HEADERS_METADATA_KEY] = request_headers
    else:
        metadata.pop(REQUEST_HEADERS_METADATA_KEY, None)

    if auth["type"] != AUTH_TYPE_NONE:
        metadata[AUTH_METADATA_KEY] = auth
    else:
        metadata.pop(AUTH_METADATA_KEY, None)
    return metadata


def describe_embedding_provider_route_request_metadata(
    runtime_metadata: Mapping[str, object] | None,
) -> EmbeddingProviderRouteRequestMetadata:
    metadata = normalize_embedding_provider_route_metadata(runtime_metadata)
    auth = dict(metadata.get(AUTH_METADATA_KEY) or {})
    return EmbeddingProviderRouteRequestMetadata(
        request_headers=dict(metadata.get(REQUEST_HEADERS_METADATA_KEY) or {}),
        auth_type=str(auth.get("type") or AUTH_TYPE_NONE),
        auth_token_env=auth.get("token_env") or auth.get("key_env"),
        auth_header_name=auth.get("header_name"),
    )


def resolve_embedding_provider_route_request_headers(
    runtime_metadata: Mapping[str, object] | None,
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    metadata = normalize_embedding_provider_route_metadata(runtime_metadata)
    request_headers = dict(metadata.get(REQUEST_HEADERS_METADATA_KEY) or {})
    auth = dict(metadata.get(AUTH_METADATA_KEY) or {})
    auth_type = str(auth.get("type") or AUTH_TYPE_NONE)
    source_environ = environ or os.environ

    if auth_type == AUTH_TYPE_NONE:
        return request_headers
    if auth_type == AUTH_TYPE_BEARER:
        token = _read_required_env(source_environ, str(auth["token_env"]))
        request_headers["Authorization"] = f"Bearer {token}"
        return request_headers
    if auth_type == AUTH_TYPE_API_KEY:
        token = _read_required_env(source_environ, str(auth["key_env"]))
        request_headers[str(auth["header_name"])] = token
        return request_headers

    raise InvalidEmbeddingProviderRouteAuthError(f"Unsupported auth type: {auth_type}")


def _normalize_request_headers(value: object) -> dict[str, str]:
    if value in (None, ""):
        return {}
    if not isinstance(value, Mapping):
        raise InvalidEmbeddingProviderRouteAuthError("request_headers must be a JSON object")
    return {
        _normalize_header_name(str(key)): _normalize_header_value(str(header_value))
        for key, header_value in value.items()
    }


def _normalize_auth_metadata(value: object) -> dict[str, str]:
    if value in (None, "", {}):
        return {"type": AUTH_TYPE_NONE}
    if not isinstance(value, Mapping):
        raise InvalidEmbeddingProviderRouteAuthError("auth must be a JSON object")
    auth_type = str(value.get("type") or AUTH_TYPE_NONE).strip().lower()
    if auth_type not in SUPPORTED_AUTH_TYPES:
        raise InvalidEmbeddingProviderRouteAuthError(f"Unsupported auth type: {auth_type}")
    if auth_type == AUTH_TYPE_NONE:
        return {"type": AUTH_TYPE_NONE}
    if auth_type == AUTH_TYPE_BEARER:
        return {
            "type": AUTH_TYPE_BEARER,
            "token_env": _normalize_env_name(str(value.get("token_env") or "")),
        }
    return {
        "type": AUTH_TYPE_API_KEY,
        "key_env": _normalize_env_name(str(value.get("key_env") or "")),
        "header_name": _normalize_header_name(str(value.get("header_name") or "X-API-Key")),
    }


def _normalize_header_name(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise InvalidEmbeddingProviderRouteAuthError("header name is required")
    if any(char in normalized for char in ("\r", "\n", ":")):
        raise InvalidEmbeddingProviderRouteAuthError("header name contains invalid characters")
    return normalized


def _normalize_header_value(value: str) -> str:
    normalized = value.strip()
    if any(char in normalized for char in ("\r", "\n")):
        raise InvalidEmbeddingProviderRouteAuthError("header value contains invalid characters")
    return normalized


def _normalize_env_name(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise InvalidEmbeddingProviderRouteAuthError("auth environment variable is required")
    if any(char.isspace() for char in normalized):
        raise InvalidEmbeddingProviderRouteAuthError(
            "auth environment variable must not contain whitespace"
        )
    return normalized


def _read_required_env(environ: Mapping[str, str], env_name: str) -> str:
    value = environ.get(env_name)
    if value is None or not value.strip():
        raise InvalidEmbeddingProviderRouteAuthError(
            f"Required auth environment variable is not set: {env_name}"
        )
    return value
