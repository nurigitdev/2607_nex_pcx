import pytest

from app.core.embedding_provider_route_auth import (
    InvalidEmbeddingProviderRouteAuthError,
    describe_embedding_provider_route_request_metadata,
    normalize_embedding_provider_route_metadata,
    resolve_embedding_provider_route_request_headers,
)


def test_embedding_provider_route_auth_resolves_bearer_and_headers() -> None:
    metadata = normalize_embedding_provider_route_metadata(
        {
            "request_headers": {" X-Tenant ": " nex "},
            "auth": {"type": " BEARER ", "token_env": "NEX_PCX_PROVIDER_TOKEN"},
            "pool": "gpu-a",
        }
    )

    assert metadata == {
        "pool": "gpu-a",
        "request_headers": {"X-Tenant": "nex"},
        "auth": {"type": "bearer", "token_env": "NEX_PCX_PROVIDER_TOKEN"},
    }
    assert resolve_embedding_provider_route_request_headers(
        metadata,
        environ={"NEX_PCX_PROVIDER_TOKEN": "secret-token"},
    ) == {
        "X-Tenant": "nex",
        "Authorization": "Bearer secret-token",
    }
    description = describe_embedding_provider_route_request_metadata(metadata)
    assert description.auth_type == "bearer"
    assert description.auth_token_env == "NEX_PCX_PROVIDER_TOKEN"
    assert description.request_headers == {"X-Tenant": "nex"}


def test_embedding_provider_route_auth_drops_empty_metadata() -> None:
    assert normalize_embedding_provider_route_metadata(None) == {}
    assert (
        normalize_embedding_provider_route_metadata(
            {"request_headers": {}, "auth": {"type": "none"}}
        )
        == {}
    )

    description = describe_embedding_provider_route_request_metadata({})

    assert description.auth_type == "none"
    assert description.auth_token_env is None
    assert description.request_headers == {}


def test_embedding_provider_route_auth_resolves_api_key_header() -> None:
    headers = resolve_embedding_provider_route_request_headers(
        {
            "auth": {
                "type": "api_key",
                "key_env": "NEX_PCX_PROVIDER_API_KEY",
            }
        },
        environ={"NEX_PCX_PROVIDER_API_KEY": "api-secret"},
    )

    assert headers == {"X-API-Key": "api-secret"}


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        ({"request_headers": []}, "request_headers"),
        ({"request_headers": {"Bad:Header": "value"}}, "header name"),
        ({"request_headers": {"X-Good": "bad\nvalue"}}, "header value"),
        ({"auth": []}, "auth"),
        ({"auth": {"type": "basic"}}, "Unsupported auth type"),
        ({"auth": {"type": "bearer", "token_env": " "}}, "environment variable"),
        ({"auth": {"type": "api_key", "key_env": "BAD ENV"}}, "whitespace"),
    ],
)
def test_embedding_provider_route_auth_rejects_invalid_metadata(
    metadata: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(InvalidEmbeddingProviderRouteAuthError, match=message):
        normalize_embedding_provider_route_metadata(metadata)


def test_embedding_provider_route_auth_rejects_missing_runtime_secret() -> None:
    with pytest.raises(InvalidEmbeddingProviderRouteAuthError, match="not set"):
        resolve_embedding_provider_route_request_headers(
            {"auth": {"type": "bearer", "token_env": "MISSING_TOKEN"}},
            environ={},
        )
