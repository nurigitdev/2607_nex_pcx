import pytest

from app.core.embedding_provider_routes import (
    EmbeddingProviderRouteInput,
    InvalidEmbeddingProviderRouteError,
    get_embedding_provider_route,
    validate_embedding_provider_route_input,
)


def test_embedding_provider_route_input_normalizes_remote_route() -> None:
    route_input = validate_embedding_provider_route_input(
        EmbeddingProviderRouteInput(
            profile_name=" kure_v1_1024 ",
            provider_name=" gpu-a ",
            provider_mode=" REMOTE ",
            provider_base_url="http://provider.local/",
            timeout_seconds=5.5,
            priority=10,
            runtime_metadata={"device": "cuda:0"},
        )
    )

    assert route_input.profile_name == "kure_v1_1024"
    assert route_input.provider_name == "gpu-a"
    assert route_input.provider_mode == "remote"
    assert route_input.provider_base_url == "http://provider.local"
    assert route_input.runtime_metadata == {"device": "cuda:0"}


@pytest.mark.parametrize(
    ("route_input", "message"),
    [
        (EmbeddingProviderRouteInput(profile_name="", provider_name="gpu"), "profile_name"),
        (EmbeddingProviderRouteInput(profile_name="p", provider_name=""), "provider_name"),
        (
            EmbeddingProviderRouteInput(
                profile_name="p",
                provider_name="gpu",
                provider_mode="local",
            ),
            "Unsupported provider_mode",
        ),
        (
            EmbeddingProviderRouteInput(profile_name="p", provider_name="gpu"),
            "provider_base_url",
        ),
        (
            EmbeddingProviderRouteInput(
                profile_name="p",
                provider_name="gpu",
                provider_base_url="http://p",
                timeout_seconds=0,
            ),
            "timeout_seconds",
        ),
        (
            EmbeddingProviderRouteInput(
                profile_name="p",
                provider_name="gpu",
                provider_base_url="http://p",
                priority=-1,
            ),
            "priority",
        ),
    ],
)
def test_embedding_provider_route_input_rejects_invalid_values(
    route_input: EmbeddingProviderRouteInput,
    message: str,
) -> None:
    with pytest.raises(InvalidEmbeddingProviderRouteError, match=message):
        validate_embedding_provider_route_input(route_input)


def test_get_embedding_provider_route_rejects_invalid_route_id() -> None:
    with pytest.raises(InvalidEmbeddingProviderRouteError, match="route_id"):
        get_embedding_provider_route("postgresql://example.invalid/db", 0)
