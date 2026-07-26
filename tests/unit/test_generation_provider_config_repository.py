import pytest

from app.core.generation_runs import (
    DGX_VLLM_GENERATION_API_KEY_ENV,
    DGX_VLLM_GENERATION_BASE_URL,
    DGX_VLLM_GENERATION_MODEL_ID,
    DGX_VLLM_GENERATION_PROVIDER_NAME,
    GENERATION_PROVIDER_MODE_MOCK,
    GENERATION_PROVIDER_MODE_REMOTE_OPENAI_COMPATIBLE,
    GenerationProviderConfigInput,
    InvalidGenerationRunError,
    build_dgx_vllm_generation_provider_config_input,
    validate_generation_provider_config_input,
)


def test_dgx_vllm_generation_provider_config_input_uses_secret_env_reference() -> None:
    config_input = build_dgx_vllm_generation_provider_config_input(is_default=True)

    assert config_input.provider_name == DGX_VLLM_GENERATION_PROVIDER_NAME
    assert config_input.provider_mode == GENERATION_PROVIDER_MODE_REMOTE_OPENAI_COMPATIBLE
    assert config_input.provider_base_url == DGX_VLLM_GENERATION_BASE_URL
    assert config_input.model_id == DGX_VLLM_GENERATION_MODEL_ID
    assert config_input.is_default is True
    assert config_input.runtime_options["api_key_env"] == DGX_VLLM_GENERATION_API_KEY_ENV
    assert config_input.runtime_options["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }
    assert "api_key" not in config_input.runtime_options


def test_validate_generation_provider_config_input_allows_mock_without_base_url() -> None:
    config_input = validate_generation_provider_config_input(
        GenerationProviderConfigInput(
            provider_name=" mock-provider ",
            provider_mode=" MOCK ",
            model_id=" qwen ",
            provider_base_url=" ",
            runtime_options={"api_key_env": "NEX_PCX_REMOTE_GENERATION_PROVIDER_API_KEY"},
        )
    )

    assert config_input.provider_name == "mock-provider"
    assert config_input.provider_mode == GENERATION_PROVIDER_MODE_MOCK
    assert config_input.provider_base_url is None
    assert config_input.model_id == "qwen"


@pytest.mark.parametrize(
    ("config_input", "message"),
    (
        (
            GenerationProviderConfigInput(
                provider_name=" ",
                provider_mode=GENERATION_PROVIDER_MODE_MOCK,
                model_id="model",
            ),
            "provider_name",
        ),
        (
            GenerationProviderConfigInput(
                provider_name="provider",
                provider_mode="unsupported",
                model_id="model",
            ),
            "provider_mode",
        ),
        (
            GenerationProviderConfigInput(
                provider_name="provider",
                provider_mode=GENERATION_PROVIDER_MODE_REMOTE_OPENAI_COMPATIBLE,
                model_id="model",
            ),
            "provider_base_url",
        ),
        (
            GenerationProviderConfigInput(
                provider_name="provider",
                provider_mode=GENERATION_PROVIDER_MODE_MOCK,
                model_id=" ",
            ),
            "model_id",
        ),
        (
            GenerationProviderConfigInput(
                provider_name="provider",
                provider_mode=GENERATION_PROVIDER_MODE_MOCK,
                model_id="model",
                request_timeout_seconds=0,
            ),
            "request_timeout_seconds",
        ),
        (
            GenerationProviderConfigInput(
                provider_name="provider",
                provider_mode=GENERATION_PROVIDER_MODE_MOCK,
                model_id="model",
                max_tokens=0,
            ),
            "max_tokens",
        ),
        (
            GenerationProviderConfigInput(
                provider_name="provider",
                provider_mode=GENERATION_PROVIDER_MODE_MOCK,
                model_id="model",
                temperature=2.1,
            ),
            "temperature",
        ),
        (
            GenerationProviderConfigInput(
                provider_name="provider",
                provider_mode=GENERATION_PROVIDER_MODE_MOCK,
                model_id="model",
                top_p=0,
            ),
            "top_p",
        ),
        (
            GenerationProviderConfigInput(
                provider_name="provider",
                provider_mode=GENERATION_PROVIDER_MODE_MOCK,
                model_id="model",
                runtime_options=["bad"],
            ),
            "runtime_options",
        ),
        (
            GenerationProviderConfigInput(
                provider_name="provider",
                provider_mode=GENERATION_PROVIDER_MODE_MOCK,
                model_id="model",
                runtime_options={"headers": {"Authorization": "Bearer secret"}},
            ),
            "runtime_options.headers.Authorization",
        ),
        (
            GenerationProviderConfigInput(
                provider_name="provider",
                provider_mode=GENERATION_PROVIDER_MODE_MOCK,
                model_id="model",
                runtime_options={"nested": [{"token": "secret"}]},
            ),
            "runtime_options.nested.0.token",
        ),
        (
            GenerationProviderConfigInput(
                provider_name="provider",
                provider_mode=GENERATION_PROVIDER_MODE_MOCK,
                model_id="model",
                created_by=" ",
            ),
            "created_by",
        ),
        (
            GenerationProviderConfigInput(
                provider_name="provider",
                provider_mode=GENERATION_PROVIDER_MODE_MOCK,
                model_id="model",
                created_by_user_id=0,
            ),
            "created_by_user_id",
        ),
    ),
)
def test_validate_generation_provider_config_input_rejects_bad_values(
    config_input: GenerationProviderConfigInput,
    message: str,
) -> None:
    with pytest.raises(InvalidGenerationRunError, match=message):
        validate_generation_provider_config_input(config_input)
