from uuid import uuid4

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.database import connect
from app.core.generation_runs import (
    DGX_VLLM_GENERATION_API_KEY_ENV,
    DGX_VLLM_GENERATION_BASE_URL,
    DGX_VLLM_GENERATION_MODEL_ID,
    GENERATION_PROVIDER_MODE_REMOTE_OPENAI_COMPATIBLE,
    get_default_generation_provider_config,
    get_generation_provider_config_by_name,
    list_generation_provider_configs,
    seed_dgx_vllm_generation_provider_config,
)
from app.main import create_app

MOCK_PROVIDER_NAME = "mock_qwen36_27b_nvfp4"


def _provider_name() -> str:
    return f"pytest_dgx_vllm_{uuid4().hex}"


def _restore_generation_provider_defaults(database_url: str, provider_name: str) -> None:
    with connect(database_url) as conn:
        conn.execute(
            """
            DELETE FROM generation_provider_configs
            WHERE provider_name = %s
            """,
            (provider_name,),
        )
        conn.execute(
            """
            UPDATE generation_provider_configs
            SET is_default = false
            WHERE is_default
              AND provider_name <> %s
            """,
            (MOCK_PROVIDER_NAME,),
        )
        conn.execute(
            """
            UPDATE generation_provider_configs
            SET is_default = true,
                is_active = true
            WHERE provider_name = %s
            """,
            (MOCK_PROVIDER_NAME,),
        )
        conn.commit()


def test_generation_provider_config_repository_seeds_dgx_vllm_and_switches_default(
    migrated_database_url: str,
) -> None:
    provider_name = _provider_name()
    try:
        initial_default = get_default_generation_provider_config(migrated_database_url)
        assert initial_default is not None
        assert initial_default.provider_name == MOCK_PROVIDER_NAME

        inactive_provider = seed_dgx_vllm_generation_provider_config(
            migrated_database_url,
            provider_name=provider_name,
            is_active=False,
        )
        active_names = {
            provider.provider_name
            for provider in list_generation_provider_configs(
                migrated_database_url,
                include_inactive=False,
            )
        }
        assert inactive_provider.provider_name == provider_name
        assert inactive_provider.is_active is False
        assert provider_name not in active_names

        default_provider = seed_dgx_vllm_generation_provider_config(
            migrated_database_url,
            provider_name=provider_name,
            provider_base_url=f"{DGX_VLLM_GENERATION_BASE_URL}/",
            is_default=True,
            max_tokens=2048,
        )
        stored = get_generation_provider_config_by_name(migrated_database_url, provider_name)
        current_default = get_default_generation_provider_config(migrated_database_url)

        assert stored == default_provider
        assert default_provider.provider_mode == GENERATION_PROVIDER_MODE_REMOTE_OPENAI_COMPATIBLE
        assert default_provider.provider_base_url == DGX_VLLM_GENERATION_BASE_URL
        assert default_provider.model_id == DGX_VLLM_GENERATION_MODEL_ID
        assert default_provider.is_default is True
        assert default_provider.is_active is True
        assert default_provider.max_tokens == 2048
        assert default_provider.runtime_options["api_key_env"] == DGX_VLLM_GENERATION_API_KEY_ENV
        assert current_default is not None
        assert current_default.provider_name == provider_name
    finally:
        _restore_generation_provider_defaults(migrated_database_url, provider_name)


def test_generation_provider_runtime_config_api_lists_and_seeds_without_secret_leak(
    migrated_database_url: str,
) -> None:
    provider_name = _provider_name()
    app = create_app(
        Settings(
            database_url=migrated_database_url,
            remote_generation_provider_api_key="secret-value-must-not-leak",
        )
    )

    try:
        with TestClient(app) as client:
            seed_response = client.post(
                "/api/admin/generation-provider-configs/seed-dgx-vllm",
                json={
                    "provider_name": provider_name,
                    "is_default": True,
                    "temperature": 0,
                    "top_p": 1,
                    "thinking_disabled": True,
                },
            )
            default_response = client.get("/api/admin/generation-provider-configs/default")
            list_response = client.get(
                "/api/admin/generation-provider-configs",
                params={"include_inactive": "false"},
            )
            invalid_seed_response = client.post(
                "/api/admin/generation-provider-configs/seed-dgx-vllm",
                json={
                    "provider_name": f"{provider_name}_invalid",
                    "provider_base_url": " ",
                },
            )

        seed_body = seed_response.json()
        default_body = default_response.json()
        list_body = list_response.json()
        serialized = str(seed_body) + str(default_body) + str(list_body)

        assert seed_response.status_code == 201
        assert seed_body["provider"]["provider_name"] == provider_name
        assert seed_body["provider"]["is_default"] is True
        assert seed_body["provider"]["runtime_options"]["api_key_env"] == (
            DGX_VLLM_GENERATION_API_KEY_ENV
        )
        assert seed_body["runtime_config"]["api_key_configured"] is True
        assert seed_body["runtime_config"]["extra_body"] == {
            "chat_template_kwargs": {"enable_thinking": False}
        }
        assert seed_body["seed"]["secret_persisted"] is False
        assert default_response.status_code == 200
        assert default_body["provider"]["provider_name"] == provider_name
        assert list_response.status_code == 200
        assert list_body["summary"]["default_provider_name"] == provider_name
        assert provider_name in [provider["provider_name"] for provider in list_body["providers"]]
        assert "secret-value-must-not-leak" not in serialized
        assert invalid_seed_response.status_code == 400
        assert "provider_base_url" in invalid_seed_response.json()["detail"]
    finally:
        _restore_generation_provider_defaults(migrated_database_url, provider_name)


def test_generation_provider_runtime_config_api_returns_503_without_database() -> None:
    app = create_app(Settings(database_url=None))

    with TestClient(app) as client:
        list_response = client.get("/api/admin/generation-provider-configs")
        default_response = client.get("/api/admin/generation-provider-configs/default")
        seed_response = client.post("/api/admin/generation-provider-configs/seed-dgx-vllm")

    assert list_response.status_code == 503
    assert default_response.status_code == 503
    assert seed_response.status_code == 503
