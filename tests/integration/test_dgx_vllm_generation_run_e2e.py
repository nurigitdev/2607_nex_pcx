import importlib.util
import json
import sys
from pathlib import Path

import pytest

from app.core.generation_provider_metrics import parse_openai_chat_completion_metrics
from app.core.generation_providers import (
    GenerationChatCompletionRequest,
    GenerationChatCompletionResponse,
)
from app.core.generation_runs import (
    DGX_VLLM_GENERATION_MODEL_ID,
    GENERATION_PROVIDER_MODE_REMOTE_OPENAI_COMPATIBLE,
    get_default_generation_provider_config,
)

pytestmark = pytest.mark.integration


def _load_script_module(script_name: str):
    script_path = Path(__file__).resolve().parents[2] / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(f"{script_name}_module", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


generation_e2e = _load_script_module("run_dgx_vllm_generation_run_e2e.py")


def test_dgx_vllm_generation_run_e2e_persists_run_with_fake_provider(
    migrated_database_url: str,
    tmp_path: Path,
) -> None:
    previous_default = get_default_generation_provider_config(migrated_database_url)
    provider = _FakeGenerationProvider(answer_text="계정 공유는 금지됩니다. [RCP-001]")
    plan = generation_e2e.build_dgx_vllm_generation_run_e2e_plan(
        database_url=migrated_database_url,
        provider_name="pytest_slice_355_dgx_vllm_e2e",
        api_key_configured=True,
    )

    report = generation_e2e.run_dgx_vllm_generation_run_e2e(
        plan,
        provider_client=provider,
        api_key="pytest-api-key",
    )

    assert report.passed is True
    assert report.cleanup_confirmed is True
    assert report.default_provider_restored is True
    assert report.fixture is not None
    assert report.result.persisted_run_found is True
    assert report.result.status == "succeeded"
    assert report.result.guardrail_status == "allowed"
    assert report.result.provider_mode == GENERATION_PROVIDER_MODE_REMOTE_OPENAI_COMPATIBLE
    assert report.result.citation_count == 1
    assert report.result.cited_count == 1
    assert report.result.total_token_count == 109
    assert report.result.provider_http_status_code == 200
    assert len(provider.requests) == 1
    assert provider.requests[0].model_id == DGX_VLLM_GENERATION_MODEL_ID
    assert provider.requests[0].runtime_metadata["search_log_id"] == report.fixture.search_log_id

    markdown_path = tmp_path / "generation_e2e.md"
    json_path = tmp_path / "generation_e2e.json"
    generation_e2e.write_markdown_report(report, markdown_path)
    generation_e2e.write_json_report(report, json_path)
    markdown = markdown_path.read_text(encoding="utf-8")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert "# DGX vLLM Generation Run Live E2E Result" in markdown
    assert "`passed`: `true`" in markdown
    assert "pytest-api-key" not in markdown
    assert "pytest-api-key" not in json.dumps(payload, ensure_ascii=False)
    assert payload["plan"]["database_url"].endswith("/nex_pcx_test")
    assert payload["plan"]["query_text"] == "<query_text>"
    assert payload["plan"]["fixture_text"] == "<fixture_text>"

    restored_default = get_default_generation_provider_config(migrated_database_url)
    assert (restored_default.provider_name if restored_default is not None else None) == (
        previous_default.provider_name if previous_default is not None else None
    )


def test_dgx_vllm_generation_run_e2e_reports_missing_citation_use(
    migrated_database_url: str,
) -> None:
    provider = _FakeGenerationProvider(answer_text="계정 공유는 금지됩니다.")
    plan = generation_e2e.build_dgx_vllm_generation_run_e2e_plan(
        database_url=migrated_database_url,
        provider_name="pytest_slice_355_missing_citation",
        api_key_env=None,
    )

    report = generation_e2e.run_dgx_vllm_generation_run_e2e(
        plan,
        provider_client=provider,
    )

    assert report.passed is False
    assert report.cleanup_confirmed is True
    assert report.result.status == "succeeded"
    assert report.result.citation_count == 1
    assert report.result.cited_count == 0
    assert "at least one generation citation must be used in the answer" in (
        report.result.mismatches
    )


def test_dgx_vllm_generation_run_e2e_plan_redacts_payload() -> None:
    plan = generation_e2e.build_dgx_vllm_generation_run_e2e_plan(
        database_url="postgresql://nex_pcx_dev:secret@127.0.0.1:5432/nex_pcx_dev",
        provider_base_url="http://192.168.20.243:12000/",
        api_key_configured=True,
        query_text="secret query text",
        fixture_text="secret fixture text",
    )

    payload = generation_e2e._plan_payload(plan)

    assert plan.provider_base_url == "http://192.168.20.243:12000"
    assert payload["database_url"] == (
        "postgresql://nex_pcx_dev:<redacted>@127.0.0.1:5432/nex_pcx_dev"
    )
    assert payload["query_text"] == "<query_text>"
    assert payload["fixture_text"] == "<fixture_text>"
    assert "secret" not in json.dumps(payload)
    assert "secret query text" not in json.dumps(payload)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"database_url": " "}, "database_url"),
        ({"provider_name": " "}, "provider_name"),
        ({"provider_base_url": " "}, "provider_base_url"),
        ({"model_id": " "}, "model_id"),
        ({"request_timeout_seconds": 0}, "request_timeout_seconds"),
        ({"max_tokens": 0}, "max_tokens"),
        ({"temperature": 2.1}, "temperature"),
        ({"top_p": 0}, "top_p"),
        ({"max_context_chars": 0}, "max_context_chars"),
        ({"max_items": 0}, "max_items"),
        ({"query_text": " "}, "query_text"),
        ({"fixture_text": " "}, "fixture_text"),
    ),
)
def test_dgx_vllm_generation_run_e2e_plan_rejects_invalid_inputs(
    kwargs: dict[str, object],
    message: str,
) -> None:
    defaults: dict[str, object] = {"database_url": "postgresql://example/test"}
    defaults.update(kwargs)

    with pytest.raises(ValueError, match=message):
        generation_e2e.build_dgx_vllm_generation_run_e2e_plan(**defaults)


class _FakeGenerationProvider:
    def __init__(self, *, answer_text: str) -> None:
        self.answer_text = answer_text
        self.requests: list[GenerationChatCompletionRequest] = []

    def complete(
        self,
        request: GenerationChatCompletionRequest,
    ) -> GenerationChatCompletionResponse:
        self.requests.append(request)
        metrics = parse_openai_chat_completion_metrics(
            {
                "id": "chatcmpl-slice-355",
                "object": "chat.completion",
                "created": 1785000000,
                "model": DGX_VLLM_GENERATION_MODEL_ID,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": self.answer_text},
                    }
                ],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 9,
                    "total_tokens": 109,
                },
            },
            provider_name="pytest_slice_355_dgx_vllm_e2e",
            provider_mode=GENERATION_PROVIDER_MODE_REMOTE_OPENAI_COMPATIBLE,
            requested_model_id=request.model_id,
            http_status_code=200,
            elapsed_ms=71,
            provider_elapsed_ms=68,
        )
        return GenerationChatCompletionResponse(
            answer_text=self.answer_text,
            finish_reason="stop",
            provider_model_id=DGX_VLLM_GENERATION_MODEL_ID,
            response_id="chatcmpl-slice-355",
            input_token_count=100,
            output_token_count=9,
            total_token_count=109,
            elapsed_ms=71,
            provider_metrics=metrics,
            response_metadata={"provider_name": "pytest_slice_355_dgx_vllm_e2e"},
            raw_response={},
        )
