import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from app.core.generation_provider_metrics import parse_openai_chat_completion_metrics
from app.core.generation_providers import (
    GenerationChatCompletionRequest,
    GenerationChatCompletionResponse,
    GenerationProviderRequestError,
    InvalidGenerationProviderError,
)


def _load_script_module(script_name: str):
    script_path = Path(__file__).resolve().parents[2] / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(f"{script_name}_module", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


generation_smoke = _load_script_module("run_dgx_vllm_generation_smoke.py")


def test_build_dgx_vllm_generation_smoke_plan_defaults_to_dgx_vllm() -> None:
    plan = generation_smoke.build_dgx_vllm_generation_smoke_plan(api_key_configured=True)

    assert plan.provider_name == "dgx-vllm-qwen3-6-27b"
    assert plan.base_url == "http://192.168.20.243:12000"
    assert plan.chat_completions_url == "http://192.168.20.243:12000/v1/chat/completions"
    assert plan.model_id == "/home/nurivoice-dgx/models/nvidia/Qwen3.6-27B-NVFP4"
    assert plan.api_key_env == "NEX_PCX_REMOTE_GENERATION_PROVIDER_API_KEY"
    assert plan.api_key_configured is True
    assert plan.serving_max_model_len_label == "200k"
    assert plan.request.model_id == plan.model_id
    assert plan.request.max_tokens == 96
    assert plan.request.temperature == 0.0
    assert plan.request.top_p == 1.0
    assert plan.thinking_disabled is True
    assert plan.request.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}
    assert [message.role for message in plan.request.messages] == ["system", "user"]


def test_build_dgx_vllm_generation_smoke_plan_accepts_custom_inputs() -> None:
    plan = generation_smoke.build_dgx_vllm_generation_smoke_plan(
        base_url="http://vllm.local:12000/",
        provider_name="custom-vllm",
        model_id="served-qwen",
        api_key_env=None,
        timeout_seconds=12.5,
        max_tokens=64,
        temperature=0.2,
        top_p=0.7,
        prompt_text="  연결 확인  ",
        thinking_disabled=False,
        require_response_model_match=True,
        serving_max_model_len_label="32k",
    )

    assert plan.base_url == "http://vllm.local:12000"
    assert plan.chat_completions_url == "http://vllm.local:12000/v1/chat/completions"
    assert plan.provider_name == "custom-vllm"
    assert plan.model_id == "served-qwen"
    assert plan.api_key_env is None
    assert plan.timeout_seconds == 12.5
    assert plan.max_tokens == 64
    assert plan.temperature == 0.2
    assert plan.top_p == 0.7
    assert plan.thinking_disabled is False
    assert plan.request.extra_body == {}
    assert plan.require_response_model_match is True
    assert plan.serving_max_model_len_label == "32k"
    assert plan.request.messages[1].content == "연결 확인"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"base_url": " "}, "base_url"),
        ({"timeout_seconds": 0}, "timeout_seconds"),
        ({"max_tokens": 0}, "max_tokens"),
        ({"temperature": -0.1}, "temperature"),
        ({"top_p": 0}, "top_p"),
        ({"model_id": " "}, "model_id"),
        ({"prompt_text": " "}, "prompt_text"),
        ({"provider_name": " "}, "provider_name"),
        ({"serving_max_model_len_label": " "}, "serving_max_model_len_label"),
    ),
)
def test_build_dgx_vllm_generation_smoke_plan_rejects_invalid_inputs(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises((InvalidGenerationProviderError, ValueError), match=message):
        generation_smoke.build_dgx_vllm_generation_smoke_plan(**kwargs)


def test_run_dgx_vllm_generation_smoke_passes_for_expected_response() -> None:
    provider = _FakeGenerationSmokeProvider()
    plan = generation_smoke.build_dgx_vllm_generation_smoke_plan()

    report = generation_smoke.run_dgx_vllm_generation_smoke(provider, plan=plan)

    assert report.passed is True
    assert len(provider.requests) == 1
    assert provider.requests[0].model_id == plan.model_id
    observation = report.observation
    assert observation.passed is True
    assert observation.http_status_code == 200
    assert observation.response_model_id == plan.model_id
    assert observation.finish_reason == "stop"
    assert observation.total_token_count == 35
    assert observation.answer_preview == "연결 확인 완료입니다."
    assert observation.provider_metrics["succeeded"] is True


def test_run_dgx_vllm_generation_smoke_reports_model_mismatch_when_required() -> None:
    provider = _FakeGenerationSmokeProvider(response_model_id="different-model")
    plan = generation_smoke.build_dgx_vllm_generation_smoke_plan(require_response_model_match=True)

    report = generation_smoke.run_dgx_vllm_generation_smoke(provider, plan=plan)

    assert report.passed is False
    assert report.observation.mismatches == (
        "response_model_id: expected "
        "'/home/nurivoice-dgx/models/nvidia/Qwen3.6-27B-NVFP4', got 'different-model'",
    )


def test_run_dgx_vllm_generation_smoke_reports_missing_answer_and_usage() -> None:
    provider = _FakeGenerationSmokeProvider(
        answer_text=" ",
        finish_reason=None,
        total_token_count=None,
    )
    plan = generation_smoke.build_dgx_vllm_generation_smoke_plan()

    report = generation_smoke.run_dgx_vllm_generation_smoke(provider, plan=plan)

    assert report.passed is False
    assert report.observation.mismatches == (
        "answer_text must not be empty",
        "finish_reason must be present",
        "total_token_count must be present",
    )


def test_run_dgx_vllm_generation_smoke_captures_provider_request_error() -> None:
    provider = _FailingGenerationSmokeProvider()
    plan = generation_smoke.build_dgx_vllm_generation_smoke_plan()

    report = generation_smoke.run_dgx_vllm_generation_smoke(provider, plan=plan)

    assert report.passed is False
    assert report.observation.error == "Remote generation provider returned HTTP 503"
    assert report.observation.http_status_code == 503
    assert report.observation.provider_metrics["error_code"] == "overloaded"


def test_run_dgx_vllm_generation_smoke_captures_invalid_provider_error() -> None:
    provider = _InvalidGenerationSmokeProvider()
    plan = generation_smoke.build_dgx_vllm_generation_smoke_plan()

    report = generation_smoke.run_dgx_vllm_generation_smoke(provider, plan=plan)

    assert report.passed is False
    assert report.observation.error == "provider unavailable"
    assert report.observation.provider_metrics == {}


def test_dgx_vllm_generation_smoke_writes_markdown_and_json_reports(tmp_path) -> None:
    provider = _FakeGenerationSmokeProvider()
    plan = generation_smoke.build_dgx_vllm_generation_smoke_plan(api_key_configured=True)
    report = generation_smoke.run_dgx_vllm_generation_smoke(provider, plan=plan)
    markdown_path = tmp_path / "nested" / "generation_smoke.md"
    json_path = tmp_path / "nested" / "generation_smoke.json"

    generation_smoke.write_markdown_report(report, markdown_path)
    generation_smoke.write_json_report(report, json_path)

    markdown = markdown_path.read_text(encoding="utf-8")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert "# DGX vLLM Generation Smoke Result" in markdown
    assert "`passed`: `true`" in markdown
    assert "`api_key_configured`: `true`" in markdown
    assert "`thinking_disabled`: `true`" in markdown
    assert "연결 확인 완료입니다." in markdown
    assert "sensitive-secret" not in markdown
    assert payload["passed"] is True
    assert payload["plan"]["request"]["messages"][0]["content"] == "<message:1>"
    assert payload["observation"]["answer_preview"] == "연결 확인 완료입니다."


def test_dgx_vllm_generation_smoke_cli_prints_json_dry_run_plan() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_dgx_vllm_generation_smoke.py",
            "--base-url",
            "http://vllm.local:12000/",
            "--model-id",
            "served-qwen",
            "--prompt-text",
            "sensitive smoke prompt",
            "--dry-run",
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)

    assert payload["dry_run"] is True
    assert payload["plan"]["base_url"] == "http://vllm.local:12000"
    assert payload["plan"]["chat_completions_url"] == "http://vllm.local:12000/v1/chat/completions"
    assert payload["plan"]["model_id"] == "served-qwen"
    assert payload["plan"]["request"]["message_count"] == 2
    assert payload["plan"]["request"]["messages"][1]["content"] == "<message:2>"
    assert payload["plan"]["request"]["extra_body_keys"] == ["chat_template_kwargs"]
    assert "sensitive smoke prompt" not in result.stdout


class _FakeGenerationSmokeProvider:
    def __init__(
        self,
        *,
        answer_text: str = "연결 확인 완료입니다.",
        response_model_id: str = "/home/nurivoice-dgx/models/nvidia/Qwen3.6-27B-NVFP4",
        finish_reason: str | None = "stop",
        total_token_count: int | None = 35,
    ) -> None:
        self.answer_text = answer_text
        self.response_model_id = response_model_id
        self.finish_reason = finish_reason
        self.total_token_count = total_token_count
        self.requests: list[GenerationChatCompletionRequest] = []

    def complete(
        self,
        request: GenerationChatCompletionRequest,
    ) -> GenerationChatCompletionResponse:
        self.requests.append(request)
        metrics = parse_openai_chat_completion_metrics(
            {
                "id": "chatcmpl-smoke",
                "object": "chat.completion",
                "model": self.response_model_id,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": self.finish_reason,
                        "message": {"role": "assistant", "content": self.answer_text},
                    }
                ],
                "usage": {
                    "prompt_tokens": 25,
                    "completion_tokens": 10,
                    **(
                        {"total_tokens": self.total_token_count}
                        if self.total_token_count is not None
                        else {}
                    ),
                },
            },
            provider_name="dgx-vllm-qwen3-6-27b",
            requested_model_id=request.model_id,
            http_status_code=200,
            elapsed_ms=15,
            provider_elapsed_ms=15,
        )
        return GenerationChatCompletionResponse(
            answer_text=self.answer_text,
            finish_reason=self.finish_reason,
            provider_model_id=self.response_model_id,
            response_id="chatcmpl-smoke",
            input_token_count=25,
            output_token_count=10,
            total_token_count=self.total_token_count,
            elapsed_ms=15,
            provider_metrics=metrics,
            response_metadata={"provider_name": "dgx-vllm-qwen3-6-27b"},
            raw_response={},
        )


class _FailingGenerationSmokeProvider:
    def complete(
        self,
        request: GenerationChatCompletionRequest,
    ) -> GenerationChatCompletionResponse:
        metrics = parse_openai_chat_completion_metrics(
            {"error": {"message": "model overloaded", "code": "overloaded"}},
            provider_name="dgx-vllm-qwen3-6-27b",
            requested_model_id=request.model_id,
            http_status_code=503,
            elapsed_ms=7,
        )
        raise GenerationProviderRequestError(
            "Remote generation provider returned HTTP 503",
            metrics=metrics,
            payload={"error": {"message": "model overloaded", "code": "overloaded"}},
        )


class _InvalidGenerationSmokeProvider:
    def complete(
        self,
        request: GenerationChatCompletionRequest,
    ) -> GenerationChatCompletionResponse:
        raise InvalidGenerationProviderError("provider unavailable")
