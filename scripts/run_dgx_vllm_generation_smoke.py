"""Run a chat completion smoke check against the DGX vLLM runtime."""

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.generation_provider_metrics import (  # noqa: E402
    generation_provider_metrics_payload,
)
from app.core.generation_providers import (  # noqa: E402
    OPENAI_CHAT_COMPLETIONS_PATH,
    GenerationChatCompletionRequest,
    GenerationProvider,
    GenerationProviderRequestError,
    InvalidGenerationProviderError,
    OpenAICompatibleGenerationProviderClient,
    generation_chat_request_from_openai_messages,
)

DEFAULT_DGX_HOST = "192.168.20.243"
DEFAULT_DGX_VLLM_PORT = 12000
DEFAULT_DGX_VLLM_MODEL_ID = "/home/nurivoice-dgx/models/nvidia/Qwen3.5-122B-A10B-NVFP4"
DEFAULT_API_KEY_ENV = "NEX_PCX_REMOTE_GENERATION_PROVIDER_API_KEY"
DEFAULT_PROVIDER_NAME = "dgx-vllm-qwen3-5-122b-a10b"
DEFAULT_TIMEOUT_SECONDS = 300.0
DEFAULT_MAX_TOKENS = 96
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_PROMPT_TEXT = (
    "NeX-PCX vLLM 연결 smoke test입니다. "
    "한국어로 한 문장만 답하고, 문장 안에 '연결 확인 완료'를 포함해 주세요."
)
ANSWER_PREVIEW_CHARS = 240


@dataclass(frozen=True)
class DgxVllmGenerationSmokePlan:
    provider_name: str
    base_url: str
    chat_completions_url: str
    model_id: str
    api_key_env: str | None
    api_key_configured: bool
    timeout_seconds: float
    max_tokens: int
    temperature: float
    top_p: float
    thinking_disabled: bool
    require_response_model_match: bool
    serving_max_model_len_label: str
    request: GenerationChatCompletionRequest


@dataclass(frozen=True)
class DgxVllmGenerationSmokeObservation:
    request_elapsed_ms: int
    provider_elapsed_ms: int | None
    http_status_code: int | None
    response_model_id: str | None
    response_id: str | None
    finish_reason: str | None
    input_token_count: int | None
    output_token_count: int | None
    total_token_count: int | None
    answer_char_count: int
    answer_preview: str
    provider_metrics: dict[str, Any]
    mismatches: tuple[str, ...]
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.error is None and not self.mismatches


@dataclass(frozen=True)
class DgxVllmGenerationSmokeReport:
    plan: DgxVllmGenerationSmokePlan
    observation: DgxVllmGenerationSmokeObservation
    total_elapsed_ms: int

    @property
    def passed(self) -> bool:
        return self.observation.passed


def build_dgx_vllm_generation_smoke_plan(
    *,
    base_url: str | None = None,
    host: str = DEFAULT_DGX_HOST,
    port: int = DEFAULT_DGX_VLLM_PORT,
    provider_name: str = DEFAULT_PROVIDER_NAME,
    model_id: str = DEFAULT_DGX_VLLM_MODEL_ID,
    api_key_env: str | None = DEFAULT_API_KEY_ENV,
    api_key_configured: bool = False,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    prompt_text: str = DEFAULT_PROMPT_TEXT,
    thinking_disabled: bool = True,
    require_response_model_match: bool = False,
    serving_max_model_len_label: str = "200k",
) -> DgxVllmGenerationSmokePlan:
    selected_base_url = _normalize_base_url(base_url or f"http://{host}:{port}")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be greater than 0")
    if max_tokens <= 0:
        raise ValueError("max_tokens must be greater than 0")
    if not 0 <= temperature <= 2:
        raise ValueError("temperature must be between 0 and 2")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be greater than 0 and less than or equal to 1")
    normalized_model_id = _validate_nonblank(model_id, "model_id")
    normalized_prompt_text = _validate_nonblank(prompt_text, "prompt_text")
    request = generation_chat_request_from_openai_messages(
        (
            {
                "role": "system",
                "content": (
                    "You are a concise NeX-PCX vLLM smoke test assistant. "
                    "Answer in Korean with one short sentence."
                ),
            },
            {"role": "user", "content": normalized_prompt_text},
        ),
        model_id=normalized_model_id,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        trace_id="slice-348-dgx-vllm-generation-smoke",
        extra_body=(
            {"chat_template_kwargs": {"enable_thinking": False}} if thinking_disabled else {}
        ),
        runtime_metadata={"smoke": "dgx_vllm_generation"},
    )
    return DgxVllmGenerationSmokePlan(
        provider_name=_validate_nonblank(provider_name, "provider_name"),
        base_url=selected_base_url,
        chat_completions_url=f"{selected_base_url}{OPENAI_CHAT_COMPLETIONS_PATH}",
        model_id=normalized_model_id,
        api_key_env=api_key_env.strip() if api_key_env else None,
        api_key_configured=api_key_configured,
        timeout_seconds=timeout_seconds,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        thinking_disabled=thinking_disabled,
        require_response_model_match=require_response_model_match,
        serving_max_model_len_label=_validate_nonblank(
            serving_max_model_len_label,
            "serving_max_model_len_label",
        ),
        request=request,
    )


def run_dgx_vllm_generation_smoke(
    provider: GenerationProvider,
    *,
    plan: DgxVllmGenerationSmokePlan,
) -> DgxVllmGenerationSmokeReport:
    started_at = perf_counter()
    observation = _run_request(provider, plan=plan)
    return DgxVllmGenerationSmokeReport(
        plan=plan,
        observation=observation,
        total_elapsed_ms=max(0, int((perf_counter() - started_at) * 1000)),
    )


def write_markdown_report(
    report: DgxVllmGenerationSmokeReport,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_markdown_report(report), encoding="utf-8")


def write_json_report(
    report: DgxVllmGenerationSmokeReport,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_report_payload(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _run_request(
    provider: GenerationProvider,
    *,
    plan: DgxVllmGenerationSmokePlan,
) -> DgxVllmGenerationSmokeObservation:
    started_at = perf_counter()
    try:
        response = provider.complete(plan.request)
    except GenerationProviderRequestError as exc:
        metrics_payload = (
            generation_provider_metrics_payload(exc.metrics) if exc.metrics is not None else {}
        )
        return DgxVllmGenerationSmokeObservation(
            request_elapsed_ms=max(0, int((perf_counter() - started_at) * 1000)),
            provider_elapsed_ms=metrics_payload.get("provider_elapsed_ms"),
            http_status_code=metrics_payload.get("http_status_code"),
            response_model_id=metrics_payload.get("response_model_id"),
            response_id=metrics_payload.get("response_id"),
            finish_reason=metrics_payload.get("finish_reason"),
            input_token_count=metrics_payload.get("input_token_count"),
            output_token_count=metrics_payload.get("output_token_count"),
            total_token_count=metrics_payload.get("total_token_count"),
            answer_char_count=0,
            answer_preview="",
            provider_metrics=metrics_payload,
            mismatches=(),
            error=str(exc),
        )
    except (InvalidGenerationProviderError, ValueError, OSError) as exc:
        return DgxVllmGenerationSmokeObservation(
            request_elapsed_ms=max(0, int((perf_counter() - started_at) * 1000)),
            provider_elapsed_ms=None,
            http_status_code=None,
            response_model_id=None,
            response_id=None,
            finish_reason=None,
            input_token_count=None,
            output_token_count=None,
            total_token_count=None,
            answer_char_count=0,
            answer_preview="",
            provider_metrics={},
            mismatches=(),
            error=str(exc),
        )

    metrics_payload = generation_provider_metrics_payload(response.provider_metrics)
    answer_text = response.answer_text.strip()
    return DgxVllmGenerationSmokeObservation(
        request_elapsed_ms=max(0, int((perf_counter() - started_at) * 1000)),
        provider_elapsed_ms=response.provider_metrics.provider_elapsed_ms,
        http_status_code=response.provider_metrics.http_status_code,
        response_model_id=response.provider_model_id,
        response_id=response.response_id,
        finish_reason=response.finish_reason,
        input_token_count=response.input_token_count,
        output_token_count=response.output_token_count,
        total_token_count=response.total_token_count,
        answer_char_count=len(answer_text),
        answer_preview=_answer_preview(answer_text),
        provider_metrics=metrics_payload,
        mismatches=_response_mismatches(response, metrics_payload=metrics_payload, plan=plan),
        error=None,
    )


def _response_mismatches(
    response: Any,
    *,
    metrics_payload: dict[str, Any],
    plan: DgxVllmGenerationSmokePlan,
) -> tuple[str, ...]:
    mismatches: list[str] = []
    if not response.answer_text.strip():
        mismatches.append("answer_text must not be empty")
    if metrics_payload.get("succeeded") is not True:
        mismatches.append("provider_metrics.succeeded must be true")
    if response.finish_reason is None:
        mismatches.append("finish_reason must be present")
    if response.total_token_count is None:
        mismatches.append("total_token_count must be present")
    if plan.require_response_model_match and response.provider_model_id != plan.model_id:
        mismatches.append(
            f"response_model_id: expected {plan.model_id!r}, got {response.provider_model_id!r}"
        )
    return tuple(mismatches)


def _answer_preview(answer_text: str) -> str:
    normalized = " ".join(answer_text.split())
    if len(normalized) <= ANSWER_PREVIEW_CHARS:
        return normalized
    return f"{normalized[:ANSWER_PREVIEW_CHARS]}..."


def _normalize_base_url(base_url: str) -> str:
    normalized = base_url.strip().rstrip("/")
    if not normalized:
        raise ValueError("base_url is required")
    return normalized


def _validate_nonblank(value: str, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} is required")
    return normalized


def _report_payload(report: DgxVllmGenerationSmokeReport) -> dict[str, Any]:
    return {
        "passed": report.passed,
        "plan": _plan_payload(report.plan),
        "observation": asdict(report.observation),
        "total_elapsed_ms": report.total_elapsed_ms,
    }


def _plan_payload(plan: DgxVllmGenerationSmokePlan) -> dict[str, Any]:
    return {
        **asdict(plan),
        "request": {
            "model_id": plan.request.model_id,
            "message_count": len(plan.request.messages),
            "messages": [
                {"role": message.role, "content": f"<message:{index}>"}
                for index, message in enumerate(plan.request.messages, start=1)
            ],
            "max_tokens": plan.request.max_tokens,
            "temperature": plan.request.temperature,
            "top_p": plan.request.top_p,
            "extra_body_keys": sorted(plan.request.extra_body.keys()),
            "trace_id": plan.request.trace_id,
        },
    }


def _print_human_report(report: DgxVllmGenerationSmokeReport) -> None:
    status = "PASS" if report.passed else "FAIL"
    observation = report.observation
    print(f"DGX vLLM generation smoke: {status}")
    print(f"- provider_name: {report.plan.provider_name}")
    print(f"- chat_completions_url: {report.plan.chat_completions_url}")
    print(f"- model_id: {report.plan.model_id}")
    print(f"- api_key_env: {report.plan.api_key_env}")
    print(f"- api_key_configured: {report.plan.api_key_configured}")
    print(f"- serving_max_model_len: {report.plan.serving_max_model_len_label}")
    print(f"- thinking_disabled: {report.plan.thinking_disabled}")
    print(f"- http_status_code: {observation.http_status_code}")
    print(f"- finish_reason: {observation.finish_reason}")
    print(f"- request_elapsed_ms: {observation.request_elapsed_ms}")
    print(f"- provider_elapsed_ms: {observation.provider_elapsed_ms}")
    print(f"- total_token_count: {observation.total_token_count}")
    if observation.answer_preview:
        print(f"- answer_preview: {observation.answer_preview}")
    if observation.mismatches:
        print("- mismatches:")
        for mismatch in observation.mismatches:
            print(f"  - {mismatch}")
    if observation.error:
        print(f"- error: {observation.error}")


def _markdown_report(report: DgxVllmGenerationSmokeReport) -> str:
    observation = report.observation
    lines = [
        "# DGX vLLM Generation Smoke Result",
        "",
        f"- `passed`: `{str(report.passed).lower()}`",
        f"- `provider_name`: `{report.plan.provider_name}`",
        f"- `chat_completions_url`: `{report.plan.chat_completions_url}`",
        f"- `model_id`: `{report.plan.model_id}`",
        f"- `api_key_env`: `{report.plan.api_key_env}`",
        f"- `api_key_configured`: `{str(report.plan.api_key_configured).lower()}`",
        f"- `serving_max_model_len`: `{report.plan.serving_max_model_len_label}`",
        f"- `timeout_seconds`: `{report.plan.timeout_seconds:g}`",
        f"- `max_tokens`: `{report.plan.max_tokens}`",
        f"- `temperature`: `{report.plan.temperature:g}`",
        f"- `top_p`: `{report.plan.top_p:g}`",
        f"- `thinking_disabled`: `{str(report.plan.thinking_disabled).lower()}`",
        f"- `http_status_code`: `{observation.http_status_code}`",
        f"- `response_model_id`: `{observation.response_model_id}`",
        f"- `response_id`: `{observation.response_id}`",
        f"- `finish_reason`: `{observation.finish_reason}`",
        f"- `request_elapsed_ms`: `{observation.request_elapsed_ms}`",
        f"- `provider_elapsed_ms`: `{observation.provider_elapsed_ms}`",
        f"- `input_token_count`: `{observation.input_token_count}`",
        f"- `output_token_count`: `{observation.output_token_count}`",
        f"- `total_token_count`: `{observation.total_token_count}`",
        f"- `answer_char_count`: `{observation.answer_char_count}`",
        "",
        "## Answer Preview",
        "",
        observation.answer_preview or "`<empty>`",
        "",
        "## Provider Metrics",
        "",
        "```json",
        json.dumps(observation.provider_metrics, ensure_ascii=False, indent=2),
        "```",
        "",
    ]
    if observation.mismatches:
        lines.extend(["## Mismatches", ""])
        lines.extend(f"- `{mismatch}`" for mismatch in observation.mismatches)
        lines.append("")
    if observation.error:
        lines.extend(["## Error", "", f"`{observation.error}`", ""])
    return "\n".join(lines)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a chat completion smoke check against the DGX vLLM runtime.",
    )
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--host", default=DEFAULT_DGX_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_DGX_VLLM_PORT)
    parser.add_argument("--provider-name", default=DEFAULT_PROVIDER_NAME)
    parser.add_argument("--model-id", default=DEFAULT_DGX_VLLM_MODEL_ID)
    parser.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--prompt-text", default=DEFAULT_PROMPT_TEXT)
    parser.add_argument("--serving-max-model-len-label", default="200k")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--require-response-model-match", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--markdown-output", default=None)
    parser.add_argument("--json-output", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    api_key = os.getenv(args.api_key_env) if args.api_key_env else None
    try:
        plan = build_dgx_vllm_generation_smoke_plan(
            base_url=args.base_url,
            host=args.host,
            port=args.port,
            provider_name=args.provider_name,
            model_id=args.model_id,
            api_key_env=args.api_key_env,
            api_key_configured=bool(api_key),
            timeout_seconds=args.timeout_seconds,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            prompt_text=args.prompt_text,
            thinking_disabled=not args.enable_thinking,
            require_response_model_match=args.require_response_model_match,
            serving_max_model_len_label=args.serving_max_model_len_label,
        )
    except (InvalidGenerationProviderError, ValueError) as exc:
        parser.error(str(exc))

    if args.dry_run:
        payload = {"dry_run": True, "plan": _plan_payload(plan)}
        print(json.dumps(payload, ensure_ascii=False, indent=None if args.json else 2))
        return 0

    provider = OpenAICompatibleGenerationProviderClient(
        plan.base_url,
        provider_name=plan.provider_name,
        model_id=plan.model_id,
        timeout_seconds=plan.timeout_seconds,
        api_key=api_key,
    )
    try:
        report = run_dgx_vllm_generation_smoke(provider, plan=plan)
    finally:
        provider.close()

    if args.markdown_output:
        write_markdown_report(report, Path(args.markdown_output))
    if args.json_output:
        write_json_report(report, Path(args.json_output))

    if args.json:
        print(json.dumps(_report_payload(report), ensure_ascii=False))
    else:
        _print_human_report(report)
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
