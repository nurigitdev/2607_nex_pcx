"""Run a retrieval-to-generation E2E check against the DGX vLLM runtime."""

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

from psycopg.types.json import Json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.database import connect  # noqa: E402
from app.core.generation_executor import execute_remote_generation_run  # noqa: E402
from app.core.generation_runs import (  # noqa: E402
    DGX_VLLM_GENERATION_API_KEY_ENV,
    DGX_VLLM_GENERATION_BASE_URL,
    DGX_VLLM_GENERATION_MODEL_ID,
    GENERATION_GUARDRAIL_ALLOWED,
    GENERATION_PROVIDER_MODE_REMOTE_OPENAI_COMPATIBLE,
    GENERATION_STATUS_SUCCEEDED,
    GenerationRunRecord,
    get_default_generation_provider_config,
    get_generation_run,
    list_generation_run_citations,
    seed_dgx_vllm_generation_provider_config,
)
from app.core.retrieval_context import (  # noqa: E402
    DEFAULT_CONTEXT_CHAR_BUDGET,
    DEFAULT_CONTEXT_MAX_ITEMS,
    RetrievalContextInput,
    build_retrieval_context_package,
)
from app.core.search_logs import (  # noqa: E402
    SearchLogInput,
    SearchLogResultInput,
    create_search_log,
    create_search_log_results,
)

DEFAULT_DATABASE_URL_ENV = "NEX_PCX_DATABASE_URL"
DEFAULT_PROVIDER_NAME = "slice_355_dgx_vllm_generation_e2e"
DEFAULT_QUERY_TEXT = "회사 보안 규정의 계정 공유 정책은 무엇인가?"
DEFAULT_FIXTURE_TEXT = (
    "NeX-PCX DGX vLLM generation run E2E fixture. "
    "회사 보안 규정은 계정 공유를 금지하고, 문서 근거 기반 답변에는 "
    "RCP citation을 포함해야 합니다."
)
DEFAULT_TIMEOUT_SECONDS = 300
DEFAULT_MAX_TOKENS = 512
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_DOCUMENT_GROUP_PREFIX = "slice-355-dgx-vllm-generation-e2e"
DEFAULT_CREATED_BY = "slice_355_dgx_vllm_generation_e2e"
ANSWER_PREVIEW_CHARS = 360


@dataclass(frozen=True)
class DgxVllmGenerationRunE2EPlan:
    database_url: str
    provider_name: str
    provider_base_url: str
    model_id: str
    api_key_env: str | None
    api_key_configured: bool
    request_timeout_seconds: int
    max_tokens: int
    temperature: float
    top_p: float
    max_context_chars: int
    max_items: int
    include_neighbors: bool
    query_text: str
    fixture_text: str
    cleanup_fixture: bool


@dataclass(frozen=True)
class DgxVllmGenerationRunE2EFixture:
    smoke_run_key: str
    file_id: int
    document_id: int
    chunk_id: int
    search_log_id: int


@dataclass(frozen=True)
class DgxVllmGenerationRunE2EResult:
    generation_run_id: int | None
    search_log_id: int | None
    provider_name: str | None
    provider_mode: str | None
    model_id: str | None
    status: str | None
    guardrail_status: str | None
    retrieval_confidence_status: str | None
    citation_readiness_status: str | None
    citation_count: int
    cited_count: int
    answer_char_count: int
    answer_preview: str
    finish_reason: str | None
    input_token_count: int | None
    output_token_count: int | None
    total_token_count: int | None
    elapsed_ms: int | None
    provider_elapsed_ms: int | None
    provider_http_status_code: int | None
    provider_response_id: str | None
    provider_metrics: dict[str, Any]
    persisted_run_found: bool
    mismatches: tuple[str, ...]
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.error is None and not self.mismatches


@dataclass(frozen=True)
class DgxVllmGenerationRunE2EReport:
    plan: DgxVllmGenerationRunE2EPlan
    fixture: DgxVllmGenerationRunE2EFixture | None
    result: DgxVllmGenerationRunE2EResult
    cleanup_attempted: bool
    cleanup_confirmed: bool
    default_provider_restored: bool
    total_elapsed_seconds: float

    @property
    def passed(self) -> bool:
        return (
            self.result.passed
            and (not self.cleanup_attempted or self.cleanup_confirmed)
            and self.default_provider_restored
        )


def build_dgx_vllm_generation_run_e2e_plan(
    *,
    database_url: str,
    provider_name: str = DEFAULT_PROVIDER_NAME,
    provider_base_url: str = DGX_VLLM_GENERATION_BASE_URL,
    model_id: str = DGX_VLLM_GENERATION_MODEL_ID,
    api_key_env: str | None = DGX_VLLM_GENERATION_API_KEY_ENV,
    api_key_configured: bool = False,
    request_timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    max_context_chars: int = DEFAULT_CONTEXT_CHAR_BUDGET,
    max_items: int = DEFAULT_CONTEXT_MAX_ITEMS,
    include_neighbors: bool = True,
    query_text: str = DEFAULT_QUERY_TEXT,
    fixture_text: str = DEFAULT_FIXTURE_TEXT,
    cleanup_fixture: bool = True,
) -> DgxVllmGenerationRunE2EPlan:
    if request_timeout_seconds <= 0:
        raise ValueError("request_timeout_seconds must be greater than 0")
    if max_tokens <= 0:
        raise ValueError("max_tokens must be greater than 0")
    if not 0 <= temperature <= 2:
        raise ValueError("temperature must be between 0 and 2")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be greater than 0 and less than or equal to 1")
    if max_context_chars <= 0:
        raise ValueError("max_context_chars must be greater than 0")
    if max_items <= 0:
        raise ValueError("max_items must be greater than 0")
    return DgxVllmGenerationRunE2EPlan(
        database_url=_validate_nonblank(database_url, "database_url"),
        provider_name=_validate_nonblank(provider_name, "provider_name"),
        provider_base_url=_normalize_base_url(provider_base_url),
        model_id=_validate_nonblank(model_id, "model_id"),
        api_key_env=api_key_env.strip() if api_key_env else None,
        api_key_configured=api_key_configured,
        request_timeout_seconds=request_timeout_seconds,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        max_context_chars=max_context_chars,
        max_items=max_items,
        include_neighbors=include_neighbors,
        query_text=_validate_nonblank(query_text, "query_text"),
        fixture_text=_validate_nonblank(fixture_text, "fixture_text"),
        cleanup_fixture=cleanup_fixture,
    )


def create_generation_run_e2e_fixture(
    database_url: str,
    *,
    query_text: str,
    fixture_text: str,
    smoke_run_key: str | None = None,
) -> DgxVllmGenerationRunE2EFixture:
    ids = _seed_owner_ids(database_url)
    run_key = smoke_run_key or f"{DEFAULT_DOCUMENT_GROUP_PREFIX}-{uuid4()}"
    checksum = f"{run_key}-{uuid4()}"
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO files (
                    original_file_name,
                    stored_file_name,
                    file_ext,
                    file_size_bytes,
                    sha256_checksum,
                    storage_path,
                    uploaded_by_user_id,
                    document_group
                )
                VALUES (%s, %s, '.md', %s, %s, %s, %s, %s)
                RETURNING file_id
                """,
                (
                    f"{run_key}.md",
                    f"{run_key}.stored.md",
                    len(fixture_text.encode("utf-8")),
                    checksum,
                    f"/tmp/{run_key}.md",
                    ids["alice.member"],
                    run_key,
                ),
            )
            file_id = int(cursor.fetchone()["file_id"])
            cursor.execute(
                """
                INSERT INTO documents (
                    file_id,
                    document_title,
                    document_group,
                    owner_user_id,
                    owner_org_unit_id,
                    access_scope
                )
                VALUES (%s, %s, %s, %s, %s, 'company')
                RETURNING document_id
                """,
                (
                    file_id,
                    "Slice 355 DGX vLLM Generation E2E Fixture",
                    run_key,
                    ids["alice.member"],
                    ids["NeX Company"],
                ),
            )
            document_id = int(cursor.fetchone()["document_id"])
            cursor.execute(
                """
                INSERT INTO chunks (
                    document_id,
                    chunk_seq,
                    chunk_type,
                    chunk_text,
                    content_markdown,
                    content_hash,
                    chunk_policy_name,
                    heading_path,
                    source_anchor,
                    page_no,
                    source_char_start,
                    source_char_end,
                    token_count,
                    char_count,
                    metadata
                )
                VALUES (
                    %s, 0, 'text', %s, %s, %s, 'heading_512_64',
                    %s, %s, 1, 0, %s, 42, %s, %s
                )
                RETURNING chunk_id
                """,
                (
                    document_id,
                    fixture_text,
                    fixture_text,
                    f"chunk-{checksum}",
                    ["운영", "보안 규정"],
                    Json({"start_line": 1, "end_line": 4, "source": "slice_355_e2e"}),
                    len(fixture_text),
                    len(fixture_text),
                    Json({"fixture": "slice_355", "smoke_run_key": run_key}),
                ),
            )
            chunk_id = int(cursor.fetchone()["chunk_id"])

    search_log = create_search_log(
        database_url,
        SearchLogInput(
            query_text=query_text,
            normalized_query_text=query_text,
            actor_user_id=ids["alice.member"],
            requested_search_scope="company",
            effective_search_scope="company",
            permission_filter_metadata={"scope": "company", "fixture": "slice_355"},
            document_group=run_key,
            file_type="md",
            top_k=1,
            profiles=("reranked_vector_cosine",),
            chunk_policy_name="heading_512_64",
            strategy_name="reranked_vector_cosine",
            similarity_metric="cosine",
            query_runtime_metadata={
                "fixture": "slice_355",
                "live_e2e": True,
                "provider": "dgx_vllm_generation_run",
            },
            total_elapsed_ms=23,
            created_by=DEFAULT_CREATED_BY,
        ),
    )
    create_search_log_results(
        database_url,
        [
            SearchLogResultInput(
                search_log_id=search_log.search_log_id,
                profile_name="reranked_vector_cosine",
                search_profile_name="reranked_vector_cosine",
                retrieval_strategy="reranked",
                rank=1,
                chunk_id=chunk_id,
                distance=0.03,
                score=0.97,
                score_components={
                    "source_score": 0.93,
                    "reranker_score": 0.98,
                    "fixture": "slice_355",
                },
                profile_elapsed_ms=17,
            )
        ],
    )
    return DgxVllmGenerationRunE2EFixture(
        smoke_run_key=run_key,
        file_id=file_id,
        document_id=document_id,
        chunk_id=chunk_id,
        search_log_id=search_log.search_log_id,
    )


def cleanup_generation_run_e2e_fixture(
    database_url: str,
    fixture: DgxVllmGenerationRunE2EFixture,
) -> bool:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "DELETE FROM search_logs WHERE document_group = %s",
                (fixture.smoke_run_key,),
            )
            cursor.execute("DELETE FROM files WHERE file_id = %s", (fixture.file_id,))
            cursor.execute(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM files
                    WHERE file_id = %s
                ) AS file_exists,
                EXISTS (
                    SELECT 1
                    FROM search_logs
                    WHERE document_group = %s
                ) AS search_log_exists
                """,
                (fixture.file_id, fixture.smoke_run_key),
            )
            row = cursor.fetchone()
    return not bool(row["file_exists"]) and not bool(row["search_log_exists"])


def run_dgx_vllm_generation_run_e2e(
    plan: DgxVllmGenerationRunE2EPlan,
    *,
    provider_client: object | None = None,
    api_key: str | None = None,
) -> DgxVllmGenerationRunE2EReport:
    started_at = perf_counter()
    previous_default = get_default_generation_provider_config(plan.database_url)
    previous_default_provider_name = previous_default.provider_name if previous_default else None
    fixture: DgxVllmGenerationRunE2EFixture | None = None
    cleanup_attempted = False
    cleanup_confirmed = False
    default_provider_restored = False
    try:
        fixture = create_generation_run_e2e_fixture(
            plan.database_url,
            query_text=plan.query_text,
            fixture_text=plan.fixture_text,
        )
        seed_dgx_vllm_generation_provider_config(
            plan.database_url,
            provider_name=plan.provider_name,
            provider_base_url=plan.provider_base_url,
            model_id=plan.model_id,
            api_key_env=plan.api_key_env or "",
            request_timeout_seconds=plan.request_timeout_seconds,
            max_tokens=plan.max_tokens,
            temperature=plan.temperature,
            top_p=plan.top_p,
            is_default=True,
            is_active=True,
            thinking_disabled=True,
            created_by=DEFAULT_CREATED_BY,
        )
        package = build_retrieval_context_package(
            plan.database_url,
            RetrievalContextInput(
                search_log_id=fixture.search_log_id,
                max_context_chars=plan.max_context_chars,
                include_neighbors=plan.include_neighbors,
                max_items=plan.max_items,
            ),
        )
        execution_report = execute_remote_generation_run(
            plan.database_url,
            package,
            provider_client=provider_client,  # type: ignore[arg-type]
            api_key=api_key,
            created_by=DEFAULT_CREATED_BY,
        )
        stored_run = get_generation_run(plan.database_url, execution_report.run.generation_run_id)
        citations = list_generation_run_citations(
            plan.database_url,
            execution_report.run.generation_run_id,
        )
        result = _result_from_persisted_run(
            stored_run,
            citations=citations,
            expected_search_log_id=fixture.search_log_id,
        )
    except Exception as exc:
        result = _failed_result(fixture.search_log_id if fixture else None, error=str(exc))
    finally:
        default_provider_restored = restore_generation_provider_default(
            plan.database_url,
            smoke_provider_name=plan.provider_name,
            previous_default_provider_name=previous_default_provider_name,
        )
        if fixture is not None and plan.cleanup_fixture:
            cleanup_attempted = True
            cleanup_confirmed = cleanup_generation_run_e2e_fixture(plan.database_url, fixture)

    return DgxVllmGenerationRunE2EReport(
        plan=plan,
        fixture=fixture,
        result=result,
        cleanup_attempted=cleanup_attempted,
        cleanup_confirmed=cleanup_confirmed,
        default_provider_restored=default_provider_restored,
        total_elapsed_seconds=round(perf_counter() - started_at, 3),
    )


def restore_generation_provider_default(
    database_url: str,
    *,
    smoke_provider_name: str,
    previous_default_provider_name: str | None,
) -> bool:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            if smoke_provider_name != previous_default_provider_name:
                cursor.execute(
                    "DELETE FROM generation_provider_configs WHERE provider_name = %s",
                    (smoke_provider_name,),
                )
            if previous_default_provider_name:
                cursor.execute("""
                    UPDATE generation_provider_configs
                    SET is_default = false,
                        updated_at = now()
                    WHERE is_default
                    """)
                cursor.execute(
                    """
                    UPDATE generation_provider_configs
                    SET is_default = true,
                        is_active = true,
                        updated_at = now()
                    WHERE provider_name = %s
                    """,
                    (previous_default_provider_name,),
                )
            cursor.execute(
                """
                SELECT provider_name
                FROM generation_provider_configs
                WHERE is_default AND is_active
                ORDER BY provider_config_id
                LIMIT 1
                """,
            )
            row = cursor.fetchone()
    if previous_default_provider_name is None:
        return row is None or row["provider_name"] != smoke_provider_name
    return row is not None and row["provider_name"] == previous_default_provider_name


def write_markdown_report(
    report: DgxVllmGenerationRunE2EReport,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_markdown_report(report), encoding="utf-8")


def write_json_report(
    report: DgxVllmGenerationRunE2EReport,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_report_payload(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _seed_owner_ids(database_url: str) -> dict[str, int]:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT login_id, user_id
                FROM app_users
                WHERE login_id = 'alice.member'
                """,
            )
            user_row = cursor.fetchone()
            cursor.execute(
                """
                SELECT org_unit_name, org_unit_id
                FROM org_units
                WHERE org_unit_name = 'NeX Company'
                """,
            )
            org_row = cursor.fetchone()
    if user_row is None:
        raise RuntimeError("seed user alice.member was not found")
    if org_row is None:
        raise RuntimeError("seed organization NeX Company was not found")
    return {
        "alice.member": int(user_row["user_id"]),
        "NeX Company": int(org_row["org_unit_id"]),
    }


def _result_from_persisted_run(
    stored_run: GenerationRunRecord | None,
    *,
    citations: tuple[object, ...],
    expected_search_log_id: int,
) -> DgxVllmGenerationRunE2EResult:
    if stored_run is None:
        return _failed_result(
            expected_search_log_id,
            error=None,
            mismatches=("generation run must be persisted",),
        )
    response_metadata = dict(stored_run.response_metadata or {})
    provider_metrics = dict(response_metadata.get("provider_metrics") or {})
    cited_count = sum(1 for citation in citations if bool(getattr(citation, "was_cited", False)))
    answer_text = stored_run.answer_text or ""
    mismatches = _run_mismatches(
        stored_run,
        provider_metrics=provider_metrics,
        citation_count=len(citations),
        cited_count=cited_count,
        expected_search_log_id=expected_search_log_id,
    )
    return DgxVllmGenerationRunE2EResult(
        generation_run_id=stored_run.generation_run_id,
        search_log_id=stored_run.search_log_id,
        provider_name=stored_run.provider_name,
        provider_mode=stored_run.provider_mode,
        model_id=stored_run.model_id,
        status=stored_run.status,
        guardrail_status=stored_run.guardrail_status,
        retrieval_confidence_status=stored_run.retrieval_confidence_status,
        citation_readiness_status=stored_run.citation_readiness_status,
        citation_count=len(citations),
        cited_count=cited_count,
        answer_char_count=len(answer_text.strip()),
        answer_preview=_answer_preview(answer_text),
        finish_reason=stored_run.finish_reason,
        input_token_count=stored_run.input_token_count,
        output_token_count=stored_run.output_token_count,
        total_token_count=stored_run.total_token_count,
        elapsed_ms=stored_run.elapsed_ms,
        provider_elapsed_ms=provider_metrics.get("provider_elapsed_ms"),
        provider_http_status_code=provider_metrics.get("http_status_code"),
        provider_response_id=provider_metrics.get("response_id"),
        provider_metrics=provider_metrics,
        persisted_run_found=True,
        mismatches=mismatches,
        error=None,
    )


def _run_mismatches(
    stored_run: GenerationRunRecord,
    *,
    provider_metrics: dict[str, Any],
    citation_count: int,
    cited_count: int,
    expected_search_log_id: int,
) -> tuple[str, ...]:
    mismatches: list[str] = []
    if stored_run.search_log_id != expected_search_log_id:
        mismatches.append("generation run search_log_id must match fixture search log")
    if stored_run.status != GENERATION_STATUS_SUCCEEDED:
        mismatches.append("generation run status must be succeeded")
    if stored_run.guardrail_status != GENERATION_GUARDRAIL_ALLOWED:
        mismatches.append("guardrail_status must be allowed")
    if stored_run.provider_mode != GENERATION_PROVIDER_MODE_REMOTE_OPENAI_COMPATIBLE:
        mismatches.append("provider_mode must be remote_openai_compatible")
    if not (stored_run.answer_text or "").strip():
        mismatches.append("answer_text must not be empty")
    if stored_run.finish_reason is None:
        mismatches.append("finish_reason must be present")
    if stored_run.total_token_count is None:
        mismatches.append("total_token_count must be present")
    if stored_run.elapsed_ms is None:
        mismatches.append("elapsed_ms must be present")
    if citation_count < 1:
        mismatches.append("at least one generation citation must be persisted")
    if cited_count < 1:
        mismatches.append("at least one generation citation must be used in the answer")
    if provider_metrics.get("succeeded") is not True:
        mismatches.append("provider_metrics.succeeded must be true")
    if provider_metrics.get("http_status_code") is None:
        mismatches.append("provider_metrics.http_status_code must be present")
    if provider_metrics.get("provider_elapsed_ms") is None:
        mismatches.append("provider_metrics.provider_elapsed_ms must be present")
    return tuple(mismatches)


def _failed_result(
    search_log_id: int | None,
    *,
    error: str | None,
    mismatches: tuple[str, ...] = (),
) -> DgxVllmGenerationRunE2EResult:
    return DgxVllmGenerationRunE2EResult(
        generation_run_id=None,
        search_log_id=search_log_id,
        provider_name=None,
        provider_mode=None,
        model_id=None,
        status=None,
        guardrail_status=None,
        retrieval_confidence_status=None,
        citation_readiness_status=None,
        citation_count=0,
        cited_count=0,
        answer_char_count=0,
        answer_preview="",
        finish_reason=None,
        input_token_count=None,
        output_token_count=None,
        total_token_count=None,
        elapsed_ms=None,
        provider_elapsed_ms=None,
        provider_http_status_code=None,
        provider_response_id=None,
        provider_metrics={},
        persisted_run_found=False,
        mismatches=mismatches,
        error=error,
    )


def _report_payload(report: DgxVllmGenerationRunE2EReport) -> dict[str, Any]:
    return {
        "passed": report.passed,
        "plan": _plan_payload(report.plan),
        "fixture": asdict(report.fixture) if report.fixture is not None else None,
        "result": asdict(report.result),
        "cleanup_attempted": report.cleanup_attempted,
        "cleanup_confirmed": report.cleanup_confirmed,
        "default_provider_restored": report.default_provider_restored,
        "total_elapsed_seconds": report.total_elapsed_seconds,
    }


def _plan_payload(plan: DgxVllmGenerationRunE2EPlan) -> dict[str, Any]:
    return {
        "database_url": _redact_database_url(plan.database_url),
        "provider_name": plan.provider_name,
        "provider_base_url": plan.provider_base_url,
        "model_id": plan.model_id,
        "api_key_env": plan.api_key_env,
        "api_key_configured": plan.api_key_configured,
        "request_timeout_seconds": plan.request_timeout_seconds,
        "max_tokens": plan.max_tokens,
        "temperature": plan.temperature,
        "top_p": plan.top_p,
        "max_context_chars": plan.max_context_chars,
        "max_items": plan.max_items,
        "include_neighbors": plan.include_neighbors,
        "query_text": "<query_text>",
        "fixture_text": "<fixture_text>",
        "cleanup_fixture": plan.cleanup_fixture,
    }


def _print_human_report(report: DgxVllmGenerationRunE2EReport) -> None:
    status = "PASS" if report.passed else "FAIL"
    print(f"DGX vLLM generation run E2E: {status}")
    print(f"- provider_name: {report.plan.provider_name}")
    print(f"- provider_base_url: {report.plan.provider_base_url}")
    print(f"- model_id: {report.plan.model_id}")
    print(f"- api_key_env: {report.plan.api_key_env}")
    print(f"- api_key_configured: {report.plan.api_key_configured}")
    print(f"- search_log_id: {report.result.search_log_id}")
    print(f"- generation_run_id: {report.result.generation_run_id}")
    print(f"- status: {report.result.status}")
    print(f"- guardrail_status: {report.result.guardrail_status}")
    print(f"- citation_count: {report.result.citation_count}")
    print(f"- cited_count: {report.result.cited_count}")
    print(f"- total_token_count: {report.result.total_token_count}")
    print(f"- elapsed_ms: {report.result.elapsed_ms}")
    print(f"- provider_elapsed_ms: {report.result.provider_elapsed_ms}")
    print(f"- http_status_code: {report.result.provider_http_status_code}")
    print(f"- cleanup_confirmed: {report.cleanup_confirmed}")
    print(f"- default_provider_restored: {report.default_provider_restored}")
    if report.result.answer_preview:
        print(f"- answer_preview: {report.result.answer_preview}")
    if report.result.mismatches:
        print("- mismatches:")
        for mismatch in report.result.mismatches:
            print(f"  - {mismatch}")
    if report.result.error:
        print(f"- error: {report.result.error}")


def _markdown_report(report: DgxVllmGenerationRunE2EReport) -> str:
    result = report.result
    lines = [
        "# DGX vLLM Generation Run Live E2E Result",
        "",
        f"- `passed`: `{str(report.passed).lower()}`",
        f"- `provider_name`: `{report.plan.provider_name}`",
        f"- `provider_base_url`: `{report.plan.provider_base_url}`",
        f"- `model_id`: `{report.plan.model_id}`",
        f"- `api_key_env`: `{report.plan.api_key_env}`",
        f"- `api_key_configured`: `{str(report.plan.api_key_configured).lower()}`",
        f"- `request_timeout_seconds`: `{report.plan.request_timeout_seconds}`",
        f"- `max_tokens`: `{report.plan.max_tokens}`",
        f"- `temperature`: `{report.plan.temperature:g}`",
        f"- `top_p`: `{report.plan.top_p:g}`",
        f"- `search_log_id`: `{result.search_log_id}`",
        f"- `generation_run_id`: `{result.generation_run_id}`",
        f"- `status`: `{result.status}`",
        f"- `guardrail_status`: `{result.guardrail_status}`",
        f"- `retrieval_confidence_status`: `{result.retrieval_confidence_status}`",
        f"- `citation_readiness_status`: `{result.citation_readiness_status}`",
        f"- `citation_count`: `{result.citation_count}`",
        f"- `cited_count`: `{result.cited_count}`",
        f"- `finish_reason`: `{result.finish_reason}`",
        f"- `input_token_count`: `{result.input_token_count}`",
        f"- `output_token_count`: `{result.output_token_count}`",
        f"- `total_token_count`: `{result.total_token_count}`",
        f"- `elapsed_ms`: `{result.elapsed_ms}`",
        f"- `provider_elapsed_ms`: `{result.provider_elapsed_ms}`",
        f"- `provider_http_status_code`: `{result.provider_http_status_code}`",
        f"- `provider_response_id`: `{result.provider_response_id}`",
        f"- `cleanup_confirmed`: `{str(report.cleanup_confirmed).lower()}`",
        f"- `default_provider_restored`: `{str(report.default_provider_restored).lower()}`",
        "",
        "## Answer Preview",
        "",
        result.answer_preview or "`<empty>`",
        "",
        "## Provider Metrics",
        "",
        "```json",
        json.dumps(result.provider_metrics, ensure_ascii=False, indent=2),
        "```",
        "",
    ]
    if report.fixture is not None:
        lines.extend(
            [
                "## Fixture",
                "",
                f"- `smoke_run_key`: `{report.fixture.smoke_run_key}`",
                f"- `file_id`: `{report.fixture.file_id}`",
                f"- `document_id`: `{report.fixture.document_id}`",
                f"- `chunk_id`: `{report.fixture.chunk_id}`",
                "",
            ],
        )
    if result.mismatches:
        lines.extend(["## Mismatches", ""])
        lines.extend(f"- `{mismatch}`" for mismatch in result.mismatches)
        lines.append("")
    if result.error:
        lines.extend(["## Error", "", f"`{result.error}`", ""])
    return "\n".join(lines)


def _answer_preview(answer_text: str) -> str:
    normalized = " ".join(answer_text.split())
    if len(normalized) <= ANSWER_PREVIEW_CHARS:
        return normalized
    return f"{normalized[:ANSWER_PREVIEW_CHARS]}..."


def _normalize_base_url(base_url: str) -> str:
    normalized = base_url.strip().rstrip("/")
    if not normalized:
        raise ValueError("provider_base_url is required")
    return normalized


def _validate_nonblank(value: str, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} is required")
    return normalized


def _redact_database_url(database_url: str) -> str:
    try:
        parts = urlsplit(database_url)
    except ValueError:
        return "<redacted-database-url>"
    if not parts.netloc:
        return "<redacted-database-url>"
    host = parts.hostname or ""
    port = f":{parts.port}" if parts.port is not None else ""
    user = parts.username or ""
    auth = f"{user}:<redacted>@" if user else ""
    return urlunsplit((parts.scheme, f"{auth}{host}{port}", parts.path, "", ""))


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run retrieval-to-remote-generation E2E verification against DGX vLLM.",
    )
    parser.add_argument(
        "--database-url",
        default=os.getenv(DEFAULT_DATABASE_URL_ENV),
        help=f"PostgreSQL URL. Defaults to ${DEFAULT_DATABASE_URL_ENV}.",
    )
    parser.add_argument("--provider-name", default=DEFAULT_PROVIDER_NAME)
    parser.add_argument("--provider-base-url", default=DGX_VLLM_GENERATION_BASE_URL)
    parser.add_argument("--model-id", default=DGX_VLLM_GENERATION_MODEL_ID)
    parser.add_argument("--api-key-env", default=DGX_VLLM_GENERATION_API_KEY_ENV)
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--max-context-chars", type=int, default=DEFAULT_CONTEXT_CHAR_BUDGET)
    parser.add_argument("--max-items", type=int, default=DEFAULT_CONTEXT_MAX_ITEMS)
    parser.add_argument("--exclude-neighbors", action="store_true")
    parser.add_argument("--query-text", default=DEFAULT_QUERY_TEXT)
    parser.add_argument("--fixture-text", default=DEFAULT_FIXTURE_TEXT)
    parser.add_argument("--no-cleanup", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--markdown-output", default=None)
    parser.add_argument("--json-output", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    if not args.database_url:
        parser.error(f"--database-url or ${DEFAULT_DATABASE_URL_ENV} is required")
    api_key = os.getenv(args.api_key_env) if args.api_key_env else None
    if args.api_key_env and not api_key and not args.dry_run:
        parser.error(f"${args.api_key_env} is required for live remote generation E2E")
    try:
        plan = build_dgx_vllm_generation_run_e2e_plan(
            database_url=args.database_url,
            provider_name=args.provider_name,
            provider_base_url=args.provider_base_url,
            model_id=args.model_id,
            api_key_env=args.api_key_env,
            api_key_configured=bool(api_key),
            request_timeout_seconds=args.timeout_seconds,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            max_context_chars=args.max_context_chars,
            max_items=args.max_items,
            include_neighbors=not args.exclude_neighbors,
            query_text=args.query_text,
            fixture_text=args.fixture_text,
            cleanup_fixture=not args.no_cleanup,
        )
    except ValueError as exc:
        parser.error(str(exc))

    if args.dry_run:
        payload = {"dry_run": True, "plan": _plan_payload(plan)}
        print(json.dumps(payload, ensure_ascii=False, indent=None if args.json else 2))
        return 0

    report = run_dgx_vllm_generation_run_e2e(plan, api_key=api_key)
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
