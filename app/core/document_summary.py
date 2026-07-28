"""Document summary orchestration from stored chunks to generation runs."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from app.core.database import connect
from app.core.document_inventory import DocumentInventoryItem, get_document_inventory_item
from app.core.generation_executor import (
    GenerationExecutionReport,
    execute_mock_generation_run,
    execute_remote_generation_run,
)
from app.core.generation_providers import GenerationProvider
from app.core.generation_runs import (
    DEFAULT_GENERATION_RUN_HISTORY_LIMIT,
    GENERATION_PROVIDER_MODE_MOCK,
    GENERATION_PROVIDER_MODE_REMOTE_OPENAI_COMPATIBLE,
    GENERATION_RUN_HISTORY_FILTER_ALL,
    GENERATION_STATUSES,
    MAX_GENERATION_RUN_HISTORY_LIMIT,
    GenerationRunRecord,
    _generation_run_from_row,
    _history_item_from_run,
)
from app.core.retrieval_context import (
    DEFAULT_CONTEXT_CHAR_BUDGET,
    RetrievalContextInput,
    RetrievalContextPackage,
    build_retrieval_context_package,
    validate_retrieval_context_input,
)
from app.core.search_logs import (
    SearchLogInput,
    SearchLogResultInput,
    create_search_log_in_connection,
    create_search_log_result_in_connection,
)

DOCUMENT_SUMMARY_PROFILE_NAME = "bm25_keyword"
DOCUMENT_SUMMARY_STRATEGY_NAME = "document_summary"
DEFAULT_DOCUMENT_SUMMARY_TEMPLATE_KEY = "summary"
DEFAULT_DOCUMENT_SUMMARY_MAX_CHUNKS = 20
MAX_DOCUMENT_SUMMARY_MAX_CHUNKS = 100
DEFAULT_DOCUMENT_SUMMARY_HISTORY_LIMIT = DEFAULT_GENERATION_RUN_HISTORY_LIMIT
MAX_DOCUMENT_SUMMARY_HISTORY_LIMIT = MAX_GENERATION_RUN_HISTORY_LIMIT

DOCUMENT_SUMMARY_PROVIDER_MODES = {
    GENERATION_PROVIDER_MODE_MOCK,
    GENERATION_PROVIDER_MODE_REMOTE_OPENAI_COMPATIBLE,
}


@dataclass(frozen=True)
class DocumentSummaryInput:
    document_id: int
    actor_user_id: int
    summary_instruction: str = ""
    provider_mode: str = GENERATION_PROVIDER_MODE_REMOTE_OPENAI_COMPATIBLE
    generation_template_key: str | None = DEFAULT_DOCUMENT_SUMMARY_TEMPLATE_KEY
    max_chunks: int = DEFAULT_DOCUMENT_SUMMARY_MAX_CHUNKS
    max_context_chars: int = DEFAULT_CONTEXT_CHAR_BUDGET
    include_neighbors: bool = False
    chunk_policy_name: str | None = None


@dataclass(frozen=True)
class DocumentSummaryResult:
    document: DocumentInventoryItem
    source_chunk_ids: tuple[int, ...]
    retrieval_package: RetrievalContextPackage
    generation_report: GenerationExecutionReport

    @property
    def search_log_id(self) -> int:
        return self.retrieval_package.search_log.search_log.search_log_id


@dataclass(frozen=True)
class DocumentSummaryHistoryFilter:
    limit: int = DEFAULT_DOCUMENT_SUMMARY_HISTORY_LIMIT
    run_status: str = GENERATION_RUN_HISTORY_FILTER_ALL
    generation_template_key: str = GENERATION_RUN_HISTORY_FILTER_ALL


@dataclass(frozen=True)
class DocumentSummaryHistoryItem:
    run: GenerationRunRecord
    document_id: int | None
    file_id: int | None
    document_title: str | None
    original_file_name: str | None
    document_group: str | None
    file_type: str | None
    template_key: str
    template_name: str
    summary_instruction: str
    source_chunk_count: int
    answer_quality_status: str
    answer_quality_reason_codes: tuple[str, ...]
    citation_coverage_percent: float | None
    expected_citation_count: int
    cited_citation_count: int
    missing_citation_count: int
    unrecognized_citation_count: int
    created_at: datetime

    @property
    def document_label(self) -> str:
        return self.document_title or self.original_file_name or "-"


@dataclass(frozen=True)
class DocumentSummaryHistorySummary:
    run_count: int
    succeeded_count: int
    failed_count: int
    no_answer_count: int
    latest_created_at: datetime | None


@dataclass(frozen=True)
class DocumentSummaryHistory:
    filters: DocumentSummaryHistoryFilter
    summary: DocumentSummaryHistorySummary
    runs: tuple[DocumentSummaryHistoryItem, ...]


@dataclass(frozen=True)
class DocumentSummaryReadiness:
    ready_document_count: int
    summarized_document_count: int
    unsummarized_document_count: int
    summary_run_count: int
    latest_summary_run_at: datetime | None


class InvalidDocumentSummaryError(ValueError):
    """Raised when a document summary run cannot be created."""


def _validate_positive(value: int, field_name: str) -> int:
    if value <= 0:
        raise InvalidDocumentSummaryError(f"{field_name} must be greater than 0")
    return value


def _validate_provider_mode(provider_mode: str) -> str:
    normalized = provider_mode.strip().lower()
    if normalized not in DOCUMENT_SUMMARY_PROVIDER_MODES:
        raise InvalidDocumentSummaryError("provider_mode is not supported")
    return normalized


def _validate_document_summary_history_limit(limit: int) -> int:
    if limit < 1 or limit > MAX_DOCUMENT_SUMMARY_HISTORY_LIMIT:
        raise InvalidDocumentSummaryError(
            "limit must be between 1 and " f"{MAX_DOCUMENT_SUMMARY_HISTORY_LIMIT}"
        )
    return limit


def _validate_document_summary_history_status(run_status: str) -> str:
    normalized = (run_status or GENERATION_RUN_HISTORY_FILTER_ALL).strip().lower()
    if not normalized:
        return GENERATION_RUN_HISTORY_FILTER_ALL
    if normalized == GENERATION_RUN_HISTORY_FILTER_ALL:
        return normalized
    if normalized not in GENERATION_STATUSES:
        raise InvalidDocumentSummaryError("run_status is not supported")
    return normalized


def _validate_document_summary_history_template_key(template_key: str) -> str:
    normalized = (template_key or GENERATION_RUN_HISTORY_FILTER_ALL).strip().lower()
    return normalized or GENERATION_RUN_HISTORY_FILTER_ALL


def validate_document_summary_history_filter(
    history_filter: DocumentSummaryHistoryFilter,
) -> DocumentSummaryHistoryFilter:
    return DocumentSummaryHistoryFilter(
        limit=_validate_document_summary_history_limit(history_filter.limit),
        run_status=_validate_document_summary_history_status(history_filter.run_status),
        generation_template_key=_validate_document_summary_history_template_key(
            history_filter.generation_template_key
        ),
    )


def _validate_chunk_policy_name(chunk_policy_name: str | None) -> str | None:
    if chunk_policy_name is None:
        return None
    normalized = chunk_policy_name.strip()
    return normalized or None


def _validate_summary_input(summary_input: DocumentSummaryInput) -> DocumentSummaryInput:
    _validate_positive(summary_input.document_id, "document_id")
    _validate_positive(summary_input.actor_user_id, "actor_user_id")
    if summary_input.max_chunks <= 0:
        raise InvalidDocumentSummaryError("max_chunks must be greater than 0")
    if summary_input.max_chunks > MAX_DOCUMENT_SUMMARY_MAX_CHUNKS:
        raise InvalidDocumentSummaryError(
            f"max_chunks must be less than or equal to {MAX_DOCUMENT_SUMMARY_MAX_CHUNKS}"
        )
    context_input = validate_retrieval_context_input(
        RetrievalContextInput(
            search_log_id=1,
            max_context_chars=summary_input.max_context_chars,
            include_neighbors=summary_input.include_neighbors,
            max_items=summary_input.max_chunks,
        )
    )
    template_key = (
        summary_input.generation_template_key.strip()
        if summary_input.generation_template_key
        else DEFAULT_DOCUMENT_SUMMARY_TEMPLATE_KEY
    )
    return DocumentSummaryInput(
        document_id=summary_input.document_id,
        actor_user_id=summary_input.actor_user_id,
        summary_instruction=summary_input.summary_instruction.strip(),
        provider_mode=_validate_provider_mode(summary_input.provider_mode),
        generation_template_key=template_key or DEFAULT_DOCUMENT_SUMMARY_TEMPLATE_KEY,
        max_chunks=summary_input.max_chunks,
        max_context_chars=context_input.max_context_chars,
        include_neighbors=context_input.include_neighbors,
        chunk_policy_name=_validate_chunk_policy_name(summary_input.chunk_policy_name),
    )


def _document_title(document: DocumentInventoryItem) -> str:
    return document.document_title or document.original_file_name


def _summary_query_text(
    document: DocumentInventoryItem,
    *,
    summary_instruction: str,
) -> str:
    title = _document_title(document)
    if summary_instruction:
        return f"{title} 요약 요청: {summary_instruction}"
    return f"{title} 문서를 핵심 내용 중심으로 요약해줘"


def _fetch_document_summary_chunk_ids(
    database_url: str,
    *,
    document_id: int,
    max_chunks: int,
    chunk_policy_name: str | None,
) -> tuple[int, ...]:
    filters = ["document_id = %s", "btrim(coalesce(chunk_text, '')) <> ''"]
    params: list[object] = [document_id]
    if chunk_policy_name is not None:
        filters.append("chunk_policy_name = %s")
        params.append(chunk_policy_name)
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT chunk_id
                FROM chunks
                WHERE {" AND ".join(filters)}
                ORDER BY chunk_seq ASC, chunk_id ASC
                LIMIT %s
                """,
                [*params, max_chunks],
            )
            return tuple(int(row["chunk_id"]) for row in cursor.fetchall())


def _document_summary_history_item_from_row(row: dict[str, Any]) -> DocumentSummaryHistoryItem:
    runtime_metadata = dict(row.get("query_runtime_metadata") or {})
    template_key = str(row.get("summary_template_key") or "").strip()
    if not template_key:
        template_key = str(runtime_metadata.get("generation_template_key") or "").strip()
    template_name = str(row.get("summary_template_name") or template_key or "-")
    run = _generation_run_from_row(row)
    quality = _history_item_from_run(run)
    return DocumentSummaryHistoryItem(
        run=run,
        document_id=row["summary_document_id"],
        file_id=row["summary_file_id"],
        document_title=row["document_title"],
        original_file_name=row["original_file_name"],
        document_group=row["summary_document_group"],
        file_type=row["summary_file_type"],
        template_key=template_key or "-",
        template_name=template_name,
        summary_instruction=str(runtime_metadata.get("summary_instruction") or ""),
        source_chunk_count=int(row["source_chunk_count"] or 0),
        answer_quality_status=quality.answer_quality_status,
        answer_quality_reason_codes=quality.answer_quality_reason_codes,
        citation_coverage_percent=quality.citation_coverage_percent,
        expected_citation_count=quality.expected_citation_count,
        cited_citation_count=quality.cited_citation_count,
        missing_citation_count=quality.missing_citation_count,
        unrecognized_citation_count=quality.unrecognized_citation_count,
        created_at=row["created_at"],
    )


def _document_summary_history_summary(
    runs: tuple[DocumentSummaryHistoryItem, ...],
) -> DocumentSummaryHistorySummary:
    return DocumentSummaryHistorySummary(
        run_count=len(runs),
        succeeded_count=sum(1 for item in runs if item.run.status == "succeeded"),
        failed_count=sum(1 for item in runs if item.run.status == "failed"),
        no_answer_count=sum(1 for item in runs if item.run.status == "no_answer"),
        latest_created_at=runs[0].created_at if runs else None,
    )


def list_document_summary_history(
    database_url: str,
    *,
    history_filter: DocumentSummaryHistoryFilter | None = None,
) -> DocumentSummaryHistory:
    validated = validate_document_summary_history_filter(
        history_filter or DocumentSummaryHistoryFilter()
    )
    with connect(database_url) as connection:
        rows = connection.execute(
            """
            SELECT
                gr.*,
                sl.query_runtime_metadata,
                sl.document_group AS summary_document_group,
                sl.file_type AS summary_file_type,
                CASE
                    WHEN (sl.query_runtime_metadata ->> 'document_id') ~ '^[0-9]+$'
                        THEN (sl.query_runtime_metadata ->> 'document_id')::bigint
                    ELSE NULL
                END AS summary_document_id,
                CASE
                    WHEN (sl.query_runtime_metadata ->> 'file_id') ~ '^[0-9]+$'
                        THEN (sl.query_runtime_metadata ->> 'file_id')::bigint
                    ELSE NULL
                END AS summary_file_id,
                CASE
                    WHEN jsonb_typeof(sl.query_runtime_metadata -> 'source_chunk_ids') = 'array'
                        THEN jsonb_array_length(sl.query_runtime_metadata -> 'source_chunk_ids')
                    ELSE 0
                END AS source_chunk_count,
                d.document_title,
                f.original_file_name,
                COALESCE(
                    gt.template_key,
                    gr.request_metadata #>> '{generation_template,template_key}',
                    sl.query_runtime_metadata ->> 'generation_template_key',
                    ''
                ) AS summary_template_key,
                COALESCE(
                    gt.template_name,
                    gr.request_metadata #>> '{generation_template,template_name}',
                    ''
                ) AS summary_template_name
            FROM generation_runs gr
            JOIN search_logs sl
              ON sl.search_log_id = gr.search_log_id
            LEFT JOIN documents d
              ON d.document_id = CASE
                    WHEN (sl.query_runtime_metadata ->> 'document_id') ~ '^[0-9]+$'
                        THEN (sl.query_runtime_metadata ->> 'document_id')::bigint
                    ELSE NULL
                 END
            LEFT JOIN files f
              ON f.file_id = COALESCE(
                    d.file_id,
                    CASE
                        WHEN (sl.query_runtime_metadata ->> 'file_id') ~ '^[0-9]+$'
                            THEN (sl.query_runtime_metadata ->> 'file_id')::bigint
                        ELSE NULL
                    END
                 )
            LEFT JOIN generation_templates gt
              ON gt.generation_template_id = gr.generation_template_id
            WHERE (
                sl.strategy_name = %s
                OR sl.query_runtime_metadata ->> 'operation' = 'document_summary'
                OR gr.created_by = 'api_document_summary'
            )
              AND (%s = 'all' OR gr.status = %s)
              AND (
                %s = 'all'
                OR COALESCE(
                    gt.template_key,
                    gr.request_metadata #>> '{generation_template,template_key}',
                    sl.query_runtime_metadata ->> 'generation_template_key',
                    ''
                ) = %s
              )
            ORDER BY gr.created_at DESC, gr.generation_run_id DESC
            LIMIT %s
            """,
            (
                DOCUMENT_SUMMARY_STRATEGY_NAME,
                validated.run_status,
                validated.run_status,
                validated.generation_template_key,
                validated.generation_template_key,
                validated.limit,
            ),
        ).fetchall()
    runs = tuple(_document_summary_history_item_from_row(dict(row)) for row in rows)
    return DocumentSummaryHistory(
        filters=validated,
        summary=_document_summary_history_summary(runs),
        runs=runs,
    )


def get_document_summary_readiness(database_url: str) -> DocumentSummaryReadiness:
    with connect(database_url) as connection:
        row = connection.execute(
            """
            WITH ready_documents AS (
                SELECT d.document_id
                FROM documents d
                JOIN chunks c
                  ON c.document_id = d.document_id
                WHERE btrim(COALESCE(c.chunk_text, '')) <> ''
                GROUP BY d.document_id
            ),
            summarized_documents AS (
                SELECT DISTINCT
                    CASE
                        WHEN (sl.query_runtime_metadata ->> 'document_id') ~ '^[0-9]+$'
                            THEN (sl.query_runtime_metadata ->> 'document_id')::bigint
                        ELSE NULL
                    END AS document_id
                FROM generation_runs gr
                JOIN search_logs sl
                  ON sl.search_log_id = gr.search_log_id
                WHERE (
                    sl.strategy_name = %s
                    OR sl.query_runtime_metadata ->> 'operation' = 'document_summary'
                    OR gr.created_by = 'api_document_summary'
                )
            ),
            summary_runs AS (
                SELECT gr.generation_run_id, gr.created_at
                FROM generation_runs gr
                JOIN search_logs sl
                  ON sl.search_log_id = gr.search_log_id
                WHERE (
                    sl.strategy_name = %s
                    OR sl.query_runtime_metadata ->> 'operation' = 'document_summary'
                    OR gr.created_by = 'api_document_summary'
                )
            )
            SELECT
                count(rd.document_id) AS ready_document_count,
                count(rd.document_id) FILTER (
                    WHERE sd.document_id IS NOT NULL
                ) AS summarized_document_count,
                count(rd.document_id) FILTER (
                    WHERE sd.document_id IS NULL
                ) AS unsummarized_document_count,
                (SELECT count(*) FROM summary_runs) AS summary_run_count,
                (SELECT max(created_at) FROM summary_runs) AS latest_summary_run_at
            FROM ready_documents rd
            LEFT JOIN summarized_documents sd
              ON sd.document_id = rd.document_id
            """,
            (DOCUMENT_SUMMARY_STRATEGY_NAME, DOCUMENT_SUMMARY_STRATEGY_NAME),
        ).fetchone()
    return DocumentSummaryReadiness(
        ready_document_count=int(row["ready_document_count"] or 0),
        summarized_document_count=int(row["summarized_document_count"] or 0),
        unsummarized_document_count=int(row["unsummarized_document_count"] or 0),
        summary_run_count=int(row["summary_run_count"] or 0),
        latest_summary_run_at=row["latest_summary_run_at"],
    )


def _create_document_summary_search_log(
    database_url: str,
    *,
    document: DocumentInventoryItem,
    summary_input: DocumentSummaryInput,
    query_text: str,
    chunk_ids: tuple[int, ...],
) -> int:
    with connect(database_url) as connection:
        search_log = create_search_log_in_connection(
            connection,
            SearchLogInput(
                query_text=query_text,
                normalized_query_text=query_text.lower(),
                actor_user_id=summary_input.actor_user_id,
                requested_search_scope="company",
                effective_search_scope="company",
                permission_filter_metadata={
                    "source": "document_summary",
                    "document_id": document.document_id,
                    "access_scope": document.access_scope,
                    "owner_user_id": document.owner_user_id,
                    "owner_org_unit_id": document.owner_org_unit_id,
                },
                document_group=document.document_group,
                file_type=document.file_ext,
                chunk_policy_name=summary_input.chunk_policy_name,
                strategy_name=DOCUMENT_SUMMARY_STRATEGY_NAME,
                top_k=len(chunk_ids),
                similarity_metric="bm25",
                profiles=(DOCUMENT_SUMMARY_PROFILE_NAME,),
                query_runtime_metadata={
                    "operation": "document_summary",
                    "document_id": document.document_id,
                    "file_id": document.file_id,
                    "summary_instruction": summary_input.summary_instruction,
                    "chunk_source_strategy": "document_chunk_sequence",
                    "source_chunk_ids": list(chunk_ids),
                    "max_chunks": summary_input.max_chunks,
                    "include_neighbors": summary_input.include_neighbors,
                    "generation_template_key": summary_input.generation_template_key,
                },
                total_elapsed_ms=0,
                created_by="document_summary",
                created_by_user_id=summary_input.actor_user_id,
            ),
        )
        for rank, chunk_id in enumerate(chunk_ids, start=1):
            create_search_log_result_in_connection(
                connection,
                SearchLogResultInput(
                    search_log_id=search_log.search_log_id,
                    profile_name=DOCUMENT_SUMMARY_PROFILE_NAME,
                    search_profile_name=DOCUMENT_SUMMARY_PROFILE_NAME,
                    retrieval_strategy=DOCUMENT_SUMMARY_STRATEGY_NAME,
                    rank=rank,
                    chunk_id=chunk_id,
                    score=1.0,
                    score_components={
                        "source": "document_chunk_sequence",
                        "document_id": document.document_id,
                        "rank_weight": max(len(chunk_ids) - rank + 1, 1),
                    },
                    profile_elapsed_ms=0,
                ),
            )
    return search_log.search_log_id


def _generation_report_for_provider_mode(
    database_url: str,
    *,
    summary_input: DocumentSummaryInput,
    retrieval_package: RetrievalContextPackage,
    api_key: str | None,
    generation_provider_client: GenerationProvider | None,
) -> GenerationExecutionReport:
    if summary_input.provider_mode == GENERATION_PROVIDER_MODE_MOCK:
        return execute_mock_generation_run(
            database_url,
            retrieval_package,
            generation_template_key=summary_input.generation_template_key,
            created_by="api_document_summary",
            created_by_user_id=summary_input.actor_user_id,
        )
    return execute_remote_generation_run(
        database_url,
        retrieval_package,
        generation_template_key=summary_input.generation_template_key,
        provider_client=generation_provider_client,
        api_key=api_key,
        created_by="api_document_summary",
        created_by_user_id=summary_input.actor_user_id,
    )


def document_summary_metadata_payload(result: DocumentSummaryResult) -> dict[str, Any]:
    return {
        "document_id": result.document.document_id,
        "file_id": result.document.file_id,
        "document_title": result.document.document_title,
        "original_file_name": result.document.original_file_name,
        "document_group": result.document.document_group,
        "file_ext": result.document.file_ext,
        "source_chunk_count": len(result.source_chunk_ids),
        "source_chunk_ids": list(result.source_chunk_ids),
    }


def run_document_summary_generation(
    database_url: str,
    summary_input: DocumentSummaryInput,
    *,
    api_key: str | None = None,
    generation_provider_client: GenerationProvider | None = None,
) -> DocumentSummaryResult:
    """Create a generation run that summarizes an already-ingested document."""

    validated = _validate_summary_input(summary_input)
    document = get_document_inventory_item(database_url, validated.document_id)
    if document is None:
        raise InvalidDocumentSummaryError("document was not found")
    chunk_ids = _fetch_document_summary_chunk_ids(
        database_url,
        document_id=document.document_id,
        max_chunks=validated.max_chunks,
        chunk_policy_name=validated.chunk_policy_name,
    )
    if not chunk_ids:
        raise InvalidDocumentSummaryError("document has no summary-ready chunks")

    search_log_id = _create_document_summary_search_log(
        database_url,
        document=document,
        summary_input=validated,
        query_text=_summary_query_text(
            document,
            summary_instruction=validated.summary_instruction,
        ),
        chunk_ids=chunk_ids,
    )
    retrieval_package = build_retrieval_context_package(
        database_url,
        RetrievalContextInput(
            search_log_id=search_log_id,
            max_context_chars=validated.max_context_chars,
            include_neighbors=validated.include_neighbors,
            max_items=validated.max_chunks,
        ),
    )
    if retrieval_package is None:
        raise InvalidDocumentSummaryError("retrieval context package was not created")

    generation_report = _generation_report_for_provider_mode(
        database_url,
        summary_input=validated,
        retrieval_package=retrieval_package,
        api_key=api_key,
        generation_provider_client=generation_provider_client,
    )
    return DocumentSummaryResult(
        document=document,
        source_chunk_ids=chunk_ids,
        retrieval_package=retrieval_package,
        generation_report=generation_report,
    )
