"""FastAPI application factory for NeX_PCX."""

import traceback as traceback_module
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from app.core.admin_logging import list_logs, log_event
from app.core.config import Settings, get_settings
from app.core.database import connect
from app.core.embedding_jobs import (
    EmbeddingJobRecord,
    InvalidEmbeddingJobError,
    get_embedding_job,
    list_active_embedding_profiles,
    list_embedding_jobs,
)
from app.core.embedding_vectors import (
    EmbeddingVectorRecord,
    InvalidEmbeddingVectorError,
    get_chunk_embedding,
)
from app.core.file_metadata import (
    SUPPORTED_FILE_EXTENSIONS,
    InvalidFileMetadataError,
    UnsupportedFileExtensionError,
)
from app.core.file_uploads import InvalidUploadFileNameError, store_upload
from app.core.permissions import InvalidPermissionError
from app.core.pipeline_jobs import (
    InvalidPipelineJobError,
    PipelineJobEventRecord,
    PipelineJobListItem,
    PipelineJobRecord,
    get_pipeline_job,
    list_pipeline_job_events,
    list_pipeline_jobs,
)
from app.core.search_compare import (
    InvalidSearchCompareError,
    SearchCompareInput,
    SearchCompareProfileResult,
    SearchCompareResult,
    run_search_compare,
)
from app.core.search_logs import InvalidSearchLogError
from app.core.vector_search import InvalidVectorSearchError, VectorSearchResult

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES = Jinja2Templates(directory=BASE_DIR / "web" / "templates")
UPLOAD_FILE_FORM = File(...)
DOCUMENT_GROUP_FORM = Form("default")
SECURITY_LEVEL_FORM = Form("internal")
UPLOADED_BY_FORM = Form(None)


class SearchCompareRequest(BaseModel):
    query_text: str
    actor_user_id: int
    requested_search_scope: str = "company"
    top_k: int = Field(default=5, ge=1)
    profiles: list[str] | None = None
    chunk_policy_name: str | None = None
    document_group: str | None = None
    file_type: str | None = None


def list_search_actor_options(database_url: str) -> list[dict[str, object]]:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT user_id, login_id, display_name
                FROM app_users
                WHERE is_active
                ORDER BY login_id ASC
                """)
            return [
                {
                    "user_id": int(row["user_id"]),
                    "login_id": str(row["login_id"]),
                    "display_name": str(row["display_name"]),
                }
                for row in cursor.fetchall()
            ]


def pipeline_job_response_payload(
    pipeline_job: PipelineJobRecord | None,
) -> dict[str, object] | None:
    if pipeline_job is None:
        return None
    return {
        "job_id": pipeline_job.job_id,
        "status": pipeline_job.status,
        "stage": pipeline_job.stage,
        "progress_percent": str(pipeline_job.progress_percent),
    }


def _datetime_response(value: object | None) -> str | None:
    return value.isoformat() if hasattr(value, "isoformat") else None


def pipeline_job_detail_payload(pipeline_job: PipelineJobRecord) -> dict[str, object]:
    return {
        "job_id": pipeline_job.job_id,
        "job_type": pipeline_job.job_type,
        "file_id": pipeline_job.file_id,
        "document_id": pipeline_job.document_id,
        "parent_job_id": pipeline_job.parent_job_id,
        "requested_by_user_id": pipeline_job.requested_by_user_id,
        "status": pipeline_job.status,
        "stage": pipeline_job.stage,
        "priority": pipeline_job.priority,
        "total_units": pipeline_job.total_units,
        "processed_units": pipeline_job.processed_units,
        "progress_percent": str(pipeline_job.progress_percent),
        "current_message": pipeline_job.current_message,
        "attempts": pipeline_job.attempts,
        "max_attempts": pipeline_job.max_attempts,
        "lease_owner": pipeline_job.lease_owner,
        "lease_expires_at": _datetime_response(pipeline_job.lease_expires_at),
        "heartbeat_at": _datetime_response(pipeline_job.heartbeat_at),
        "error_code": pipeline_job.error_code,
        "error_message": pipeline_job.error_message,
        "metadata": pipeline_job.metadata,
        "queued_at": _datetime_response(pipeline_job.queued_at),
        "started_at": _datetime_response(pipeline_job.started_at),
        "finished_at": _datetime_response(pipeline_job.finished_at),
        "updated_at": _datetime_response(pipeline_job.updated_at),
    }


def pipeline_job_list_item_payload(item: PipelineJobListItem) -> dict[str, object]:
    return {
        "job": pipeline_job_detail_payload(item.job),
        "original_file_name": item.original_file_name,
        "document_title": item.document_title,
        "requested_by_login_id": item.requested_by_login_id,
        "requested_by_display_name": item.requested_by_display_name,
    }


def pipeline_job_event_payload(event: PipelineJobEventRecord) -> dict[str, object]:
    return {
        "event_id": event.event_id,
        "job_id": event.job_id,
        "event_type": event.event_type,
        "stage": event.stage,
        "status": event.status,
        "message": event.message,
        "event_metadata": event.event_metadata,
        "created_at": _datetime_response(event.created_at),
    }


def embedding_job_payload(job: EmbeddingJobRecord) -> dict[str, object]:
    return {
        "job_id": job.job_id,
        "chunk_id": job.chunk_id,
        "profile_name": job.profile_name,
        "status": job.status,
        "attempts": job.attempts,
        "max_attempts": job.max_attempts,
        "lease_owner": job.lease_owner,
        "lease_expires_at": _datetime_response(job.lease_expires_at),
        "error_code": job.error_code,
        "error_message": job.error_message,
        "last_error_at": _datetime_response(job.last_error_at),
        "runtime_metadata": job.runtime_metadata,
        "created_at": _datetime_response(job.created_at),
        "started_at": _datetime_response(job.started_at),
        "finished_at": _datetime_response(job.finished_at),
        "updated_at": _datetime_response(job.updated_at),
    }


def embedding_vector_payload(vector: EmbeddingVectorRecord | None) -> dict[str, object] | None:
    if vector is None:
        return None
    return {
        "chunk_id": vector.chunk_id,
        "profile_name": vector.profile_name,
        "table_name": vector.table_name,
        "dimension": vector.dimension,
        "storage_type": vector.storage_type,
        "elapsed_ms": vector.elapsed_ms,
        "created_at": _datetime_response(vector.created_at),
    }


def vector_search_result_payload(result: VectorSearchResult) -> dict[str, object]:
    return {
        "profile_name": result.profile_name,
        "rank": result.rank,
        "chunk_id": result.chunk_id,
        "document_id": result.document_id,
        "file_id": result.file_id,
        "distance": result.distance,
        "score": result.score,
        "chunk_preview": result.chunk_preview,
        "content_hash": result.content_hash,
        "chunk_policy_name": result.chunk_policy_name,
        "heading_path": list(result.heading_path),
        "page_no": result.page_no,
        "slide_no": result.slide_no,
        "sheet_name": result.sheet_name,
        "cell_range": result.cell_range,
        "document_title": result.document_title,
        "document_group": result.document_group,
        "original_file_name": result.original_file_name,
        "file_ext": result.file_ext,
        "embedding_elapsed_ms": result.embedding_elapsed_ms,
    }


def search_compare_profile_payload(profile: SearchCompareProfileResult) -> dict[str, object]:
    return {
        "profile_name": profile.profile_name,
        "elapsed_ms": profile.elapsed_ms,
        "results": [vector_search_result_payload(result) for result in profile.results],
    }


def search_compare_payload(result: SearchCompareResult) -> dict[str, object]:
    return {
        "search_log_id": result.search_log_id,
        "query_text": result.query_text,
        "actor_user_id": result.actor_user_id,
        "requested_search_scope": result.requested_search_scope,
        "effective_search_scope": result.effective_search_scope,
        "permission_filter_metadata": result.permission_filter.metadata,
        "top_k": result.top_k,
        "total_elapsed_ms": result.total_elapsed_ms,
        "profiles": [search_compare_profile_payload(profile) for profile in result.profiles],
    }


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(title=settings.app_name, version=settings.app_version)
    app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

    def template_context(request: Request, **context: object) -> dict[str, object]:
        return {
            "request": request,
            "app_name": settings.app_name,
            "app_version": settings.app_version,
            "environment": settings.environment,
            **context,
        }

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        if settings.database_url:
            try:
                log_event(
                    settings.database_url,
                    level="ERROR",
                    event_type="unhandled_exception",
                    source="fastapi",
                    message=str(exc) or exc.__class__.__name__,
                    traceback=traceback_module.format_exc(),
                    request_path=request.url.path,
                )
            except Exception:
                pass
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {
            "status": "ok",
            "service": settings.app_name,
            "version": settings.app_version,
            "environment": settings.environment,
        }

    def upload_template_context(
        request: Request,
        *,
        result: dict[str, object] | None = None,
        duplicate: bool = False,
        error_message: str | None = None,
        form_values: dict[str, str] | None = None,
    ) -> dict[str, object]:
        return template_context(
            request,
            database_configured=bool(settings.database_url),
            supported_file_extensions=sorted(SUPPORTED_FILE_EXTENSIONS),
            result=result,
            duplicate=duplicate,
            error_message=error_message,
            form_values=form_values
            or {
                "document_group": "default",
                "security_level": "internal",
                "uploaded_by": "",
            },
        )

    @app.post("/api/files")
    async def api_upload_file(
        file: UploadFile = UPLOAD_FILE_FORM,
        document_group: str = DOCUMENT_GROUP_FORM,
        security_level: str = SECURITY_LEVEL_FORM,
        uploaded_by: str | None = UPLOADED_BY_FORM,
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            result = store_upload(
                database_url=settings.database_url,
                upload_stream=file.file,
                original_file_name=file.filename,
                storage_dir=settings.upload_storage_dir,
                mime_type=file.content_type,
                document_group=document_group.strip() or "default",
                security_level=security_level.strip() or "internal",
                uploaded_by=uploaded_by.strip() if uploaded_by and uploaded_by.strip() else None,
            )
        except UnsupportedFileExtensionError as exc:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=str(exc),
            ) from exc
        except (InvalidUploadFileNameError, InvalidFileMetadataError) as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(
            status_code=status.HTTP_200_OK if result.duplicate else status.HTTP_201_CREATED,
            content={
                "duplicate": result.duplicate,
                "file": asdict(result.file),
                "pipeline_job_id": (
                    result.pipeline_job.job_id if result.pipeline_job is not None else None
                ),
                "pipeline_job": pipeline_job_response_payload(result.pipeline_job),
            },
        )

    @app.get("/api/pipeline/jobs")
    def api_list_pipeline_jobs(
        status_filter: str | None = Query(default=None, alias="status"),
        limit: int = 100,
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            jobs = list_pipeline_jobs(
                settings.database_url,
                status=status_filter,
                limit=limit,
            )
        except InvalidPipelineJobError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(content={"jobs": [pipeline_job_list_item_payload(job) for job in jobs]})

    @app.get("/api/pipeline/jobs/{job_id}")
    def api_get_pipeline_job(job_id: int) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            job = get_pipeline_job(settings.database_url, job_id)
            if job is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Pipeline job not found.",
                )
            events = list_pipeline_job_events(settings.database_url, job_id)
        except InvalidPipelineJobError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(
            content={
                "job": pipeline_job_detail_payload(job),
                "events": [pipeline_job_event_payload(event) for event in events],
            },
        )

    @app.get("/api/embedding/jobs")
    def api_list_embedding_jobs(
        status_filter: str | None = Query(default=None, alias="status"),
        profile_name: str | None = None,
        limit: int = 100,
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            jobs = list_embedding_jobs(
                settings.database_url,
                status=status_filter,
                profile_name=profile_name,
                limit=limit,
            )
        except InvalidEmbeddingJobError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(content={"jobs": [embedding_job_payload(job) for job in jobs]})

    @app.get("/api/embedding/jobs/{job_id}")
    def api_get_embedding_job(job_id: int) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            job = get_embedding_job(settings.database_url, job_id)
            if job is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Embedding job not found.",
                )
            vector = get_chunk_embedding(
                settings.database_url,
                profile_name=job.profile_name,
                chunk_id=job.chunk_id,
            )
        except InvalidEmbeddingJobError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        except InvalidEmbeddingVectorError:
            vector = None

        return JSONResponse(
            content={
                "job": embedding_job_payload(job),
                "embedding": embedding_vector_payload(vector),
            },
        )

    @app.post("/api/search/compare")
    def api_search_compare(payload: SearchCompareRequest) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            result = run_search_compare(
                settings.database_url,
                SearchCompareInput(
                    query_text=payload.query_text,
                    actor_user_id=payload.actor_user_id,
                    requested_search_scope=payload.requested_search_scope,
                    top_k=payload.top_k,
                    profiles=tuple(payload.profiles) if payload.profiles is not None else None,
                    chunk_policy_name=payload.chunk_policy_name,
                    document_group=payload.document_group,
                    file_type=payload.file_type,
                ),
            )
        except (
            InvalidSearchCompareError,
            InvalidPermissionError,
            InvalidVectorSearchError,
            InvalidSearchLogError,
        ) as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(content=search_compare_payload(result))

    @app.get("/files/upload", response_class=HTMLResponse)
    def upload_file_page(request: Request) -> HTMLResponse:
        error_message = None
        if not settings.database_url:
            error_message = "NEX_PCX_DATABASE_URL is not configured."

        return TEMPLATES.TemplateResponse(
            request,
            "file_upload.html",
            upload_template_context(request, error_message=error_message),
        )

    @app.post("/files/upload", response_class=HTMLResponse)
    async def submit_upload_file(
        request: Request,
        file: UploadFile = UPLOAD_FILE_FORM,
        document_group: str = DOCUMENT_GROUP_FORM,
        security_level: str = SECURITY_LEVEL_FORM,
        uploaded_by: str | None = UPLOADED_BY_FORM,
    ) -> HTMLResponse:
        form_values = {
            "document_group": document_group.strip() or "default",
            "security_level": security_level.strip() or "internal",
            "uploaded_by": uploaded_by.strip() if uploaded_by and uploaded_by.strip() else "",
        }
        result_payload = None
        duplicate = False
        error_message = None

        if not settings.database_url:
            error_message = "NEX_PCX_DATABASE_URL is not configured."
        else:
            try:
                result = store_upload(
                    database_url=settings.database_url,
                    upload_stream=file.file,
                    original_file_name=file.filename,
                    storage_dir=settings.upload_storage_dir,
                    mime_type=file.content_type,
                    document_group=form_values["document_group"],
                    security_level=form_values["security_level"],
                    uploaded_by=form_values["uploaded_by"] or None,
                )
                result_payload = asdict(result.file)
                result_payload["pipeline_job"] = pipeline_job_response_payload(
                    result.pipeline_job,
                )
                result_payload["pipeline_job_id"] = (
                    result.pipeline_job.job_id if result.pipeline_job is not None else None
                )
                duplicate = result.duplicate
            except (
                UnsupportedFileExtensionError,
                InvalidUploadFileNameError,
                InvalidFileMetadataError,
            ) as exc:
                error_message = str(exc)

        return TEMPLATES.TemplateResponse(
            request,
            "file_upload.html",
            upload_template_context(
                request,
                result=result_payload,
                duplicate=duplicate,
                error_message=error_message,
                form_values=form_values,
            ),
        )

    @app.get("/", response_class=HTMLResponse)
    def dashboard(request: Request) -> HTMLResponse:
        return TEMPLATES.TemplateResponse(
            request,
            "dashboard.html",
            template_context(request),
        )

    @app.get("/search", response_class=HTMLResponse)
    def search_compare_page(request: Request) -> HTMLResponse:
        actor_options: list[dict[str, object]] = []
        profile_options: list[str] = []
        error_message = None

        if not settings.database_url:
            error_message = "NEX_PCX_DATABASE_URL is not configured."
        else:
            try:
                actor_options = list_search_actor_options(settings.database_url)
                profile_options = [
                    profile.profile_name
                    for profile in list_active_embedding_profiles(settings.database_url)
                ]
            except Exception as exc:
                error_message = str(exc)

        return TEMPLATES.TemplateResponse(
            request,
            "search_compare.html",
            template_context(
                request,
                actor_options=actor_options,
                profile_options=profile_options,
                default_actor_id=actor_options[0]["user_id"] if actor_options else "",
                error_message=error_message,
                database_configured=bool(settings.database_url),
            ),
        )

    @app.get("/admin/jobs", response_class=HTMLResponse)
    def pipeline_jobs_page(
        request: Request,
        status_filter: str | None = Query(default=None, alias="status"),
        job_id: int | None = None,
    ) -> HTMLResponse:
        jobs: list[PipelineJobListItem] = []
        selected_job = None
        selected_events: list[PipelineJobEventRecord] = []
        error_message = None

        if not settings.database_url:
            error_message = "NEX_PCX_DATABASE_URL is not configured."
        else:
            try:
                jobs = list_pipeline_jobs(settings.database_url, status=status_filter)
                if job_id is not None:
                    selected_job = get_pipeline_job(settings.database_url, job_id)
                    if selected_job is None:
                        error_message = f"Pipeline job not found: {job_id}"
                    else:
                        selected_events = list_pipeline_job_events(settings.database_url, job_id)
            except InvalidPipelineJobError as exc:
                error_message = str(exc)

        return TEMPLATES.TemplateResponse(
            request,
            "pipeline_jobs.html",
            template_context(
                request,
                jobs=jobs,
                selected_job=selected_job,
                selected_events=selected_events,
                selected_status=status_filter or "",
                selected_job_id=job_id,
                error_message=error_message,
                database_configured=bool(settings.database_url),
            ),
        )

    @app.get("/admin/embedding-jobs", response_class=HTMLResponse)
    def embedding_jobs_page(
        request: Request,
        status_filter: str | None = Query(default=None, alias="status"),
        profile_name: str | None = None,
        job_id: int | None = None,
    ) -> HTMLResponse:
        jobs: list[EmbeddingJobRecord] = []
        selected_job = None
        selected_vector = None
        error_message = None

        if not settings.database_url:
            error_message = "NEX_PCX_DATABASE_URL is not configured."
        else:
            try:
                jobs = list_embedding_jobs(
                    settings.database_url,
                    status=status_filter,
                    profile_name=profile_name,
                )
                if job_id is not None:
                    selected_job = get_embedding_job(settings.database_url, job_id)
                    if selected_job is None:
                        error_message = f"Embedding job not found: {job_id}"
                    else:
                        try:
                            selected_vector = get_chunk_embedding(
                                settings.database_url,
                                profile_name=selected_job.profile_name,
                                chunk_id=selected_job.chunk_id,
                            )
                        except InvalidEmbeddingVectorError:
                            selected_vector = None
            except InvalidEmbeddingJobError as exc:
                error_message = str(exc)

        return TEMPLATES.TemplateResponse(
            request,
            "embedding_jobs.html",
            template_context(
                request,
                jobs=jobs,
                selected_job=selected_job,
                selected_vector=selected_vector,
                selected_status=status_filter or "",
                selected_profile_name=profile_name or "",
                selected_job_id=job_id,
                error_message=error_message,
                database_configured=bool(settings.database_url),
            ),
        )

    @app.get("/admin/logs", response_class=HTMLResponse)
    def admin_logs(request: Request, level: str | None = None) -> HTMLResponse:
        logs = []
        error_message = None
        if settings.database_url:
            try:
                logs = list_logs(settings.database_url, level=level)
            except Exception as exc:
                error_message = str(exc)
        else:
            error_message = "NEX_PCX_DATABASE_URL is not configured."

        return TEMPLATES.TemplateResponse(
            request,
            "admin_logs.html",
            template_context(
                request,
                logs=logs,
                selected_level=level or "",
                error_message=error_message,
            ),
        )

    return app


app = create_app()
