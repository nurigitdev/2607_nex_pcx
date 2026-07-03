"""FastAPI application factory for NeX_PCX."""

import traceback as traceback_module
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.core.admin_logging import list_logs, log_event
from app.core.config import Settings, get_settings
from app.core.file_metadata import InvalidFileMetadataError, UnsupportedFileExtensionError
from app.core.file_uploads import InvalidUploadFileNameError, store_upload

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES = Jinja2Templates(directory=BASE_DIR / "web" / "templates")
UPLOAD_FILE_FORM = File(...)
DOCUMENT_GROUP_FORM = Form("default")
SECURITY_LEVEL_FORM = Form("internal")
UPLOADED_BY_FORM = Form(None)


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

    @app.post("/api/files")
    async def upload_file(
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
            content={"duplicate": result.duplicate, "file": asdict(result.file)},
        )

    @app.get("/", response_class=HTMLResponse)
    def dashboard(request: Request) -> HTMLResponse:
        return TEMPLATES.TemplateResponse(
            request,
            "dashboard.html",
            template_context(request),
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
