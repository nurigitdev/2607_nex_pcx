"""Standalone reranker provider service for Qwen-style reranking."""

from dataclasses import dataclass
from math import isfinite
from os import getenv
from pathlib import Path
from time import perf_counter
from typing import Any

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.core.rerankers import (
    DEFAULT_RERANKER_MODEL_ID,
    DEFAULT_RERANKER_PROFILE_NAME,
    REMOTE_RERANKER_PROVIDER_MODE,
    RERANK_RETRIEVAL_STRATEGY,
    InvalidRerankerError,
    MockLexicalOverlapReranker,
    RerankCandidate,
    RerankRequest,
    RerankResult,
    RerankResultItem,
    validate_rerank_request,
)

RERANKER_PROVIDER_BACKEND_MOCK = "mock"
RERANKER_PROVIDER_BACKEND_QWEN = "qwen_reranker"
DEFAULT_RERANKER_MODEL_DIR_NAME = "qwen3_reranker_4b"


@dataclass(frozen=True)
class RerankerProviderServiceSettings:
    backend: str = RERANKER_PROVIDER_BACKEND_MOCK
    provider_model_id: str = DEFAULT_RERANKER_MODEL_ID
    reranker_profile_name: str = DEFAULT_RERANKER_PROFILE_NAME
    device: str = "cpu"
    ready: bool = True
    models_dir: Path = Path("models")
    model_dir_name: str = DEFAULT_RERANKER_MODEL_DIR_NAME

    @property
    def local_model_dir(self) -> Path:
        return self.models_dir / self.model_dir_name


class RerankCandidateBody(BaseModel):
    candidate_key: str
    rank: int
    text: str
    source_profile_name: str
    source_retrieval_strategy: str
    source_score: float
    chunk_id: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RerankRequestBody(BaseModel):
    query_text: str
    top_k: int
    reranker_profile_name: str = DEFAULT_RERANKER_PROFILE_NAME
    reranker_model_id: str = DEFAULT_RERANKER_MODEL_ID
    candidates: list[RerankCandidateBody]


class SkeletonRerankerProvider:
    def __init__(self, settings: RerankerProviderServiceSettings) -> None:
        self.settings = settings
        self._reranker = MockLexicalOverlapReranker()

    def rerank(self, request: RerankRequest) -> RerankResult:
        validated = _validate_service_request(request, self.settings)
        started_at = perf_counter()
        mock_result = self._reranker.rerank(validated)
        elapsed_ms = max(0, int((perf_counter() - started_at) * 1000))
        return _copy_result_for_remote_service(
            mock_result,
            provider_model_id=self.settings.provider_model_id,
            backend=RERANKER_PROVIDER_BACKEND_MOCK,
            device=self.settings.device,
            model_source="mock",
            elapsed_ms=elapsed_ms,
        )


class QwenCrossEncoderRerankerProvider:
    def __init__(self, settings: RerankerProviderServiceSettings) -> None:
        self.settings = settings
        self.model_source = str(settings.local_model_dir)
        if not settings.local_model_dir.is_dir():
            raise ValueError(f"Reranker model directory does not exist: {settings.local_model_dir}")
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ValueError(
                "sentence-transformers is required for qwen_reranker. Install it with "
                '`./.venv/bin/pip install -e ".[models]"`.'
            ) from exc

        model_kwargs = {}
        if settings.device:
            model_kwargs["device"] = settings.device
        self._model = CrossEncoder(self.model_source, **model_kwargs)

    def rerank(self, request: RerankRequest) -> RerankResult:
        validated = _validate_service_request(request, self.settings)
        started_at = perf_counter()
        pairs = [(validated.query_text, candidate.text) for candidate in validated.candidates]
        scores = _coerce_scores(self._model.predict(pairs), len(pairs))  # type: ignore[attr-defined]
        ordered_items = sorted(
            zip(validated.candidates, scores, strict=True),
            key=lambda item: (-item[1], item[0].rank),
        )[: validated.top_k]
        elapsed_ms = max(0, int((perf_counter() - started_at) * 1000))
        return RerankResult(
            query_text=validated.query_text,
            reranker_profile_name=validated.reranker_profile_name,
            reranker_model_id=validated.reranker_model_id,
            provider_type=REMOTE_RERANKER_PROVIDER_MODE,
            retrieval_strategy=RERANK_RETRIEVAL_STRATEGY,
            candidate_count=len(validated.candidates),
            returned_count=len(ordered_items),
            top_k=validated.top_k,
            results=tuple(
                RerankResultItem(
                    candidate=candidate,
                    rank=index,
                    score=score,
                    score_components={
                        "rerank_rank": index,
                        "raw_cross_encoder_score": score,
                        "source_rank": candidate.rank,
                    },
                )
                for index, (candidate, score) in enumerate(ordered_items, start=1)
            ),
            runtime_metadata=_runtime_metadata(
                backend=RERANKER_PROVIDER_BACKEND_QWEN,
                device=self.settings.device,
                model_source=self.model_source,
                elapsed_ms=elapsed_ms,
                input_count=len(validated.candidates),
            ),
        )


def get_reranker_provider_service_settings() -> RerankerProviderServiceSettings:
    app_settings = get_settings()
    return RerankerProviderServiceSettings(
        backend=getenv(
            "NEX_PCX_RERANKER_PROVIDER_BACKEND",
            RERANKER_PROVIDER_BACKEND_MOCK,
        )
        .strip()
        .lower(),
        provider_model_id=getenv("NEX_PCX_RERANKER_PROVIDER_MODEL_ID", DEFAULT_RERANKER_MODEL_ID),
        reranker_profile_name=getenv(
            "NEX_PCX_RERANKER_PROVIDER_PROFILE_NAME",
            DEFAULT_RERANKER_PROFILE_NAME,
        ),
        device=getenv("NEX_PCX_RERANKER_PROVIDER_DEVICE", "cpu"),
        ready=_parse_bool(getenv("NEX_PCX_RERANKER_PROVIDER_READY", "true")),
        models_dir=Path(
            getenv(
                "NEX_PCX_RERANKER_PROVIDER_MODELS_DIR",
                str(app_settings.embedding_models_dir),
            )
        ),
        model_dir_name=getenv(
            "NEX_PCX_RERANKER_PROVIDER_MODEL_DIR_NAME",
            DEFAULT_RERANKER_MODEL_DIR_NAME,
        ),
    )


def create_app(
    settings: RerankerProviderServiceSettings | None = None,
    *,
    provider: object | None = None,
) -> FastAPI:
    provider_settings = settings or get_reranker_provider_service_settings()
    reranker_provider = provider or _build_provider(provider_settings)
    app_settings = get_settings()
    app = FastAPI(
        title=f"{app_settings.app_name} Reranker Provider",
        version=app_settings.app_version,
    )

    @app.get("/healthz")
    def healthz() -> dict[str, object]:
        return {
            "ready": provider_settings.ready,
            "provider_type": REMOTE_RERANKER_PROVIDER_MODE,
            "provider_model_id": provider_settings.provider_model_id,
            "reranker_profile_name": provider_settings.reranker_profile_name,
            "device": provider_settings.device,
            "runtime_metadata": {
                "service": "nex_pcx_reranker_provider_service",
                "backend": provider_settings.backend,
                "models_dir": str(provider_settings.models_dir),
                "model_dir": str(provider_settings.local_model_dir),
                "model_dir_exists": provider_settings.local_model_dir.is_dir(),
            },
        }

    @app.post("/v1/rerank")
    def create_reranking(payload: RerankRequestBody) -> dict[str, object]:
        if not provider_settings.ready:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Reranker provider is not ready.",
            )
        try:
            response = reranker_provider.rerank(_request_from_payload(payload))  # type: ignore[attr-defined]
        except (InvalidRerankerError, ValueError) as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        return _response_payload(response)

    return app


def _validate_service_request(
    request: RerankRequest,
    settings: RerankerProviderServiceSettings,
) -> RerankRequest:
    validated = validate_rerank_request(request)
    if validated.reranker_profile_name != settings.reranker_profile_name:
        raise ValueError(f"Unsupported reranker_profile_name: {validated.reranker_profile_name}")
    if validated.reranker_model_id != settings.provider_model_id:
        raise ValueError(f"Unsupported reranker_model_id: {validated.reranker_model_id}")
    return validated


def _request_from_payload(payload: RerankRequestBody) -> RerankRequest:
    return RerankRequest(
        query_text=payload.query_text,
        top_k=payload.top_k,
        reranker_profile_name=payload.reranker_profile_name,
        reranker_model_id=payload.reranker_model_id,
        candidates=tuple(
            RerankCandidate(
                candidate_key=candidate.candidate_key,
                rank=candidate.rank,
                text=candidate.text,
                source_profile_name=candidate.source_profile_name,
                source_retrieval_strategy=candidate.source_retrieval_strategy,
                source_score=candidate.source_score,
                chunk_id=candidate.chunk_id,
                metadata=candidate.metadata,
            )
            for candidate in payload.candidates
        ),
    )


def _copy_result_for_remote_service(
    result: RerankResult,
    *,
    provider_model_id: str,
    backend: str,
    device: str,
    model_source: str,
    elapsed_ms: int,
) -> RerankResult:
    return RerankResult(
        query_text=result.query_text,
        reranker_profile_name=result.reranker_profile_name,
        reranker_model_id=provider_model_id,
        provider_type=REMOTE_RERANKER_PROVIDER_MODE,
        retrieval_strategy=result.retrieval_strategy,
        candidate_count=result.candidate_count,
        returned_count=result.returned_count,
        top_k=result.top_k,
        results=tuple(
            RerankResultItem(
                candidate=item.candidate,
                rank=item.rank,
                score=item.score,
                score_components={
                    **item.score_components,
                    "backend_reranker_provider_type": result.provider_type,
                },
            )
            for item in result.results
        ),
        runtime_metadata=_runtime_metadata(
            backend=backend,
            device=device,
            model_source=model_source,
            elapsed_ms=elapsed_ms,
            input_count=result.candidate_count,
        ),
    )


def _response_payload(response: RerankResult) -> dict[str, object]:
    return {
        "query_text": response.query_text,
        "reranker_profile_name": response.reranker_profile_name,
        "reranker_model_id": response.reranker_model_id,
        "provider_type": response.provider_type,
        "retrieval_strategy": response.retrieval_strategy,
        "candidate_count": response.candidate_count,
        "returned_count": response.returned_count,
        "top_k": response.top_k,
        "results": [
            {
                "candidate_key": item.candidate.candidate_key,
                "rank": item.rank,
                "score": item.score,
                "score_components": dict(item.score_components),
            }
            for item in response.results
        ],
        "runtime_metadata": dict(response.runtime_metadata),
    }


def _runtime_metadata(
    *,
    backend: str,
    device: str,
    model_source: str,
    elapsed_ms: int,
    input_count: int,
) -> dict[str, object]:
    return {
        "service": "nex_pcx_reranker_provider_service",
        "backend": backend,
        "device": device,
        "model_source": model_source,
        "elapsed_ms": elapsed_ms,
        "input_count": input_count,
    }


def _coerce_scores(scores: object, expected_count: int) -> tuple[float, ...]:
    values = scores.tolist() if hasattr(scores, "tolist") else scores
    try:
        coerced = tuple(_coerce_score(value) for value in values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError("reranker scores must be iterable") from exc
    if len(coerced) != expected_count:
        raise ValueError(f"reranker score count mismatch: expected {expected_count}")
    if not all(isfinite(score) for score in coerced):
        raise ValueError("reranker scores must be finite")
    return coerced


def _coerce_score(value: object) -> float:
    if hasattr(value, "item"):
        value = value.item()  # type: ignore[attr-defined]
    if isinstance(value, list | tuple):
        if len(value) != 1:
            raise ValueError("reranker score must be scalar")
        value = value[0]
    return float(value)


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _build_provider(settings: RerankerProviderServiceSettings) -> object:
    if settings.backend == RERANKER_PROVIDER_BACKEND_MOCK:
        return SkeletonRerankerProvider(settings)
    if settings.backend == RERANKER_PROVIDER_BACKEND_QWEN:
        return QwenCrossEncoderRerankerProvider(settings)
    raise ValueError(f"Unsupported reranker provider backend: {settings.backend}")


app = create_app()
