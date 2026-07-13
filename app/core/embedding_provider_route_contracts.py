"""Contract checks for embedding provider routes."""

from dataclasses import dataclass
from time import perf_counter

from app.core.embedding_model_distribution import (
    InvalidEmbeddingModelDistributionError,
    get_embedding_model_distribution_for_profile,
)
from app.core.embedding_provider_route_auth import (
    InvalidEmbeddingProviderRouteAuthError,
    resolve_embedding_provider_route_request_headers,
)
from app.core.embedding_provider_route_health import (
    EmbeddingProviderRouteHealthResult,
    check_embedding_provider_route_health,
    provider_health_dimension_for_profile,
)
from app.core.embedding_provider_routes import EmbeddingProviderRouteRecord
from app.core.embedding_providers import (
    MOCK_EMBEDDING_PROVIDER_TYPE,
    REMOTE_EMBEDDING_PROVIDER_TYPE,
    EmbeddingProviderRequest,
    EmbeddingProviderResponse,
    InvalidEmbeddingProviderError,
    MockEmbeddingProvider,
    RemoteEmbeddingProviderClient,
)
from app.core.embedding_vectors import InvalidEmbeddingVectorError, get_embedding_vector_table

DEFAULT_CONTRACT_SAMPLE_TEXTS = ("NeX-PCX embedding provider contract check sample.",)


@dataclass(frozen=True)
class EmbeddingProviderRouteContractResult:
    route: EmbeddingProviderRouteRecord
    passed: bool
    status: str
    elapsed_ms: int
    health: EmbeddingProviderRouteHealthResult | None
    input_type: str
    sample_text_count: int
    expected_dimension: int | None
    provider_type: str | None
    provider_model_id: str | None
    model_key: str | None
    dimension: int | None
    input_count: int | None
    runtime_metadata: dict[str, object]
    validation_errors: tuple[str, ...] = ()
    error_message: str | None = None


def check_embedding_provider_route_contract(
    route: EmbeddingProviderRouteRecord,
    *,
    sample_texts: tuple[str, ...] = DEFAULT_CONTRACT_SAMPLE_TEXTS,
    input_type: str = "document",
    sample_set_name: str | None = None,
    http_client: object | None = None,
) -> EmbeddingProviderRouteContractResult:
    started_at = perf_counter()
    sample_text_count = len(sample_texts)
    try:
        distribution = get_embedding_model_distribution_for_profile(route.profile_name)
        vector_table = get_embedding_vector_table(route.profile_name)
    except (InvalidEmbeddingModelDistributionError, InvalidEmbeddingVectorError) as exc:
        return _contract_result(
            route,
            started_at=started_at,
            status="invalid_route",
            input_type=input_type,
            sample_text_count=sample_text_count,
            expected_dimension=None,
            runtime_metadata=_contract_runtime_metadata(sample_set_name),
            validation_errors=(str(exc),),
            error_message=str(exc),
        )

    health = check_embedding_provider_route_health(route, http_client=http_client)
    if not health.ready:
        return _contract_result(
            route,
            started_at=started_at,
            status=f"health_{health.status}",
            health=health,
            input_type=input_type,
            sample_text_count=sample_text_count,
            expected_dimension=vector_table.dimension,
            provider_type=health.provider_type,
            provider_model_id=health.provider_model_id,
            model_key=health.model_key,
            dimension=health.dimension,
            runtime_metadata=_contract_runtime_metadata(sample_set_name),
            validation_errors=health.validation_errors,
            error_message=health.error_message,
        )

    provider = _build_provider(route, http_client=http_client)
    request = EmbeddingProviderRequest(
        profile_name=route.profile_name,
        model_key=distribution.model_key,
        input_type=input_type,
        texts=sample_texts,
        output_dimension=vector_table.dimension,
        normalize_embeddings=True,
        trace_id=f"route-contract-{route.route_id}",
        runtime_metadata={
            "contract_check": True,
            "provider_route_id": route.route_id,
            **_contract_runtime_metadata(sample_set_name),
        },
    )
    try:
        response = provider.embed(request)
    except InvalidEmbeddingProviderError as exc:
        return _contract_result(
            route,
            started_at=started_at,
            status="embedding_failed",
            health=health,
            input_type=input_type,
            sample_text_count=sample_text_count,
            expected_dimension=vector_table.dimension,
            provider_type=health.provider_type,
            provider_model_id=health.provider_model_id,
            model_key=health.model_key,
            dimension=health.dimension,
            runtime_metadata=_contract_runtime_metadata(sample_set_name),
            validation_errors=health.validation_errors,
            error_message=str(exc),
        )
    finally:
        close = getattr(provider, "close", None)
        if callable(close):
            close()

    validation_errors = _validate_embedding_contract(route, health, response)
    return _contract_result(
        route,
        started_at=started_at,
        status="passed" if not validation_errors else "mismatch",
        passed=not validation_errors,
        health=health,
        input_type=input_type,
        sample_text_count=sample_text_count,
        expected_dimension=vector_table.dimension,
        provider_type=response.provider_type,
        provider_model_id=response.provider_model_id,
        model_key=distribution.model_key,
        dimension=response.dimension,
        input_count=response.input_count,
        runtime_metadata={
            **dict(response.runtime_metadata),
            **_contract_runtime_metadata(sample_set_name),
        },
        validation_errors=validation_errors,
    )


def _build_provider(route: EmbeddingProviderRouteRecord, *, http_client: object | None) -> object:
    if route.provider_mode == MOCK_EMBEDDING_PROVIDER_TYPE:
        return MockEmbeddingProvider()
    if route.provider_mode == REMOTE_EMBEDDING_PROVIDER_TYPE:
        try:
            headers = resolve_embedding_provider_route_request_headers(route.runtime_metadata)
        except InvalidEmbeddingProviderRouteAuthError as exc:
            raise InvalidEmbeddingProviderError(str(exc)) from exc
        return RemoteEmbeddingProviderClient(
            route.provider_base_url or "",
            timeout_seconds=route.timeout_seconds,
            headers=headers,
            http_client=http_client,
        )
    raise InvalidEmbeddingProviderError(f"Unsupported provider_mode: {route.provider_mode}")


def _validate_embedding_contract(
    route: EmbeddingProviderRouteRecord,
    health: EmbeddingProviderRouteHealthResult,
    response: EmbeddingProviderResponse,
) -> tuple[str, ...]:
    validation_errors = list(health.validation_errors)
    if response.provider_type != route.provider_mode:
        validation_errors.append(f"provider_type mismatch: {response.provider_type}")
    if health.provider_model_id and response.provider_model_id != health.provider_model_id:
        validation_errors.append(
            "provider_model_id mismatch: "
            f"health={health.provider_model_id}, embedding={response.provider_model_id}"
        )
    provider_dimension = provider_health_dimension_for_profile(
        health.dimension,
        profile_name=route.profile_name,
        runtime_metadata=health.runtime_metadata,
    )
    if provider_dimension is not None and response.dimension != provider_dimension:
        validation_errors.append(
            f"dimension mismatch: health={provider_dimension}, embedding={response.dimension}"
        )
    return tuple(validation_errors)


def _contract_runtime_metadata(sample_set_name: str | None) -> dict[str, object]:
    return {"contract_sample_set_name": sample_set_name} if sample_set_name else {}


def _contract_result(
    route: EmbeddingProviderRouteRecord,
    *,
    started_at: float,
    status: str,
    input_type: str,
    sample_text_count: int,
    expected_dimension: int | None,
    passed: bool = False,
    health: EmbeddingProviderRouteHealthResult | None = None,
    provider_type: str | None = None,
    provider_model_id: str | None = None,
    model_key: str | None = None,
    dimension: int | None = None,
    input_count: int | None = None,
    runtime_metadata: dict[str, object] | None = None,
    validation_errors: tuple[str, ...] = (),
    error_message: str | None = None,
) -> EmbeddingProviderRouteContractResult:
    return EmbeddingProviderRouteContractResult(
        route=route,
        passed=passed,
        status=status,
        elapsed_ms=max(0, int((perf_counter() - started_at) * 1000)),
        health=health,
        input_type=input_type,
        sample_text_count=sample_text_count,
        expected_dimension=expected_dimension,
        provider_type=provider_type,
        provider_model_id=provider_model_id,
        model_key=model_key,
        dimension=dimension,
        input_count=input_count,
        runtime_metadata=dict(runtime_metadata or {}),
        validation_errors=validation_errors,
        error_message=error_message,
    )
