from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.core.database import connect
from app.core.embedding_jobs import EmbeddingJobInput, create_embedding_job, get_embedding_job
from app.core.embedding_provider_routes import (
    EmbeddingProviderRouteInput,
    upsert_embedding_provider_route,
)
from app.core.embedding_providers import (
    EmbeddingProviderRequest,
    EmbeddingProviderResponse,
    EmbeddingProviderRuntimeConfig,
    RemoteEmbeddingProviderClient,
)
from app.core.embedding_vectors import get_chunk_embedding
from app.core.embedding_worker import (
    ERROR_CODE_UNSUPPORTED_EMBEDDING_PROFILE,
    EMBEDDING_WORKER_BATCH_STOP_QUEUE_EMPTY,
    process_embedding_worker_batch,
    process_next_embedding_job_with_provider,
    process_next_embedding_job_with_provider_routes,
    process_next_mock_embedding_job,
)
from app.embedding_provider_service import (
    EmbeddingProviderServiceSettings,
)
from app.embedding_provider_service import (
    create_app as create_provider_app,
)

pytestmark = pytest.mark.integration


def _create_chunk(database_url: str, chunk_text: str) -> tuple[int, int]:
    checksum = f"mock-embedding-worker-{uuid4()}"
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
                    storage_path
                )
                VALUES (%s, %s, '.md', 1, %s, %s)
                RETURNING file_id
                """,
                (
                    f"{checksum}.md",
                    f"{checksum}.stored.md",
                    checksum,
                    f"/tmp/{checksum}.md",
                ),
            )
            file_id = cursor.fetchone()["file_id"]
            cursor.execute(
                """
                INSERT INTO documents (file_id, document_title)
                VALUES (%s, %s)
                RETURNING document_id
                """,
                (file_id, f"Mock embedding worker fixture {checksum}"),
            )
            document_id = cursor.fetchone()["document_id"]
            cursor.execute(
                """
                INSERT INTO chunks (
                    document_id,
                    chunk_seq,
                    chunk_text,
                    content_hash,
                    chunk_policy_name,
                    char_count
                )
                VALUES (%s, 0, %s, %s, 'heading_512_64', %s)
                RETURNING chunk_id
                """,
                (document_id, chunk_text, f"chunk-{checksum}", len(chunk_text)),
            )
            chunk_id = cursor.fetchone()["chunk_id"]
    return file_id, chunk_id


def _create_inactive_profile(database_url: str) -> str:
    profile_name = f"unsupported_profile_{uuid4().hex}"
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO embedding_profiles (
                    profile_name,
                    model_name,
                    dimension,
                    storage_type,
                    is_active
                )
                VALUES (%s, 'example/unsupported', 3, 'vector', false)
                """,
                (profile_name,),
            )
    return profile_name


def _cleanup_fixture(database_url: str, file_id: int, profile_name: str | None = None) -> None:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute("DELETE FROM files WHERE file_id = %s", (file_id,))
            if profile_name is not None:
                cursor.execute(
                    "DELETE FROM embedding_profiles WHERE profile_name = %s",
                    (profile_name,),
                )


def _cleanup_routes(database_url: str, provider_names: list[str]) -> None:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "DELETE FROM embedding_provider_routes WHERE provider_name = ANY(%s)",
                (provider_names,),
            )


def test_mock_embedding_worker_stores_vector_and_marks_job_succeeded(
    migrated_database_url: str,
) -> None:
    chunk_text = "Mock embedding worker stores a deterministic vector."
    file_id, chunk_id = _create_chunk(migrated_database_url, chunk_text)
    try:
        created = create_embedding_job(
            migrated_database_url,
            EmbeddingJobInput(chunk_id=chunk_id, profile_name="kure_v1_1024"),
        )

        result = process_next_mock_embedding_job(
            migrated_database_url,
            worker_name="mock-worker-one",
            profile_name="kure_v1_1024",
        )

        stored_job = get_embedding_job(migrated_database_url, created.job.job_id)
        stored_vector = get_chunk_embedding(
            migrated_database_url,
            profile_name="kure_v1_1024",
            chunk_id=chunk_id,
        )

        assert result.processed is True
        assert result.job is not None
        assert result.job.job_id == created.job.job_id
        assert result.job.status == "succeeded"
        assert result.vector is not None
        assert result.vector.dimension == 1024
        assert result.elapsed_ms is not None
        assert stored_job is not None
        assert stored_job.status == "succeeded"
        assert stored_job.attempts == 1
        assert stored_job.runtime_metadata["adapter"] == "mock"
        assert stored_job.runtime_metadata["provider_type"] == "mock"
        assert stored_job.runtime_metadata["provider_model_id"] == "mock-provider"
        assert stored_job.runtime_metadata["dimension"] == 1024
        assert stored_job.runtime_metadata["table_name"] == "chunk_embeddings_kure_v1_1024"
        assert stored_vector == result.vector
    finally:
        _cleanup_fixture(migrated_database_url, file_id)


def test_mock_embedding_worker_marks_unsupported_profile_failed(
    migrated_database_url: str,
) -> None:
    file_id, chunk_id = _create_chunk(migrated_database_url, "Unsupported profile")
    profile_name = _create_inactive_profile(migrated_database_url)
    try:
        created = create_embedding_job(
            migrated_database_url,
            EmbeddingJobInput(chunk_id=chunk_id, profile_name=profile_name),
        )

        result = process_next_mock_embedding_job(
            migrated_database_url,
            worker_name="mock-worker-one",
            profile_name=profile_name,
        )

        stored_job = get_embedding_job(migrated_database_url, created.job.job_id)

        assert result.processed is True
        assert result.job is not None
        assert result.job.job_id == created.job.job_id
        assert result.job.status == "failed"
        assert result.job.error_code == ERROR_CODE_UNSUPPORTED_EMBEDDING_PROFILE
        assert stored_job is not None
        assert stored_job.status == "failed"
        assert stored_job.error_code == ERROR_CODE_UNSUPPORTED_EMBEDDING_PROFILE
        assert "Unsupported embedding profile" in stored_job.error_message
    finally:
        _cleanup_fixture(migrated_database_url, file_id, profile_name)


def test_mock_embedding_worker_batch_processes_until_queue_is_empty(
    migrated_database_url: str,
) -> None:
    first_file_id, first_chunk_id = _create_chunk(
        migrated_database_url,
        "First batch embedding worker chunk.",
    )
    second_file_id, second_chunk_id = _create_chunk(
        migrated_database_url,
        "Second batch embedding worker chunk.",
    )
    try:
        first_job = create_embedding_job(
            migrated_database_url,
            EmbeddingJobInput(chunk_id=first_chunk_id, profile_name="kure_v1_1024"),
        )
        second_job = create_embedding_job(
            migrated_database_url,
            EmbeddingJobInput(chunk_id=second_chunk_id, profile_name="kure_v1_1024"),
        )

        batch = process_embedding_worker_batch(
            lambda: process_next_mock_embedding_job(
                migrated_database_url,
                worker_name="mock-batch-worker",
                profile_name="kure_v1_1024",
            ),
            limit=5,
        )

        first_stored_job = get_embedding_job(migrated_database_url, first_job.job.job_id)
        second_stored_job = get_embedding_job(migrated_database_url, second_job.job.job_id)
        first_vector = get_chunk_embedding(
            migrated_database_url,
            profile_name="kure_v1_1024",
            chunk_id=first_chunk_id,
        )
        second_vector = get_chunk_embedding(
            migrated_database_url,
            profile_name="kure_v1_1024",
            chunk_id=second_chunk_id,
        )

        assert batch.stopped_reason == EMBEDDING_WORKER_BATCH_STOP_QUEUE_EMPTY
        assert batch.processed_count == 2
        assert batch.succeeded_count == 2
        assert batch.idle_count == 1
        assert first_stored_job is not None
        assert second_stored_job is not None
        assert first_stored_job.status == "succeeded"
        assert second_stored_job.status == "succeeded"
        assert first_vector is not None
        assert second_vector is not None
        assert first_vector.dimension == 1024
        assert second_vector.dimension == 1024
    finally:
        _cleanup_fixture(migrated_database_url, first_file_id)
        _cleanup_fixture(migrated_database_url, second_file_id)


def test_embedding_worker_can_store_vector_from_custom_provider(
    migrated_database_url: str,
) -> None:
    chunk_text = "Provider worker stores a vector from an injected provider."
    file_id, chunk_id = _create_chunk(migrated_database_url, chunk_text)
    try:
        created = create_embedding_job(
            migrated_database_url,
            EmbeddingJobInput(chunk_id=chunk_id, profile_name="kure_v1_1024"),
        )
        provider = _StaticEmbeddingProvider()

        result = process_next_embedding_job_with_provider(
            migrated_database_url,
            worker_name="provider-worker-one",
            provider=provider,
            profile_name="kure_v1_1024",
            success_message="Provider embedding stored",
        )

        stored_job = get_embedding_job(migrated_database_url, created.job.job_id)
        stored_vector = get_chunk_embedding(
            migrated_database_url,
            profile_name="kure_v1_1024",
            chunk_id=chunk_id,
        )

        assert result.processed is True
        assert result.job is not None
        assert result.job.status == "succeeded"
        assert result.message == "Provider embedding stored"
        assert stored_job is not None
        assert stored_job.runtime_metadata["provider_type"] == "remote"
        assert stored_job.runtime_metadata["provider_model_id"] == "static-gpu-provider"
        assert stored_job.runtime_metadata["adapter"] == "remote"
        assert stored_job.runtime_metadata["model_key"] == "kure_v1"
        assert provider.requests[0].trace_id == f"embedding-job-{created.job.job_id}"
        assert provider.requests[0].texts == (chunk_text,)
        assert stored_vector is not None
        assert stored_vector.dimension == 1024
    finally:
        _cleanup_fixture(migrated_database_url, file_id)


def test_embedding_worker_stores_vector_from_remote_provider_service(
    migrated_database_url: str,
) -> None:
    chunk_text = "Remote provider service stores an embedding through the worker."
    file_id, chunk_id = _create_chunk(migrated_database_url, chunk_text)
    provider_app = create_provider_app(
        EmbeddingProviderServiceSettings(
            provider_model_id="remote-provider-e2e",
            model_key="kure_v1",
            profile_names=("kure_v1_1024",),
            dimension=1024,
            device="cuda:0",
        )
    )
    try:
        created = create_embedding_job(
            migrated_database_url,
            EmbeddingJobInput(chunk_id=chunk_id, profile_name="kure_v1_1024"),
        )

        with TestClient(provider_app, base_url="http://provider.test") as http_client:
            provider = RemoteEmbeddingProviderClient(
                "http://provider.test",
                timeout_seconds=3.0,
                http_client=_TestClientEmbeddingProviderHTTPClient(http_client),
            )

            health = provider.health()
            result = process_next_embedding_job_with_provider(
                migrated_database_url,
                worker_name="remote-provider-worker",
                provider=provider,
                profile_name="kure_v1_1024",
                success_message="Remote embedding stored",
            )

        stored_job = get_embedding_job(migrated_database_url, created.job.job_id)
        stored_vector = get_chunk_embedding(
            migrated_database_url,
            profile_name="kure_v1_1024",
            chunk_id=chunk_id,
        )

        assert health.ready is True
        assert health.provider_model_id == "remote-provider-e2e"
        assert health.device == "cuda:0"
        assert result.processed is True
        assert result.job is not None
        assert result.job.status == "succeeded"
        assert result.message == "Remote embedding stored"
        assert stored_job is not None
        assert stored_job.status == "succeeded"
        assert stored_job.runtime_metadata["adapter"] == "remote"
        assert stored_job.runtime_metadata["provider_model_id"] == "remote-provider-e2e"
        assert stored_job.runtime_metadata["provider_type"] == "remote"
        assert stored_job.runtime_metadata["service"] == "nex_pcx_embedding_provider_skeleton"
        assert stored_job.runtime_metadata["device"] == "cuda:0"
        assert stored_job.runtime_metadata["trace_id"] == f"embedding-job-{created.job.job_id}"
        assert stored_vector is not None
        assert stored_vector.dimension == 1024
        assert stored_vector.table_name == "chunk_embeddings_kure_v1_1024"
    finally:
        _cleanup_fixture(migrated_database_url, file_id)


def test_route_aware_embedding_worker_uses_profile_provider_route(
    migrated_database_url: str,
) -> None:
    chunk_text = "Route-aware worker selects the provider URL from the route table."
    file_id, chunk_id = _create_chunk(migrated_database_url, chunk_text)
    provider_name = f"route-aware-provider-{uuid4().hex}"
    provider_app = create_provider_app(
        EmbeddingProviderServiceSettings(
            provider_model_id="route-aware-provider-e2e",
            model_key="kure_v1",
            profile_names=("kure_v1_1024",),
            dimension=1024,
            device="cuda:1",
        )
    )
    captured_runtime_config = None
    try:
        created = create_embedding_job(
            migrated_database_url,
            EmbeddingJobInput(chunk_id=chunk_id, profile_name="kure_v1_1024"),
        )
        route = upsert_embedding_provider_route(
            migrated_database_url,
            EmbeddingProviderRouteInput(
                profile_name="kure_v1_1024",
                provider_name=provider_name,
                provider_base_url="http://provider.test/",
                timeout_seconds=4.0,
                priority=0,
                runtime_metadata={"pool": "gpu-a"},
            ),
        )

        with TestClient(provider_app, base_url="http://provider.test") as http_client:

            def provider_builder(runtime_config: EmbeddingProviderRuntimeConfig):
                nonlocal captured_runtime_config
                captured_runtime_config = runtime_config
                return RemoteEmbeddingProviderClient(
                    runtime_config.remote_base_url or "",
                    timeout_seconds=runtime_config.remote_timeout_seconds,
                    http_client=_TestClientEmbeddingProviderHTTPClient(http_client),
                )

            result = process_next_embedding_job_with_provider_routes(
                migrated_database_url,
                worker_name="route-aware-worker",
                profile_name="kure_v1_1024",
                fallback_runtime_config=EmbeddingProviderRuntimeConfig(mode="mock"),
                provider_builder=provider_builder,
                route_selector=lambda _database_url, _profile_name: route,
            )

        stored_job = get_embedding_job(migrated_database_url, created.job.job_id)
        stored_vector = get_chunk_embedding(
            migrated_database_url,
            profile_name="kure_v1_1024",
            chunk_id=chunk_id,
        )

        assert captured_runtime_config == EmbeddingProviderRuntimeConfig(
            mode="remote",
            remote_base_url="http://provider.test",
            remote_timeout_seconds=4.0,
        )
        assert result.processed is True
        assert result.job is not None
        assert result.job.status == "succeeded"
        assert stored_job is not None
        assert stored_job.runtime_metadata["provider_runtime_source"] == "route"
        assert stored_job.runtime_metadata["provider_route_id"] == route.route_id
        assert stored_job.runtime_metadata["provider_route_name"] == provider_name
        assert stored_job.runtime_metadata["provider_route_priority"] == 0
        assert stored_job.runtime_metadata["provider_runtime_base_url"] == "http://provider.test"
        assert stored_job.runtime_metadata["provider_runtime_timeout_seconds"] == 4.0
        assert stored_job.runtime_metadata["provider_route_metadata"] == {"pool": "gpu-a"}
        assert stored_job.runtime_metadata["provider_model_id"] == "route-aware-provider-e2e"
        assert stored_job.runtime_metadata["device"] == "cuda:1"
        assert stored_vector is not None
        assert stored_vector.dimension == 1024
    finally:
        _cleanup_fixture(migrated_database_url, file_id)
        _cleanup_routes(migrated_database_url, [provider_name])


class _StaticEmbeddingProvider:
    def __init__(self) -> None:
        self.requests: list[EmbeddingProviderRequest] = []

    def embed(self, request: EmbeddingProviderRequest) -> EmbeddingProviderResponse:
        self.requests.append(request)
        return EmbeddingProviderResponse(
            embeddings=(tuple(0.001 for _ in range(request.output_dimension)),),
            dimension=request.output_dimension,
            provider_model_id="static-gpu-provider",
            provider_type="remote",
            elapsed_ms=7,
            input_count=1,
            runtime_metadata={
                "provider": "remote",
                "model_key": request.model_key,
                "device": "cuda:0",
            },
        )


class _TestClientEmbeddingProviderHTTPClient:
    def __init__(self, client: TestClient) -> None:
        self.client = client

    def request(self, method: str, url: str, **kwargs):
        kwargs.pop("timeout", None)
        return self.client.request(method, url, **kwargs)
