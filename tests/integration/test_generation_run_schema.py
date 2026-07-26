import pytest
from psycopg import errors

from app.core.database import connect, fetch_one

pytestmark = pytest.mark.integration


def _create_search_log(database_url: str) -> int:
    with connect(database_url) as conn:
        row = conn.execute("""
            INSERT INTO search_logs (
                query_text,
                normalized_query_text,
                top_k,
                profiles,
                query_runtime_metadata
            )
            VALUES (
                '생성 run schema smoke query',
                '생성 run schema smoke query',
                3,
                '["reranked_vector_cosine"]'::jsonb,
                '{"test": "generation_run_schema"}'::jsonb
            )
            RETURNING search_log_id
            """).fetchone()
        conn.commit()
    assert row is not None
    return int(row["search_log_id"])


def test_generation_provider_config_default_seeded(migrated_database_url: str) -> None:
    provider = fetch_one(
        migrated_database_url,
        """
        SELECT provider_name, provider_mode, model_id, runtime_options
        FROM generation_provider_configs
        WHERE is_default
        """,
    )

    assert provider["provider_name"] == "mock_qwen36_27b_nvfp4"
    assert provider["provider_mode"] == "mock"
    assert provider["model_id"] == "nvidia/Qwen3.6-27B-NVFP4"
    assert provider["runtime_options"]["endpoint"] == "/v1/chat/completions"


def test_generation_run_links_search_log_and_retrieval_package(
    migrated_database_url: str,
) -> None:
    search_log_id = _create_search_log(migrated_database_url)
    provider = fetch_one(
        migrated_database_url,
        """
        SELECT provider_config_id, provider_name, provider_mode, model_id
        FROM generation_provider_configs
        WHERE is_default
        """,
    )

    with connect(migrated_database_url) as conn:
        run = conn.execute(
            """
            INSERT INTO generation_runs (
                search_log_id,
                retrieval_package_key,
                provider_config_id,
                provider_name,
                provider_mode,
                model_id,
                retrieval_confidence_status,
                citation_readiness_status,
                query_text,
                request_metadata,
                guardrail_metadata
            )
            VALUES (
                %s,
                'package-337',
                %s,
                %s,
                %s,
                %s,
                'answerable',
                'ready',
                '생성 run schema smoke query',
                '{"prompt_version": "grounded_answer_v1"}'::jsonb,
                '{"status": "allowed"}'::jsonb
            )
            RETURNING generation_run_id, status, guardrail_status
            """,
            (
                search_log_id,
                provider["provider_config_id"],
                provider["provider_name"],
                provider["provider_mode"],
                provider["model_id"],
            ),
        ).fetchone()
        assert run is not None
        conn.execute(
            """
            INSERT INTO generation_run_citations (
                generation_run_id,
                citation_key,
                citation_index,
                source_label,
                source_anchor,
                citation_payload
            )
            VALUES (
                %s,
                'RCP-001',
                1,
                'fixture.md / p.1',
                '{"page_no": 1}'::jsonb,
                '{"chunk_id": 101}'::jsonb
            )
            """,
            (run["generation_run_id"],),
        )
        conn.commit()

    stored = fetch_one(
        migrated_database_url,
        """
        SELECT gr.status, gr.guardrail_status, gr.retrieval_package_key, grc.citation_key
        FROM generation_runs gr
        JOIN generation_run_citations grc
          ON grc.generation_run_id = gr.generation_run_id
        WHERE gr.search_log_id = %s
        """,
        (search_log_id,),
    )

    assert stored["status"] == "pending"
    assert stored["guardrail_status"] == "allowed"
    assert stored["retrieval_package_key"] == "package-337"
    assert stored["citation_key"] == "RCP-001"


def test_remote_openai_compatible_provider_requires_base_url(migrated_database_url: str) -> None:
    with connect(migrated_database_url) as conn:
        with pytest.raises(errors.CheckViolation):
            conn.execute("""
                INSERT INTO generation_provider_configs (
                    provider_name,
                    provider_mode,
                    model_id
                )
                VALUES (
                    'invalid_remote_generation_provider',
                    'remote_openai_compatible',
                    'nvidia/Qwen3.6-27B-NVFP4'
                )
                """)
        conn.rollback()


def test_generation_run_guardrail_status_is_constrained(migrated_database_url: str) -> None:
    search_log_id = _create_search_log(migrated_database_url)

    with connect(migrated_database_url) as conn:
        with pytest.raises(errors.CheckViolation):
            conn.execute(
                """
                INSERT INTO generation_runs (
                    search_log_id,
                    retrieval_package_key,
                    provider_name,
                    provider_mode,
                    model_id,
                    status,
                    guardrail_status,
                    retrieval_confidence_status,
                    citation_readiness_status,
                    query_text
                )
                VALUES (
                    %s,
                    'package-invalid',
                    'mock_qwen36_27b_nvfp4',
                    'mock',
                    'nvidia/Qwen3.6-27B-NVFP4',
                    'pending',
                    'unsafe',
                    'answerable',
                    'ready',
                    '생성 run schema smoke query'
                )
                """,
                (search_log_id,),
            )
        conn.rollback()
