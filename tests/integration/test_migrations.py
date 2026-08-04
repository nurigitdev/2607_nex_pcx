import pytest
from alembic.script import ScriptDirectory

from app.core.database import fetch_one
from app.core.migrations import downgrade, make_alembic_config, upgrade

pytestmark = pytest.mark.integration
HEAD_REVISION = "20260804_0044"


def test_alembic_upgrade_head_enables_pgvector(test_database_url: str) -> None:
    downgrade("base", test_database_url)
    upgrade("head", test_database_url)

    revision = fetch_one(test_database_url, "SELECT version_num FROM alembic_version")
    extension = fetch_one(
        test_database_url,
        "SELECT extversion FROM pg_extension WHERE extname = 'vector'",
    )
    dashboard_threshold_settings = fetch_one(
        test_database_url,
        """
        SELECT count(*) AS count
        FROM app_log_settings
        WHERE setting_name LIKE 'dashboard_%%_threshold'
        """,
    )
    search_experiment_table = fetch_one(
        test_database_url,
        "SELECT to_regclass('public.search_experiment_runs') AS table_name",
    )
    golden_batch_metric_snapshot_table = fetch_one(
        test_database_url,
        """
        SELECT to_regclass(
            'public.golden_search_experiment_batch_metric_snapshots'
        ) AS table_name
        """,
    )
    dgx_benchmark_run_table = fetch_one(
        test_database_url,
        """
        SELECT to_regclass(
            'public.dgx_ingestion_benchmark_runs'
        ) AS table_name
        """,
    )
    extraction_artifact_table = fetch_one(
        test_database_url,
        """
        SELECT to_regclass(
            'public.extraction_artifacts'
        ) AS table_name
        """,
    )
    extraction_quality_snapshot_table = fetch_one(
        test_database_url,
        """
        SELECT to_regclass(
            'public.extraction_quality_snapshots'
        ) AS table_name
        """,
    )
    local_extraction_profile_count = fetch_one(
        test_database_url,
        """
        SELECT count(*) AS count
        FROM extraction_profiles
        WHERE extraction_profile_name LIKE 'local_%%_default'
        """,
    )
    search_profile_table = fetch_one(
        test_database_url,
        "SELECT to_regclass('public.search_profiles') AS table_name",
    )
    bm25_profile_count = fetch_one(
        test_database_url,
        """
        SELECT count(*) AS count
        FROM search_profiles
        WHERE search_profile_name = 'bm25_keyword'
          AND profile_kind = 'keyword'
        """,
    )
    reranked_profile_count = fetch_one(
        test_database_url,
        """
        SELECT count(*) AS count
        FROM search_profiles
        WHERE search_profile_name = 'reranked_vector_cosine'
          AND profile_kind = 'rerank'
          AND strategy_name = 'reranked_vector_cosine'
          AND is_active
        """,
    )
    reranked_profile_runtime = fetch_one(
        test_database_url,
        """
        SELECT runtime_parameters
        FROM search_profiles
        WHERE search_profile_name = 'reranked_vector_cosine'
        """,
    )
    generation_run_table = fetch_one(
        test_database_url,
        "SELECT to_regclass('public.generation_runs') AS table_name",
    )
    generation_run_prompt_version_default = fetch_one(
        test_database_url,
        """
        SELECT column_default
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'generation_runs'
          AND column_name = 'prompt_version'
        """,
    )
    generation_citation_table = fetch_one(
        test_database_url,
        "SELECT to_regclass('public.generation_run_citations') AS table_name",
    )
    generation_template_table = fetch_one(
        test_database_url,
        "SELECT to_regclass('public.generation_templates') AS table_name",
    )
    default_generation_template = fetch_one(
        test_database_url,
        """
        SELECT template_key, template_family, document_type, output_format, change_note
        FROM generation_templates
        WHERE is_default
        """,
    )
    generation_template_family_index = fetch_one(
        test_database_url,
        "SELECT to_regclass('public.idx_generation_templates_family_version') AS index_name",
    )
    long_form_template_required_summary = fetch_one(
        test_database_url,
        """
        SELECT bool_and((section ->> 'required')::boolean) AS all_required
        FROM generation_templates,
             jsonb_array_elements(section_schema) AS section
        WHERE template_key IN ('report', 'proposal')
        """,
    )
    summary_template_preset_count = fetch_one(
        test_database_url,
        """
        SELECT count(*) AS count
        FROM generation_templates
        WHERE template_family = 'summary'
          AND document_type = 'summary'
          AND template_key IN (
              'summary_executive',
              'summary_risk_action',
              'summary_working'
          )
          AND is_active
        """,
    )
    chat_session_table = fetch_one(
        test_database_url,
        "SELECT to_regclass('public.chat_sessions') AS table_name",
    )
    chat_message_table = fetch_one(
        test_database_url,
        "SELECT to_regclass('public.chat_messages') AS table_name",
    )
    chat_message_link_table = fetch_one(
        test_database_url,
        "SELECT to_regclass('public.chat_message_links') AS table_name",
    )
    chat_message_sequence_index = fetch_one(
        test_database_url,
        "SELECT to_regclass('public.idx_chat_messages_session_sequence') AS index_name",
    )
    default_generation_provider = fetch_one(
        test_database_url,
        """
        SELECT provider_mode, model_id
        FROM generation_provider_configs
        WHERE provider_name = 'mock_qwen35_122b_a10b_nvfp4'
          AND is_default
        """,
    )
    dgx_generation_provider = fetch_one(
        test_database_url,
        """
        SELECT provider_mode, provider_base_url, model_id, is_active, is_default, runtime_options
        FROM generation_provider_configs
        WHERE provider_name = 'dgx_vllm_qwen35_122b_a10b_nvfp4'
        """,
    )
    keyword_index_table = fetch_one(
        test_database_url,
        "SELECT to_regclass('public.chunk_keyword_terms') AS table_name",
    )
    keyword_indexing_stage_constraint = fetch_one(
        test_database_url,
        """
        SELECT pg_get_constraintdef(oid) LIKE '%%keyword_indexing%%' AS enabled
        FROM pg_constraint
        WHERE conname = 'pipeline_jobs_stage_check'
        """,
    )
    vllm_metric_snapshot_table = fetch_one(
        test_database_url,
        """
        SELECT to_regclass(
            'public.vllm_runtime_metric_snapshots'
        ) AS table_name
        """,
    )
    vllm_metric_snapshot_provider_index = fetch_one(
        test_database_url,
        """
        SELECT to_regclass(
            'public.idx_vllm_runtime_metric_snapshots_provider_sampled'
        ) AS index_name
        """,
    )
    vllm_readiness_threshold_settings = fetch_one(
        test_database_url,
        """
        SELECT count(*) AS count
        FROM app_log_settings
        WHERE setting_name LIKE 'vllm_runtime_%%'
        """,
    )
    provider_resource_snapshot_table = fetch_one(
        test_database_url,
        """
        SELECT to_regclass(
            'public.provider_resource_snapshots'
        ) AS table_name
        """,
    )
    provider_resource_snapshot_provider_index = fetch_one(
        test_database_url,
        """
        SELECT to_regclass(
            'public.idx_provider_resource_snapshots_provider_collected'
        ) AS index_name
        """,
    )
    provider_resource_snapshot_status_constraint = fetch_one(
        test_database_url,
        """
        SELECT pg_get_constraintdef(oid) LIKE '%%critical%%' AS enabled
        FROM pg_constraint
        WHERE conname = 'provider_resource_snapshots_status_check'
        """,
    )
    provider_resource_snapshot_settings = fetch_one(
        test_database_url,
        """
        SELECT count(*) AS count
        FROM app_log_settings
        WHERE setting_name LIKE 'provider_resource_%%'
        """,
    )

    assert revision["version_num"] == HEAD_REVISION
    assert extension["extversion"]
    assert dashboard_threshold_settings["count"] == 9
    assert search_experiment_table["table_name"] == "search_experiment_runs"
    assert (
        golden_batch_metric_snapshot_table["table_name"]
        == "golden_search_experiment_batch_metric_snapshots"
    )
    assert dgx_benchmark_run_table["table_name"] == "dgx_ingestion_benchmark_runs"
    assert extraction_artifact_table["table_name"] == "extraction_artifacts"
    assert extraction_quality_snapshot_table["table_name"] == "extraction_quality_snapshots"
    assert local_extraction_profile_count["count"] == 7
    assert search_profile_table["table_name"] == "search_profiles"
    assert bm25_profile_count["count"] == 1
    assert reranked_profile_count["count"] == 1
    assert reranked_profile_runtime["runtime_parameters"]["reranker_profile_name"] == (
        "qwen3_reranker_0_6b"
    )
    assert reranked_profile_runtime["runtime_parameters"]["reranker_model_id"] == (
        "Qwen/Qwen3-Reranker-0.6B"
    )
    assert generation_run_table["table_name"] == "generation_runs"
    assert "grounded_answer_v1_prompt_v1" in generation_run_prompt_version_default["column_default"]
    assert generation_citation_table["table_name"] == "generation_run_citations"
    assert generation_template_table["table_name"] == "generation_templates"
    assert default_generation_template["template_key"] == "grounded_answer"
    assert default_generation_template["template_family"] == "grounded_answer"
    assert default_generation_template["document_type"] == "grounded_answer"
    assert default_generation_template["output_format"] == "markdown"
    assert default_generation_template["change_note"] == ""
    assert (
        generation_template_family_index["index_name"] == "idx_generation_templates_family_version"
    )
    assert long_form_template_required_summary["all_required"] is True
    assert summary_template_preset_count["count"] == 3
    assert chat_session_table["table_name"] == "chat_sessions"
    assert chat_message_table["table_name"] == "chat_messages"
    assert chat_message_link_table["table_name"] == "chat_message_links"
    assert chat_message_sequence_index["index_name"] == "idx_chat_messages_session_sequence"
    assert default_generation_provider["provider_mode"] == "mock"
    assert default_generation_provider["model_id"] == "nvidia/Qwen3.5-122B-A10B-NVFP4"
    assert dgx_generation_provider["provider_mode"] == "remote_openai_compatible"
    assert dgx_generation_provider["provider_base_url"] == "http://192.168.20.243:12000"
    assert (
        dgx_generation_provider["model_id"]
        == "/home/nurivoice-dgx/models/nvidia/Qwen3.5-122B-A10B-NVFP4"
    )
    assert dgx_generation_provider["is_active"] is False
    assert dgx_generation_provider["is_default"] is False
    assert (
        dgx_generation_provider["runtime_options"]["api_key_env"]
        == "NEX_PCX_REMOTE_GENERATION_PROVIDER_API_KEY"
    )
    assert keyword_index_table["table_name"] == "chunk_keyword_terms"
    assert keyword_indexing_stage_constraint["enabled"] is True
    assert vllm_metric_snapshot_table["table_name"] == "vllm_runtime_metric_snapshots"
    assert (
        vllm_metric_snapshot_provider_index["index_name"]
        == "idx_vllm_runtime_metric_snapshots_provider_sampled"
    )
    assert vllm_readiness_threshold_settings["count"] == 14
    assert provider_resource_snapshot_table["table_name"] == "provider_resource_snapshots"
    assert (
        provider_resource_snapshot_provider_index["index_name"]
        == "idx_provider_resource_snapshots_provider_collected"
    )
    assert provider_resource_snapshot_status_constraint["enabled"] is True
    assert provider_resource_snapshot_settings["count"] == 3


def test_alembic_downgrade_base_clears_revision(test_database_url: str) -> None:
    upgrade("head", test_database_url)
    downgrade("base", test_database_url)
    try:
        revision_count = fetch_one(
            test_database_url,
            "SELECT count(*) AS revision_count FROM alembic_version",
        )
        extension = fetch_one(
            test_database_url,
            "SELECT extversion FROM pg_extension WHERE extname = 'vector'",
        )

        assert revision_count["revision_count"] == 0
        assert extension["extversion"]
        assert (
            fetch_one(test_database_url, "SELECT to_regclass('public.files') AS table_name")[
                "table_name"
            ]
            is None
        )
    finally:
        upgrade("head", test_database_url)


def test_alembic_config_points_at_project_migrations(test_database_url: str) -> None:
    config = make_alembic_config(test_database_url)
    script = ScriptDirectory.from_config(config)

    assert script.get_current_head() == HEAD_REVISION
