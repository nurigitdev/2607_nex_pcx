from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRS_PATH = PROJECT_ROOT / "260702_NeX_PCX_SRS_v1.3.md"
STRATEGY_DOC_PATH = PROJECT_ROOT / "docs" / "generation_provider_strategy_vllm.md"
PROMPT_DOC_PATH = PROJECT_ROOT / "docs" / "generation_prompt_package_builder.md"
MOCK_EXECUTOR_DOC_PATH = PROJECT_ROOT / "docs" / "generation_mock_executor.md"
GENERATION_API_DOC_PATH = PROJECT_ROOT / "docs" / "generation_run_api.md"
GENERATION_UI_DOC_PATH = PROJECT_ROOT / "docs" / "generation_run_ui_mvp.md"
GENERATION_DETAIL_UI_DOC_PATH = PROJECT_ROOT / "docs" / "generation_run_detail_ui.md"
GENERATION_PROMPT_PREVIEW_DOC_PATH = PROJECT_ROOT / "docs" / "generation_prompt_preview_api_ui.md"
GENERATION_PROVIDER_METRICS_DOC_PATH = (
    PROJECT_ROOT / "docs" / "generation_provider_metrics_contract.md"
)
GENERATION_PROVIDER_METRIC_SNAPSHOT_DOC_PATH = (
    PROJECT_ROOT / "docs" / "generation_provider_metric_snapshot_api.md"
)
GENERATION_PROVIDER_METRIC_SNAPSHOT_UI_DOC_PATH = (
    PROJECT_ROOT / "docs" / "generation_provider_metric_snapshot_ui.md"
)
GENERATION_OPENAI_VLLM_CLIENT_DOC_PATH = PROJECT_ROOT / "docs" / "generation_openai_vllm_client.md"
DGX_VLLM_GENERATION_SMOKE_RUNNER_DOC_PATH = (
    PROJECT_ROOT / "docs" / "dgx_vllm_generation_smoke_runner.md"
)
GENERATION_PROVIDER_RUNTIME_CONFIG_API_DOC_PATH = (
    PROJECT_ROOT / "docs" / "generation_provider_runtime_config_api.md"
)
GENERATION_PROVIDER_RUNTIME_CONFIG_UI_DOC_PATH = (
    PROJECT_ROOT / "docs" / "generation_provider_runtime_config_ui.md"
)
GENERATION_REMOTE_EXECUTOR_DOC_PATH = PROJECT_ROOT / "docs" / "generation_remote_executor.md"
GENERATION_REMOTE_RUN_API_DOC_PATH = PROJECT_ROOT / "docs" / "generation_remote_run_api.md"
GENERATION_REMOTE_RUN_UI_DOC_PATH = PROJECT_ROOT / "docs" / "generation_remote_run_ui.md"
DGX_VLLM_GENERATION_RUN_E2E_DOC_PATH = PROJECT_ROOT / "docs" / "dgx_vllm_generation_run_live_e2e.md"
GENERATION_ANSWER_CITATION_GUARDRAIL_DOC_PATH = (
    PROJECT_ROOT / "docs" / "generation_answer_citation_guardrail.md"
)
GENERATION_ANSWER_QUALITY_BADGE_UI_DOC_PATH = (
    PROJECT_ROOT / "docs" / "generation_answer_quality_badge_ui.md"
)
GENERATION_RUN_HISTORY_QUALITY_FILTER_DOC_PATH = (
    PROJECT_ROOT / "docs" / "generation_run_history_quality_filter.md"
)
DIRECT_GENERATION_QUERY_ORCHESTRATOR_DOC_PATH = (
    PROJECT_ROOT / "docs" / "direct_generation_query_orchestrator_api.md"
)
DIRECT_GENERATION_UI_DOC_PATH = PROJECT_ROOT / "docs" / "direct_generation_ui_mvp.md"
GENERATION_TEMPLATE_STRATEGY_DOC_PATH = PROJECT_ROOT / "docs" / "generation_template_strategy.md"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_srs_documents_generation_provider_strategy() -> None:
    srs_text = _read(SRS_PATH)

    assert "Software Requirements Specification v1.62" in srs_text
    assert "Generation provider strategy" in srs_text
    assert "vLLM runtime contract" in srs_text
    assert "Qwen3.5-122B-A10B-NVFP4" in srs_text
    assert "FR-051" in srs_text
    assert "FR-052" in srs_text
    assert "FR-053" in srs_text
    assert "FR-054" in srs_text
    assert "FR-055" in srs_text
    assert "FR-056" in srs_text
    assert "FR-057" in srs_text
    assert "FR-058" in srs_text
    assert "FR-059" in srs_text
    assert "FR-060" in srs_text
    assert "FR-061" in srs_text
    assert "FR-062" in srs_text
    assert "FR-063" in srs_text
    assert "FR-064" in srs_text
    assert "FR-065" in srs_text
    assert "FR-066" in srs_text
    assert "FR-067" in srs_text
    assert "FR-068" in srs_text
    assert "FR-069" in srs_text
    assert "FR-070" in srs_text
    assert "FR-071" in srs_text
    assert "FR-072" in srs_text
    assert "FR-073" in srs_text
    assert "FR-074" in srs_text
    assert "FR-075" in srs_text
    assert "FR-076" in srs_text
    assert "FR-077" in srs_text
    assert "FR-078" in srs_text
    assert "FR-079" in srs_text
    assert "FR-080" in srs_text
    assert "FR-081" in srs_text
    assert "FR-082" in srs_text
    assert "FR-083" in srs_text
    assert "FR-084" in srs_text
    assert "FR-085" in srs_text
    assert "FR-086" in srs_text
    assert "FR-087" in srs_text
    assert "remote_openai_compatible" in srs_text
    assert "/v1/chat/completions" in srs_text
    assert "OpenAI-compatible vLLM client foundation" in srs_text
    assert "DGX vLLM generation smoke evidence" in srs_text
    assert "Generation provider runtime config API" in srs_text
    assert "Generation provider runtime config UI" in srs_text
    assert "Remote vLLM generation executor foundation" in srs_text
    assert "Remote generation run API" in srs_text
    assert "Remote generation run UI controls" in srs_text
    assert "Generation provider metrics mode breakdown" in srs_text
    assert "DGX vLLM generation run live E2E verification" in srs_text
    assert "Generation answer citation guardrail" in srs_text
    assert "Generation answer quality badge/detail UI" in srs_text
    assert "Generation run history" in srs_text
    assert "Direct generation query orchestrator API" in srs_text
    assert "Direct Generation UI MVP" in srs_text
    assert "Generation template strategy" in srs_text
    assert "generation_templates" in srs_text
    assert "Template-based Generation" in srs_text
    assert "Conversational workspace strategy" in srs_text
    assert "Chat intent routing contract" in srs_text
    assert "chat_sessions" in srs_text
    assert "chat_messages" in srs_text
    assert "chat_message_links" in srs_text


def test_generation_template_strategy_doc_defines_reproducible_contract() -> None:
    strategy_text = _read(GENERATION_TEMPLATE_STRATEGY_DOC_PATH)

    assert "Generation Template Strategy" in strategy_text
    assert "FR-077" in strategy_text
    assert "`generation_templates`" in strategy_text
    assert "`grounded_answer`" in strategy_text
    assert "`report`" in strategy_text
    assert "`proposal`" in strategy_text
    assert "`summary`" in strategy_text
    assert "`meeting_minutes`" in strategy_text
    assert "Markdown" in strategy_text
    assert "template snapshot" in strategy_text
    assert "`prompt_hash`" in strategy_text


def test_generation_provider_strategy_doc_defines_mock_first_vllm_contract() -> None:
    strategy_text = _read(STRATEGY_DOC_PATH)

    assert "`mock`" in strategy_text
    assert "`remote_openai_compatible`" in strategy_text
    assert "/v1/chat/completions" in strategy_text
    assert "nvidia/Qwen3.5-122B-A10B-NVFP4" in strategy_text
    assert "low_confidence" in strategy_text
    assert "no_relevant_context" in strategy_text


def test_generation_prompt_package_doc_defines_guarded_openai_messages() -> None:
    prompt_text = _read(PROMPT_DOC_PATH)

    assert "OpenAI-compatible chat" in prompt_text
    assert "`messages`" in prompt_text
    assert "`prompt_hash`" in prompt_text
    assert "`context_hash`" in prompt_text
    assert "`blocked`" in prompt_text
    assert "`block_reason`" in prompt_text
    assert "no-answer response" in prompt_text


def test_generation_mock_executor_doc_defines_persistent_no_answer_path() -> None:
    executor_text = _read(MOCK_EXECUTOR_DOC_PATH)

    assert "Generation Mock Executor" in executor_text
    assert "`generation_runs`" in executor_text
    assert "`generation_run_citations`" in executor_text
    assert "`mock_completed`" in executor_text
    assert "`guardrail_no_answer`" in executor_text
    assert "Remote vLLM Handoff" in executor_text


def test_generation_run_api_doc_defines_mock_create_and_detail_contract() -> None:
    api_text = _read(GENERATION_API_DOC_PATH)

    assert "Generation Run API" in api_text
    assert "POST /api/search/logs/{search_log_id}/generation-runs/mock" in api_text
    assert "GET /api/generation/runs/{generation_run_id}" in api_text
    assert "`201 Created`" in api_text
    assert "provider-neutral" in api_text


def test_generation_run_ui_doc_defines_mock_result_panel() -> None:
    ui_text = _read(GENERATION_UI_DOC_PATH)

    assert "Generation Run UI MVP" in ui_text
    assert "`GET /generation`" in ui_text
    assert "`POST /generation/runs/mock`" in ui_text
    assert "citation trace" in ui_text
    assert "`response_metadata.answer_quality`" in ui_text
    assert "vLLM generation runs" in ui_text


def test_generation_run_detail_ui_doc_defines_reproducibility_panel() -> None:
    detail_text = _read(GENERATION_DETAIL_UI_DOC_PATH)

    assert "Generation Run Detail UI" in detail_text
    assert "`GET /generation/runs/{generation_run_id}`" in detail_text
    assert "prompt hash" in detail_text
    assert "guardrail metadata" in detail_text
    assert "answer quality status" in detail_text
    assert "vLLM runs" in detail_text


def test_generation_prompt_preview_doc_defines_non_persistent_preview() -> None:
    preview_text = _read(GENERATION_PROMPT_PREVIEW_DOC_PATH)

    assert "Generation Prompt Preview API/UI" in preview_text
    assert "`GET /api/search/logs/{search_log_id}/generation-prompt/preview`" in preview_text
    assert "`generation_runs`" in preview_text
    assert "does not create rows" in preview_text
    assert "OpenAI-compatible `messages`" in preview_text


def test_generation_provider_metrics_doc_defines_vllm_usage_contract() -> None:
    metrics_text = _read(GENERATION_PROVIDER_METRICS_DOC_PATH)

    assert "Generation Provider Metrics Contract" in metrics_text
    assert "`parse_openai_chat_completion_metrics(...)`" in metrics_text
    assert "`choices[0].finish_reason`" in metrics_text
    assert "token usage" in metrics_text
    assert "DGX runtime" in metrics_text


def test_generation_provider_metric_snapshot_doc_defines_admin_api() -> None:
    snapshot_text = _read(GENERATION_PROVIDER_METRIC_SNAPSHOT_DOC_PATH)

    assert "Generation Provider Metric Snapshot API" in snapshot_text
    assert "`GET /api/admin/generation-provider-metrics/snapshot`" in snapshot_text
    assert "`generation_runs.response_metadata.provider_metrics`" in snapshot_text
    assert "`metric_present=false`" in snapshot_text
    assert "`mode_summaries`" in snapshot_text
    assert "`remote_openai_compatible`" in snapshot_text


def test_generation_provider_metric_snapshot_ui_doc_defines_operator_panel() -> None:
    snapshot_ui_text = _read(GENERATION_PROVIDER_METRIC_SNAPSHOT_UI_DOC_PATH)

    assert "Generation Provider Metric Snapshot UI" in snapshot_ui_text
    assert "`GET /admin/generation-provider-metrics`" in snapshot_ui_text
    assert "Summary cards" in snapshot_ui_text
    assert "provider mode breakdown table" in snapshot_ui_text
    assert "`remote_openai_compatible`" in snapshot_ui_text
    assert "Raw snapshot JSON" in snapshot_ui_text


def test_generation_openai_vllm_client_doc_defines_client_contract() -> None:
    client_text = _read(GENERATION_OPENAI_VLLM_CLIENT_DOC_PATH)

    assert "OpenAI-Compatible vLLM Client Foundation" in client_text
    assert "`GenerationProviderRuntimeConfig`" in client_text
    assert "`POST /v1/chat/completions`" in client_text
    assert "`GenerationProviderMetrics`" in client_text
    assert "`GenerationProviderRequestError`" in client_text
    assert "`error_code=invalid_json`" in client_text
    assert "`extra_body`" in client_text


def test_dgx_vllm_generation_smoke_runner_doc_defines_live_evidence_contract() -> None:
    smoke_text = _read(DGX_VLLM_GENERATION_SMOKE_RUNNER_DOC_PATH)

    assert "DGX vLLM Generation Smoke Runner" in smoke_text
    assert "`192.168.20.243`" in smoke_text
    assert "`12000`" in smoke_text
    assert "`/v1/chat/completions`" in smoke_text
    assert "`NEX_PCX_REMOTE_GENERATION_PROVIDER_API_KEY`" in smoke_text
    assert "not the secret value" in smoke_text
    assert "provider_metrics.succeeded=true" in smoke_text
    assert "enable_thinking" in smoke_text


def test_generation_provider_runtime_config_api_doc_defines_dgx_seed_contract() -> None:
    config_text = _read(GENERATION_PROVIDER_RUNTIME_CONFIG_API_DOC_PATH)

    assert "Generation Provider Runtime Config API" in config_text
    assert "`GET /api/admin/generation-provider-configs`" in config_text
    assert "`GET /api/admin/generation-provider-configs/default`" in config_text
    assert "`POST /api/admin/generation-provider-configs/seed-dgx-vllm`" in config_text
    assert "`remote_openai_compatible`" in config_text
    assert "`http://192.168.20.243:12000`" in config_text
    assert "`NEX_PCX_REMOTE_GENERATION_PROVIDER_API_KEY`" in config_text
    assert "environment-only" in config_text


def test_generation_provider_runtime_config_ui_doc_defines_operator_page() -> None:
    ui_text = _read(GENERATION_PROVIDER_RUNTIME_CONFIG_UI_DOC_PATH)

    assert "Generation Provider Runtime Config UI" in ui_text
    assert "`GET /admin/generation-provider-configs`" in ui_text
    assert "`POST /admin/generation-provider-configs/seed-dgx-vllm`" in ui_text
    assert "runtime validation status" in ui_text
    assert "`NEX_PCX_REMOTE_GENERATION_PROVIDER_API_KEY`" in ui_text
    assert "Korean is the default UI language" in ui_text


def test_generation_remote_executor_doc_defines_persistence_contract() -> None:
    executor_text = _read(GENERATION_REMOTE_EXECUTOR_DOC_PATH)

    assert "Remote vLLM Generation Executor Foundation" in executor_text
    assert "`execute_remote_generation_run(...)`" in executor_text
    assert "`remote_openai_compatible`" in executor_text
    assert "`status=no_answer`" in executor_text
    assert "`status=failed`" in executor_text
    assert "GenerationProviderRequestError" in executor_text
    assert "injected fake provider" in executor_text


def test_generation_remote_run_api_doc_defines_env_only_contract() -> None:
    api_text = _read(GENERATION_REMOTE_RUN_API_DOC_PATH)

    assert "Remote Generation Run API" in api_text
    assert "POST /api/search/logs/{search_log_id}/generation-runs/remote" in api_text
    assert "`remote_openai_compatible`" in api_text
    assert "`runtime_options.api_key_env`" in api_text
    assert "environment-only" in api_text
    assert "`status=failed`" in api_text


def test_generation_remote_run_ui_doc_defines_readiness_controls() -> None:
    ui_text = _read(GENERATION_REMOTE_RUN_UI_DOC_PATH)

    assert "Remote Generation Run UI Controls" in ui_text
    assert "`GET /generation`" in ui_text
    assert "`POST /generation/runs/remote`" in ui_text
    assert "`Remote vLLM Runtime`" in ui_text
    assert "`remote_openai_compatible`" in ui_text
    assert "`runtime_options.api_key_env`" in ui_text
    assert "Korean remains the default UI language" in ui_text


def test_dgx_vllm_generation_run_live_e2e_doc_defines_persistence_evidence() -> None:
    e2e_text = _read(DGX_VLLM_GENERATION_RUN_E2E_DOC_PATH)

    assert "DGX vLLM Generation Run Live E2E Verification" in e2e_text
    assert "scripts/run_dgx_vllm_generation_run_e2e.py" in e2e_text
    assert "`192.168.20.243`" in e2e_text
    assert "`12000`" in e2e_text
    assert "`generation_runs`" in e2e_text
    assert "`generation_run_citations`" in e2e_text
    assert "`execute_remote_generation_run(...)`" in e2e_text
    assert "`remote_openai_compatible`" in e2e_text
    assert "environment variable name" in e2e_text


def test_generation_answer_citation_guardrail_doc_defines_quality_metadata() -> None:
    quality_text = _read(GENERATION_ANSWER_CITATION_GUARDRAIL_DOC_PATH)

    assert "Generation Answer Citation Guardrail + Quality Metadata" in quality_text
    assert "`response_metadata.answer_quality`" in quality_text
    assert "`guardrail_metadata`" in quality_text
    assert "`generation_answer_quality_v1`" in quality_text
    assert "`not_evaluated`" in quality_text
    assert "missing all required citations" in quality_text
    assert "provider execution health" in quality_text
    assert "grounded answer" in quality_text
    assert "quality independently measurable" in quality_text


def test_generation_answer_quality_badge_ui_doc_defines_operator_panels() -> None:
    ui_text = _read(GENERATION_ANSWER_QUALITY_BADGE_UI_DOC_PATH)

    assert "Generation Answer Quality Badge + Detail UI" in ui_text
    assert "`GET /generation`" in ui_text
    assert "`GET /generation/runs/{generation_run_id}`" in ui_text
    assert "`response_metadata.answer_quality`" in ui_text
    assert "expected/used/missing/unknown" in ui_text
    assert "FR-073" in _read(SRS_PATH)


def test_generation_run_history_quality_filter_doc_defines_filter_contract() -> None:
    history_text = _read(GENERATION_RUN_HISTORY_QUALITY_FILTER_DOC_PATH)

    assert "Generation Run History + Quality Filter API/UI" in history_text
    assert "`GET /api/generation/runs`" in history_text
    assert "`GET /generation/runs`" in history_text
    assert "`answer_quality_status`" in history_text
    assert "`remote_openai_compatible`" in history_text
    assert "raw JSON evidence" in history_text
    assert "FR-074" in _read(SRS_PATH)


def test_direct_generation_query_orchestrator_doc_defines_api_contract() -> None:
    direct_text = _read(DIRECT_GENERATION_QUERY_ORCHESTRATOR_DOC_PATH)

    assert "Direct Generation Query Orchestrator API" in direct_text
    assert "`POST /api/generation/direct-runs`" in direct_text
    assert "`remote_openai_compatible`" in direct_text
    assert "`retrieval_context`" in direct_text
    assert "`generation`" in direct_text
    assert "Secrets are not included" in direct_text
    assert "FR-075" in _read(SRS_PATH)


def test_direct_generation_ui_doc_defines_form_to_result_contract() -> None:
    ui_text = _read(DIRECT_GENERATION_UI_DOC_PATH)

    assert "Direct Generation UI MVP" in ui_text
    assert "`GET /generation`" in ui_text
    assert "`POST /generation/direct-runs`" in ui_text
    assert "`data-direct-generation-form`" in ui_text
    assert "`remote_openai_compatible`" in ui_text
    assert "answer quality" in ui_text
    assert "FR-076" in _read(SRS_PATH)
