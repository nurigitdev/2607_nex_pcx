from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRS_PATH = PROJECT_ROOT / "260702_NeX_PCX_SRS_v1.3.md"
STRATEGY_DOC_PATH = PROJECT_ROOT / "docs" / "generation_provider_strategy_vllm.md"
PROMPT_DOC_PATH = PROJECT_ROOT / "docs" / "generation_prompt_package_builder.md"
MOCK_EXECUTOR_DOC_PATH = PROJECT_ROOT / "docs" / "generation_mock_executor.md"
GENERATION_API_DOC_PATH = PROJECT_ROOT / "docs" / "generation_run_api.md"
GENERATION_UI_DOC_PATH = PROJECT_ROOT / "docs" / "generation_run_ui_mvp.md"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_srs_documents_generation_provider_strategy() -> None:
    srs_text = _read(SRS_PATH)

    assert "Software Requirements Specification v1.36" in srs_text
    assert "Generation provider strategy" in srs_text
    assert "vLLM runtime contract" in srs_text
    assert "Qwen3.6-27B-NVFP4" in srs_text
    assert "FR-051" in srs_text
    assert "FR-052" in srs_text
    assert "FR-053" in srs_text
    assert "FR-054" in srs_text
    assert "FR-055" in srs_text
    assert "FR-056" in srs_text
    assert "FR-057" in srs_text
    assert "remote_openai_compatible" in srs_text
    assert "/v1/chat/completions" in srs_text


def test_generation_provider_strategy_doc_defines_mock_first_vllm_contract() -> None:
    strategy_text = _read(STRATEGY_DOC_PATH)

    assert "`mock`" in strategy_text
    assert "`remote_openai_compatible`" in strategy_text
    assert "/v1/chat/completions" in strategy_text
    assert "nvidia/Qwen3.6-27B-NVFP4" in strategy_text
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
    assert "vLLM generation runs" in ui_text
