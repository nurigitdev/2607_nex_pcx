from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRS_PATH = PROJECT_ROOT / "260702_NeX_PCX_SRS_v1.3.md"
STRATEGY_DOC_PATH = PROJECT_ROOT / "docs" / "generation_provider_strategy_vllm.md"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_srs_documents_generation_provider_strategy() -> None:
    srs_text = _read(SRS_PATH)

    assert "Software Requirements Specification v1.31" in srs_text
    assert "Generation provider strategy" in srs_text
    assert "vLLM runtime contract" in srs_text
    assert "Qwen3.6-27B-NVFP4" in srs_text
    assert "FR-051" in srs_text
    assert "FR-052" in srs_text
    assert "FR-053" in srs_text
    assert "FR-054" in srs_text
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
