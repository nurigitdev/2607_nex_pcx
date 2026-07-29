from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRS_PATH = PROJECT_ROOT / "260702_NeX_PCX_SRS_v1.3.md"
STRATEGY_DOC_PATH = PROJECT_ROOT / "docs" / "dgx_provider_resource_monitor_strategy.md"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_srs_documents_dgx_provider_resource_monitor_strategy() -> None:
    srs_text = _read(SRS_PATH)

    assert "Software Requirements Specification v1.58" in srs_text
    assert "DGX provider resource monitor strategy" in srs_text
    assert "FR-087" in srs_text
    assert "provider_resource_snapshots" in srs_text
    assert "RAM RSS" in srs_text
    assert "GPU memory" in srs_text
    assert "swap pressure" in srs_text
    assert "remote embedding provider" in srs_text
    assert "remote reranker provider" in srs_text
    assert "vLLM runtime" in srs_text


def test_strategy_doc_defines_provider_resource_monitor_contract() -> None:
    strategy_text = _read(STRATEGY_DOC_PATH)

    assert "DGX Provider Resource Monitor Strategy" in strategy_text
    assert "`provider_type`" in strategy_text
    assert "`embedding`, `reranker`, or `vllm`" in strategy_text
    assert "`192.168.20.243`" in strategy_text
    assert "Port binding should be the primary local process match" in strategy_text
    assert "RSS bytes" in strategy_text
    assert "GPU resource counters" in strategy_text
    assert "memory bytes" in strategy_text
    assert "`ok`, `warning`, `critical`, or `unknown`" in strategy_text
    assert "`process_not_found`" in strategy_text
    assert "Secrets, prompt text, document text" in strategy_text
    assert "Slice 409: Remote Provider Resource Probe Script" in strategy_text
    assert "Slice 412: Dashboard Provider Resource Card" in strategy_text
