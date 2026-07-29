from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRS_PATH = PROJECT_ROOT / "260702_NeX_PCX_SRS_v1.3.md"
STRATEGY_DOC_PATH = PROJECT_ROOT / "docs" / "dgx_provider_resource_monitor_strategy.md"
PROBE_SCRIPT_DOC_PATH = PROJECT_ROOT / "docs" / "dgx_provider_resource_probe_script.md"
SNAPSHOT_SCHEMA_DOC_PATH = PROJECT_ROOT / "docs" / "provider_resource_snapshot_schema.md"


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


def test_probe_script_doc_defines_read_only_dgx_evidence_contract() -> None:
    probe_text = _read(PROBE_SCRIPT_DOC_PATH)

    assert "DGX Provider Resource Probe Script" in probe_text
    assert "scripts/probe_provider_resources.py" in probe_text
    assert "`kure-primary`" in probe_text
    assert "`qwen-reranker-primary`" in probe_text
    assert "`dgx_vllm_qwen36_27b_nvfp4`" in probe_text
    assert "--ssh-user nexpcx" in probe_text
    assert "--local-only" in probe_text
    assert "read-only" in probe_text
    assert "`process_not_found`" in probe_text
    assert "SHA-256 hash" in probe_text


def test_provider_resource_snapshot_schema_doc_defines_persistence_contract() -> None:
    schema_text = _read(SNAPSHOT_SCHEMA_DOC_PATH)

    assert "Provider Resource Snapshot Schema" in schema_text
    assert "`provider_resource_snapshots`" in schema_text
    assert "one row per provider target" in schema_text
    assert "`probe_run_id`" in schema_text
    assert "`process_rss_bytes`" in schema_text
    assert "`gpu_memory_used_bytes`" in schema_text
    assert "`system_swap_used_percent`" in schema_text
    assert "`ok`, `warning`, `critical`, or `unknown`" in schema_text
    assert "`idx_provider_resource_snapshots_provider_collected`" in schema_text
    assert "`provider_resource_snapshot_retention_days`" in schema_text
    assert "stores no prompt text" in schema_text
