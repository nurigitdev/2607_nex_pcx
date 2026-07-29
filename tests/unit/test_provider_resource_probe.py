from datetime import UTC, datetime

import pytest

from app.core.provider_resource_probe import (
    InvalidProviderResourceProbeError,
    ProviderResourceTarget,
    build_provider_resource_probe_report,
    parse_nvidia_smi_processes,
    parse_proc_meminfo,
    parse_ps_processes,
    parse_ss_listeners,
    provider_resource_probe_report_payload,
    render_provider_resource_probe_markdown,
    select_provider_resource_targets,
)


def test_select_provider_resource_targets_supports_aliases_and_host_override() -> None:
    targets = select_provider_resource_targets(("embedding",), host="dgx.local")

    assert [target.provider_name for target in targets] == [
        "kure-primary",
        "bge-primary",
        "qwen-primary",
    ]
    assert {target.host for target in targets} == {"dgx.local"}

    vllm_target = select_provider_resource_targets(("generation",))[0]
    assert vllm_target.provider_type == "vllm"
    assert vllm_target.port == 12000


def test_select_provider_resource_targets_rejects_empty_or_unknown_inputs() -> None:
    with pytest.raises(InvalidProviderResourceProbeError, match="unsupported provider selector"):
        select_provider_resource_targets(("missing",))

    with pytest.raises(InvalidProviderResourceProbeError, match="host is required"):
        select_provider_resource_targets(("all",), host=" ")


def test_provider_resource_parsers_normalize_process_port_gpu_and_memory() -> None:
    ps_text = "\n".join(
        [
            "PID PPID USER RSS VSZ %CPU ELAPSED COMMAND",
            (
                "101 1 nexpcx 204800 409600 7.5 3600 "
                "/home/x/.venv/bin/python -m uvicorn "
                "app.embedding_provider_service:app --api-key=secret"
            ),
            "bad row",
        ]
    )
    ss_text = """
    State Recv-Q Send-Q Local Address:Port Peer Address:Port Process
    LISTEN 0 2048 0.0.0.0:9101 0.0.0.0:* users:(("python",pid=101,fd=3))
    LISTEN 0 2048 [::]:12000 [::]:* users:(("vllm",pid=202,fd=4))
    """
    gpu_text = """
    101, python, 24576
    202, vllm, 65536 MiB
    No running processes found
    """
    meminfo_text = """
    MemTotal:       131072000 kB
    MemAvailable:   65536000 kB
    SwapTotal:       8388608 kB
    SwapFree:        4194304 kB
    """

    processes = parse_ps_processes(ps_text)
    listeners = parse_ss_listeners(ss_text)
    gpu_processes = parse_nvidia_smi_processes(gpu_text)
    memory = parse_proc_meminfo(meminfo_text)

    assert processes[0].pid == 101
    assert processes[0].rss_bytes == 204800 * 1024
    assert processes[0].cpu_percent == 7.5
    assert "<redacted>" in processes[0].command_preview
    assert listeners[0].port == 9101
    assert listeners[0].pid == 101
    assert listeners[1].port == 12000
    assert gpu_processes[0].used_gpu_memory_bytes == 24576 * 1024 * 1024
    assert gpu_processes[1].used_gpu_memory_bytes == 65536 * 1024 * 1024
    assert memory.total_ram_bytes == 131072000 * 1024
    assert memory.memory_available_percent == 50.0
    assert memory.swap_used_percent == 50.0


def test_provider_resource_report_marks_ready_provider_with_swap_pressure() -> None:
    target = ProviderResourceTarget(
        provider_name="pytest-kure",
        provider_type="embedding",
        host="dgx.local",
        port=9101,
        process_match="embedding_provider_service",
        model_id="kure_v1_1024",
        ram_warning_bytes=512 * 1024**2,
    )

    report = build_provider_resource_probe_report(
        targets=(target,),
        ps_text=(
            "101 1 nexpcx 204800 409600 7.5 3600 "
            "/home/x/.venv/bin/python -m uvicorn app.embedding_provider_service:app"
        ),
        ss_text='LISTEN 0 2048 0.0.0.0:9101 0.0.0.0:* users:(("python",pid=101,fd=3))',
        nvidia_smi_text="101, python, 1024",
        meminfo_text="""
        MemTotal:       104857600 kB
        MemAvailable:   52428800 kB
        SwapTotal:       8388608 kB
        SwapFree:        4194304 kB
        """,
        collected_at=datetime(2026, 7, 29, 9, 30, tzinfo=UTC),
    )
    snapshot = report.snapshots[0]
    payload = provider_resource_probe_report_payload(report)
    markdown = render_provider_resource_probe_markdown(report)

    assert report.status == "critical"
    assert snapshot.match_confidence == "port"
    assert snapshot.process is not None
    assert snapshot.gpu_process is not None
    assert snapshot.reason_codes == ("swap_pressure",)
    assert payload["collected_at"] == "2026-07-29T09:30:00+00:00"
    assert payload["snapshots"][0]["process"]["pid"] == 101
    assert "# DGX Provider Resource Probe" in markdown
    assert (
        "| pytest-kure | embedding | 9101 | critical | 101 | "
        "200.00 MiB | 1.00 GiB | swap_pressure |"
    ) in markdown


def test_provider_resource_report_marks_missing_and_command_fallback_paths() -> None:
    missing_target = ProviderResourceTarget(
        provider_name="missing-vllm",
        provider_type="vllm",
        host="dgx.local",
        port=12000,
        process_match="vllm",
    )
    fallback_target = ProviderResourceTarget(
        provider_name="fallback-reranker",
        provider_type="reranker",
        host="dgx.local",
        port=9104,
        process_match="reranker_provider_service",
        required=False,
    )

    report = build_provider_resource_probe_report(
        targets=(missing_target, fallback_target),
        ps_text=(
            "301 1 nexpcx 1024 2048 1.0 10 "
            "/home/x/.venv/bin/python -m uvicorn app.reranker_provider_service:app"
        ),
        ss_text="",
        meminfo_text="""
        MemTotal:       1000 kB
        MemAvailable:    500 kB
        SwapTotal:         0 kB
        SwapFree:          0 kB
        """,
    )

    assert report.status == "critical"
    assert report.critical_count == 1
    assert report.warning_count == 1
    assert report.snapshots[0].reason_codes == ("port_not_listening", "process_not_found")
    assert report.snapshots[1].match_confidence == "command"
    assert report.snapshots[1].reason_codes == (
        "port_not_listening",
        "command_match_without_port",
    )


def test_provider_resource_report_rejects_empty_targets() -> None:
    with pytest.raises(InvalidProviderResourceProbeError, match="at least one"):
        build_provider_resource_probe_report(
            targets=(),
            ps_text="",
            ss_text="",
        )
