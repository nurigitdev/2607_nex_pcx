"""Provider process resource probe helpers for a shared DGX host."""

import hashlib
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

DEFAULT_DGX_PROVIDER_RESOURCE_HOST = "192.168.20.243"

PROVIDER_RESOURCE_STATUS_OK = "ok"
PROVIDER_RESOURCE_STATUS_WARNING = "warning"
PROVIDER_RESOURCE_STATUS_CRITICAL = "critical"
PROVIDER_RESOURCE_STATUS_UNKNOWN = "unknown"

_STATUS_RANK = {
    PROVIDER_RESOURCE_STATUS_OK: 0,
    PROVIDER_RESOURCE_STATUS_UNKNOWN: 1,
    PROVIDER_RESOURCE_STATUS_WARNING: 2,
    PROVIDER_RESOURCE_STATUS_CRITICAL: 3,
}

DEFAULT_RAM_AVAILABLE_WARNING_PERCENT = 20.0
DEFAULT_RAM_AVAILABLE_CRITICAL_PERCENT = 10.0
DEFAULT_SWAP_WARNING_BYTES = 1024**3
DEFAULT_SWAP_CRITICAL_BYTES = 8 * 1024**3
DEFAULT_SWAP_WARNING_PERCENT = 10.0
DEFAULT_SWAP_CRITICAL_PERCENT = 30.0


@dataclass(frozen=True)
class ProviderResourceTarget:
    provider_name: str
    provider_type: str
    host: str
    port: int
    process_match: str
    model_id: str | None = None
    required: bool = True
    ram_warning_bytes: int | None = None
    ram_critical_bytes: int | None = None
    gpu_memory_warning_bytes: int | None = None
    gpu_memory_critical_bytes: int | None = None


@dataclass(frozen=True)
class ProcessSnapshot:
    pid: int
    ppid: int | None
    user: str
    rss_bytes: int
    vms_bytes: int
    cpu_percent: float
    uptime_seconds: int | None
    command_preview: str
    command_hash: str


@dataclass(frozen=True)
class PortListener:
    port: int
    pid: int | None
    process_name: str | None
    raw_line: str


@dataclass(frozen=True)
class GpuProcessSnapshot:
    pid: int
    process_name: str
    used_gpu_memory_bytes: int


@dataclass(frozen=True)
class SystemMemorySnapshot:
    total_ram_bytes: int | None
    available_ram_bytes: int | None
    memory_available_percent: float | None
    swap_total_bytes: int | None
    swap_used_bytes: int | None
    swap_used_percent: float | None


@dataclass(frozen=True)
class ProviderResourceSnapshot:
    provider_name: str
    provider_type: str
    host: str
    port: int
    model_id: str | None
    status: str
    reason_codes: tuple[str, ...]
    match_confidence: str
    process: ProcessSnapshot | None
    listener: PortListener | None
    gpu_process: GpuProcessSnapshot | None
    system_memory: SystemMemorySnapshot
    collected_at: datetime
    collector_error: str | None = None


@dataclass(frozen=True)
class ProviderResourceProbeReport:
    host: str
    collected_at: datetime
    status: str
    target_count: int
    ok_count: int
    warning_count: int
    critical_count: int
    unknown_count: int
    system_memory: SystemMemorySnapshot
    snapshots: tuple[ProviderResourceSnapshot, ...]
    collector_errors: tuple[str, ...] = ()


DEFAULT_DGX_PROVIDER_RESOURCE_TARGETS: tuple[ProviderResourceTarget, ...] = (
    ProviderResourceTarget(
        provider_name="kure-primary",
        provider_type="embedding",
        host=DEFAULT_DGX_PROVIDER_RESOURCE_HOST,
        port=9101,
        process_match="embedding_provider_service",
        model_id="kure_v1_1024",
    ),
    ProviderResourceTarget(
        provider_name="bge-primary",
        provider_type="embedding",
        host=DEFAULT_DGX_PROVIDER_RESOURCE_HOST,
        port=9102,
        process_match="embedding_provider_service",
        model_id="bge_m3_1024",
    ),
    ProviderResourceTarget(
        provider_name="qwen-primary",
        provider_type="embedding",
        host=DEFAULT_DGX_PROVIDER_RESOURCE_HOST,
        port=9103,
        process_match="embedding_provider_service",
        model_id="qwen3_embedding_4b",
        ram_warning_bytes=48 * 1024**3,
        ram_critical_bytes=64 * 1024**3,
    ),
    ProviderResourceTarget(
        provider_name="qwen-reranker-primary",
        provider_type="reranker",
        host=DEFAULT_DGX_PROVIDER_RESOURCE_HOST,
        port=9104,
        process_match="reranker_provider_service",
        model_id="qwen3_reranker_0_6b",
        ram_warning_bytes=32 * 1024**3,
        ram_critical_bytes=48 * 1024**3,
    ),
    ProviderResourceTarget(
        provider_name="dgx_vllm_qwen35_122b_a10b_nvfp4",
        provider_type="vllm",
        host=DEFAULT_DGX_PROVIDER_RESOURCE_HOST,
        port=12000,
        process_match="vllm",
        model_id="/home/nurivoice-dgx/models/nvidia/Qwen3.5-122B-A10B-NVFP4",
        ram_warning_bytes=96 * 1024**3,
        ram_critical_bytes=112 * 1024**3,
    ),
)


class InvalidProviderResourceProbeError(ValueError):
    """Raised when provider resource probe inputs are invalid."""


def select_provider_resource_targets(
    selectors: tuple[str, ...] | list[str] | None = None,
    *,
    host: str = DEFAULT_DGX_PROVIDER_RESOURCE_HOST,
) -> tuple[ProviderResourceTarget, ...]:
    normalized_selectors = tuple(
        selector.strip().lower() for selector in (selectors or ("all",)) if selector.strip()
    )
    if not normalized_selectors:
        normalized_selectors = ("all",)

    selected: list[ProviderResourceTarget] = []
    for target in DEFAULT_DGX_PROVIDER_RESOURCE_TARGETS:
        aliases = {
            "all",
            target.provider_type,
            target.provider_name.lower(),
            target.model_id.lower() if target.model_id else "",
        }
        if target.provider_name == "kure-primary":
            aliases.add("kure")
        elif target.provider_name == "bge-primary":
            aliases.add("bge")
        elif target.provider_name == "qwen-primary":
            aliases.add("qwen")
        elif target.provider_type == "reranker":
            aliases.add("reranker")
        elif target.provider_type == "vllm":
            aliases.add("generation")

        if any(selector in aliases for selector in normalized_selectors):
            selected.append(_replace_target_host(target, host))

    if not selected:
        raise InvalidProviderResourceProbeError(
            f"unsupported provider selector: {normalized_selectors[0]}"
        )
    return tuple(selected)


def build_provider_resource_probe_report(
    *,
    targets: tuple[ProviderResourceTarget, ...],
    ps_text: str,
    ss_text: str,
    nvidia_smi_text: str = "",
    meminfo_text: str = "",
    collector_errors: tuple[str, ...] = (),
    collected_at: datetime | None = None,
) -> ProviderResourceProbeReport:
    if not targets:
        raise InvalidProviderResourceProbeError("at least one provider target is required")

    selected_collected_at = _ensure_aware(collected_at or datetime.now(UTC))
    processes = {process.pid: process for process in parse_ps_processes(ps_text)}
    listeners = _listeners_by_port(parse_ss_listeners(ss_text))
    gpu_processes = {
        process.pid: process for process in parse_nvidia_smi_processes(nvidia_smi_text)
    }
    system_memory = parse_proc_meminfo(meminfo_text)
    snapshots = tuple(
        _build_target_snapshot(
            target,
            processes=processes,
            listeners=listeners,
            gpu_processes=gpu_processes,
            system_memory=system_memory,
            collected_at=selected_collected_at,
        )
        for target in targets
    )
    status = _worst_status([snapshot.status for snapshot in snapshots])
    return ProviderResourceProbeReport(
        host=targets[0].host,
        collected_at=selected_collected_at,
        status=status,
        target_count=len(snapshots),
        ok_count=sum(1 for snapshot in snapshots if snapshot.status == PROVIDER_RESOURCE_STATUS_OK),
        warning_count=sum(
            1 for snapshot in snapshots if snapshot.status == PROVIDER_RESOURCE_STATUS_WARNING
        ),
        critical_count=sum(
            1 for snapshot in snapshots if snapshot.status == PROVIDER_RESOURCE_STATUS_CRITICAL
        ),
        unknown_count=sum(
            1 for snapshot in snapshots if snapshot.status == PROVIDER_RESOURCE_STATUS_UNKNOWN
        ),
        system_memory=system_memory,
        snapshots=snapshots,
        collector_errors=collector_errors,
    )


def parse_ps_processes(ps_text: str) -> tuple[ProcessSnapshot, ...]:
    processes: list[ProcessSnapshot] = []
    for line in ps_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split(maxsplit=7)
        if len(parts) < 8 or parts[0].upper() == "PID":
            continue
        try:
            pid = int(parts[0])
            ppid = int(parts[1])
            rss_bytes = int(float(parts[3])) * 1024
            vms_bytes = int(float(parts[4])) * 1024
            cpu_percent = float(parts[5])
            uptime_seconds = int(float(parts[6]))
        except ValueError:
            continue
        command = parts[7]
        processes.append(
            ProcessSnapshot(
                pid=pid,
                ppid=ppid,
                user=parts[2],
                rss_bytes=rss_bytes,
                vms_bytes=vms_bytes,
                cpu_percent=cpu_percent,
                uptime_seconds=uptime_seconds,
                command_preview=_redacted_command_preview(command),
                command_hash=_command_hash(command),
            )
        )
    return tuple(processes)


def parse_ss_listeners(ss_text: str) -> tuple[PortListener, ...]:
    listeners: list[PortListener] = []
    for line in ss_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.lower().startswith("state "):
            continue
        port = _extract_listener_port(stripped)
        if port is None:
            continue
        pid_match = re.search(r"pid=(\d+)", stripped)
        process_match = re.search(r'users:\(\("([^"]+)"', stripped)
        listeners.append(
            PortListener(
                port=port,
                pid=int(pid_match.group(1)) if pid_match else None,
                process_name=process_match.group(1) if process_match else None,
                raw_line=stripped,
            )
        )
    return tuple(listeners)


def parse_nvidia_smi_processes(nvidia_smi_text: str) -> tuple[GpuProcessSnapshot, ...]:
    processes: list[GpuProcessSnapshot] = []
    for line in nvidia_smi_text.splitlines():
        stripped = line.strip()
        if not stripped or "no running" in stripped.lower():
            continue
        parts = [part.strip() for part in stripped.split(",")]
        if len(parts) < 3 or parts[0].lower() == "pid":
            continue
        try:
            pid = int(parts[0])
            memory_mib = int(re.search(r"\d+", parts[2]).group(0))  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            continue
        processes.append(
            GpuProcessSnapshot(
                pid=pid,
                process_name=parts[1],
                used_gpu_memory_bytes=memory_mib * 1024 * 1024,
            )
        )
    return tuple(processes)


def parse_proc_meminfo(meminfo_text: str) -> SystemMemorySnapshot:
    values: dict[str, int] = {}
    for line in meminfo_text.splitlines():
        if ":" not in line:
            continue
        key, raw_value = line.split(":", maxsplit=1)
        key = key.strip()
        match = re.search(r"\d+", raw_value)
        if match:
            values[key] = int(match.group(0)) * 1024

    total_ram = values.get("MemTotal")
    available_ram = values.get("MemAvailable")
    swap_total = values.get("SwapTotal")
    swap_free = values.get("SwapFree")
    swap_used = None
    if swap_total is not None and swap_free is not None:
        swap_used = max(0, swap_total - swap_free)

    return SystemMemorySnapshot(
        total_ram_bytes=total_ram,
        available_ram_bytes=available_ram,
        memory_available_percent=_percent(available_ram, total_ram),
        swap_total_bytes=swap_total,
        swap_used_bytes=swap_used,
        swap_used_percent=_percent(swap_used, swap_total),
    )


def provider_resource_probe_report_payload(
    report: ProviderResourceProbeReport,
) -> dict[str, Any]:
    payload = asdict(report)
    payload["collected_at"] = report.collected_at.isoformat()
    for snapshot in payload["snapshots"]:
        snapshot["collected_at"] = snapshot["collected_at"].isoformat()
    return payload


def render_provider_resource_probe_markdown(
    report: ProviderResourceProbeReport,
) -> str:
    lines = [
        "# DGX Provider Resource Probe",
        "",
        f"- `host`: `{report.host}`",
        f"- `collected_at`: `{report.collected_at.isoformat()}`",
        f"- `status`: `{report.status}`",
        f"- `targets`: `{report.target_count}`",
        "",
        "| Provider | Type | Port | Status | PID | RAM RSS | GPU Memory | Reasons |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for snapshot in report.snapshots:
        process = snapshot.process
        gpu_process = snapshot.gpu_process
        lines.append(
            "| "
            f"{snapshot.provider_name} | "
            f"{snapshot.provider_type} | "
            f"{snapshot.port} | "
            f"{snapshot.status} | "
            f"{process.pid if process else '-'} | "
            f"{_bytes_label(process.rss_bytes if process else None)} | "
            f"{_bytes_label(gpu_process.used_gpu_memory_bytes if gpu_process else None)} | "
            f"{', '.join(snapshot.reason_codes) or '-'} |"
        )
    if report.collector_errors:
        lines.extend(["", "## Collector Errors", ""])
        for error in report.collector_errors:
            lines.append(f"- {error}")
    return "\n".join(lines) + "\n"


def _build_target_snapshot(
    target: ProviderResourceTarget,
    *,
    processes: dict[int, ProcessSnapshot],
    listeners: dict[int, PortListener],
    gpu_processes: dict[int, GpuProcessSnapshot],
    system_memory: SystemMemorySnapshot,
    collected_at: datetime,
) -> ProviderResourceSnapshot:
    listener = listeners.get(target.port)
    process = processes.get(listener.pid) if listener and listener.pid is not None else None
    match_confidence = "port" if process is not None else "missing"
    if process is None:
        process = _find_process_by_command(processes, target.process_match)
        if process is not None:
            match_confidence = "command"
    gpu_process = gpu_processes.get(process.pid) if process else None
    status, reason_codes = _target_status(
        target,
        process=process,
        listener=listener,
        gpu_process=gpu_process,
        system_memory=system_memory,
        match_confidence=match_confidence,
    )
    return ProviderResourceSnapshot(
        provider_name=target.provider_name,
        provider_type=target.provider_type,
        host=target.host,
        port=target.port,
        model_id=target.model_id,
        status=status,
        reason_codes=reason_codes,
        match_confidence=match_confidence,
        process=process,
        listener=listener,
        gpu_process=gpu_process,
        system_memory=system_memory,
        collected_at=collected_at,
    )


def _target_status(
    target: ProviderResourceTarget,
    *,
    process: ProcessSnapshot | None,
    listener: PortListener | None,
    gpu_process: GpuProcessSnapshot | None,
    system_memory: SystemMemorySnapshot,
    match_confidence: str,
) -> tuple[str, tuple[str, ...]]:
    reasons: list[tuple[str, str]] = []
    if listener is None:
        reasons.append(("port_not_listening", PROVIDER_RESOURCE_STATUS_WARNING))
    if process is None:
        missing_status = (
            PROVIDER_RESOURCE_STATUS_CRITICAL
            if target.required
            else PROVIDER_RESOURCE_STATUS_WARNING
        )
        reasons.append(("process_not_found", missing_status))
    elif match_confidence == "command":
        reasons.append(("command_match_without_port", PROVIDER_RESOURCE_STATUS_WARNING))

    if process is not None:
        _append_budget_reason(
            reasons,
            code="ram_budget_pressure",
            value=process.rss_bytes,
            warning=target.ram_warning_bytes,
            critical=target.ram_critical_bytes,
        )
    if gpu_process is not None:
        _append_budget_reason(
            reasons,
            code="gpu_memory_pressure",
            value=gpu_process.used_gpu_memory_bytes,
            warning=target.gpu_memory_warning_bytes,
            critical=target.gpu_memory_critical_bytes,
        )

    if _below(system_memory.memory_available_percent, DEFAULT_RAM_AVAILABLE_CRITICAL_PERCENT):
        reasons.append(("system_ram_pressure", PROVIDER_RESOURCE_STATUS_CRITICAL))
    elif _below(system_memory.memory_available_percent, DEFAULT_RAM_AVAILABLE_WARNING_PERCENT):
        reasons.append(("system_ram_pressure", PROVIDER_RESOURCE_STATUS_WARNING))

    swap_used = system_memory.swap_used_bytes
    swap_percent = system_memory.swap_used_percent
    if (
        swap_used is not None
        and swap_percent is not None
        and (
            swap_used >= DEFAULT_SWAP_CRITICAL_BYTES
            or swap_percent >= DEFAULT_SWAP_CRITICAL_PERCENT
        )
    ):
        reasons.append(("swap_pressure", PROVIDER_RESOURCE_STATUS_CRITICAL))
    elif (
        swap_used is not None
        and swap_percent is not None
        and (
            swap_used >= DEFAULT_SWAP_WARNING_BYTES or swap_percent >= DEFAULT_SWAP_WARNING_PERCENT
        )
    ):
        reasons.append(("swap_pressure", PROVIDER_RESOURCE_STATUS_WARNING))

    status = _worst_status([status for _, status in reasons])
    return status, tuple(code for code, _ in reasons)


def _append_budget_reason(
    reasons: list[tuple[str, str]],
    *,
    code: str,
    value: int,
    warning: int | None,
    critical: int | None,
) -> None:
    if critical is not None and value >= critical:
        reasons.append((code, PROVIDER_RESOURCE_STATUS_CRITICAL))
    elif warning is not None and value >= warning:
        reasons.append((code, PROVIDER_RESOURCE_STATUS_WARNING))


def _find_process_by_command(
    processes: dict[int, ProcessSnapshot],
    process_match: str,
) -> ProcessSnapshot | None:
    normalized_match = process_match.strip().lower()
    if not normalized_match:
        return None
    for process in processes.values():
        if normalized_match in process.command_preview.lower():
            return process
    return None


def _listeners_by_port(listeners: tuple[PortListener, ...]) -> dict[int, PortListener]:
    by_port: dict[int, PortListener] = {}
    for listener in listeners:
        by_port.setdefault(listener.port, listener)
    return by_port


def _extract_listener_port(line: str) -> int | None:
    for token in line.split():
        match = re.search(r":(\d+)$", token)
        if match:
            return int(match.group(1))
    return None


def _replace_target_host(
    target: ProviderResourceTarget,
    host: str,
) -> ProviderResourceTarget:
    selected_host = host.strip()
    if not selected_host:
        raise InvalidProviderResourceProbeError("host is required")
    return ProviderResourceTarget(
        provider_name=target.provider_name,
        provider_type=target.provider_type,
        host=selected_host,
        port=target.port,
        process_match=target.process_match,
        model_id=target.model_id,
        required=target.required,
        ram_warning_bytes=target.ram_warning_bytes,
        ram_critical_bytes=target.ram_critical_bytes,
        gpu_memory_warning_bytes=target.gpu_memory_warning_bytes,
        gpu_memory_critical_bytes=target.gpu_memory_critical_bytes,
    )


def _redacted_command_preview(command: str) -> str:
    tokens = command.split()
    redacted_tokens = [
        "<redacted>" if re.search(r"api[-_]?key|password|secret|token", token, re.I) else token
        for token in tokens
    ]
    preview = " ".join(redacted_tokens)
    return preview if len(preview) <= 180 else f"{preview[:177]}..."


def _command_hash(command: str) -> str:
    return hashlib.sha256(command.encode("utf-8")).hexdigest()


def _ensure_aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _worst_status(statuses: list[str] | tuple[str, ...]) -> str:
    if not statuses:
        return PROVIDER_RESOURCE_STATUS_OK
    return max(statuses, key=lambda status: _STATUS_RANK.get(status, 0))


def _percent(numerator: int | None, denominator: int | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return round(numerator / denominator * 100, 2)


def _below(value: float | None, threshold: float) -> bool:
    return value is not None and value < threshold


def _bytes_label(value: int | None) -> str:
    if value is None:
        return "-"
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    amount = float(value)
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            return f"{amount:.2f} {unit}" if unit != "B" else f"{int(amount)} B"
        amount /= 1024
    return f"{value} B"
