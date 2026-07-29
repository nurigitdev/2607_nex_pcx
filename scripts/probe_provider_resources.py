"""Probe provider process RAM/GPU/swap resource usage."""

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.provider_resource_probe import (  # noqa: E402
    DEFAULT_DGX_PROVIDER_RESOURCE_HOST,
    InvalidProviderResourceProbeError,
    build_provider_resource_probe_report,
    provider_resource_probe_report_payload,
    render_provider_resource_probe_markdown,
    select_provider_resource_targets,
)
from app.core.provider_resource_snapshots import (  # noqa: E402
    InvalidProviderResourceSnapshotError,
    provider_resource_snapshot_record_payload,
    record_provider_resource_probe_payload,
)

DEFAULT_REMOTE_WORKDIR = "/home/nexpcx/2607_nex_pcx"
DEFAULT_REMOTE_PYTHON_BIN = "/home/nexpcx/2607_nex_pcx/.venv/bin/python"
DEFAULT_TIMEOUT_SECONDS = 15


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Collect read-only resource evidence for DGX remote providers.",
    )
    parser.add_argument("--host", default=DEFAULT_DGX_PROVIDER_RESOURCE_HOST)
    parser.add_argument(
        "--provider",
        action="append",
        default=[],
        help="Provider selector: all, embedding, reranker, vllm, kure, bge, qwen, generation.",
    )
    parser.add_argument(
        "--ssh-user",
        default=None,
        help="When provided, execute this probe on the remote host through SSH.",
    )
    parser.add_argument("--remote-workdir", default=DEFAULT_REMOTE_WORKDIR)
    parser.add_argument("--remote-python-bin", default=DEFAULT_REMOTE_PYTHON_BIN)
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print JSON to stdout.")
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--markdown-output", default=None)
    parser.add_argument(
        "--database-url",
        default=os.getenv("NEX_PCX_DATABASE_URL"),
        help="Database URL used with --persist. Defaults to NEX_PCX_DATABASE_URL.",
    )
    parser.add_argument(
        "--persist",
        action="store_true",
        help="Persist provider resource snapshots to the database.",
    )
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args(argv)

    try:
        targets = select_provider_resource_targets(
            tuple(args.provider) or None,
            host=args.host,
        )
    except InvalidProviderResourceProbeError as exc:
        print(f"provider resource probe plan failed: {exc}", file=sys.stderr)
        return 2

    report = None
    if args.dry_run:
        if args.persist:
            print(
                "provider resource persistence failed: dry-run cannot be persisted", file=sys.stderr
            )
            return 2
        payload = _dry_run_payload(args, targets)
    elif args.ssh_user and not args.local_only:
        try:
            payload = _run_remote_probe(args)
        except (json.JSONDecodeError, subprocess.SubprocessError, OSError, RuntimeError) as exc:
            print(f"remote provider resource probe failed: {exc}", file=sys.stderr)
            return 1
    else:
        report = collect_local_report(args, targets)
        payload = provider_resource_probe_report_payload(report)

    if args.persist:
        if not args.database_url:
            print("provider resource persistence failed: database URL is required", file=sys.stderr)
            return 1
        try:
            records = record_provider_resource_probe_payload(
                args.database_url,
                payload,
                runtime_metadata={
                    "source": "scripts/probe_provider_resources.py",
                    "remote": bool(args.ssh_user and not args.local_only),
                },
            )
        except InvalidProviderResourceSnapshotError as exc:
            print(f"provider resource persistence failed: {exc}", file=sys.stderr)
            return 1
        except Exception as exc:  # pragma: no cover - defensive CLI boundary
            print(f"provider resource persistence failed: {exc}", file=sys.stderr)
            return 1
        payload["snapshot_records"] = [
            provider_resource_snapshot_record_payload(record) for record in records
        ]

    json_text = json.dumps(payload, ensure_ascii=False, indent=2 if args.pretty else None)
    if args.json_output:
        _write_text(Path(args.json_output), json_text + "\n")
    if args.markdown_output:
        if report is not None:
            markdown_text = render_provider_resource_probe_markdown(report)
        elif "snapshots" in payload:
            markdown_text = _render_payload_markdown(payload)
        else:
            markdown_text = _render_dry_run_markdown(payload)
        _write_text(Path(args.markdown_output), markdown_text)
    if args.json or not args.json_output:
        print(json_text)
    return 0 if payload.get("status") != "critical" else 1


def collect_local_report(args: argparse.Namespace, targets):
    collector_errors: list[str] = []
    ps_text = _run_command_text(
        ["ps", "-eo", "pid=,ppid=,user=,rss=,vsz=,pcpu=,etimes=,args="],
        timeout_seconds=args.timeout_seconds,
        collector_errors=collector_errors,
    )
    ss_text = _run_command_text(
        ["ss", "-ltnp"],
        timeout_seconds=args.timeout_seconds,
        collector_errors=collector_errors,
    )
    nvidia_smi_text = _run_command_text(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        timeout_seconds=args.timeout_seconds,
        collector_errors=collector_errors,
        required=False,
    )
    meminfo_text = ""
    try:
        meminfo_text = Path("/proc/meminfo").read_text(encoding="utf-8")
    except OSError as exc:
        collector_errors.append(f"/proc/meminfo failed: {exc}")
    return build_provider_resource_probe_report(
        targets=targets,
        ps_text=ps_text,
        ss_text=ss_text,
        nvidia_smi_text=nvidia_smi_text,
        meminfo_text=meminfo_text,
        collector_errors=tuple(collector_errors),
    )


def build_remote_probe_command(args: argparse.Namespace) -> list[str]:
    remote_args = [
        args.remote_python_bin,
        "scripts/probe_provider_resources.py",
        "--local-only",
        "--host",
        args.host,
        "--json",
    ]
    for provider in args.provider:
        remote_args.extend(["--provider", provider])
    remote_command = f"cd {shlex.quote(args.remote_workdir)} && " + " ".join(
        shlex.quote(part) for part in remote_args
    )
    return ["ssh", f"{args.ssh_user}@{args.host}", remote_command]


def _run_remote_probe(args: argparse.Namespace) -> dict[str, object]:
    command = build_remote_probe_command(args)
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=args.timeout_seconds,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or "ssh failed")
    return json.loads(completed.stdout)


def _run_command_text(
    command: list[str],
    *,
    timeout_seconds: int,
    collector_errors: list[str],
    required: bool = True,
) -> str:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        collector_errors.append(f"{command[0]} failed: {exc}")
        return ""
    if completed.returncode != 0 and required:
        collector_errors.append(f"{command[0]} exited {completed.returncode}: {completed.stderr}")
    return completed.stdout


def _dry_run_payload(args: argparse.Namespace, targets) -> dict[str, object]:
    payload: dict[str, object] = {
        "status": "dry_run",
        "host": args.host,
        "remote": bool(args.ssh_user and not args.local_only),
        "targets": [
            {
                "provider_name": target.provider_name,
                "provider_type": target.provider_type,
                "host": target.host,
                "port": target.port,
                "process_match": target.process_match,
                "model_id": target.model_id,
            }
            for target in targets
        ],
        "commands": [
            "ps -eo pid=,ppid=,user=,rss=,vsz=,pcpu=,etimes=,args=",
            "ss -ltnp",
            (
                "nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory "
                "--format=csv,noheader,nounits"
            ),
            "cat /proc/meminfo",
        ],
    }
    if args.ssh_user and not args.local_only:
        payload["ssh_command"] = build_remote_probe_command(args)
    return payload


def _render_dry_run_markdown(payload: dict[str, object]) -> str:
    lines = [
        "# DGX Provider Resource Probe Dry Run",
        "",
        f"- `host`: `{payload.get('host')}`",
        f"- `remote`: `{payload.get('remote')}`",
        "",
        "| Provider | Type | Port | Process Match |",
        "| --- | --- | ---: | --- |",
    ]
    for target in payload.get("targets", []):
        if not isinstance(target, dict):
            continue
        lines.append(
            "| "
            f"{target.get('provider_name')} | "
            f"{target.get('provider_type')} | "
            f"{target.get('port')} | "
            f"{target.get('process_match')} |"
        )
    return "\n".join(lines) + "\n"


def _render_payload_markdown(payload: dict[str, object]) -> str:
    lines = [
        "# DGX Provider Resource Probe",
        "",
        f"- `host`: `{payload.get('host')}`",
        f"- `collected_at`: `{payload.get('collected_at')}`",
        f"- `status`: `{payload.get('status')}`",
        "",
        "| Provider | Type | Port | Status | PID | RAM RSS | GPU Memory | Reasons |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for snapshot in payload.get("snapshots", []):
        if not isinstance(snapshot, dict):
            continue
        process = snapshot.get("process") if isinstance(snapshot.get("process"), dict) else {}
        gpu_process = (
            snapshot.get("gpu_process") if isinstance(snapshot.get("gpu_process"), dict) else {}
        )
        reason_codes = snapshot.get("reason_codes")
        reasons = ", ".join(reason_codes) if isinstance(reason_codes, list) else "-"
        lines.append(
            "| "
            f"{snapshot.get('provider_name')} | "
            f"{snapshot.get('provider_type')} | "
            f"{snapshot.get('port')} | "
            f"{snapshot.get('status')} | "
            f"{process.get('pid', '-')} | "
            f"{process.get('rss_bytes', '-')} | "
            f"{gpu_process.get('used_gpu_memory_bytes', '-')} | "
            f"{reasons or '-'} |"
        )
    return "\n".join(lines) + "\n"


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
