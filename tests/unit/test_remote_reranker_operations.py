from typing import Any

import pytest

from app.core import remote_reranker_operations as operations
from app.core.config import Settings
from app.core.rerankers import (
    REMOTE_RERANKER_PROVIDER_MODE,
    RERANK_RETRIEVAL_STRATEGY,
    InvalidRerankerError,
    RerankerHealth,
    RerankResult,
    RerankResultItem,
)


class FakeCompletedProcess:
    def __init__(self, *, stdout: str, stderr: str = "", returncode: int = 0) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


class FakeRemoteRerankerClient:
    health_payload = RerankerHealth(
        ready=True,
        provider_type=REMOTE_RERANKER_PROVIDER_MODE,
        provider_model_id="Qwen/Qwen3-Reranker-4B",
        reranker_profile_name="qwen3_reranker_4b",
        device="cuda:0",
        runtime_metadata={
            "service": "nex_pcx_reranker_provider_service",
            "backend": "qwen_reranker",
            "device": "cuda:0",
        },
    )

    def __init__(self, base_url: str, *, timeout_seconds: float) -> None:
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds

    def health(self) -> RerankerHealth:
        return self.health_payload

    def rerank(self, request):
        return RerankResult(
            query_text=request.query_text,
            reranker_profile_name=request.reranker_profile_name,
            reranker_model_id=request.reranker_model_id,
            provider_type=REMOTE_RERANKER_PROVIDER_MODE,
            retrieval_strategy=RERANK_RETRIEVAL_STRATEGY,
            candidate_count=len(request.candidates),
            returned_count=2,
            top_k=2,
            results=(
                RerankResultItem(
                    candidate=request.candidates[0],
                    rank=1,
                    score=8.756176,
                    score_components={"source_rank": 1},
                ),
                RerankResultItem(
                    candidate=request.candidates[1],
                    rank=2,
                    score=6.445219,
                    score_components={"source_rank": 2},
                ),
            ),
            runtime_metadata={
                "service": "nex_pcx_reranker_provider_service",
                "backend": "qwen_reranker",
                "device": "cuda:0",
                "elapsed_ms": 11,
            },
        )

    def close(self) -> None:
        return None


def _health_payload(**overrides: Any) -> dict[str, Any]:
    payload = {
        "ready": True,
        "provider_type": "remote",
        "provider_model_id": "Qwen/Qwen3-Reranker-4B",
        "reranker_profile_name": "qwen3_reranker_4b",
        "device": "cuda:0",
        "runtime_metadata": {
            "service": "nex_pcx_reranker_provider_service",
            "backend": "qwen_reranker",
            "device": "cuda:0",
        },
    }
    payload.update(overrides)
    return payload


def _report(
    plan: operations.RemoteRerankerOperationsPlan,
    *,
    status: str = "running",
    command_ok: bool = True,
    health_ok: bool = True,
    health_mismatches: tuple[str, ...] = (),
    request_smoke_checked: bool = False,
    request_smoke_passed: bool | None = None,
    request_smoke_summary: dict[str, Any] | None = None,
    error: str | None = None,
) -> operations.RemoteRerankerOperationsReport:
    return operations.RemoteRerankerOperationsReport(
        plan=plan,
        status=status,
        pid="2437559" if status == "running" else None,
        command_observation=operations.RemoteCommandObservation(
            ok=command_ok,
            exit_code=0 if command_ok else 255,
            stdout=(
                f"status={status}\npid=2437559\n" if status == "running" else f"status={status}\n"
            ),
            stderr="" if command_ok else "ssh failed",
            values=(
                {"status": status, "pid": "2437559"} if status == "running" else {"status": status}
            ),
        ),
        health_checked=True,
        health_ok=health_ok,
        health_status_code=200 if health_ok else None,
        health_payload=_health_payload() if health_ok else None,
        health_error=None if health_ok else "connection refused",
        health_mismatches=health_mismatches,
        request_smoke_checked=request_smoke_checked,
        request_smoke_passed=request_smoke_passed,
        request_smoke_summary=request_smoke_summary,
        elapsed_ms=42,
        error=error,
    )


def test_remote_reranker_operations_status_uses_configured_remote_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_run(
        plan: operations.RemoteRerankerOperationsPlan,
        **kwargs: Any,
    ) -> operations.RemoteRerankerOperationsReport:
        captured["plan"] = plan
        captured["kwargs"] = kwargs
        return _report(
            plan,
            request_smoke_checked=True,
            request_smoke_passed=True,
            request_smoke_summary={
                "passed": True,
                "request_elapsed_ms": 17,
                "provider_elapsed_ms": 11,
                "candidate_count": 3,
                "returned_count": 2,
                "result_previews": [{"candidate_key": "candidate-1", "rank": 1, "score": 8.75}],
                "runtime_metadata": {"backend": "qwen_reranker"},
                "mismatches": [],
                "error": None,
            },
        )

    monkeypatch.setattr(operations, "run_remote_reranker_operations_status", fake_run)
    status = operations.get_remote_reranker_operations_status(
        Settings(
            reranker_provider_mode="remote",
            remote_reranker_provider_url="http://reranker.local:9199/",
            remote_reranker_provider_timeout_seconds=71.0,
        ),
        request_smoke=True,
    )

    assert status.status_code == 200
    assert captured["plan"].base_url == "http://reranker.local:9199"
    assert captured["kwargs"]["request_smoke"] is True
    assert status.payload["operations_status"] == "ready"
    assert status.payload["app_runtime"]["status"] == "remote_selected"
    assert status.payload["app_runtime"]["timeout_seconds"] == 71.0
    assert status.payload["request_smoke"]["summary"]["returned_count"] == 2


def test_remote_reranker_operations_status_defaults_to_dgx_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, operations.RemoteRerankerOperationsPlan] = {}

    def fake_run(
        plan: operations.RemoteRerankerOperationsPlan,
        **_: Any,
    ) -> operations.RemoteRerankerOperationsReport:
        captured["plan"] = plan
        return _report(plan)

    monkeypatch.setattr(operations, "run_remote_reranker_operations_status", fake_run)
    status = operations.get_remote_reranker_operations_status(Settings())

    assert status.status_code == 200
    assert captured["plan"].base_url == "http://192.168.20.243:9104"
    assert captured["plan"].ssh_target == "nexpcx@192.168.20.243"
    assert status.payload["app_runtime"]["status"] == "mock_selected"
    assert status.payload["provider"]["pid_file"].endswith("run/remote_reranker_provider_9104.pid")


@pytest.mark.parametrize(
    ("report_kwargs", "expected_status"),
    [
        ({"command_ok": False}, "command_failed"),
        ({"status": "stopped", "health_ok": False}, "stopped"),
        ({"health_mismatches": ("device mismatch",)}, "contract_mismatch"),
        ({"health_ok": False}, "health_unreachable"),
        (
            {"request_smoke_checked": True, "request_smoke_passed": False},
            "request_smoke_failed",
        ),
        ({"error": "unexpected failure"}, "failed"),
        ({"status": "unknown"}, "warning"),
    ],
)
def test_remote_reranker_operations_status_classifies_failures(
    monkeypatch: pytest.MonkeyPatch,
    report_kwargs: dict[str, Any],
    expected_status: str,
) -> None:
    def fake_run(
        plan: operations.RemoteRerankerOperationsPlan,
        **_: Any,
    ) -> operations.RemoteRerankerOperationsReport:
        return _report(plan, **report_kwargs)

    monkeypatch.setattr(operations, "run_remote_reranker_operations_status", fake_run)
    status = operations.get_remote_reranker_operations_status(Settings())

    assert status.status_code == 503
    assert status.payload["passed"] is False
    assert status.payload["operations_status"] == expected_status


def test_remote_reranker_operations_status_reports_misconfigured_runtime() -> None:
    status = operations.get_remote_reranker_operations_status(
        Settings(reranker_provider_mode="remote"),
        request_smoke=True,
    )

    assert status.status_code == 503
    assert status.payload["operations_status"] == "misconfigured"
    assert status.payload["app_runtime"]["status"] == "misconfigured"
    assert "remote_reranker_provider_url" in status.payload["error"]


def test_remote_reranker_operations_route_uses_default_https_port() -> None:
    route_host, port = operations._operations_route_from_runtime_config(
        {"remote_base_url": "https://reranker.example"},
        default_host="default.local",
        default_port=9104,
    )

    assert route_host == "reranker.example"
    assert port == 443


def test_remote_reranker_operations_route_uses_default_http_port() -> None:
    route_host, port = operations._operations_route_from_runtime_config(
        {"remote_base_url": "http://reranker.example"},
        default_host="default.local",
        default_port=9104,
    )

    assert route_host == "reranker.example"
    assert port == 80


def test_remote_reranker_operations_rejects_invalid_plan_inputs() -> None:
    with pytest.raises(ValueError, match="host"):
        operations.build_remote_reranker_operations_plan(host=" ")
    with pytest.raises(ValueError, match="port"):
        operations.build_remote_reranker_operations_plan(port=0)


def test_remote_reranker_operations_status_reports_invalid_runtime_url() -> None:
    status = operations.get_remote_reranker_operations_status(
        Settings(
            reranker_provider_mode="remote",
            remote_reranker_provider_url="not-a-url",
        )
    )

    assert status.status_code == 503
    assert status.payload["operations_status"] == "misconfigured"
    assert "hostname" in status.payload["error"]


def test_run_remote_reranker_operations_status_checks_ssh_health_and_smoke(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[tuple[str, ...]] = []

    def fake_run(command: tuple[str, ...], **kwargs: Any) -> FakeCompletedProcess:
        commands.append(command)
        assert kwargs["timeout"] == operations.DEFAULT_REMOTE_RERANKER_SSH_TIMEOUT_SECONDS
        return FakeCompletedProcess(
            stdout=(
                "status=running\n"
                "pid=2437559\n"
                "file_pid=2437559\n"
                "pid_file=run/remote_reranker_provider_9104.pid\n"
                "log_file=logs/remote_reranker_provider_9104.log\n"
            )
        )

    monkeypatch.setattr(operations.subprocess, "run", fake_run)
    monkeypatch.setattr(operations, "RemoteRerankerProviderClient", FakeRemoteRerankerClient)
    plan = operations.build_remote_reranker_operations_plan()

    report = operations.run_remote_reranker_operations_status(plan, request_smoke=True)

    assert report.passed is True
    assert report.pid == "2437559"
    assert report.request_smoke_passed is True
    assert report.request_smoke_summary["provider_elapsed_ms"] == 11
    assert commands[0][:5] == ("ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5")
    assert commands[0][5] == "nexpcx@192.168.20.243"


def test_run_remote_reranker_operations_status_skips_smoke_on_health_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class MismatchHealthClient(FakeRemoteRerankerClient):
        health_payload = RerankerHealth(
            ready=True,
            provider_type=REMOTE_RERANKER_PROVIDER_MODE,
            provider_model_id="Qwen/Qwen3-Reranker-4B",
            reranker_profile_name="qwen3_reranker_4b",
            device="cpu",
            runtime_metadata={
                "service": "nex_pcx_reranker_provider_service",
                "backend": "qwen_reranker",
                "device": "cpu",
            },
        )

    monkeypatch.setattr(
        operations.subprocess,
        "run",
        lambda *_args, **_kwargs: FakeCompletedProcess(stdout="status=running\npid=2437559\n"),
    )
    monkeypatch.setattr(operations, "RemoteRerankerProviderClient", MismatchHealthClient)
    plan = operations.build_remote_reranker_operations_plan()

    report = operations.run_remote_reranker_operations_status(plan, request_smoke=True)

    assert report.passed is False
    assert "device: expected 'cuda:0', got 'cpu'" in report.health_mismatches
    assert report.request_smoke_summary is None
    assert report.request_smoke_passed is None


def test_run_remote_reranker_operations_status_reports_ssh_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(*_args: Any, **_kwargs: Any) -> None:
        raise operations.subprocess.TimeoutExpired(cmd="ssh", timeout=8)

    monkeypatch.setattr(operations.subprocess, "run", fake_run)
    plan = operations.build_remote_reranker_operations_plan()

    report = operations.run_remote_reranker_operations_status(plan, request_smoke=True)

    assert report.passed is False
    assert report.command_observation.ok is False
    assert report.request_smoke_checked is True
    assert "timed out" in (report.error or "")


def test_probe_health_reports_client_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class FailingHealthClient(FakeRemoteRerankerClient):
        def health(self) -> RerankerHealth:
            raise InvalidRerankerError("health failed")

    monkeypatch.setattr(operations, "RemoteRerankerProviderClient", FailingHealthClient)
    plan = operations.build_remote_reranker_operations_plan()

    observation = operations._probe_health_once(plan, timeout_seconds=5)

    assert observation.ok is False
    assert observation.status_code is None
    assert observation.error == "health failed"


def test_run_request_smoke_reports_client_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class FailingRerankClient(FakeRemoteRerankerClient):
        def rerank(self, request):
            raise InvalidRerankerError("rerank failed")

    monkeypatch.setattr(operations, "RemoteRerankerProviderClient", FailingRerankClient)
    plan = operations.build_remote_reranker_operations_plan()

    summary = operations._run_request_smoke(plan, timeout_seconds=5)

    assert summary["passed"] is False
    assert summary["returned_count"] is None
    assert summary["error"] == "rerank failed"


def test_request_smoke_mismatches_report_shape_rank_score_and_metadata() -> None:
    plan = operations.build_remote_reranker_operations_plan()
    request = operations._build_request_smoke_request(plan)
    result = RerankResult(
        query_text=request.query_text,
        reranker_profile_name=request.reranker_profile_name,
        reranker_model_id=request.reranker_model_id,
        provider_type=REMOTE_RERANKER_PROVIDER_MODE,
        retrieval_strategy=RERANK_RETRIEVAL_STRATEGY,
        candidate_count=len(request.candidates),
        returned_count=3,
        top_k=2,
        results=(
            RerankResultItem(
                candidate=request.candidates[0],
                rank=2,
                score=float("inf"),
                score_components={},
            ),
            RerankResultItem(
                candidate=request.candidates[1],
                rank=3,
                score=1.0,
                score_components={},
            ),
        ),
        runtime_metadata={"service": "unexpected"},
    )

    mismatches = operations._request_smoke_mismatches(result, plan=plan, request=request)

    assert "result_count: expected 3, got 2" in mismatches
    assert "result ranks: expected (1, 2), got (2, 3)" in mismatches
    assert "score must be finite for 'candidate-1'" in mismatches
    assert any(item.startswith("runtime_metadata.service") for item in mismatches)
    assert any(item.startswith("runtime_metadata.backend") for item in mismatches)
    assert any(item.startswith("runtime_metadata.device") for item in mismatches)


def test_small_helpers_handle_missing_and_invalid_values() -> None:
    assert (
        operations._health_mismatches(None, plan=operations.build_remote_reranker_operations_plan())
        == ()
    )
    assert operations._parse_key_values("noise\nstatus=running\n") == {"status": "running"}
    assert operations._metadata_int({}, "elapsed_ms") is None
    assert operations._metadata_int({"elapsed_ms": "slow"}, "elapsed_ms") is None
