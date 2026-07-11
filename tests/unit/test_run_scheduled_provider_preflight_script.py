import importlib.util
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from app.core.embedding_provider_preflight_schedules import (
    EmbeddingProviderPreflightScheduleRecord,
    ScheduledProviderRoutePreflightRun,
)


def _load_run_scheduled_provider_preflight_module():
    script_path = (
        Path(__file__).resolve().parents[2] / "scripts" / "run_scheduled_provider_preflight.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_scheduled_provider_preflight_script",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


run_scheduled_provider_preflight = _load_run_scheduled_provider_preflight_module()


def test_run_scheduled_preflight_serializes_due_runs(monkeypatch) -> None:
    schedule = make_schedule(next_run_at=datetime(2026, 7, 11, 13, 0, tzinfo=UTC))
    updated_schedule = make_schedule(
        next_run_at=datetime(2026, 7, 11, 14, 0, tzinfo=UTC),
        last_status="succeeded",
        run_count=1,
    )
    calls = {}

    def fake_run_due(database_url: str, **kwargs):
        calls["run_due"] = (database_url, kwargs)
        return [
            ScheduledProviderRoutePreflightRun(
                schedule=schedule,
                status="succeeded",
                result={"route_count": 1, "failed_count": 0},
                updated_schedule=updated_schedule,
            )
        ]

    monkeypatch.setattr(
        run_scheduled_provider_preflight,
        "run_due_embedding_provider_preflight_schedules",
        fake_run_due,
    )

    payload = run_scheduled_provider_preflight.run_scheduled_preflight(
        "postgresql://example/db",
        limit=3,
        schedule_name="hourly-kure",
    )

    assert calls["run_due"] == (
        "postgresql://example/db",
        {"limit": 3, "schedule_name": "hourly-kure"},
    )
    assert payload["run_count"] == 1
    assert payload["failed_count"] == 0
    assert payload["results"][0]["schedule_name"] == "hourly-kure"
    assert payload["results"][0]["next_run_at"] == "2026-07-11T14:00:00+00:00"


def test_main_returns_failure_exit_code_when_schedule_fails(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        run_scheduled_provider_preflight,
        "get_settings",
        lambda: SimpleNamespace(database_url="postgresql://example/db"),
    )
    monkeypatch.setattr(
        run_scheduled_provider_preflight,
        "run_scheduled_preflight",
        lambda *args, **kwargs: {
            "run_count": 1,
            "failed_count": 1,
            "results": [],
        },
    )
    monkeypatch.setattr(sys, "argv", ["run_scheduled_provider_preflight.py"])

    exit_code = run_scheduled_provider_preflight.main()

    assert exit_code == 1
    assert '"failed_count": 1' in capsys.readouterr().out


def test_main_allows_failed_schedules_when_requested(monkeypatch) -> None:
    monkeypatch.setattr(
        run_scheduled_provider_preflight,
        "get_settings",
        lambda: SimpleNamespace(database_url="postgresql://example/db"),
    )
    monkeypatch.setattr(
        run_scheduled_provider_preflight,
        "run_scheduled_preflight",
        lambda *args, **kwargs: {
            "run_count": 1,
            "failed_count": 1,
            "results": [],
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_scheduled_provider_preflight.py", "--allow-failures"],
    )

    assert run_scheduled_provider_preflight.main() == 0


def make_schedule(**overrides) -> EmbeddingProviderPreflightScheduleRecord:
    now = datetime(2026, 7, 11, 12, 0, tzinfo=UTC)
    values = {
        "schedule_name": "hourly-kure",
        "description": "Hourly KURE preflight",
        "profile_name": "kure_v1_1024",
        "active_only": True,
        "interval_minutes": 60,
        "is_enabled": True,
        "next_run_at": now,
        "last_run_at": None,
        "last_status": "never_run",
        "last_result": {},
        "run_count": 0,
        "failure_count": 0,
        "created_at": now,
        "updated_at": now,
    }
    values.update(overrides)
    return EmbeddingProviderPreflightScheduleRecord(**values)
