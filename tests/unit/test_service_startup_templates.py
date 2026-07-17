from pathlib import Path

import pytest

from app.core.service_startup_templates import (
    build_service_startup_template_plan,
    render_env_file,
    render_operator_readme,
    render_service_startup_template_plan_json,
    render_systemd_unit,
    service_startup_template_plan_payload,
    write_service_startup_templates,
)


def test_build_plan_contains_web_and_worker_services(tmp_path) -> None:
    plan = build_service_startup_template_plan(
        workdir=str(tmp_path / "repo"),
        user="nexpcx",
        group="nexpcx",
        output_dir=str(tmp_path / "deployment"),
        web_port=8080,
        chunk_policy_names=("small", "large"),
    )
    payload = service_startup_template_plan_payload(plan)

    assert plan.env_file.endswith("deployment/env/nex-pcx.env")
    assert [service.service_name for service in plan.services] == [
        "nex-pcx-web",
        "nex-pcx-pipeline-worker",
        "nex-pcx-embedding-worker",
    ]
    assert plan.services[0].restart == "on-failure"
    assert plan.services[1].restart == "always"
    assert "--port 8080" in plan.services[0].shell_command
    assert "--chunk-policy-names small large" in plan.services[1].shell_command
    assert payload["user_systemd"] is False
    assert payload["environment"]["NEX_PCX_DATABASE_URL"] == "***"


def test_render_env_and_systemd_templates_include_expected_controls(tmp_path) -> None:
    plan = build_service_startup_template_plan(
        workdir=str(tmp_path / "repo path"),
        user="svc",
        output_dir=str(tmp_path / "deployment"),
        database_url_placeholder="postgresql://svc:secret@db/nex_pcx",
    )
    env_text = render_env_file(plan)
    unit_text = render_systemd_unit(plan, plan.services[0])
    readme = render_operator_readme(plan)

    assert "NEX_PCX_EMBEDDING_PROVIDER_MODE=remote" in env_text
    assert "NEX_PCX_DATABASE_URL=postgresql://svc:secret@db/nex_pcx" in env_text
    assert "WorkingDirectory=" in unit_text
    assert "ExecStart=" in unit_text
    assert "Restart=on-failure" in unit_text
    assert "PrivateTmp=true" in unit_text
    assert "nex-pcx-web.service" in readme


def test_render_user_systemd_templates_omit_system_only_controls(tmp_path) -> None:
    plan = build_service_startup_template_plan(
        workdir=str(tmp_path / "repo"),
        user="svc",
        output_dir=str(tmp_path / "deployment"),
        user_systemd=True,
    )

    unit_text = render_systemd_unit(plan, plan.services[0])
    readme = render_operator_readme(plan)

    assert "User=" not in unit_text
    assert "Group=" not in unit_text
    assert "After=network-online.target" not in unit_text
    assert "NoNewPrivileges=true" not in unit_text
    assert "PrivateTmp=true" not in unit_text
    assert "WantedBy=default.target" in unit_text
    assert "systemctl --user daemon-reload" in readme
    assert "Systemd scope: `user`" in readme


def test_write_service_startup_templates_creates_env_units_and_readme(tmp_path) -> None:
    plan = build_service_startup_template_plan(
        workdir=str(tmp_path / "repo"),
        user="nexpcx",
        output_dir=str(tmp_path / "deployment"),
    )

    paths = write_service_startup_templates(plan)

    assert len(paths) == 5
    assert Path(plan.env_file).exists()
    assert all(Path(service.unit_file).exists() for service in plan.services)
    assert (tmp_path / "deployment" / "README.md").exists()


def test_plan_json_can_be_pretty_printed(tmp_path) -> None:
    plan = build_service_startup_template_plan(
        workdir=str(tmp_path / "repo"),
        user="nexpcx",
    )

    json_text = render_service_startup_template_plan_json(plan, pretty=True)

    assert '"service_name": "nex-pcx-web"' in json_text
    assert "\n  " in json_text


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"workdir": ""}, "workdir is required"),
        ({"user": ""}, "user is required"),
        ({"web_port": 0}, "web_port must be greater than zero"),
        ({"restart_seconds": 0}, "restart_seconds must be greater than zero"),
        ({"chunk_policy_names": ()}, "at least one chunk policy name is required"),
    ],
)
def test_build_plan_validates_required_values(tmp_path, kwargs, message) -> None:
    values = {
        "workdir": str(tmp_path / "repo"),
        "user": "nexpcx",
    }
    values.update(kwargs)

    with pytest.raises(ValueError, match=message):
        build_service_startup_template_plan(**values)
