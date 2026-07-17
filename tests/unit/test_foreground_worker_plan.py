from datetime import UTC, datetime

import pytest

from app.core.foreground_worker_plan import (
    build_foreground_worker_plan,
    foreground_worker_plan_payload,
    render_foreground_worker_plan_markdown,
)


def test_foreground_worker_plan_renders_bounded_commands() -> None:
    plan = build_foreground_worker_plan(
        workdir="/repo",
        database_url_source="${DATABASE_URL}",
        lease_seconds=123,
        embedding_limit=7,
        chunk_policy_names=("small", "large"),
        generated_at=datetime(2026, 7, 17, 11, 12, 13, tzinfo=UTC),
    )
    payload = foreground_worker_plan_payload(plan)
    markdown = render_foreground_worker_plan_markdown(payload)

    assert payload["generated_at_label"] == "2026-07-17 11:12:13"
    assert payload["embedding_limit"] == 7
    assert [command["code"] for command in payload["commands"]] == [
        "pipeline_worker_help",
        "embedding_worker_help",
        "pipeline_worker_once",
        "embedding_worker_batch",
    ]
    pipeline_command = payload["commands"][2]["shell_command"]
    embedding_command = payload["commands"][3]["shell_command"]
    assert "--chunk-policy-names small large" in pipeline_command
    assert "--limit 7" in embedding_command
    assert "--lease-seconds 123" in embedding_command
    assert "Foreground Worker Command Plan" in markdown


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"workdir": ""}, "workdir is required"),
        ({"python_bin": ""}, "python_bin is required"),
        ({"database_url_source": ""}, "database_url_source is required"),
        ({"lease_seconds": 0}, "lease_seconds"),
        ({"embedding_limit": 0}, "embedding_limit"),
        ({"chunk_policy_names": ()}, "chunk policy"),
    ],
)
def test_foreground_worker_plan_rejects_invalid_values(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        build_foreground_worker_plan(**kwargs)
