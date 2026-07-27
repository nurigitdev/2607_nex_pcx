from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.database import connect
from app.main import create_app

pytestmark = pytest.mark.integration


def _template_key(prefix: str = "pytest_generation_template") -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


def _template_payload(template_key: str, *, is_default: bool = False) -> dict[str, object]:
    return {
        "template_key": template_key,
        "template_family": template_key,
        "template_name": "Pytest 생성 템플릿",
        "template_version": "v1",
        "document_type": "report",
        "language": "ko",
        "output_format": "markdown",
        "section_schema": [
            {"key": "title", "heading": "제목", "required": True},
            {"key": "evidence", "heading": "근거", "required": True},
        ],
        "system_instruction": "검색 근거를 보고서 형식으로 정리한다.",
        "user_instruction_suffix": "각 주장에 citation key를 포함한다.",
        "style_guidance": {"tone": "formal"},
        "citation_policy": {"required": True, "minimum_citations": 1},
        "is_default": is_default,
        "is_active": True,
        "change_note": "pytest 관리 API smoke",
        "created_by": "pytest-api",
    }


def _cleanup_generation_templates(database_url: str, *template_keys: str) -> None:
    with connect(database_url) as conn:
        conn.execute(
            """
            UPDATE generation_templates
            SET is_default = false,
                updated_at = now()
            WHERE is_default
            """,
        )
        conn.execute(
            """
            UPDATE generation_templates
            SET is_default = true,
                is_active = true,
                updated_at = now()
            WHERE template_key = 'grounded_answer'
            """,
        )
        if template_keys:
            conn.execute(
                """
                DELETE FROM generation_templates
                WHERE template_key = ANY(%s)
                """,
                (list(template_keys),),
            )
        conn.commit()


def test_generation_template_management_api_crud_and_guardrails(
    migrated_database_url: str,
) -> None:
    template_key = _template_key()
    inactive_key = _template_key("pytest_inactive_generation_template")
    clone_key = _template_key("pytest_clone_generation_template")
    app = create_app(Settings(database_url=migrated_database_url))

    try:
        with TestClient(app) as client:
            create_response = client.post(
                "/api/admin/generation-templates",
                json=_template_payload(template_key),
            )
            read_response = client.get(f"/api/admin/generation-templates/{template_key}")
            update_payload = _template_payload(template_key, is_default=True)
            update_payload["template_name"] = "Pytest 생성 템플릿 v2"
            update_payload["template_version"] = "v2"
            update_response = client.put(
                f"/api/admin/generation-templates/{template_key}",
                json=update_payload,
            )
            list_response = client.get(
                "/api/admin/generation-templates",
                params={"include_inactive": "true"},
            )
            clone_response = client.post(
                f"/api/admin/generation-templates/{template_key}/clone",
                json={
                    "target_template_key": clone_key,
                    "target_template_version": "v3",
                    "target_template_name": "Pytest 생성 템플릿 clone",
                    "change_note": "rollback 후보",
                    "created_by": "pytest-api",
                },
            )
            duplicate_clone_response = client.post(
                f"/api/admin/generation-templates/{template_key}/clone",
                json={
                    "target_template_key": clone_key,
                    "target_template_version": "v4",
                },
            )
            rollback_response = client.post(f"/api/admin/generation-templates/{clone_key}/rollback")
            deactivate_default_response = client.patch(
                f"/api/admin/generation-templates/{clone_key}/active",
                json={"is_active": False},
            )
            inactive_create_response = client.post(
                "/api/admin/generation-templates",
                json={
                    **_template_payload(inactive_key),
                    "is_active": False,
                    "is_default": False,
                },
            )
            inactive_default_response = client.post(
                f"/api/admin/generation-templates/{inactive_key}/default"
            )
            mismatch_response = client.put(
                f"/api/admin/generation-templates/{template_key}_mismatch",
                json=_template_payload(template_key),
            )
            invalid_response = client.post(
                "/api/admin/generation-templates",
                json={**_template_payload(_template_key()), "section_schema": []},
            )
            missing_read_response = client.get("/api/admin/generation-templates/missing_template")
            missing_active_response = client.patch(
                "/api/admin/generation-templates/missing_template/active",
                json={"is_active": True},
            )
            missing_default_response = client.post(
                "/api/admin/generation-templates/missing_template/default"
            )
            missing_clone_response = client.post(
                "/api/admin/generation-templates/missing_template/clone",
                json={
                    "target_template_key": _template_key("pytest_missing_clone"),
                    "target_template_version": "v1",
                },
            )
            missing_rollback_response = client.post(
                "/api/admin/generation-templates/missing_template/rollback"
            )

        assert create_response.status_code == 201
        assert create_response.json()["template"]["template_key"] == template_key
        assert read_response.status_code == 200
        assert read_response.json()["template"]["system_instruction"].startswith("검색 근거")
        assert update_response.status_code == 200
        assert update_response.json()["template"]["template_name"] == "Pytest 생성 템플릿 v2"
        assert update_response.json()["template"]["is_default"] is True
        assert list_response.status_code == 200
        assert list_response.json()["summary"]["default_template_key"] == template_key
        assert list_response.json()["summary"]["default_template_version"] == "v2"
        assert list_response.json()["summary"]["applied_template_label"].startswith(
            "Pytest 생성 템플릿 v2"
        )
        assert template_key in [
            template["template_key"] for template in list_response.json()["templates"]
        ]
        assert clone_response.status_code == 201
        assert clone_response.json()["template"]["template_key"] == clone_key
        assert clone_response.json()["template"]["template_family"] == template_key
        assert clone_response.json()["template"]["template_version"] == "v3"
        assert clone_response.json()["template"]["change_note"] == "rollback 후보"
        assert clone_response.json()["template"]["clone_source_template_id"] is not None
        assert duplicate_clone_response.status_code == 400
        assert "already exists" in duplicate_clone_response.json()["detail"]
        assert rollback_response.status_code == 200
        assert rollback_response.json()["template"]["template_key"] == clone_key
        assert rollback_response.json()["template"]["is_default"] is True
        assert deactivate_default_response.status_code == 400
        assert "default" in deactivate_default_response.json()["detail"]
        assert inactive_create_response.status_code == 201
        assert inactive_create_response.json()["template"]["is_active"] is False
        assert inactive_default_response.status_code == 400
        assert mismatch_response.status_code == 400
        assert "path and payload" in mismatch_response.json()["detail"]
        assert invalid_response.status_code == 400
        assert "section_schema" in invalid_response.json()["detail"]
        assert missing_read_response.status_code == 404
        assert missing_active_response.status_code == 404
        assert missing_default_response.status_code == 404
        assert missing_clone_response.status_code == 404
        assert missing_rollback_response.status_code == 404
    finally:
        _cleanup_generation_templates(migrated_database_url, template_key, inactive_key, clone_key)


def test_generation_template_management_api_returns_503_without_database() -> None:
    app = create_app(Settings(database_url=None))

    with TestClient(app) as client:
        list_response = client.get("/api/admin/generation-templates")
        read_response = client.get("/api/admin/generation-templates/report")
        create_response = client.post(
            "/api/admin/generation-templates",
            json=_template_payload("pytest_no_database"),
        )
        update_response = client.put(
            "/api/admin/generation-templates/report",
            json=_template_payload("report"),
        )
        active_response = client.patch(
            "/api/admin/generation-templates/report/active",
            json={"is_active": True},
        )
        default_response = client.post("/api/admin/generation-templates/report/default")
        clone_response = client.post(
            "/api/admin/generation-templates/report/clone",
            json={"target_template_key": "report_v2", "target_template_version": "v2"},
        )
        rollback_response = client.post("/api/admin/generation-templates/report/rollback")

    assert list_response.status_code == 503
    assert read_response.status_code == 503
    assert create_response.status_code == 503
    assert update_response.status_code == 503
    assert active_response.status_code == 503
    assert default_response.status_code == 503
    assert clone_response.status_code == 503
    assert rollback_response.status_code == 503


def test_generation_template_management_ui_saves_and_toggles_template(
    migrated_database_url: str,
) -> None:
    template_key = _template_key("pytest_ui_generation_template")
    clone_key = _template_key("pytest_ui_generation_template_clone")
    app = create_app(Settings(database_url=migrated_database_url))

    try:
        with TestClient(app) as client:
            initial_response = client.get("/admin/generation-templates")
            new_response = client.get("/admin/generation-templates?new_template=true")
            save_response = client.post(
                "/admin/generation-templates/upsert",
                data={
                    "template_key": template_key,
                    "template_family": "pytest_ui_family",
                    "template_name": "Pytest UI 보고서",
                    "template_version": "v1",
                    "document_type": "report",
                    "language": "ko",
                    "output_format": "markdown",
                    "section_schema": """
                    [
                      {"key": "title", "heading": "제목", "required": true},
                      {"key": "evidence", "heading": "근거", "required": true}
                    ]
                    """,
                    "system_instruction": "UI에서 저장한 보고서 템플릿이다.",
                    "user_instruction_suffix": "근거를 명확히 작성한다.",
                    "style_guidance": '{"tone": "formal"}',
                    "citation_policy": '{"required": true}',
                    "change_note": "UI 생성",
                    "is_active": "true",
                },
                follow_redirects=False,
            )
            saved_response = client.get(save_response.headers["location"])
            clone_response = client.post(
                f"/admin/generation-templates/{template_key}/clone",
                data={
                    "target_template_key": clone_key,
                    "target_template_version": "v2",
                    "target_template_name": "Pytest UI 보고서 v2",
                    "change_note": "UI clone",
                    "is_active": "true",
                    "include_inactive": "true",
                },
                follow_redirects=False,
            )
            cloned_response = client.get(clone_response.headers["location"])
            rollback_response = client.post(
                f"/admin/generation-templates/{clone_key}/rollback",
                data={"include_inactive": "true"},
                follow_redirects=False,
            )
            rolled_back_response = client.get(rollback_response.headers["location"])
            deactivate_response = client.post(
                f"/admin/generation-templates/{template_key}/active",
                data={"is_active": "false", "include_inactive": "true"},
                follow_redirects=False,
            )
            inactive_response = client.get(deactivate_response.headers["location"])
            invalid_json_response = client.post(
                "/admin/generation-templates/upsert",
                data={
                    "template_key": template_key,
                    "template_name": "Invalid JSON",
                    "document_type": "report",
                    "section_schema": "[",
                    "system_instruction": "invalid json smoke",
                    "style_guidance": "{}",
                    "citation_policy": "{}",
                    "is_active": "true",
                },
                follow_redirects=False,
            )
            invalid_page_response = client.get(invalid_json_response.headers["location"])

        assert initial_response.status_code == 200
        assert "생성 Template 관리" in initial_response.text
        assert "data-generation-template-management-summary" in initial_response.text
        assert "data-generation-template-management-form" in initial_response.text
        assert "data-generation-template-management-table" in initial_response.text
        assert "현재 적용 Template" in initial_response.text
        assert new_response.status_code == 200
        assert "새 Template 작성" in new_response.text
        assert save_response.status_code == 303
        assert f"saved_template={template_key}" in save_response.headers["location"]
        assert saved_response.status_code == 200
        assert template_key in saved_response.text
        assert "Pytest UI 보고서" in saved_response.text
        assert "data-generation-template-clone" in saved_response.text
        assert "pytest_ui_family" in saved_response.text
        assert clone_response.status_code == 303
        assert f"cloned_template={clone_key}" in clone_response.headers["location"]
        assert cloned_response.status_code == 200
        assert "템플릿 버전을 복제했습니다" in cloned_response.text
        assert clone_key in cloned_response.text
        assert rollback_response.status_code == 303
        assert f"rolled_back_template={clone_key}" in rollback_response.headers["location"]
        assert rolled_back_response.status_code == 200
        assert "현재 적용" in rolled_back_response.text
        assert deactivate_response.status_code == 303
        assert inactive_response.status_code == 200
        assert "비활성" in inactive_response.text
        assert invalid_json_response.status_code == 303
        assert invalid_page_response.status_code == 200
        assert "section_schema must be valid JSON" in invalid_page_response.text
    finally:
        _cleanup_generation_templates(migrated_database_url, template_key, clone_key)


def test_generation_template_management_ui_reports_missing_database() -> None:
    app = create_app(Settings(database_url=None))

    with TestClient(app) as client:
        page_response = client.get("/admin/generation-templates")
        post_response = client.post(
            "/admin/generation-templates/upsert",
            data={
                "template_key": "pytest_no_database",
                "template_name": "No DB",
                "document_type": "report",
                "section_schema": "[]",
                "system_instruction": "No DB",
            },
            follow_redirects=False,
        )
        active_response = client.post(
            "/admin/generation-templates/report/active",
            follow_redirects=False,
        )
        default_response = client.post(
            "/admin/generation-templates/report/default",
            follow_redirects=False,
        )
        clone_response = client.post(
            "/admin/generation-templates/report/clone",
            data={
                "target_template_key": "report_v2",
                "target_template_version": "v2",
            },
            follow_redirects=False,
        )
        rollback_response = client.post(
            "/admin/generation-templates/report/rollback",
            follow_redirects=False,
        )

    assert page_response.status_code == 200
    assert "NEX_PCX_DATABASE_URL is not configured." in page_response.text
    assert post_response.status_code == 303
    assert "template_error=NEX_PCX_DATABASE_URL" in post_response.headers["location"]
    assert active_response.status_code == 303
    assert "template_error=NEX_PCX_DATABASE_URL" in active_response.headers["location"]
    assert default_response.status_code == 303
    assert "template_error=NEX_PCX_DATABASE_URL" in default_response.headers["location"]
    assert clone_response.status_code == 303
    assert "template_error=NEX_PCX_DATABASE_URL" in clone_response.headers["location"]
    assert rollback_response.status_code == 303
    assert "template_error=NEX_PCX_DATABASE_URL" in rollback_response.headers["location"]
