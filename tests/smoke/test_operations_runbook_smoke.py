def test_operations_runbook_page_loads_document_source(client) -> None:
    response = client.get("/admin/operations-runbook")

    assert response.status_code == 200
    assert 'lang="ko"' in response.text
    assert "운영 Runbook" in response.text
    assert "data-operations-runbook-page" in response.text
    assert "docs/operations_runbook.md" in response.text
    assert "NeX_PCX Operations Runbook" in response.text
    assert "docs/service_process_startup.md" in response.text
    assert "docs/backup_restore_smoke.md" in response.text
    assert "docs/go_live_smoke.md" in response.text
    assert "docs/operational_retention_cleanup.md" in response.text
    assert "docs/emergency_recovery_commands.md" in response.text
    assert "docs/operator_handoff_bundle.md" in response.text
    assert "docs/release_tag_version_snapshot.md" in response.text
    assert "Startup Checklist" in response.text
    assert "scripts/render_service_startup_templates.py" in response.text
    assert "scripts/audit_runtime_config.py" in response.text
    assert "scripts/verify_operational_retention.py" in response.text
    assert "scripts/run_backup_restore_smoke.py" in response.text
    assert "scripts/run_go_live_smoke.py" in response.text
    assert "scripts/render_emergency_recovery_index.py" in response.text
    assert "scripts/export_operator_handoff_bundle.py" in response.text
    assert "scripts/export_release_version_snapshot.py" in response.text
    assert "scripts/validate_operations_startup.py" in response.text
    assert "scripts/export_go_live_evidence.py" in response.text
    assert "scripts/check_shutdown_drain.py" in response.text
    assert "Shutdown Checklist" in response.text
    assert 'href="/admin/operations-runbook"' in response.text


def test_operations_runbook_page_supports_english(client) -> None:
    response = client.get("/admin/operations-runbook?lang=en")

    assert response.status_code == 200
    assert response.cookies.get("nex_pcx_lang") == "en"
    assert 'lang="en"' in response.text
    assert "Operations Runbook" in response.text
    assert "Startup, shutdown, daily checks" in response.text
