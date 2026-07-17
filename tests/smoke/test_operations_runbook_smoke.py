def test_operations_runbook_page_loads_document_source(client) -> None:
    response = client.get("/admin/operations-runbook")

    assert response.status_code == 200
    assert 'lang="ko"' in response.text
    assert "운영 Runbook" in response.text
    assert "data-operations-runbook-page" in response.text
    assert "docs/operations_runbook.md" in response.text
    assert "NeX_PCX Operations Runbook" in response.text
    assert "Startup Checklist" in response.text
    assert "scripts/validate_operations_startup.py" in response.text
    assert "Shutdown Checklist" in response.text
    assert 'href="/admin/operations-runbook"' in response.text


def test_operations_runbook_page_supports_english(client) -> None:
    response = client.get("/admin/operations-runbook?lang=en")

    assert response.status_code == 200
    assert response.cookies.get("nex_pcx_lang") == "en"
    assert 'lang="en"' in response.text
    assert "Operations Runbook" in response.text
    assert "Startup, shutdown, daily checks" in response.text
