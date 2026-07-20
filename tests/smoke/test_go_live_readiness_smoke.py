def test_go_live_readiness_page_renders_without_database(client) -> None:
    response = client.get("/admin/go-live-readiness")

    assert response.status_code == 200
    assert 'lang="ko"' in response.text
    assert "운영 준비도" in response.text
    assert "운영 차단" in response.text
    assert "Database URL 설정" in response.text
    assert "NEX_PCX_DATABASE_URL is not configured." in response.text
    assert "/api/admin/go-live-readiness" in response.text
    assert 'href="/admin/go-live-readiness"' in response.text


def test_go_live_readiness_page_supports_english(client) -> None:
    response = client.get("/admin/go-live-readiness?lang=en")

    assert response.status_code == 200
    assert response.cookies.get("nex_pcx_lang") == "en"
    assert 'lang="en"' in response.text
    assert "Go-Live Readiness" in response.text
    assert "Blocked" in response.text
    assert "Database URL" in response.text


def test_go_live_readiness_api_reports_missing_database(client) -> None:
    response = client.get("/api/admin/go-live-readiness")

    assert response.status_code == 200
    payload = response.json()["go_live_readiness"]
    assert payload["status"] == "blocked"
    checks = {
        check["code"]: check for section in payload["sections"] for check in section["checks"]
    }
    assert checks["database_configured"]["status"] == "failed"
    assert checks["database_connectivity"]["status"] == "skipped"
    assert checks["database_backed_checks"]["status"] == "skipped"
