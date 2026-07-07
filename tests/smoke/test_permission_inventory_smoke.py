def test_permission_inventory_page_renders_empty_state(client) -> None:
    response = client.get("/admin/permissions")

    assert response.status_code == 200
    assert 'lang="ko"' in response.text
    assert "권한 시뮬레이션" in response.text
    assert "데이터베이스가 설정되지 않았습니다." in response.text
    assert 'href="/api/admin/permissions"' in response.text
    assert 'href="/admin/permissions"' in response.text


def test_permission_inventory_supports_english_language_switch(client) -> None:
    response = client.get("/admin/permissions?lang=en")

    assert response.status_code == 200
    assert 'lang="en"' in response.text
    assert "Permission Simulation" in response.text
    assert "No database configured" in response.text


def test_permission_inventory_apis_require_database(client) -> None:
    for path in (
        "/api/admin/permissions",
        "/api/admin/permissions/users",
        "/api/admin/permissions/org-units",
        "/api/admin/permissions/memberships",
    ):
        response = client.get(path)

        assert response.status_code == 503
