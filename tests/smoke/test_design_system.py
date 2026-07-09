def test_design_system_page_renders_living_style_guide(client) -> None:
    response = client.get("/admin/design-system")

    assert response.status_code == 200
    assert 'lang="ko"' in response.text
    assert "디자인 시스템" in response.text
    assert "Design Principles" in response.text
    assert "Design Tokens" in response.text
    assert "Component Library" in response.text
    assert "Documentation Rules" in response.text
    assert "--pcx-accent" in response.text
    assert "Permission Summary Panel" in response.text
    assert 'href="/admin/design-system"' in response.text


def test_design_system_page_supports_english_language_switch(client) -> None:
    response = client.get("/admin/design-system?lang=en")

    assert response.status_code == 200
    assert response.cookies.get("nex_pcx_lang") == "en"
    assert 'lang="en"' in response.text
    assert "Design System" in response.text
    assert "Principles, tokens, and component rules" in response.text
    assert "Operational Clarity" in response.text
    assert "Reproducibility First" in response.text
