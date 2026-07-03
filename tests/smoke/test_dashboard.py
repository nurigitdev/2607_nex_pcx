def test_dashboard_renders_empty_state(client) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert "Dashboard" in response.text
    assert "RAG experiment bench skeleton" in response.text
    assert "Documents" in response.text
    assert "Upload" in response.text
    assert "Logs" in response.text
