def test_dashboard_renders_empty_state(client) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert "Dashboard" in response.text
    assert "RAG experiment bench operations" in response.text
    assert "Documents" in response.text
    assert "Golden Evaluation Snapshot" in response.text
    assert "Active Question Sets" in response.text
    assert "/api/dashboard/evaluations" in response.text
    assert "Upload" in response.text
    assert "Logs" in response.text


def test_dashboard_evaluation_api_requires_database(client) -> None:
    response = client.get("/api/dashboard/evaluations")

    assert response.status_code == 503
