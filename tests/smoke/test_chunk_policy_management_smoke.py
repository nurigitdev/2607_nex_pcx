def test_chunk_policy_page_renders_empty_state(client) -> None:
    response = client.get("/admin/chunk-policies")

    assert response.status_code == 200
    assert "Chunk Policies" in response.text
    assert "Chunk size candidates, overlap settings, and observed usage" in response.text
    assert "No database configured" in response.text
    assert 'href="/admin/chunk-policies"' in response.text


def test_chunk_policy_api_requires_database(client) -> None:
    response = client.get("/api/chunk-policies")

    assert response.status_code == 503


def test_chunk_policy_detail_api_requires_database(client) -> None:
    response = client.get("/api/chunk-policies/heading_512_64")

    assert response.status_code == 503
