def test_document_inventory_page_renders_empty_state(client) -> None:
    response = client.get("/documents")

    assert response.status_code == 200
    assert "Documents" in response.text
    assert "Uploaded files, parse status, chunks, and pipeline activity" in response.text
    assert "No database configured" in response.text
    assert 'href="/documents"' in response.text


def test_document_inventory_api_requires_database(client) -> None:
    response = client.get("/api/documents")

    assert response.status_code == 503


def test_document_detail_page_renders_empty_state(client) -> None:
    response = client.get("/documents/1")

    assert response.status_code == 200
    assert "Document Detail" in response.text
    assert "NEX_PCX_DATABASE_URL is not configured." in response.text


def test_document_detail_api_requires_database(client) -> None:
    response = client.get("/api/documents/1")

    assert response.status_code == 503
