from fastapi import status


def test_dgx_ingestion_benchmarks_page_without_database(client) -> None:
    response = client.get("/admin/dgx-ingestion-benchmarks")

    assert response.status_code == status.HTTP_200_OK
    assert "DGX Ingestion Benchmark 이력" in response.text
    assert "data-dgx-ingestion-benchmarks-page" in response.text
    assert "/api/admin/dgx-ingestion-benchmarks" in response.text
    assert "NEX_PCX_DATABASE_URL is not configured." in response.text


def test_dgx_ingestion_benchmark_apis_require_database(client) -> None:
    list_response = client.get("/api/admin/dgx-ingestion-benchmarks")
    detail_response = client.get("/api/admin/dgx-ingestion-benchmarks/1")

    assert list_response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert detail_response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert list_response.json()["detail"] == "NEX_PCX_DATABASE_URL is not configured."
