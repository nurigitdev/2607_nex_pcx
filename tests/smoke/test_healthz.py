def test_healthz_returns_ok(client) -> None:
    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "NeX_PCX",
        "version": "0.1.0",
        "environment": "local",
    }
