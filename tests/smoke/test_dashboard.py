def test_dashboard_renders_empty_state(client) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert 'lang="ko"' in response.text
    assert "대시보드" in response.text
    assert "RAG 실험 벤치 운영 현황" in response.text
    assert "운영 상태" in response.text
    assert "정상" in response.text
    assert "/api/dashboard/operational-health" in response.text
    assert "문서" in response.text
    assert "Core Metrics" in response.text
    assert "마지막 갱신" in response.text
    assert "자동 갱신" in response.text
    assert 'data-dashboard-refresh' in response.text
    assert 'data-refresh-seconds="0"' in response.text
    assert "/?refresh_seconds=30" in response.text
    assert "처리량 / Latency 스냅샷" in response.text
    assert 'aria-label="조회 기간"' in response.text
    assert "/?lookback_hours=1" in response.text
    assert "/?lookback_hours=720" in response.text
    assert "최근 24시간" in response.text
    assert "Pipeline Queue 스냅샷" in response.text
    assert "최근 운영 실패" in response.text
    assert "임베딩 Queue 스냅샷" in response.text
    assert "골든 평가 스냅샷" in response.text
    assert "활성 질문 세트" in response.text
    assert "/api/dashboard/core-metrics" in response.text
    assert "/api/dashboard/throughput-latency" in response.text
    assert "/api/dashboard/pipeline-queue" in response.text
    assert "/api/dashboard/recent-failures" in response.text
    assert "data-failure-detail-panel" in response.text
    assert "/api/dashboard/embedding-backlog" in response.text
    assert "/api/dashboard/evaluations" in response.text
    assert "업로드" in response.text
    assert "로그" in response.text


def test_dashboard_supports_english_language_switch(client) -> None:
    response = client.get("/?lang=en")

    assert response.status_code == 200
    assert response.cookies.get("nex_pcx_lang") == "en"
    assert 'lang="en"' in response.text
    assert "Dashboard" in response.text
    assert "RAG experiment bench operations" in response.text
    assert "Operational Health" in response.text
    assert "Healthy" in response.text
    assert "Last Updated" in response.text
    assert 'aria-label="Auto Refresh"' in response.text
    assert "Core Metrics" in response.text
    assert "Throughput / Latency Snapshot" in response.text
    assert 'aria-label="Time Window"' in response.text
    assert "/?lang=en&amp;lookback_hours=6" in response.text
    assert "Pipeline Queue Snapshot" in response.text
    assert "Recent Operational Failures" in response.text
    assert "Embedding Queue Snapshot" in response.text
    assert "Upload" in response.text
    assert "Logs" in response.text


def test_dashboard_uses_language_cookie(client) -> None:
    client.cookies.set("nex_pcx_lang", "en")

    response = client.get("/")

    assert response.status_code == 200
    assert 'lang="en"' in response.text
    assert "Dashboard" in response.text


def test_dashboard_invalid_time_window_falls_back_to_default(client) -> None:
    response = client.get("/?lookback_hours=0")

    assert response.status_code == 200
    assert "lookback_hours must be greater than 0" in response.text
    assert "최근 24시간" in response.text


def test_dashboard_invalid_refresh_interval_falls_back_to_off(client) -> None:
    response = client.get("/?refresh_seconds=15")

    assert response.status_code == 200
    assert "refresh_seconds must be one of 0, 30, or 60" in response.text
    assert 'data-refresh-seconds="0"' in response.text


def test_dashboard_evaluation_api_requires_database(client) -> None:
    response = client.get("/api/dashboard/evaluations")

    assert response.status_code == 503


def test_dashboard_core_metrics_api_requires_database(client) -> None:
    response = client.get("/api/dashboard/core-metrics")

    assert response.status_code == 503


def test_dashboard_operational_health_api_requires_database(client) -> None:
    response = client.get("/api/dashboard/operational-health")

    assert response.status_code == 503


def test_dashboard_pipeline_queue_api_requires_database(client) -> None:
    response = client.get("/api/dashboard/pipeline-queue")

    assert response.status_code == 503


def test_dashboard_throughput_latency_api_requires_database(client) -> None:
    response = client.get("/api/dashboard/throughput-latency")

    assert response.status_code == 503


def test_dashboard_embedding_backlog_api_requires_database(client) -> None:
    response = client.get("/api/dashboard/embedding-backlog")

    assert response.status_code == 503


def test_dashboard_recent_failures_api_requires_database(client) -> None:
    response = client.get("/api/dashboard/recent-failures")

    assert response.status_code == 503


def test_dashboard_failure_detail_api_requires_database(client) -> None:
    response = client.get("/api/dashboard/recent-failures/pipeline/1")

    assert response.status_code == 503
