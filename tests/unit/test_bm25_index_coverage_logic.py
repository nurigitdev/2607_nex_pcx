from app.core.bm25_index_coverage import _status_for_bm25_index_cell


def test_bm25_index_coverage_status_variants() -> None:
    assert (
        _status_for_bm25_index_cell(
            policy_chunk_count=0,
            chunk_count=0,
            indexed_chunk_count=0,
            statistics_term_count=0,
            statistics_corpus_chunk_count=0,
        )
        == "not_chunked"
    )
    assert (
        _status_for_bm25_index_cell(
            policy_chunk_count=2,
            chunk_count=2,
            indexed_chunk_count=2,
            statistics_term_count=4,
            statistics_corpus_chunk_count=2,
        )
        == "complete"
    )
    assert (
        _status_for_bm25_index_cell(
            policy_chunk_count=3,
            chunk_count=3,
            indexed_chunk_count=2,
            statistics_term_count=4,
            statistics_corpus_chunk_count=3,
        )
        == "partial"
    )
    assert (
        _status_for_bm25_index_cell(
            policy_chunk_count=3,
            chunk_count=3,
            indexed_chunk_count=2,
            statistics_term_count=4,
            statistics_corpus_chunk_count=2,
        )
        == "stale"
    )
    assert (
        _status_for_bm25_index_cell(
            policy_chunk_count=2,
            chunk_count=2,
            indexed_chunk_count=0,
            statistics_term_count=0,
            statistics_corpus_chunk_count=0,
        )
        == "missing"
    )
