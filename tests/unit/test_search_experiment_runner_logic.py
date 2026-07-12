import pytest

from app.core.search_experiment_runner import (
    InvalidSearchExperimentExecutionError,
    _filter_results_by_threshold,
    _normalize_profiles,
)
from app.core.vector_search import VectorSearchResult


def _vector_result(rank: int, score: float) -> VectorSearchResult:
    return VectorSearchResult(
        profile_name="bge_m3_1024",
        rank=rank,
        chunk_id=rank,
        document_id=1,
        file_id=1,
        distance=1 - score,
        score=score,
        chunk_text="chunk",
        chunk_preview="chunk",
        content_hash=f"hash-{rank}",
        chunk_policy_name="heading_512_64",
        heading_path=(),
        page_no=None,
        slide_no=None,
        sheet_name=None,
        cell_range=None,
        document_title="Doc",
        document_group="default",
        original_file_name="doc.md",
        file_ext=".md",
        embedding_elapsed_ms=1,
    )


def test_search_experiment_runner_normalizes_profiles() -> None:
    assert _normalize_profiles((" bge_m3_1024 ", "bge_m3_1024", "kure_v1_1024"), "") == (
        "bge_m3_1024",
        "kure_v1_1024",
    )
    with pytest.raises(InvalidSearchExperimentExecutionError, match="profile_name"):
        _normalize_profiles((" ",), "")


def test_search_experiment_runner_filters_results_by_threshold() -> None:
    results = (_vector_result(1, 0.91), _vector_result(2, 0.42))

    assert _filter_results_by_threshold(results, None) == results
    assert _filter_results_by_threshold(results, 0.9) == (results[0],)
    assert _filter_results_by_threshold(results, 0.99) == ()
