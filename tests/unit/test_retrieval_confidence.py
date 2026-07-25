from types import SimpleNamespace

from app.core.retrieval_confidence import (
    RETRIEVAL_CONFIDENCE_ANSWERABLE,
    RETRIEVAL_CONFIDENCE_FAILED,
    RETRIEVAL_CONFIDENCE_LOW,
    RETRIEVAL_CONFIDENCE_NO_CONTEXT,
    assess_profile_retrieval_confidence,
    assess_retrieval_confidence,
    retrieval_confidence_assessment_payload,
)


def _result(
    *,
    score: float | None,
    profile_name: str = "qwen3_4b_2560",
    search_profile_name: str | None = None,
    retrieval_strategy: str | None = "vector_cosine",
    score_components: dict[str, object] | None = None,
):
    return SimpleNamespace(
        profile_name=profile_name,
        search_profile_name=search_profile_name,
        retrieval_strategy=retrieval_strategy,
        rank=1,
        chunk_id=10,
        score=score,
        score_components=score_components or {},
    )


def test_reranked_profile_is_low_confidence_when_source_and_reranker_are_weak() -> None:
    assessment = assess_profile_retrieval_confidence(
        "reranked_vector_cosine",
        (
            _result(
                score=-3.43,
                profile_name="reranked_vector_cosine",
                search_profile_name="reranked_vector_cosine",
                retrieval_strategy="reranked",
                score_components={
                    "source_score": 0.39,
                    "raw_cross_encoder_score": -3.43,
                },
            ),
        ),
    )

    assert assessment.status == RETRIEVAL_CONFIDENCE_LOW
    assert assessment.answerable is False
    assert assessment.top_source_score == 0.39
    assert assessment.top_reranker_score == -3.43
    assert set(assessment.reason_codes) == {
        "weak_source_vector_score",
        "weak_reranker_score",
    }


def test_reranked_profile_is_answerable_when_one_signal_is_strong() -> None:
    assessment = assess_profile_retrieval_confidence(
        "reranked_vector_cosine",
        (
            _result(
                score=-0.5,
                profile_name="reranked_vector_cosine",
                search_profile_name="reranked_vector_cosine",
                retrieval_strategy="reranked",
                score_components={
                    "source_score": 0.72,
                    "raw_cross_encoder_score": -0.5,
                },
            ),
        ),
    )

    assert assessment.status == RETRIEVAL_CONFIDENCE_ANSWERABLE
    assert assessment.answerable is True
    assert assessment.reason_codes == ()


def test_vector_bm25_and_hybrid_profiles_use_strategy_specific_score_signals() -> None:
    vector = assess_profile_retrieval_confidence(
        "qwen3_4b_2560",
        (_result(score=0.49),),
    )
    bm25 = assess_profile_retrieval_confidence(
        "bm25_keyword",
        (
            _result(
                score=0.02,
                profile_name="bm25_keyword",
                search_profile_name="bm25_keyword",
                retrieval_strategy="bm25_keyword",
            ),
        ),
    )
    hybrid = assess_profile_retrieval_confidence(
        "hybrid_keyword_vector",
        (
            _result(
                score=0.03,
                profile_name="hybrid_keyword_vector",
                search_profile_name="hybrid_keyword_vector",
                retrieval_strategy="hybrid_keyword_vector",
                score_components={
                    "source_count": 2,
                    "vector_score": 0.2,
                    "keyword_score": 0.0,
                },
            ),
        ),
    )

    assert vector.status == RETRIEVAL_CONFIDENCE_LOW
    assert vector.reason_codes == ("weak_vector_score",)
    assert bm25.status == RETRIEVAL_CONFIDENCE_ANSWERABLE
    assert hybrid.status == RETRIEVAL_CONFIDENCE_ANSWERABLE


def test_overall_assessment_withholds_generation_context_without_answerable_profiles() -> None:
    no_results = assess_profile_retrieval_confidence("qwen3_4b_2560", ())
    failed = assess_profile_retrieval_confidence(
        "reranked_vector_cosine",
        (),
        profile_status="failed",
    )
    assessment = assess_retrieval_confidence((no_results, failed))
    payload = retrieval_confidence_assessment_payload(assessment)

    assert no_results.status == RETRIEVAL_CONFIDENCE_NO_CONTEXT
    assert failed.status == RETRIEVAL_CONFIDENCE_FAILED
    assert assessment.status == RETRIEVAL_CONFIDENCE_NO_CONTEXT
    assert assessment.withhold_generation_context is True
    assert payload["withhold_generation_context"] is True
    assert payload["no_context_profile_count"] == 1
    assert payload["failed_profile_count"] == 1
