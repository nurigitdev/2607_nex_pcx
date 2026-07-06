import math

import pytest

from app.core.embedding_vectors import (
    EmbeddingVectorInput,
    InvalidEmbeddingVectorError,
    generate_mock_embedding,
    get_embedding_vector_table,
    validate_embedding_vector_input,
    vector_to_pg_literal,
)


def test_get_embedding_vector_table_returns_profile_mapping() -> None:
    table = get_embedding_vector_table("qwen3_4b_2560")

    assert table.table_name == "chunk_embeddings_qwen3_4b_2560"
    assert table.dimension == 2560
    assert table.storage_type == "halfvec"


@pytest.mark.parametrize("profile_name", [" ", "unknown"])
def test_get_embedding_vector_table_rejects_unknown_profile(profile_name: str) -> None:
    with pytest.raises(InvalidEmbeddingVectorError, match="profile|Unsupported"):
        get_embedding_vector_table(profile_name)


def test_vector_to_pg_literal_formats_finite_values() -> None:
    assert vector_to_pg_literal((0.0, 1.25, -0.5)) == "[0,1.25,-0.5]"


def test_vector_to_pg_literal_rejects_non_finite_values() -> None:
    with pytest.raises(InvalidEmbeddingVectorError, match="finite"):
        vector_to_pg_literal((0.0, float("nan")))


def test_validate_embedding_vector_input_rejects_bad_shape_and_elapsed() -> None:
    with pytest.raises(InvalidEmbeddingVectorError, match="chunk_id"):
        validate_embedding_vector_input(
            EmbeddingVectorInput(
                chunk_id=0,
                profile_name="kure_v1_1024",
                embedding=tuple(0.0 for _ in range(1024)),
            )
        )

    with pytest.raises(InvalidEmbeddingVectorError, match="elapsed_ms"):
        validate_embedding_vector_input(
            EmbeddingVectorInput(
                chunk_id=1,
                profile_name="kure_v1_1024",
                embedding=tuple(0.0 for _ in range(1024)),
                elapsed_ms=-1,
            )
        )

    with pytest.raises(InvalidEmbeddingVectorError, match="1024 dimensions"):
        validate_embedding_vector_input(
            EmbeddingVectorInput(
                chunk_id=1,
                profile_name="kure_v1_1024",
                embedding=(0.0, 1.0),
            )
        )


def test_generate_mock_embedding_is_deterministic_and_normalized() -> None:
    first = generate_mock_embedding("hello", profile_name="kure_v1_1024", dimension=8)
    second = generate_mock_embedding("hello", profile_name="kure_v1_1024", dimension=8)
    other = generate_mock_embedding("hello", profile_name="bge_m3_1024", dimension=8)
    magnitude = math.sqrt(sum(value * value for value in first))

    assert first == second
    assert first != other
    assert len(first) == 8
    assert magnitude == pytest.approx(1.0)


def test_generate_mock_embedding_rejects_non_positive_dimension() -> None:
    with pytest.raises(InvalidEmbeddingVectorError, match="dimension"):
        generate_mock_embedding("hello", profile_name="kure_v1_1024", dimension=0)
