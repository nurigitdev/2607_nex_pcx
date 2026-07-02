import pytest

pytestmark = pytest.mark.integration


def test_halfvec_2560_type_is_available(pgvector_connection) -> None:
    with pgvector_connection.cursor() as cursor:
        cursor.execute("CREATE TEMP TABLE tmp_halfvec_check (embedding halfvec(2560))")
        cursor.execute(
            "INSERT INTO tmp_halfvec_check (embedding) "
            "SELECT ('[' || string_agg('0', ',') || ']')::halfvec "
            "FROM generate_series(1, 2560)"
        )
        cursor.execute("SELECT vector_dims(embedding::vector) AS dimensions FROM tmp_halfvec_check")
        row = cursor.fetchone()

    assert row["dimensions"] == 2560
