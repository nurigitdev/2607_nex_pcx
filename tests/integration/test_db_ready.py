import pytest

pytestmark = pytest.mark.integration


def test_database_connection_is_ready(db_connection) -> None:
    with db_connection.cursor() as cursor:
        cursor.execute("SELECT current_database() AS database_name")
        row = cursor.fetchone()

    assert row["database_name"]
