import pytest

from app.core.search_compare import (
    InvalidSearchCompareError,
    SearchPermissionMatrixEntryInput,
    SearchPermissionMatrixInput,
    run_permission_search_matrix,
)


@pytest.mark.parametrize(
    ("matrix_input", "message"),
    [
        (
            SearchPermissionMatrixInput(
                query_text="hello",
                entries=(),
            ),
            "entries must not be empty",
        ),
        (
            SearchPermissionMatrixInput(
                query_text="hello",
                entries=tuple(
                    SearchPermissionMatrixEntryInput(
                        actor_user_id=index + 1,
                        requested_search_scope="company",
                    )
                    for index in range(13)
                ),
            ),
            "entries must be 12 or fewer",
        ),
        (
            SearchPermissionMatrixInput(
                query_text="hello",
                entries=(
                    SearchPermissionMatrixEntryInput(
                        actor_user_id=1,
                        requested_search_scope="company",
                    ),
                    SearchPermissionMatrixEntryInput(
                        actor_user_id=1,
                        requested_search_scope="company",
                    ),
                ),
            ),
            "entries must be unique",
        ),
        (
            SearchPermissionMatrixInput(
                query_text="hello",
                entries=(
                    SearchPermissionMatrixEntryInput(
                        actor_user_id=0,
                        requested_search_scope="company",
                    ),
                ),
            ),
            "actor_user_id",
        ),
        (
            SearchPermissionMatrixInput(
                query_text=" ",
                entries=(
                    SearchPermissionMatrixEntryInput(
                        actor_user_id=1,
                        requested_search_scope="company",
                    ),
                ),
            ),
            "query_text",
        ),
    ],
)
def test_run_permission_search_matrix_rejects_invalid_input_before_db(
    matrix_input: SearchPermissionMatrixInput,
    message: str,
) -> None:
    with pytest.raises(InvalidSearchCompareError, match=message):
        run_permission_search_matrix("postgresql://unused", matrix_input)
