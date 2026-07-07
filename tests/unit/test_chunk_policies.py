import pytest

from app.core.chunk_policies import InvalidChunkPolicyManagementError, get_chunk_policy_summary


def test_get_chunk_policy_summary_rejects_blank_policy_name() -> None:
    with pytest.raises(InvalidChunkPolicyManagementError, match="chunk_policy_name"):
        get_chunk_policy_summary("postgresql://unused", " ")
