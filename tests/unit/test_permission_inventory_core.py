import pytest

from app.core.permission_inventory import (
    InvalidPermissionInventoryError,
    get_permission_readiness_summary,
    list_permission_memberships,
    list_permission_org_units,
    list_permission_users,
)


def test_permission_inventory_rejects_invalid_user_limit() -> None:
    with pytest.raises(InvalidPermissionInventoryError, match="greater than 0"):
        list_permission_users("postgresql://unused", limit=0)


def test_permission_inventory_rejects_invalid_org_unit_limit() -> None:
    with pytest.raises(InvalidPermissionInventoryError, match="less than or equal to 500"):
        list_permission_org_units("postgresql://unused", limit=501)


def test_permission_inventory_rejects_invalid_membership_limit() -> None:
    with pytest.raises(InvalidPermissionInventoryError, match="less than or equal to 1000"):
        list_permission_memberships("postgresql://unused", limit=1001)


def test_permission_readiness_rejects_invalid_issue_limit() -> None:
    with pytest.raises(InvalidPermissionInventoryError, match="less than or equal to 100"):
        get_permission_readiness_summary("postgresql://unused", issue_limit=101)
