import pytest

from app.core.document_inventory import (
    DocumentPermissionUpdateInput,
    InvalidDocumentInventoryError,
    get_document_inventory_item,
    list_document_inventory,
    update_document_permission,
)


def test_document_inventory_validates_limit_before_db_connection() -> None:
    with pytest.raises(InvalidDocumentInventoryError, match="limit"):
        list_document_inventory("postgresql://unused", limit=0)

    with pytest.raises(InvalidDocumentInventoryError, match="less than or equal"):
        list_document_inventory("postgresql://unused", limit=501)


def test_document_inventory_validates_filters_before_db_connection() -> None:
    with pytest.raises(InvalidDocumentInventoryError, match="parse_status"):
        list_document_inventory("postgresql://unused", parse_status="done")

    with pytest.raises(InvalidDocumentInventoryError, match="document_group"):
        list_document_inventory("postgresql://unused", document_group=" ")


def test_document_inventory_validates_document_id_before_db_connection() -> None:
    with pytest.raises(InvalidDocumentInventoryError, match="document_id"):
        get_document_inventory_item("postgresql://unused", 0)


def test_document_permission_update_validates_inputs_before_db_connection() -> None:
    with pytest.raises(InvalidDocumentInventoryError, match="document_id"):
        update_document_permission(
            "postgresql://unused",
            0,
            DocumentPermissionUpdateInput(access_scope="personal"),
        )

    with pytest.raises(InvalidDocumentInventoryError, match="owner_user_id"):
        update_document_permission(
            "postgresql://unused",
            1,
            DocumentPermissionUpdateInput(owner_user_id=0),
        )

    with pytest.raises(InvalidDocumentInventoryError, match="owner_org_unit_id"):
        update_document_permission(
            "postgresql://unused",
            1,
            DocumentPermissionUpdateInput(owner_org_unit_id=-1),
        )

    with pytest.raises(InvalidDocumentInventoryError, match="Unsupported access_scope"):
        update_document_permission(
            "postgresql://unused",
            1,
            DocumentPermissionUpdateInput(access_scope="private"),
        )
