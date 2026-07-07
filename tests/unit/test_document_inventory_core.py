import pytest

from app.core.document_inventory import (
    InvalidDocumentInventoryError,
    list_document_inventory,
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
