"""Permission simulation inventory read-model helpers."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from app.core.database import connect
from app.core.permissions import InvalidPermissionError, get_actor_permission_context


@dataclass(frozen=True)
class PermissionAccessScopeCount:
    access_scope: str
    document_count: int


@dataclass(frozen=True)
class PermissionInventorySummary:
    active_user_count: int
    inactive_user_count: int
    org_unit_count: int
    active_org_unit_count: int
    membership_count: int
    document_count: int
    access_scope_counts: tuple[PermissionAccessScopeCount, ...]


@dataclass(frozen=True)
class PermissionReadinessIssueRecord:
    document_id: int
    file_id: int
    document_title: str | None
    original_file_name: str
    access_scope: str
    issue_codes: tuple[str, ...]


@dataclass(frozen=True)
class PermissionReadinessSummary:
    document_count: int
    ready_document_count: int
    issue_document_count: int
    missing_uploader_count: int
    personal_missing_owner_count: int
    scoped_missing_org_count: int
    readiness_percent: float | None
    issues: tuple[PermissionReadinessIssueRecord, ...]


@dataclass(frozen=True)
class PermissionUserInventoryRecord:
    user_id: int
    login_id: str
    display_name: str
    email: str | None
    is_active: bool
    primary_role_name: str | None
    primary_org_unit_id: int | None
    primary_org_unit_name: str | None
    primary_org_unit_type: str | None
    membership_count: int
    uploaded_file_count: int
    owned_document_count: int
    managed_org_unit_count: int
    ancestor_org_unit_count: int
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class PermissionOrgUnitInventoryRecord:
    org_unit_id: int
    parent_org_unit_id: int | None
    parent_org_unit_name: str | None
    org_unit_name: str
    org_unit_type: str
    is_active: bool
    depth: int
    org_path: str
    membership_count: int
    primary_membership_count: int
    owned_document_count: int
    child_org_unit_count: int
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class PermissionMembershipInventoryRecord:
    membership_id: int
    user_id: int
    login_id: str
    display_name: str
    org_unit_id: int
    org_unit_name: str
    org_unit_type: str
    role_name: str
    is_primary: bool
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class PermissionInventory:
    summary: PermissionInventorySummary
    users: tuple[PermissionUserInventoryRecord, ...]
    org_units: tuple[PermissionOrgUnitInventoryRecord, ...]
    memberships: tuple[PermissionMembershipInventoryRecord, ...]


class InvalidPermissionInventoryError(ValueError):
    """Raised when permission inventory query input is invalid."""


def _validate_limit(limit: int, *, max_limit: int = 500) -> int:
    if limit <= 0:
        raise InvalidPermissionInventoryError("limit must be greater than 0")
    if limit > max_limit:
        raise InvalidPermissionInventoryError(f"limit must be less than or equal to {max_limit}")
    return limit


def _row_to_access_scope_count(row: dict[str, Any]) -> PermissionAccessScopeCount:
    return PermissionAccessScopeCount(
        access_scope=str(row["access_scope"]),
        document_count=int(row["document_count"]),
    )


def _readiness_issue_codes(row: dict[str, Any]) -> tuple[str, ...]:
    issue_codes: list[str] = []
    if row["uploaded_by_user_id"] is None:
        issue_codes.append("missing_uploader")
    if row["access_scope"] == "personal" and row["owner_user_id"] is None:
        issue_codes.append("personal_missing_owner")
    if row["access_scope"] in {"team", "org_tree"} and row["owner_org_unit_id"] is None:
        issue_codes.append("scoped_missing_org")
    return tuple(issue_codes)


def _row_to_readiness_issue(row: dict[str, Any]) -> PermissionReadinessIssueRecord:
    return PermissionReadinessIssueRecord(
        document_id=int(row["document_id"]),
        file_id=int(row["file_id"]),
        document_title=row["document_title"],
        original_file_name=str(row["original_file_name"]),
        access_scope=str(row["access_scope"]),
        issue_codes=_readiness_issue_codes(row),
    )


def _row_to_permission_user(
    database_url: str,
    row: dict[str, Any],
) -> PermissionUserInventoryRecord:
    ancestor_org_unit_count = 0
    managed_org_unit_count = 0
    if row["is_active"] and row["primary_org_unit_id"] is not None:
        try:
            context = get_actor_permission_context(database_url, int(row["user_id"]))
        except InvalidPermissionError:
            pass
        else:
            ancestor_org_unit_count = len(context.ancestor_org_unit_ids)
            managed_org_unit_count = len(context.managed_org_unit_ids)

    return PermissionUserInventoryRecord(
        user_id=int(row["user_id"]),
        login_id=str(row["login_id"]),
        display_name=str(row["display_name"]),
        email=row["email"],
        is_active=bool(row["is_active"]),
        primary_role_name=row["primary_role_name"],
        primary_org_unit_id=(
            int(row["primary_org_unit_id"]) if row.get("primary_org_unit_id") is not None else None
        ),
        primary_org_unit_name=row["primary_org_unit_name"],
        primary_org_unit_type=row["primary_org_unit_type"],
        membership_count=int(row["membership_count"]),
        uploaded_file_count=int(row["uploaded_file_count"]),
        owned_document_count=int(row["owned_document_count"]),
        managed_org_unit_count=managed_org_unit_count,
        ancestor_org_unit_count=ancestor_org_unit_count,
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _row_to_permission_org_unit(row: dict[str, Any]) -> PermissionOrgUnitInventoryRecord:
    return PermissionOrgUnitInventoryRecord(
        org_unit_id=int(row["org_unit_id"]),
        parent_org_unit_id=(
            int(row["parent_org_unit_id"]) if row.get("parent_org_unit_id") is not None else None
        ),
        parent_org_unit_name=row["parent_org_unit_name"],
        org_unit_name=str(row["org_unit_name"]),
        org_unit_type=str(row["org_unit_type"]),
        is_active=bool(row["is_active"]),
        depth=int(row["depth"]),
        org_path=str(row["org_path"]),
        membership_count=int(row["membership_count"]),
        primary_membership_count=int(row["primary_membership_count"]),
        owned_document_count=int(row["owned_document_count"]),
        child_org_unit_count=int(row["child_org_unit_count"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _row_to_permission_membership(row: dict[str, Any]) -> PermissionMembershipInventoryRecord:
    return PermissionMembershipInventoryRecord(
        membership_id=int(row["membership_id"]),
        user_id=int(row["user_id"]),
        login_id=str(row["login_id"]),
        display_name=str(row["display_name"]),
        org_unit_id=int(row["org_unit_id"]),
        org_unit_name=str(row["org_unit_name"]),
        org_unit_type=str(row["org_unit_type"]),
        role_name=str(row["role_name"]),
        is_primary=bool(row["is_primary"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def get_permission_inventory_summary(database_url: str) -> PermissionInventorySummary:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT
                    count(*) FILTER (WHERE is_active) AS active_user_count,
                    count(*) FILTER (WHERE NOT is_active) AS inactive_user_count
                FROM app_users
                """)
            user_counts = cursor.fetchone()
            cursor.execute("""
                SELECT
                    count(*) AS org_unit_count,
                    count(*) FILTER (WHERE is_active) AS active_org_unit_count
                FROM org_units
                """)
            org_counts = cursor.fetchone()
            cursor.execute("SELECT count(*) AS membership_count FROM user_org_memberships")
            membership_counts = cursor.fetchone()
            cursor.execute("SELECT count(*) AS document_count FROM documents")
            document_counts = cursor.fetchone()
            cursor.execute("""
                SELECT access_scope, count(*) AS document_count
                FROM documents
                GROUP BY access_scope
                ORDER BY access_scope ASC
                """)
            access_scope_rows = cursor.fetchall()

    return PermissionInventorySummary(
        active_user_count=int(user_counts["active_user_count"]),
        inactive_user_count=int(user_counts["inactive_user_count"]),
        org_unit_count=int(org_counts["org_unit_count"]),
        active_org_unit_count=int(org_counts["active_org_unit_count"]),
        membership_count=int(membership_counts["membership_count"]),
        document_count=int(document_counts["document_count"]),
        access_scope_counts=tuple(
            _row_to_access_scope_count(dict(row)) for row in access_scope_rows
        ),
    )


def get_permission_readiness_summary(
    database_url: str,
    *,
    issue_limit: int = 20,
) -> PermissionReadinessSummary:
    validated_limit = _validate_limit(issue_limit, max_limit=100)
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute("""
                WITH document_readiness AS (
                    SELECT
                        d.document_id,
                        d.file_id,
                        d.document_title,
                        d.access_scope,
                        d.owner_user_id,
                        d.owner_org_unit_id,
                        f.original_file_name,
                        f.uploaded_by_user_id,
                        (
                            f.uploaded_by_user_id IS NOT NULL
                            AND (
                                d.access_scope != 'personal'
                                OR d.owner_user_id IS NOT NULL
                            )
                            AND (
                                d.access_scope NOT IN ('team', 'org_tree')
                                OR d.owner_org_unit_id IS NOT NULL
                            )
                        ) AS is_ready
                    FROM documents d
                    JOIN files f ON f.file_id = d.file_id
                )
                SELECT
                    count(*) AS document_count,
                    count(*) FILTER (WHERE is_ready) AS ready_document_count,
                    count(*) FILTER (WHERE NOT is_ready) AS issue_document_count,
                    count(*) FILTER (
                        WHERE uploaded_by_user_id IS NULL
                    ) AS missing_uploader_count,
                    count(*) FILTER (
                        WHERE access_scope = 'personal'
                          AND owner_user_id IS NULL
                    ) AS personal_missing_owner_count,
                    count(*) FILTER (
                        WHERE access_scope IN ('team', 'org_tree')
                          AND owner_org_unit_id IS NULL
                    ) AS scoped_missing_org_count
                FROM document_readiness
                """)
            summary_row = cursor.fetchone()
            cursor.execute(
                """
                SELECT
                    d.document_id,
                    d.file_id,
                    d.document_title,
                    d.access_scope,
                    d.owner_user_id,
                    d.owner_org_unit_id,
                    f.original_file_name,
                    f.uploaded_by_user_id
                FROM documents d
                JOIN files f ON f.file_id = d.file_id
                WHERE f.uploaded_by_user_id IS NULL
                   OR (d.access_scope = 'personal' AND d.owner_user_id IS NULL)
                   OR (
                       d.access_scope IN ('team', 'org_tree')
                       AND d.owner_org_unit_id IS NULL
                   )
                ORDER BY d.updated_at DESC, d.document_id DESC
                LIMIT %s
                """,
                (validated_limit,),
            )
            issue_rows = cursor.fetchall()

    document_count = int(summary_row["document_count"])
    ready_document_count = int(summary_row["ready_document_count"])
    readiness_percent = round(ready_document_count / document_count, 4) if document_count else None
    return PermissionReadinessSummary(
        document_count=document_count,
        ready_document_count=ready_document_count,
        issue_document_count=int(summary_row["issue_document_count"]),
        missing_uploader_count=int(summary_row["missing_uploader_count"]),
        personal_missing_owner_count=int(summary_row["personal_missing_owner_count"]),
        scoped_missing_org_count=int(summary_row["scoped_missing_org_count"]),
        readiness_percent=readiness_percent,
        issues=tuple(_row_to_readiness_issue(dict(row)) for row in issue_rows),
    )


def list_permission_users(
    database_url: str,
    *,
    limit: int = 100,
) -> list[PermissionUserInventoryRecord]:
    validated_limit = _validate_limit(limit)
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                WITH primary_membership AS (
                    SELECT DISTINCT ON (m.user_id)
                        m.user_id,
                        m.role_name AS primary_role_name,
                        o.org_unit_id AS primary_org_unit_id,
                        o.org_unit_name AS primary_org_unit_name,
                        o.org_unit_type AS primary_org_unit_type
                    FROM user_org_memberships m
                    JOIN org_units o ON o.org_unit_id = m.org_unit_id
                    ORDER BY m.user_id, m.is_primary DESC, m.membership_id ASC
                ),
                membership_counts AS (
                    SELECT user_id, count(*) AS membership_count
                    FROM user_org_memberships
                    GROUP BY user_id
                ),
                uploaded_counts AS (
                    SELECT uploaded_by_user_id AS user_id, count(*) AS uploaded_file_count
                    FROM files
                    WHERE uploaded_by_user_id IS NOT NULL
                    GROUP BY uploaded_by_user_id
                ),
                owned_counts AS (
                    SELECT owner_user_id AS user_id, count(*) AS owned_document_count
                    FROM documents
                    WHERE owner_user_id IS NOT NULL
                    GROUP BY owner_user_id
                )
                SELECT
                    u.user_id,
                    u.login_id,
                    u.display_name,
                    u.email,
                    u.is_active,
                    pm.primary_role_name,
                    pm.primary_org_unit_id,
                    pm.primary_org_unit_name,
                    pm.primary_org_unit_type,
                    COALESCE(mc.membership_count, 0) AS membership_count,
                    COALESCE(uc.uploaded_file_count, 0) AS uploaded_file_count,
                    COALESCE(oc.owned_document_count, 0) AS owned_document_count,
                    u.created_at,
                    u.updated_at
                FROM app_users u
                LEFT JOIN primary_membership pm ON pm.user_id = u.user_id
                LEFT JOIN membership_counts mc ON mc.user_id = u.user_id
                LEFT JOIN uploaded_counts uc ON uc.user_id = u.user_id
                LEFT JOIN owned_counts oc ON oc.user_id = u.user_id
                ORDER BY u.is_active DESC, u.login_id ASC
                LIMIT %s
                """,
                (validated_limit,),
            )
            rows = cursor.fetchall()
    return [_row_to_permission_user(database_url, dict(row)) for row in rows]


def list_permission_org_units(
    database_url: str,
    *,
    limit: int = 100,
) -> list[PermissionOrgUnitInventoryRecord]:
    validated_limit = _validate_limit(limit)
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                WITH RECURSIVE org_tree AS (
                    SELECT
                        o.org_unit_id,
                        o.parent_org_unit_id,
                        o.org_unit_name,
                        o.org_unit_type,
                        o.is_active,
                        o.created_at,
                        o.updated_at,
                        0 AS depth,
                        ARRAY[o.org_unit_name]::TEXT[] AS path_names
                    FROM org_units o
                    WHERE o.parent_org_unit_id IS NULL
                    UNION ALL
                    SELECT
                        child.org_unit_id,
                        child.parent_org_unit_id,
                        child.org_unit_name,
                        child.org_unit_type,
                        child.is_active,
                        child.created_at,
                        child.updated_at,
                        parent.depth + 1 AS depth,
                        parent.path_names || child.org_unit_name
                    FROM org_units child
                    JOIN org_tree parent
                        ON parent.org_unit_id = child.parent_org_unit_id
                ),
                membership_counts AS (
                    SELECT
                        org_unit_id,
                        count(*) AS membership_count,
                        count(*) FILTER (WHERE is_primary) AS primary_membership_count
                    FROM user_org_memberships
                    GROUP BY org_unit_id
                ),
                document_counts AS (
                    SELECT owner_org_unit_id AS org_unit_id, count(*) AS owned_document_count
                    FROM documents
                    WHERE owner_org_unit_id IS NOT NULL
                    GROUP BY owner_org_unit_id
                ),
                child_counts AS (
                    SELECT parent_org_unit_id AS org_unit_id, count(*) AS child_org_unit_count
                    FROM org_units
                    WHERE parent_org_unit_id IS NOT NULL
                    GROUP BY parent_org_unit_id
                )
                SELECT
                    ot.org_unit_id,
                    ot.parent_org_unit_id,
                    parent.org_unit_name AS parent_org_unit_name,
                    ot.org_unit_name,
                    ot.org_unit_type,
                    ot.is_active,
                    ot.depth,
                    array_to_string(ot.path_names, ' / ') AS org_path,
                    COALESCE(mc.membership_count, 0) AS membership_count,
                    COALESCE(mc.primary_membership_count, 0) AS primary_membership_count,
                    COALESCE(dc.owned_document_count, 0) AS owned_document_count,
                    COALESCE(cc.child_org_unit_count, 0) AS child_org_unit_count,
                    ot.created_at,
                    ot.updated_at
                FROM org_tree ot
                LEFT JOIN org_units parent ON parent.org_unit_id = ot.parent_org_unit_id
                LEFT JOIN membership_counts mc ON mc.org_unit_id = ot.org_unit_id
                LEFT JOIN document_counts dc ON dc.org_unit_id = ot.org_unit_id
                LEFT JOIN child_counts cc ON cc.org_unit_id = ot.org_unit_id
                ORDER BY ot.path_names ASC
                LIMIT %s
                """,
                (validated_limit,),
            )
            rows = cursor.fetchall()
    return [_row_to_permission_org_unit(dict(row)) for row in rows]


def list_permission_memberships(
    database_url: str,
    *,
    limit: int = 200,
) -> list[PermissionMembershipInventoryRecord]:
    validated_limit = _validate_limit(limit, max_limit=1000)
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    m.membership_id,
                    u.user_id,
                    u.login_id,
                    u.display_name,
                    o.org_unit_id,
                    o.org_unit_name,
                    o.org_unit_type,
                    m.role_name,
                    m.is_primary,
                    m.created_at,
                    m.updated_at
                FROM user_org_memberships m
                JOIN app_users u ON u.user_id = m.user_id
                JOIN org_units o ON o.org_unit_id = m.org_unit_id
                ORDER BY u.login_id ASC, m.is_primary DESC, o.org_unit_name ASC
                LIMIT %s
                """,
                (validated_limit,),
            )
            rows = cursor.fetchall()
    return [_row_to_permission_membership(dict(row)) for row in rows]


def get_permission_inventory(database_url: str) -> PermissionInventory:
    return PermissionInventory(
        summary=get_permission_inventory_summary(database_url),
        users=tuple(list_permission_users(database_url)),
        org_units=tuple(list_permission_org_units(database_url)),
        memberships=tuple(list_permission_memberships(database_url)),
    )
