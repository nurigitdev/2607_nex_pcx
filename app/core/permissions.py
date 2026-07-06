"""Permission scope resolution for permission-aware search."""

from dataclasses import dataclass
from typing import Any

from app.core.database import connect

SEARCH_SCOPES = {"mine", "team", "managed_org", "company"}
MANAGER_ROLES = {"admin", "group_lead", "team_lead"}


@dataclass(frozen=True)
class ActorPermissionContext:
    user_id: int
    login_id: str
    display_name: str
    role_name: str
    primary_org_unit_id: int
    primary_org_unit_name: str
    ancestor_org_unit_ids: tuple[int, ...]
    managed_org_unit_ids: tuple[int, ...]


@dataclass(frozen=True)
class PermissionSearchFilter:
    actor_user_id: int
    requested_search_scope: str
    effective_search_scope: str
    where_sql: str
    params: tuple[object, ...]
    metadata: dict[str, Any]


class InvalidPermissionError(ValueError):
    """Raised when permission scope resolution fails before search."""


def _validate_user_id(user_id: int) -> None:
    if user_id <= 0:
        raise InvalidPermissionError("actor_user_id must be greater than 0")


def _validate_scope(requested_search_scope: str) -> str:
    scope = requested_search_scope.strip()
    if not scope:
        raise InvalidPermissionError("requested_search_scope is required")
    if scope not in SEARCH_SCOPES:
        raise InvalidPermissionError(f"Unsupported requested_search_scope: {scope}")
    return scope


def _fetch_org_tree_ids(database_url: str, org_unit_id: int, *, direction: str) -> tuple[int, ...]:
    if direction not in {"ancestors", "descendants"}:
        raise InvalidPermissionError(f"Unsupported org tree direction: {direction}")
    if direction == "ancestors":
        sql = """
            WITH RECURSIVE org_tree AS (
                SELECT org_unit_id, parent_org_unit_id
                FROM org_units
                WHERE org_unit_id = %s
                UNION ALL
                SELECT parent.org_unit_id, parent.parent_org_unit_id
                FROM org_units parent
                JOIN org_tree child ON child.parent_org_unit_id = parent.org_unit_id
            )
            SELECT org_unit_id
            FROM org_tree
            ORDER BY org_unit_id ASC
            """
    else:
        sql = """
            WITH RECURSIVE org_tree AS (
                SELECT org_unit_id, parent_org_unit_id
                FROM org_units
                WHERE org_unit_id = %s
                UNION ALL
                SELECT child.org_unit_id, child.parent_org_unit_id
                FROM org_units child
                JOIN org_tree parent ON child.parent_org_unit_id = parent.org_unit_id
            )
            SELECT org_unit_id
            FROM org_tree
            ORDER BY org_unit_id ASC
            """
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(sql, (org_unit_id,))
            return tuple(int(row["org_unit_id"]) for row in cursor.fetchall())


def get_actor_permission_context(
    database_url: str,
    actor_user_id: int,
) -> ActorPermissionContext:
    _validate_user_id(actor_user_id)
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    u.user_id,
                    u.login_id,
                    u.display_name,
                    m.role_name,
                    o.org_unit_id,
                    o.org_unit_name
                FROM app_users u
                JOIN user_org_memberships m ON m.user_id = u.user_id
                JOIN org_units o ON o.org_unit_id = m.org_unit_id
                WHERE u.user_id = %s
                  AND u.is_active
                ORDER BY m.is_primary DESC, m.membership_id ASC
                LIMIT 1
                """,
                (actor_user_id,),
            )
            row = cursor.fetchone()

    if row is None:
        raise InvalidPermissionError(f"Active actor was not found: {actor_user_id}")

    primary_org_unit_id = int(row["org_unit_id"])
    role_name = str(row["role_name"])
    ancestor_org_unit_ids = _fetch_org_tree_ids(
        database_url,
        primary_org_unit_id,
        direction="ancestors",
    )
    if role_name == "admin":
        managed_root = ancestor_org_unit_ids[0]
        managed_org_unit_ids = _fetch_org_tree_ids(
            database_url,
            managed_root,
            direction="descendants",
        )
    elif role_name in MANAGER_ROLES:
        managed_org_unit_ids = _fetch_org_tree_ids(
            database_url,
            primary_org_unit_id,
            direction="descendants",
        )
    else:
        managed_org_unit_ids = ()

    return ActorPermissionContext(
        user_id=int(row["user_id"]),
        login_id=str(row["login_id"]),
        display_name=str(row["display_name"]),
        role_name=role_name,
        primary_org_unit_id=primary_org_unit_id,
        primary_org_unit_name=str(row["org_unit_name"]),
        ancestor_org_unit_ids=ancestor_org_unit_ids,
        managed_org_unit_ids=managed_org_unit_ids,
    )


def _in_clause(column_name: str, values: tuple[int, ...]) -> tuple[str, list[object]]:
    if not values:
        return "FALSE", []
    placeholders = ", ".join(["%s"] * len(values))
    return f"{column_name} IN ({placeholders})", list(values)


def _owner_in_org_clause(values: tuple[int, ...]) -> tuple[str, list[object]]:
    if not values:
        return "FALSE", []
    placeholders = ", ".join(["%s"] * len(values))
    return (
        """
        EXISTS (
            SELECT 1
            FROM user_org_memberships permission_uom
            WHERE permission_uom.user_id = d.owner_user_id
              AND permission_uom.org_unit_id IN ("""
        + placeholders
        + """)
        )
        """,
        list(values),
    )


def _append_clause(
    clauses: list[str],
    params: list[object],
    clause: str,
    clause_params: list[object],
) -> None:
    if clause == "FALSE":
        return
    clauses.append(clause)
    params.extend(clause_params)


def _effective_scope(context: ActorPermissionContext, requested_scope: str) -> str:
    if requested_scope == "managed_org" and not context.managed_org_unit_ids:
        return "team"
    return requested_scope


def resolve_permission_search_filter(
    database_url: str,
    *,
    actor_user_id: int,
    requested_search_scope: str,
) -> PermissionSearchFilter:
    requested_scope = _validate_scope(requested_search_scope)
    context = get_actor_permission_context(database_url, actor_user_id)
    effective_scope = _effective_scope(context, requested_scope)

    clauses: list[str] = []
    params: list[object] = []

    if context.role_name == "admin" and effective_scope == "company":
        clauses.append("TRUE")
    else:
        clauses.append("(d.access_scope = 'company')")
        clauses.append("(d.access_scope = 'personal' AND d.owner_user_id = %s)")
        params.append(context.user_id)

        if effective_scope in {"team", "managed_org", "company"}:
            clauses.append("(d.access_scope = 'team' AND d.owner_org_unit_id = %s)")
            params.append(context.primary_org_unit_id)
            ancestor_clause, ancestor_params = _in_clause(
                "d.owner_org_unit_id",
                context.ancestor_org_unit_ids,
            )
            _append_clause(
                clauses,
                params,
                f"(d.access_scope = 'org_tree' AND {ancestor_clause})",
                ancestor_params,
            )

        if effective_scope in {"managed_org", "company"} and context.managed_org_unit_ids:
            managed_owner_clause, managed_owner_params = _owner_in_org_clause(
                context.managed_org_unit_ids
            )
            _append_clause(
                clauses,
                params,
                f"(d.access_scope = 'personal' AND {managed_owner_clause})",
                managed_owner_params,
            )
            managed_org_clause, managed_org_params = _in_clause(
                "d.owner_org_unit_id",
                context.managed_org_unit_ids,
            )
            _append_clause(
                clauses,
                params,
                f"(d.access_scope IN ('team', 'org_tree') AND {managed_org_clause})",
                managed_org_params,
            )

    where_sql = "(" + " OR ".join(clauses) + ")"
    metadata = {
        "actor_user_id": context.user_id,
        "login_id": context.login_id,
        "primary_org_unit_id": context.primary_org_unit_id,
        "primary_org_unit_name": context.primary_org_unit_name,
        "role_name": context.role_name,
        "requested_search_scope": requested_scope,
        "effective_search_scope": effective_scope,
        "ancestor_org_unit_ids": list(context.ancestor_org_unit_ids),
        "managed_org_unit_ids": list(context.managed_org_unit_ids),
        "includes_company_documents": True,
        "filter_clause_count": len(clauses),
    }
    return PermissionSearchFilter(
        actor_user_id=context.user_id,
        requested_search_scope=requested_scope,
        effective_search_scope=effective_scope,
        where_sql=where_sql,
        params=tuple(params),
        metadata=metadata,
    )
