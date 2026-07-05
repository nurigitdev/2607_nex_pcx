"""Create identity and permission metadata schema.

Revision ID: 20260705_0004
Revises: 20260703_0003
Create Date: 2026-07-05
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20260705_0004"
down_revision: str | None = "20260703_0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE app_users (
            user_id BIGSERIAL PRIMARY KEY,
            login_id TEXT NOT NULL UNIQUE,
            display_name TEXT NOT NULL,
            email TEXT,
            is_active BOOLEAN NOT NULL DEFAULT true,
            metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute("CREATE INDEX idx_app_users_active ON app_users (is_active)")

    op.execute(
        """
        CREATE TABLE org_units (
            org_unit_id BIGSERIAL PRIMARY KEY,
            parent_org_unit_id BIGINT REFERENCES org_units(org_unit_id),
            org_unit_name TEXT NOT NULL,
            org_unit_type TEXT NOT NULL
                CHECK (org_unit_type IN ('company', 'division', 'group', 'team')),
            is_active BOOLEAN NOT NULL DEFAULT true,
            metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute("CREATE INDEX idx_org_units_parent ON org_units (parent_org_unit_id)")
    op.execute("CREATE INDEX idx_org_units_active ON org_units (is_active)")

    op.execute(
        """
        CREATE TABLE user_org_memberships (
            membership_id BIGSERIAL PRIMARY KEY,
            user_id BIGINT NOT NULL REFERENCES app_users(user_id) ON DELETE CASCADE,
            org_unit_id BIGINT NOT NULL REFERENCES org_units(org_unit_id) ON DELETE CASCADE,
            role_name TEXT NOT NULL
                CHECK (role_name IN ('member', 'team_lead', 'group_lead', 'admin')),
            is_primary BOOLEAN NOT NULL DEFAULT false,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (user_id, org_unit_id, role_name)
        )
        """
    )
    op.execute("CREATE INDEX idx_user_org_memberships_user ON user_org_memberships (user_id)")
    op.execute("CREATE INDEX idx_user_org_memberships_org ON user_org_memberships (org_unit_id)")
    op.execute("CREATE INDEX idx_user_org_memberships_role ON user_org_memberships (role_name)")

    op.execute(
        """
        ALTER TABLE files
        ADD COLUMN uploaded_by_user_id BIGINT REFERENCES app_users(user_id)
        """
    )
    op.execute("CREATE INDEX idx_files_uploaded_by_user_id ON files (uploaded_by_user_id)")

    op.execute(
        """
        ALTER TABLE documents
        ADD COLUMN owner_user_id BIGINT REFERENCES app_users(user_id),
        ADD COLUMN owner_org_unit_id BIGINT REFERENCES org_units(org_unit_id),
        ADD COLUMN access_scope TEXT NOT NULL DEFAULT 'personal'
            CHECK (access_scope IN ('personal', 'team', 'org_tree', 'company')),
        ADD COLUMN permission_metadata JSONB NOT NULL DEFAULT '{}'::jsonb
        """
    )
    op.execute("CREATE INDEX idx_documents_owner_user_id ON documents (owner_user_id)")
    op.execute("CREATE INDEX idx_documents_owner_org_unit_id ON documents (owner_org_unit_id)")
    op.execute("CREATE INDEX idx_documents_access_scope ON documents (access_scope)")

    op.execute(
        """
        INSERT INTO app_users (login_id, display_name, email, metadata)
        VALUES
            ('pcx.admin', 'PCX Admin', 'pcx.admin@example.local', '{"seed_role": "admin"}'),
            ('alice.member', 'Alice Member', 'alice.member@example.local', '{"seed_role": "member"}'),
            ('bob.member', 'Bob Member', 'bob.member@example.local', '{"seed_role": "member"}'),
            (
                'chloe.teamlead',
                'Chloe Team Lead',
                'chloe.teamlead@example.local',
                '{"seed_role": "team_lead"}'
            ),
            (
                'dana.grouplead',
                'Dana Group Lead',
                'dana.grouplead@example.local',
                '{"seed_role": "group_lead"}'
            )
        ON CONFLICT (login_id) DO NOTHING
        """
    )

    op.execute(
        """
        WITH company AS (
            INSERT INTO org_units (org_unit_name, org_unit_type, metadata)
            VALUES ('NeX Company', 'company', '{"seed": true}')
            RETURNING org_unit_id
        ),
        division AS (
            INSERT INTO org_units (parent_org_unit_id, org_unit_name, org_unit_type, metadata)
            SELECT org_unit_id, 'CX Division', 'division', '{"seed": true}'
            FROM company
            RETURNING org_unit_id
        ),
        grp AS (
            INSERT INTO org_units (parent_org_unit_id, org_unit_name, org_unit_type, metadata)
            SELECT org_unit_id, 'Platform Group', 'group', '{"seed": true}'
            FROM division
            RETURNING org_unit_id
        ),
        platform_team AS (
            INSERT INTO org_units (parent_org_unit_id, org_unit_name, org_unit_type, metadata)
            SELECT org_unit_id, 'Platform Team', 'team', '{"seed": true}'
            FROM grp
            RETURNING org_unit_id
        )
        INSERT INTO org_units (parent_org_unit_id, org_unit_name, org_unit_type, metadata)
        SELECT org_unit_id, 'Business Team', 'team', '{"seed": true}'
        FROM grp
        """
    )

    op.execute(
        """
        INSERT INTO user_org_memberships (user_id, org_unit_id, role_name, is_primary)
        SELECT u.user_id, o.org_unit_id, seed.role_name, seed.is_primary
        FROM (
            VALUES
                ('pcx.admin', 'NeX Company', 'admin', true),
                ('alice.member', 'Platform Team', 'member', true),
                ('bob.member', 'Business Team', 'member', true),
                ('chloe.teamlead', 'Platform Team', 'team_lead', true),
                ('dana.grouplead', 'Platform Group', 'group_lead', true)
        ) AS seed(login_id, org_unit_name, role_name, is_primary)
        JOIN app_users u ON u.login_id = seed.login_id
        JOIN org_units o ON o.org_unit_name = seed.org_unit_name
        ON CONFLICT (user_id, org_unit_id, role_name) DO NOTHING
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_documents_access_scope")
    op.execute("DROP INDEX IF EXISTS idx_documents_owner_org_unit_id")
    op.execute("DROP INDEX IF EXISTS idx_documents_owner_user_id")
    op.execute(
        """
        ALTER TABLE documents
        DROP COLUMN IF EXISTS permission_metadata,
        DROP COLUMN IF EXISTS access_scope,
        DROP COLUMN IF EXISTS owner_org_unit_id,
        DROP COLUMN IF EXISTS owner_user_id
        """
    )

    op.execute("DROP INDEX IF EXISTS idx_files_uploaded_by_user_id")
    op.execute("ALTER TABLE files DROP COLUMN IF EXISTS uploaded_by_user_id")

    op.execute("DROP TABLE IF EXISTS user_org_memberships")
    op.execute("DROP TABLE IF EXISTS org_units")
    op.execute("DROP TABLE IF EXISTS app_users")
