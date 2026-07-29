"""Create provider resource snapshot persistence schema.

Revision ID: 20260729_0042
Revises: 20260729_0041
Create Date: 2026-07-29
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20260729_0042"
down_revision: str | None = "20260729_0041"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE provider_resource_snapshots (
            snapshot_id BIGSERIAL PRIMARY KEY,
            probe_run_id UUID NOT NULL,
            host TEXT NOT NULL CHECK (length(btrim(host)) > 0),
            provider_name TEXT NOT NULL CHECK (length(btrim(provider_name)) > 0),
            provider_type TEXT NOT NULL CHECK (
                provider_type IN ('embedding', 'reranker', 'vllm')
            ),
            model_id TEXT,
            port INT NOT NULL CHECK (port > 0 AND port <= 65535),
            status TEXT NOT NULL CHECK (
                status IN ('ok', 'warning', 'critical', 'unknown')
            ),
            reason_codes JSONB NOT NULL DEFAULT '[]'::jsonb
                CHECK (jsonb_typeof(reason_codes) = 'array'),
            match_confidence TEXT NOT NULL CHECK (
                match_confidence IN ('port', 'command', 'missing', 'unknown')
            ),
            process_pid INT CHECK (process_pid IS NULL OR process_pid > 0),
            process_ppid INT CHECK (process_ppid IS NULL OR process_ppid >= 0),
            process_user TEXT,
            process_rss_bytes BIGINT CHECK (
                process_rss_bytes IS NULL OR process_rss_bytes >= 0
            ),
            process_vms_bytes BIGINT CHECK (
                process_vms_bytes IS NULL OR process_vms_bytes >= 0
            ),
            process_cpu_percent DOUBLE PRECISION CHECK (
                process_cpu_percent IS NULL OR process_cpu_percent >= 0
            ),
            process_uptime_seconds INT CHECK (
                process_uptime_seconds IS NULL OR process_uptime_seconds >= 0
            ),
            process_command_preview TEXT,
            process_command_hash TEXT,
            listener_process_name TEXT,
            listener_raw_line TEXT,
            gpu_process_name TEXT,
            gpu_memory_used_bytes BIGINT CHECK (
                gpu_memory_used_bytes IS NULL OR gpu_memory_used_bytes >= 0
            ),
            system_total_ram_bytes BIGINT CHECK (
                system_total_ram_bytes IS NULL OR system_total_ram_bytes >= 0
            ),
            system_available_ram_bytes BIGINT CHECK (
                system_available_ram_bytes IS NULL OR system_available_ram_bytes >= 0
            ),
            system_memory_available_percent DOUBLE PRECISION CHECK (
                system_memory_available_percent IS NULL
                OR (
                    system_memory_available_percent >= 0
                    AND system_memory_available_percent <= 100
                )
            ),
            system_swap_total_bytes BIGINT CHECK (
                system_swap_total_bytes IS NULL OR system_swap_total_bytes >= 0
            ),
            system_swap_used_bytes BIGINT CHECK (
                system_swap_used_bytes IS NULL OR system_swap_used_bytes >= 0
            ),
            system_swap_used_percent DOUBLE PRECISION CHECK (
                system_swap_used_percent IS NULL
                OR (
                    system_swap_used_percent >= 0
                    AND system_swap_used_percent <= 100
                )
            ),
            collector_error TEXT,
            collector_errors JSONB NOT NULL DEFAULT '[]'::jsonb
                CHECK (jsonb_typeof(collector_errors) = 'array'),
            report_status TEXT NOT NULL CHECK (
                report_status IN ('ok', 'warning', 'critical', 'unknown')
            ),
            report_target_count INT NOT NULL CHECK (report_target_count >= 0),
            report_ok_count INT NOT NULL DEFAULT 0 CHECK (report_ok_count >= 0),
            report_warning_count INT NOT NULL DEFAULT 0 CHECK (report_warning_count >= 0),
            report_critical_count INT NOT NULL DEFAULT 0 CHECK (report_critical_count >= 0),
            report_unknown_count INT NOT NULL DEFAULT 0 CHECK (report_unknown_count >= 0),
            runtime_metadata JSONB NOT NULL DEFAULT '{}'::jsonb
                CHECK (jsonb_typeof(runtime_metadata) = 'object'),
            raw_snapshot JSONB NOT NULL DEFAULT '{}'::jsonb
                CHECK (jsonb_typeof(raw_snapshot) = 'object'),
            collected_at TIMESTAMPTZ NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """)
    op.execute("""
        CREATE INDEX idx_provider_resource_snapshots_probe_run
        ON provider_resource_snapshots (probe_run_id, snapshot_id)
        """)
    op.execute("""
        CREATE INDEX idx_provider_resource_snapshots_provider_collected
        ON provider_resource_snapshots (
            provider_name,
            collected_at DESC,
            snapshot_id DESC
        )
        """)
    op.execute("""
        CREATE INDEX idx_provider_resource_snapshots_status_collected
        ON provider_resource_snapshots (
            status,
            collected_at DESC,
            snapshot_id DESC
        )
        """)
    op.execute("""
        CREATE INDEX idx_provider_resource_snapshots_host_collected
        ON provider_resource_snapshots (
            host,
            collected_at DESC,
            snapshot_id DESC
        )
        """)
    op.execute("""
        CREATE INDEX idx_provider_resource_snapshots_process_pid
        ON provider_resource_snapshots (process_pid)
        WHERE process_pid IS NOT NULL
        """)

    op.execute("""
        INSERT INTO app_log_settings
            (setting_name, setting_value, value_type, description)
        VALUES
            (
                'provider_resource_snapshot_retention_days',
                '7',
                'int',
                'Provider resource snapshot retention in days'
            ),
            (
                'provider_resource_stale_snapshot_warning_minutes',
                '10',
                'int',
                'Provider resource snapshot age in minutes that triggers a warning'
            ),
            (
                'provider_resource_stale_snapshot_critical_minutes',
                '30',
                'int',
                'Provider resource snapshot age in minutes that triggers a critical signal'
            )
        ON CONFLICT (setting_name) DO NOTHING
        """)


def downgrade() -> None:
    op.execute("""
        DELETE FROM app_log_settings
        WHERE setting_name IN (
            'provider_resource_snapshot_retention_days',
            'provider_resource_stale_snapshot_warning_minutes',
            'provider_resource_stale_snapshot_critical_minutes'
        )
        """)
    op.execute("DROP INDEX IF EXISTS idx_provider_resource_snapshots_process_pid")
    op.execute("DROP INDEX IF EXISTS idx_provider_resource_snapshots_host_collected")
    op.execute("DROP INDEX IF EXISTS idx_provider_resource_snapshots_status_collected")
    op.execute("DROP INDEX IF EXISTS idx_provider_resource_snapshots_provider_collected")
    op.execute("DROP INDEX IF EXISTS idx_provider_resource_snapshots_probe_run")
    op.execute("DROP TABLE IF EXISTS provider_resource_snapshots")
