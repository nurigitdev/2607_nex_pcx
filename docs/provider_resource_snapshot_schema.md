# Provider Resource Snapshot Schema

Slice 410 adds `provider_resource_snapshots` so DGX provider resource probes can
be persisted as normalized operational evidence.

## Table

`provider_resource_snapshots` stores one row per provider target in a probe run.
Rows from the same probe execution share `probe_run_id`.

Core identity fields:

- `host`
- `provider_name`
- `provider_type`
- `model_id`
- `port`
- `collected_at`

Resource fields:

- process identity: `process_pid`, `process_ppid`, `process_user`
- process usage: `process_rss_bytes`, `process_vms_bytes`,
  `process_cpu_percent`, `process_uptime_seconds`
- safe command evidence: `process_command_preview`, `process_command_hash`
- listener evidence: `listener_process_name`, `listener_raw_line`
- GPU usage: `gpu_process_name`, `gpu_memory_used_bytes`
- host memory: `system_total_ram_bytes`, `system_available_ram_bytes`,
  `system_memory_available_percent`
- swap usage: `system_swap_total_bytes`, `system_swap_used_bytes`,
  `system_swap_used_percent`

Derived API fields:

- `process_resident_memory_share_percent` is calculated as
  `process_rss_bytes / system_total_ram_bytes * 100`
- `process_resident_memory_share_label` is formatted for operator UI display

The resident memory share fields are intentionally derived from stored snapshot
columns rather than duplicated as persistence fields.

Status fields:

- `status`: `ok`, `warning`, `critical`, or `unknown`
- `reason_codes`: JSON array of machine-readable reasons
- `match_confidence`: `port`, `command`, `missing`, or `unknown`
- `collector_error` and `collector_errors`

The table intentionally stores no prompt text, document text, request payload,
API key, or secret value. Command evidence must remain redacted before
persistence.

## Indexes

- `idx_provider_resource_snapshots_probe_run`
- `idx_provider_resource_snapshots_provider_collected`
- `idx_provider_resource_snapshots_status_collected`
- `idx_provider_resource_snapshots_host_collected`
- `idx_provider_resource_snapshots_process_pid`

## Settings

Slice 410 seeds the initial operational defaults in `app_log_settings`:

- `provider_resource_snapshot_retention_days`: `7`
- `provider_resource_stale_snapshot_warning_minutes`: `10`
- `provider_resource_stale_snapshot_critical_minutes`: `30`

Future API/UI slices should use these settings when showing freshness and
cleanup guidance.
