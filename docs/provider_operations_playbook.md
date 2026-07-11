# Embedding Provider Operations Playbook

Slice 120 consolidates the operator workflow for remote embedding provider routes.

Use this playbook after the provider architecture and GPU deployment shape are already
understood:

- `docs/embedding_provider_architecture.md`
- `docs/gpu_embedding_provider_deployment.md`

## Operator Goals

The embedding provider operations loop answers four questions:

1. Is each configured route reachable and compatible with its profile?
2. Is the route healthy enough for the worker to use?
3. Which contract sample set was used when a route was checked?
4. Which failures were acknowledged, and which still need action?

NeX_PCX keeps these answers in PostgreSQL through route metadata, health snapshots,
contract snapshots, app log alerts, and contract sample set records.

## Daily Check

Run this sequence before benchmark ingestion or after changing provider routes:

```bash
export NEX_PCX_DATABASE_URL="postgresql://USER:PASSWORD@127.0.0.1:5432/nex_pcx_dev"
bash scripts/migrate.sh upgrade head
./.venv/bin/python scripts/preflight_provider_routes.py
```

For scheduled operation, enable a row in `embedding_provider_preflight_schedules` and call
the due-runner from cron or a systemd timer:

```bash
./.venv/bin/python scripts/run_scheduled_provider_preflight.py --limit 20
```

The seeded `default_provider_route_preflight` schedule is disabled by default so a newly
installed environment does not contact remote providers until an operator opts in.

Then review the UI:

- `/admin/embedding-provider-routes`
- Provider health summary
- Contract snapshot history
- Route readiness status
- Unacknowledged route alerts

The preflight command uses the active default contract sample set and persists both health
and contract snapshots. A route should not be treated as ready only because the provider
responds to `GET /healthz`; it also needs to pass the embedding contract check for the
configured profile and vector dimension.

## Route Lifecycle

| Stage | Expected state | Operator action |
| --- | --- | --- |
| Planned | Route exists but is inactive | Confirm URL, profile name, model key, and dimension. |
| Smoke | Route is active but has no recent snapshot | Run preflight or a single route contract check. |
| Ready | Health and contract checks pass | Allow route-aware workers to process jobs. |
| Degraded | Health passes but contract fails | Check dimension, profile mapping, and provider output. |
| Down | Health fails or provider is unreachable | Check provider process, network path, and model preload. |
| Retired | Route is inactive | Keep snapshots for audit, but exclude from active checks. |

## Contract Sample Sets

Contract checks use `embedding_provider_contract_sample_sets`.

The default migration seeds:

- `sample_set_name`: `default_route_contract`
- `input_type`: `document`
- `sample_texts`: one deterministic document sample
- `is_active`: `true`
- `is_default`: `true`

Use sample sets to keep contract checks reproducible when comparing providers. The default
sample set should be short, stable, and safe to store in operational metadata. Avoid using
company confidential content in contract samples unless the deployment policy explicitly
allows it.

API inspection endpoint:

```text
GET /api/admin/embedding-provider-routes/contract-sample-sets?active_only=true
```

Preflight and route contract checks record the sample set name in runtime metadata as:

```json
{
  "contract_sample_set_name": "default_route_contract"
}
```

## Readiness Gate

Route-aware embedding workers can require route readiness before processing a job. Use the
readiness gate when provider routes are managed through PostgreSQL and preflight checks are
part of the deployment process.

Expected behavior:

- If a ready route exists, the worker uses that route for the job profile.
- If the route is unhealthy or has a failing contract snapshot, the worker leaves the job
  unprocessed or fails according to the configured worker path.
- If no route is ready, the operator should run preflight and inspect snapshots before
  retrying ingestion.

Relevant API:

```text
GET /api/admin/embedding-provider-routes/readiness
```

## Alert Acknowledgement

Provider route contract failures are logged through the admin logging system. Alerts remain
actionable until an operator acknowledges them.

Recommended acknowledgement policy:

- Acknowledge only after the root cause is understood or accepted.
- Include an operator note when the failure is caused by an intentional test or retired
  route.
- Do not acknowledge recurring failures without checking the latest contract snapshot.

UI location:

```text
/admin/embedding-provider-routes
```

The alert panel is an operational queue. It is not a substitute for full application logs at
`/admin/logs`.

## Failure Triage

| Symptom | Likely cause | First check |
| --- | --- | --- |
| Provider unreachable | Process down, port blocked, wrong URL | `GET /healthz` from the app host. |
| `ready=false` | Model still loading or failed preload | Provider startup logs and model bundle path. |
| Dimension mismatch | Route profile and provider output disagree | Route dimension, provider profile names, model config. |
| Input count mismatch | Provider dropped or merged inputs | Provider `/v1/embeddings` implementation. |
| Non-finite vector value | Provider returned invalid numeric output | Provider normalization and dtype conversion. |
| Slow contract check | CPU fallback or overloaded GPU | Provider device metadata and elapsed time. |

## Release Checklist

Before enabling a new route in a benchmark or customer-site environment:

1. Confirm migrations are at head.
2. Confirm model bundle version and checksum on the provider host.
3. Confirm provider `/healthz` reports the expected model key and profile names.
4. Run `scripts/preflight_provider_routes.py --json`.
5. Confirm route readiness is passing.
6. Confirm the contract sample set name in the latest snapshot.
7. Review and acknowledge any route alerts created during testing.
8. Start workers with a small batch before scaling ingestion.

## Evidence To Keep

For reproducible provider experiments, keep these records with benchmark notes:

- Application commit SHA.
- Migration head revision.
- Provider route IDs and URLs.
- Provider model IDs.
- Model bundle checksum or release ID.
- Contract sample set name.
- Latest health snapshot status.
- Latest contract snapshot status.
- Worker runtime metadata for completed embedding jobs.

This makes provider comparisons auditable when the same document corpus is reprocessed
with a different model, output dimension, route URL, or contract sample set.
