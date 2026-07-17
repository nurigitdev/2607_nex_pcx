# Production Database Revision Alignment

Date: 2026-07-17

## Scope

Slice 282 aligned the production database schema revision with the current
application Alembic head.

This operation only applies schema migrations. It does not register provider
routes, run provider preflights, enqueue ingestion jobs, or mutate application
data beyond migration-managed seed/default records.

## Result

- Target database: `nex_pcx_app`
- Migration command: `NEX_PCX_DATABASE_URL=postgresql://nex_pcx_app:***@127.0.0.1:5432/nex_pcx_app bash scripts/migrate.sh upgrade head`
- Alembic head: `20260715_0029`
- Production database revision after migration: `20260715_0029`
- pgvector extension: `vector 0.8.4`
- Public table count after migration: `48`

## Production Validation

The production environment validation was re-run after migration.

- Overall status: `blocked`
- Guard checks: `5` checked, `5` passed, `0` failed, `0` warnings
- Startup validation:
  - Database connectivity: passed
  - Alembic revision: passed
  - Application `/healthz`: passed
- Go-live readiness: blocked

The remaining go-live blocker is provider route readiness:

- `provider_route_readiness`: failed because `0/0` active provider routes are ready.
- `provider_preflight_schedule`: warning because `0/1` provider route preflight schedules are enabled.

This means the production DB revision is aligned, but the environment is not
ready for go-live until production provider routes are registered, activated,
preflighted, and validated.

## Follow-Up

1. Register and activate the production DGX provider routes.
2. Enable or confirm the provider preflight schedule.
3. Run provider route preflight validation.
4. Re-run `scripts/validate_production_environment.py` and confirm the go-live
   readiness status changes from `blocked` to `ready`.
