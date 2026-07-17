# NeX_PCX Service Process Startup Guide

This guide defines the app-host process layout for a go-live style deployment.
Remote GPU provider unit generation remains in `scripts/setup_remote_gpu_provider.py`;
this guide focuses on the web app and local queue workers.

## Process Layout

| Process | Unit | Purpose | Restart |
| --- | --- | --- | --- |
| Web app | `nex-pcx-web.service` | FastAPI admin UI and APIs | `on-failure` |
| Pipeline worker | `nex-pcx-pipeline-worker.service` | Upload extraction, chunking, and embedding job creation | `always` |
| Embedding worker | `nex-pcx-embedding-worker.service` | Route-aware embedding job execution | `always` |

The worker scripts process a bounded amount of work and exit. The generated
systemd templates use `Restart=always` with a short delay so workers keep polling
without a separate process supervisor.

## Generate Templates

Preview the templates without writing files:

```bash
./.venv/bin/python scripts/render_service_startup_templates.py --pretty
```

Write the environment file, unit files, and operator README:

```bash
./.venv/bin/python scripts/render_service_startup_templates.py \
  --workdir /home/tprover/2607_nex_pcx \
  --user nexpcx \
  --output-dir deployment \
  --write \
  --pretty
```

The generated files are:

```text
deployment/env/nex-pcx.env
deployment/systemd/nex-pcx-web.service
deployment/systemd/nex-pcx-pipeline-worker.service
deployment/systemd/nex-pcx-embedding-worker.service
deployment/README.md
```

## Required Review Before Install

- Replace the placeholder `NEX_PCX_DATABASE_URL` with the production database URL
  through the host secret store or a protected environment file.
- Confirm `NEX_PCX_UPLOAD_STORAGE_DIR` points to durable storage.
- Confirm `NEX_PCX_MODELS_DIR` points to the model directory used by local checks.
- Confirm the app host can reach remote provider route base URLs.
- Keep generated files out of git unless the environment values are sanitized.

## Suggested Start Order

1. Install the reviewed environment file.
2. Install or copy the reviewed unit files to the host systemd directory.
3. Run `systemctl daemon-reload`.
4. Start `nex-pcx-web.service`.
5. Run `scripts/validate_operations_startup.py --app-url http://127.0.0.1:8000`.
6. Start `nex-pcx-pipeline-worker.service`.
7. Start `nex-pcx-embedding-worker.service`.
8. Export go-live evidence after the first readiness pass.

## Suggested Stop Order

1. Pause scheduled ingestion and preflight.
2. Run `scripts/check_shutdown_drain.py`.
3. Stop `nex-pcx-embedding-worker.service`.
4. Stop `nex-pcx-pipeline-worker.service`.
5. Stop `nex-pcx-web.service`.
6. Stop remote provider units on the GPU host.
