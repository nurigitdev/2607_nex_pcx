# Service Boundary

Status: Draft bootstrap.

This document defines the first NeX-Platform service boundaries. The names are
stable enough to use in SRS and architecture drafts, but the internal module
layout of each service remains open until implementation planning.

## Boundary Matrix

| Service | Owns | Does Not Own | Primary Consumers |
| --- | --- | --- | --- |
| `nex-cx` | Source document repository, original files, extraction artifacts, normalized Markdown, chunk records, chunk adjacency, embedding vectors, BM25 keyword index, graph metadata, retrieval APIs. | User-facing chat UX, model runtime lifecycle, platform-wide auth, admin governance UI. | `nex-ae-back`, `nex-ag`, `nex-mo`. |
| `nex-ae-web` | User UI/UX for chat, search compare, generation, summaries, report artifacts, downloads, previews, Korean default and English support. | Retrieval storage, provider hosting, auth authority, governance policies. | End users and administrators using the workspace. |
| `nex-ae-back` | Agent orchestration, prompt intent detection, search requests to `nex-cx`, generation requests through `nex-mo`, answer packaging, citation formatting, artifact creation coordination. | Raw vector storage, provider deployment, identity issuing, global policy ownership. | `nex-ae-web`, `nex-cx`, `nex-mo`, `nex-oa`. |
| `nex-mo` | Embedding provider, reranker provider, generation provider registration, provider route health, runtime metrics, readiness snapshots, vLLM metrics, provider resource monitoring. | User sessions, source document ownership, business document templates, enterprise auth claims. | `nex-ae-back`, `nex-ag`, operations users. |
| `nex-oa` | NeX Open Auth, user authentication, service-to-service authentication, token/session/API key lifecycle, permission claims, trust boundary enforcement. | Document retrieval ranking, model serving, UI-specific navigation, platform metrics visualization. | All services. |
| `nex-ag` | Admin & governance, operations dashboard, logs, policy settings, audit trails, readiness checks, data retention controls, monitoring views. | End-user authoring UX, core retrieval storage, model inference execution, identity issuance. | Administrators, operators, governance users. |

## Shared Concepts

| Concept | Canonical Owner | Notes |
| --- | --- | --- |
| User identity | `nex-oa` | Other services consume signed claims and authorization decisions. |
| Document ownership and visibility metadata | `nex-cx` | Must preserve who uploaded what, when, and with what visibility scope. |
| Retrieval package | `nex-cx` produces, `nex-ae-back` composes | Includes chunks, source anchors, scores, policies, and no-answer signals. |
| Prompt/runtime package | `nex-ae-back` | Combines user intent, template, retrieval context, and provider settings. |
| Provider route | `nex-mo` | Includes provider id, model, URL, port, profile, health, and runtime metadata. |
| Audit event | `nex-ag` collects, all services emit | Must include actor, action, target, timestamp, result, and correlation id. |

## Integration Style

- Services communicate through explicit APIs, not shared database ownership.
- `nex-cx` exposes retrieval and source context APIs to `nex-ae-back`.
- `nex-ae-back` calls `nex-mo` for model-provider route selection and runtime status.
- `nex-oa` issues identity and permission claims that every service validates.
- `nex-ag` reads operational data through service APIs or event streams.
- Shared libraries should contain stable contracts only: errors, response envelopes,
  auth claim models, correlation ids, and observability metadata.

## Boundary Risks

| Risk | Guardrail |
| --- | --- |
| `nex-ae-back` slowly becomes a data repository | Keep source files, chunks, embedding, BM25, graph, and retrieval indexes in `nex-cx`. |
| `nex-mo` becomes an admin console | Keep governance policy and operator UX in `nex-ag`; `nex-mo` owns provider operations APIs and telemetry. |
| `nex-oa` is treated as a utility library | Keep token/session/API key issuance and trust boundary decisions in the service. |
| `nex-ag` is confused with an agent runtime | Use `nex-ag` only for admin & governance; agent behavior belongs to `nex-ae-back`. |
