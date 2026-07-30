# Common Modules Skeleton

Status: Draft bootstrap.

Common modules should be defined by stable cross-service contracts, not by early
code sharing. Start with shared schemas and error behavior; delay reusable
packages until service boundaries have survived implementation.

## Candidate Common Contracts

| Module | Purpose | Consumers |
| --- | --- | --- |
| Config | Environment parsing, profile selection, default values, secret references. | All services |
| Error envelope | Consistent API errors with code, message, retryability, and correlation id. | All services |
| Logging | Structured event fields, retention hints, severity, service name, actor. | All services |
| Audit event | Actor/action/target/result/event metadata for governance review. | All services, `nex-ag` |
| Auth claims | User id, groups, roles, scopes, service principal, trust boundary. | All services, `nex-oa` |
| Service identity | Service-to-service authentication and API key metadata. | All backend services |
| Health/readiness | Liveness, dependency readiness, degraded reason, last check time. | `nex-mo`, `nex-ag`, all services |
| Provider contract | Embedding, reranker, generation request/response and runtime metadata. | `nex-mo`, `nex-ae-back` |
| Retrieval package | Query, profiles, chunks, scores, source anchors, no-answer evidence. | `nex-cx`, `nex-ae-back` |
| Artifact contract | Generated document metadata, preview, export, download link, lineage. | `nex-ae-back`, `nex-ae-web` |
| Feature flag | Runtime toggles for experimental providers, tokenizers, templates, policies. | All services |

## Contract Style

- Use explicit version fields for provider, retrieval, prompt, and artifact contracts.
- Include correlation ids for cross-service tracing.
- Include actor and authorization context where a user action changes state.
- Separate public API fields from internal diagnostic metadata.
- Keep JSON-compatible schemas as the first contract artifact.

## Early Guardrails

| Guardrail | Reason |
| --- | --- |
| Do not create a large shared utility package first | It can hide boundary mistakes. |
| Do not let services share private database tables | It makes ownership unclear. |
| Do not duplicate auth validation logic | `nex-oa` owns trust decisions. |
| Do not bury provider runtime state inside UI code | `nex-mo` owns provider metadata. |
| Do not store generated artifacts only in chat messages | Artifact lineage and downloads need durable records. |

## Documentation To Add Later

- JSON schema files for each shared contract, starting from
  [Common Contract Freeze Candidate Map](11_common_contract_freeze_candidate_map.md).
- Error code catalog.
- Claim and scope catalog.
- Audit event taxonomy.
- OpenAPI generation strategy.
- Version compatibility policy.
