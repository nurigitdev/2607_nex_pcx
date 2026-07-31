# NeX-Platform Documentation Workspace

Status: Draft bootstrap for Slice 418.

This workspace captures the minimum documentation framework needed to turn the
NeX-PCX pre-CX experiment into actionable NeX-Platform planning material.
NeX-PCX source code direct reuse is not the primary goal. The primary goal is to
reuse the validated requirements, implementation lessons, operational evidence,
and slice history as design input for a smaller, buildable platform baseline.

## Reading Order

1. [Documentation Framework](00_documentation_framework.md)
2. [Service Boundary](01_service_boundary.md)
3. [MVP SRS Skeleton](02_mvp_srs_skeleton.md)
4. [Design System Skeleton](03_design_system_skeleton.md)
5. [Development Environment Skeleton](04_development_environment_skeleton.md)
6. [Testing Strategy Skeleton](05_testing_strategy_skeleton.md)
7. [Common Modules Skeleton](06_common_modules_skeleton.md)
8. [PCX Lessons Learned Seed](07_pcx_lessons_learned_seed.md)
9. [Source Document Review Matrix](08_source_document_review_matrix.md)
10. [Source Material Inventory](09_source_material_inventory.md)
11. [2-Week MVP Capability Map](10_2week_mvp_capability_map.md)
12. [Common Contract Freeze Candidate Map](11_common_contract_freeze_candidate_map.md)
13. [Service Boundary Decision Record](12_service_boundary_decision_record.md)
14. [AE Agent Orchestration Contract](13_ae_agent_orchestration_contract.md)
15. [CX-to-AE Retrieval Context Package Contract](14_cx_ae_retrieval_context_package_contract.md)
16. [Generation Routing Boundary Reconciliation](15_generation_routing_boundary_reconciliation.md)
17. [AE-to-CX Generation Request Package Contract](16_ae_cx_generation_request_package_contract.md)
18. [CX-to-MO Generation Provider Contract](17_cx_mo_generation_provider_contract.md)
19. [CX Generation Execution Record + Lineage Contract](18_cx_generation_execution_record_lineage_contract.md)
20. [Structured Draft + Citation Schema Contract](19_structured_draft_citation_schema_contract.md)
21. [AE Artifact Rendering Handoff Contract](20_ae_artifact_rendering_handoff_contract.md)

## Platform Services

The current target service split is:

| Service | Role |
| --- | --- |
| `nex-cx` | Content experience repository: source files, extracted text, chunks, embeddings, BM25, graph, and retrieval APIs. |
| `nex-ae-web` | User-facing workspace UI/UX for chat, search, generation, summaries, artifacts, and downloads. |
| `nex-ae-api` | Agent execution backend for intent routing, retrieval orchestration, generation orchestration, formatting, and artifact creation. |
| `nex-mo` | Model operations service for embedding, reranker, generation provider connectivity and monitoring. |
| `nex-oa` | NeX Open Auth: user auth, service auth, token/session/API key management, permission claims, and trust boundaries. |
| `nex-ag` | Admin & governance service for operations, logs, policies, monitoring, readiness, and audit views. |

## Source Inputs

This framework is designed to absorb four source streams:

| Source | How It Will Be Used |
| --- | --- |
| NeX-PCX SRS | Seed requirements and requirement naming patterns. |
| NeX-PCX slice/commit history | Evidence for what was actually implemented, tested, or deferred. |
| 400,000-token platform design document | Broad architecture ideas to distill through the review matrix. |
| Reduced 2-week MVP document | Scope constraint for the first buildable platform baseline. |

## Immediate Output

Slice 418 created the skeleton and review method. Slice 419 registered the
uploaded source material inventory and seeded the review matrix. Slice 420
distilled the 2-week barebone SRS into a service-owned MVP capability map.
Slice 421 distilled canonical terminology, state, API, error, job, logging, and
trace contracts into freeze candidates. Slice 422 reconciled the service-specific
source documents into a boundary decision record and ownership freeze candidate.
Slice 423 defined `nex-ae-api` as the bounded user-facing agent orchestrator for
intent, retrieval, prompt/template packaging, generation calls, artifact links,
and chat workspace responses. Slice 424 froze the first CX-to-AE retrieval
context package direction: AE requests corpus-grounded evidence from CX, CX
returns permission-filtered evidence/no-answer metadata, and AE owns the next
prompt/template/generation hand-off. Slice 425 reconciled generation routing so
document-grounded generation is AE-orchestrated but CX-mediated before MO
provider execution. Slice 426 froze the first AE-to-CX generation request
package so AE can send intent, template, output, quality, and retrieval package
references to CX without leaking provider runtime ownership. Slice 427 froze the
first CX-to-MO generation provider contract so CX can call MO by stable alias
while MO owns admission, routing, provider execution, streaming/cancel, and
usage metadata. Slice 428 froze the CX generation execution record and lineage
contract so retrieval packages, prompt package hashes, MO calls, structured
draft validation, citation validation, and safe AE/AG read views remain tied
together. Slice 429 froze the structured draft and citation schema contract so
CX can validate sections, blocks, citation claims, evidence anchors, and
template completeness before AE renders artifacts. Slice 430 froze the AE
artifact rendering handoff contract so validated CX drafts can become AE-owned
artifact records, render jobs, files, previews, downloads, and chat links
without losing generation lineage.
Later slices should fill the skeleton with source-backed decisions instead of
copying large documents wholesale.
