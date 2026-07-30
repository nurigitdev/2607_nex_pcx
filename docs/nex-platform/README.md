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

## Platform Services

The current target service split is:

| Service | Role |
| --- | --- |
| `nex-cx` | Content experience repository: source files, extracted text, chunks, embeddings, BM25, graph, and retrieval APIs. |
| `nex-ae-web` | User-facing workspace UI/UX for chat, search, generation, summaries, artifacts, and downloads. |
| `nex-ae-back` | Agent execution backend for intent routing, retrieval orchestration, generation orchestration, formatting, and artifact creation. |
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

Slice 418 created the skeleton and review method. Slice 419 registers the
uploaded source material inventory and seeds the review matrix. Later slices
should fill the skeleton with source-backed decisions instead of copying large
documents wholesale.
