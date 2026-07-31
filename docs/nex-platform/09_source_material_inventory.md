# Source Material Inventory

Status: Draft seed for Slice 419.

This inventory registers the 15 source documents uploaded to
`artifacts/nex-platform/source-materials/`. The raw source files remain outside
the committed documentation set unless the user explicitly approves committing
them. The committed docs should reference source IDs, hashes, and distilled
decisions rather than copying large source material wholesale.

## Intake Summary

| Field | Value |
| --- | --- |
| Source directory | `artifacts/nex-platform/source-materials/` |
| Uploaded source files | 15 Markdown files |
| Total line count | 24,342 lines |
| Commit policy | Do not commit raw source files by default |
| Review method | Inventory, distill, map, decide, normalize, trace |
| First scope filter | `NP-SRC-13` 2-week barebone SRS |
| Broad cross-check | `NP-SRC-01` full platform SRS |

## Registered Files

| Source ID | File | Lines | Bytes | SHA-256 Prefix | Primary Service Lane | Review Priority |
| --- | --- | ---: | ---: | --- | --- | --- |
| `NP-SRC-01` | `01_260723_NeX_Platform_v1.11_SRS.md` | 3,464 | 127,156 | `6d8af1cd9f8e` | Shared SRS | P3 broad cross-check |
| `NP-SRC-02` | `02_260723_NeX_Platform_Canonical_Terminology_State_API_Contract_v1.0.md` | 1,211 | 20,423 | `76ec309c9e6c` | Shared contract | P1 contract |
| `NP-SRC-03` | `03_260723_NeX_Platform_Common_Foundation_Design_v1.6.md` | 2,160 | 41,031 | `29d6523ee4a6` | Shared foundation | P1 contract |
| `NP-SRC-04` | `04_260723_NeX_Platform_Initial_Minimal_Baseline_v0.7.md` | 1,677 | 33,839 | `9fd35ca830b6` | Shared baseline | P1 MVP |
| `NP-SRC-05` | `05_260723_NeX_Platform_Service_Lifecycle_Host_Control_Design_v1.2.md` | 1,407 | 26,501 | `ee9e9082c4aa` | Lifecycle/operations | P3 deferred |
| `NP-SRC-06` | `06_260723_NeX_Platform_Installation_Bootstrap_License_Security_Baseline_Design_v1.0.md` | 1,153 | 16,843 | `57017da93cff` | Installation/security | P3 deferred |
| `NP-SRC-07` | `07_260723_NeX_OA_Operations_Administration_Design_v1.2.md` | 1,739 | 37,145 | `95d3fa9a5774` | `nex-oa` identity boundary | P1 conflict review |
| `NP-SRC-08` | `08_260723_NeX_AG_Operations_Administration_Design_v1.6.md` | 1,692 | 41,146 | `80587282bd8f` | `nex-ag` admin & governance | P1 MVP |
| `NP-SRC-09` | `09_260723_NeX_CX_Enterprise_Context_Knowledge_Lifecycle_Design_v1.2.md` | 2,161 | 36,545 | `704d8409db7e` | `nex-cx` content lifecycle | P1 MVP |
| `NP-SRC-10` | `10_260723_NeX_AE_Work_Assistant_Workspace_Design_v1.4.md` | 2,090 | 41,681 | `04d334307f69` | `nex-ae-web`, `nex-ae-api` | P1 MVP |
| `NP-SRC-11` | `11_260723_NeX_MO_Model_Operations_Design_v1.3.md` | 2,007 | 37,651 | `f7c614ce29ad` | `nex-mo` model operations | P1 MVP |
| `NP-SRC-12` | `12_260723_NeX_Platform_v2.0_Communication_Intelligence_Customer_Timeline_Concept_v0.1.md` | 718 | 13,332 | `a5264c70303a` | v2.0 roadmap | P4 deferred |
| `NP-SRC-13` | `13_260724_NeX_Platform_2Week_Barebone_SRS_v1.1.md` | 1,258 | 23,868 | `5fd6b3492216` | 2-week MVP | P0 scope gate |
| `NP-SRC-14` | `14_260724_NeX_Platform_Common_Functions_Definition_v1.1.md` | 809 | 12,123 | `fa0ba6ca241e` | Shared functions | P2 reconcile |
| `NP-SRC-15` | `15_260724_NeX_Platform_Development_Environment_Directory_Structure_v1.1.md` | 796 | 12,763 | `0d7334048ed3` | Development environment | P1 MVP |

## Priority Lanes

| Priority | Documents | Reason |
| --- | --- | --- |
| P0 scope gate | `NP-SRC-13` | The reduced 2-week MVP document should keep the first NeX-Platform baseline small. Distilled in [2-Week MVP Capability Map](10_2week_mvp_capability_map.md). |
| P1 MVP/contract | `NP-SRC-02`, `NP-SRC-03`, `NP-SRC-04`, `NP-SRC-07`, `NP-SRC-08`, `NP-SRC-09`, `NP-SRC-10`, `NP-SRC-11`, `NP-SRC-15` | These define the minimum service spine, contracts, service responsibilities, and development setup. `NP-SRC-02` and `NP-SRC-03` are distilled in [Common Contract Freeze Candidate Map](11_common_contract_freeze_candidate_map.md). `NP-SRC-07` through `NP-SRC-11` are distilled in [Service Boundary Decision Record](12_service_boundary_decision_record.md). |
| P2 reconcile | `NP-SRC-14` | Common functions must be reconciled after service ownership is fixed. |
| P3 deferred hardening | `NP-SRC-01`, `NP-SRC-05`, `NP-SRC-06` | These are important but broad enough to bloat the first baseline if read too early. |
| P4 roadmap | `NP-SRC-12` | v2.0 communication intelligence should remain a future extension unless a current MVP hook is required. |

## Initial Conflict Notes

| Source | Note | Handling |
| --- | --- | --- |
| `NP-SRC-07` | The file name uses `NeX_OA_Operations_Administration`, but the user-confirmed boundary defines `nex-oa` as NeX Open Auth. | Review identity/auth content first and move operations/admin content to `nex-ag` if needed. |
| `NP-SRC-01` | Full SRS likely duplicates many smaller focused documents. | Use it after P0/P1 review as a completeness cross-check. |
| `NP-SRC-12` | v2.0 communication/customer timeline scope can pull the MVP away from document intelligence. | Keep as roadmap/deferred unless it reveals a required extension point. |
| `NP-SRC-10` | AE workspace material includes broad future agent capability. | Freeze bounded MVP agent orchestration in [AE Agent Orchestration Contract](13_ae_agent_orchestration_contract.md); defer autonomous multi-step domain agents. |
| `NP-SRC-09` | CX material includes broad search, generation, structured draft, and artifact scope. | Freeze the CX-to-AE retrieval/evidence package in [CX-to-AE Retrieval Context Package Contract](14_cx_ae_retrieval_context_package_contract.md); keep final user-facing generation orchestration in AE. |
| `NP-SRC-09`, `NP-SRC-10`, `NP-SRC-11` | Source documents state that AE should not call MO directly for document generation. | Reconcile direct-call wording in [Generation Routing Boundary Reconciliation](15_generation_routing_boundary_reconciliation.md): AE orchestrates, CX mediates document-grounded generation, MO executes providers. |
| `NP-SRC-02`, `NP-SRC-09`, `NP-SRC-10` | Source documents repeat a compact NeX-CX Generation Request schema. | Freeze the expanded AE-to-CX request package in [AE-to-CX Generation Request Package Contract](16_ae_cx_generation_request_package_contract.md). |
| `NP-SRC-11` | MO material defines stable generation API, alias resolution, admission, routing, streaming, cancel, and usage metadata. | Freeze the CX-to-MO provider-facing generation contract in [CX-to-MO Generation Provider Contract](17_cx_mo_generation_provider_contract.md). |
| `NP-SRC-02`, `NP-SRC-09`, `NP-SRC-10`, `NP-SRC-11` | Source documents connect generation request, evidence, prompt package hash, structured draft, citation, and MO usage metadata. | Freeze the CX execution and lineage record in [CX Generation Execution Record + Lineage Contract](18_cx_generation_execution_record_lineage_contract.md). |
| `NP-SRC-02`, `NP-SRC-09`, `NP-SRC-10` | Source documents require generated answers and documents to remain source-grounded, template-aware, and renderable. | Freeze structured draft sections, blocks, citation claims, validation statuses, and safe read shape in [Structured Draft + Citation Schema Contract](19_structured_draft_citation_schema_contract.md). |

## Next Review Sequence

1. Completed: Distill `NP-SRC-13` into a small MVP capability map.
2. Completed: Reconcile `NP-SRC-02` and `NP-SRC-03` into common contract and foundation rules.
3. Completed: Map `NP-SRC-07` through `NP-SRC-11` to the user-confirmed service boundaries.
4. Completed: Freeze bounded `nex-ae-api` agent orchestration from `NP-SRC-10`.
5. Completed: Freeze CX-to-AE retrieval context package direction from `NP-SRC-09`.
6. Completed: Reconcile document-grounded generation routing through CX before MO.
7. Completed: Freeze AE-to-CX generation request package from `NP-SRC-02`,
   `NP-SRC-09`, and `NP-SRC-10`.
8. Completed: Freeze CX-to-MO generation provider contract from `NP-SRC-11`.
9. Completed: Freeze CX generation execution and lineage record across request,
   evidence, prompt package, MO call, draft, and validation refs.
10. Completed: Freeze structured draft and citation schema across generated
    sections, blocks, citation claims, evidence anchors, and validation status.
11. Use `NP-SRC-15` to settle development environment and directory assumptions.
12. Use `NP-SRC-01` as a final cross-check for missing requirements.
13. Keep `NP-SRC-05`, `NP-SRC-06`, and `NP-SRC-12` mostly deferred unless they reveal an MVP blocker.
