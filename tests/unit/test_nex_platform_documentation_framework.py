from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOC_ROOT = PROJECT_ROOT / "docs" / "nex-platform"

EXPECTED_DOCS = [
    "README.md",
    "00_documentation_framework.md",
    "01_service_boundary.md",
    "02_mvp_srs_skeleton.md",
    "03_design_system_skeleton.md",
    "04_development_environment_skeleton.md",
    "05_testing_strategy_skeleton.md",
    "06_common_modules_skeleton.md",
    "07_pcx_lessons_learned_seed.md",
    "08_source_document_review_matrix.md",
    "09_source_material_inventory.md",
    "10_2week_mvp_capability_map.md",
    "11_common_contract_freeze_candidate_map.md",
    "12_service_boundary_decision_record.md",
    "13_ae_agent_orchestration_contract.md",
    "14_cx_ae_retrieval_context_package_contract.md",
    "15_generation_routing_boundary_reconciliation.md",
]


def _read(name: str) -> str:
    return (DOC_ROOT / name).read_text(encoding="utf-8")


def test_nex_platform_minimum_document_set_exists() -> None:
    for doc_name in EXPECTED_DOCS:
        assert (DOC_ROOT / doc_name).is_file(), doc_name


def test_readme_links_minimum_document_set_and_service_split() -> None:
    readme = _read("README.md")

    for doc_name in EXPECTED_DOCS[1:]:
        assert doc_name in readme

    for service_name in (
        "nex-cx",
        "nex-ae-web",
        "nex-ae-api",
        "nex-mo",
        "nex-oa",
        "nex-ag",
    ):
        assert service_name in readme

    assert "NeX Open Auth" in readme
    assert "Admin & governance" in readme
    assert "NeX-PCX source code direct reuse is not the primary goal" in readme


def test_service_boundary_preserves_user_confirmed_ownership() -> None:
    boundary = _read("01_service_boundary.md")

    assert "original files" in boundary or "Source document repository" in boundary
    assert "chunk records" in boundary
    assert "embedding vectors" in boundary
    assert "BM25 keyword index" in boundary
    assert "graph metadata" in boundary
    assert "prompt intent detection" in boundary
    assert "search and generation requests to `nex-cx`" in boundary
    assert "Direct `nex-ae-api` to `nex-mo` generation is a later policy decision" in (boundary)
    assert "Embedding provider" in boundary
    assert "reranker provider" in boundary
    assert "generation provider" in boundary
    assert "NeX Open Auth" in boundary
    assert "token/session/API key" in boundary
    assert "permission claims" in boundary
    assert "Admin & governance" in boundary
    assert "agent behavior belongs to `nex-ae-api`" in boundary


def test_srs_skeleton_keeps_two_week_mvp_and_traceability_scope() -> None:
    srs = _read("02_mvp_srs_skeleton.md")

    assert "2-week MVP" in srs
    assert "400,000-token design document" in srs
    assert "NeX-PCX SRS and slice history" in srs
    assert "FR-CX-001" in srs
    assert "FR-AE-001" in srs
    assert "FR-MO-001" in srs
    assert "FR-OA-001" in srs
    assert "FR-AG-001" in srs
    assert "statement coverage" in srs
    assert "branch coverage" in srs


def test_testing_strategy_requires_single_pass_separate_coverage_reporting() -> None:
    testing = _read("05_testing_strategy_skeleton.md")

    assert "one pytest invocation" in testing
    assert "scripts/quality_gate.sh" in testing
    assert "Statement coverage percentage" in testing
    assert "Branch coverage percentage" in testing
    assert "No-answer and low-confidence" in testing
    assert "Playwright screenshot" in testing


def test_review_matrix_accepts_large_docs_without_scope_bloat() -> None:
    matrix = _read("08_source_document_review_matrix.md")

    for source in (
        "400,000-token",
        "2-week MVP",
        "PCX SRS",
        "NeX-PCX commit history",
    ):
        assert source in matrix

    for status in ("MVP", "Deferred", "Rejected", "Duplicate", "Needs Review"):
        assert status in matrix

    assert "choose the smaller MVP" in matrix


def test_review_matrix_registers_all_uploaded_source_materials() -> None:
    matrix = _read("08_source_document_review_matrix.md")

    for index in range(1, 16):
        assert f"NP-SRC-{index:02d}" in matrix

    assert "09_source_material_inventory.md" in matrix
    assert "2-week MVP constraint" in matrix
    assert "Identity boundary review" in matrix
    assert "Deferred v2.0 concept" in matrix
    assert "Use as the first scope filter" in matrix
    assert "10_2week_mvp_capability_map.md" in matrix
    assert "11_common_contract_freeze_candidate_map.md" in matrix
    assert "Seeded" in matrix
    assert "Mock-first, live smoke as optional evidence" in matrix


def test_common_modules_focus_on_contracts_not_early_code_sharing() -> None:
    common = _read("06_common_modules_skeleton.md")

    assert "stable cross-service contracts" in common
    assert "Error envelope" in common
    assert "Auth claims" in common
    assert "Provider contract" in common
    assert "Retrieval package" in common
    assert "Agent orchestration package" in common
    assert "Generation routing contract" in common
    assert "Do not create a large shared utility package first" in common
    assert "11_common_contract_freeze_candidate_map.md" in common
    assert "12_service_boundary_decision_record.md" in common
    assert "13_ae_agent_orchestration_contract.md" in common
    assert "14_cx_ae_retrieval_context_package_contract.md" in common
    assert "15_generation_routing_boundary_reconciliation.md" in common


def test_common_contract_freeze_map_identifies_sources_and_frozen_api_basics() -> None:
    freeze_map = _read("11_common_contract_freeze_candidate_map.md")

    assert "NP-SRC-02" in freeze_map
    assert "NP-SRC-03" in freeze_map
    assert "Freeze Now" in freeze_map
    assert "Freeze Candidate" in freeze_map
    assert "Conflict" in freeze_map
    assert "/api/v1/..." in freeze_map
    assert "/admin/v1/..." in freeze_map
    assert "/health" in freeze_map
    assert "/ready" in freeze_map
    assert "/version" in freeze_map
    assert "application/problem+json" in freeze_map
    assert "Idempotency-Key" in freeze_map
    assert "X-Request-ID" in freeze_map
    assert "traceparent" in freeze_map
    assert "cursor" in freeze_map
    assert "next_cursor" in freeze_map


def test_common_contract_freeze_map_separates_states_and_job_stage() -> None:
    freeze_map = _read("11_common_contract_freeze_candidate_map.md")

    assert "`lifecycle_state`" in freeze_map
    assert "`health_status`" in freeze_map
    assert "`job_status`" in freeze_map
    assert "PENDING" in freeze_map
    assert "CANCEL_REQUESTED" in freeze_map
    assert "TIMEOUT" in freeze_map
    assert "DETERMINATE" in freeze_map
    assert "INDETERMINATE" in freeze_map
    assert "STREAMING" in freeze_map
    assert "DEGRADED` belongs to `health_status`, not `lifecycle_state`" in freeze_map
    assert "current_stage`, not `job_status`" in freeze_map


def test_common_contract_freeze_map_tracks_db_logging_and_security_contracts() -> None:
    freeze_map = _read("11_common_contract_freeze_candidate_map.md")

    assert "cross-service database access" in freeze_map
    assert "Application log core fields" in freeze_map
    assert "password" in freeze_map
    assert "authorization" in freeze_map
    assert "access_token" in freeze_map
    assert "service_secret" in freeze_map
    assert "Audit fields" in freeze_map
    assert "Security log fields" in freeze_map
    assert "raw tokens" in freeze_map


def test_common_contract_freeze_map_keeps_unsettled_contracts_out_of_freeze_now() -> None:
    freeze_map = _read("11_common_contract_freeze_candidate_map.md")

    assert "OpenAPI/JSON Schema generation" in freeze_map
    assert "Shared `nex_common` package" in freeze_map
    assert "Evidence contract" in freeze_map
    assert "Structured draft contract" in freeze_map
    assert "Artifact rendering contract" in freeze_map
    assert "Generation ownership" in freeze_map
    assert "Statement coverage target differs" in freeze_map
    assert "Freeze contracts first" in freeze_map


def test_source_material_inventory_tracks_files_without_committing_raw_sources() -> None:
    inventory = _read("09_source_material_inventory.md")

    assert "15 Markdown files" in inventory
    assert "24,342 lines" in inventory
    assert "Do not commit raw source files by default" in inventory
    assert "artifacts/nex-platform/source-materials/" in inventory

    for source_id in ("NP-SRC-01", "NP-SRC-07", "NP-SRC-13", "NP-SRC-15"):
        assert source_id in inventory

    assert "01_260723_NeX_Platform_v1.11_SRS.md" in inventory
    assert "13_260724_NeX_Platform_2Week_Barebone_SRS_v1.1.md" in inventory
    assert "nex-oa` as NeX Open Auth" in inventory
    assert "P0 scope gate" in inventory
    assert "10_2week_mvp_capability_map.md" in inventory
    assert "11_common_contract_freeze_candidate_map.md" in inventory
    assert "12_service_boundary_decision_record.md" in inventory
    assert "13_ae_agent_orchestration_contract.md" in inventory
    assert "14_cx_ae_retrieval_context_package_contract.md" in inventory
    assert "15_generation_routing_boundary_reconciliation.md" in inventory
    assert "P4 roadmap" in inventory


def test_two_week_mvp_capability_map_distills_source_into_service_owners() -> None:
    mvp_map = _read("10_2week_mvp_capability_map.md")

    assert "NP-SRC-13" in mvp_map
    assert "MVP Core" in mvp_map
    assert "MVP Stretch" in mvp_map
    assert "Deferred" in mvp_map
    assert "Boundary Review" in mvp_map

    for service_name in (
        "nex-oa",
        "nex-ag",
        "nex-ae-web",
        "nex-ae-api",
        "nex-cx",
        "nex-mo",
    ):
        assert service_name in mvp_map

    assert "Browser -> nex-ae-web -> nex-ae-api -> nex-cx -> nex-mo" in mvp_map
    assert "nex-ae-api`" in mvp_map
    assert "route document-grounded generation through `nex-cx`" in mvp_map
    assert "Direct `nex-ae-api` to `nex-mo` generation requires a later explicit policy" in (
        mvp_map
    )


def test_two_week_mvp_map_preserves_core_retrieval_generation_and_ops_choices() -> None:
    mvp_map = _read("10_2week_mvp_capability_map.md")

    assert "heading_1000_100" in mvp_map
    assert "prev_chunk_id" in mvp_map
    assert "next_chunk_id" in mvp_map
    assert "BM25" in mvp_map
    assert "Qwen3 embedding 2560" in mvp_map
    assert "Qwen3 reranker" in mvp_map
    assert "Qwen LLM" in mvp_map
    assert "5-service health/ready/version" in mvp_map
    assert "Markdown artifact" in mvp_map


def test_two_week_mvp_map_marks_risky_or_large_scope_as_stretch_or_deferred() -> None:
    mvp_map = _read("10_2week_mvp_capability_map.md")

    assert "MeCab BM25 as preferred Korean tokenizer" in mvp_map
    assert "HWP/HWPX Kordoc MCP as stretch" in mvp_map
    assert "Defer GraphDB" in mvp_map
    assert "Defer provider failover and ensemble" in mvp_map
    assert "Defer service lifecycle UI and host agent" in mvp_map
    assert "Email verification" in mvp_map
    assert "complex RBAC" in mvp_map


def test_two_week_mvp_map_keeps_acceptance_and_quality_gate_targets() -> None:
    mvp_map = _read("10_2week_mvp_capability_map.md")

    assert "First Acceptance Scenario" in mvp_map
    assert "Regression passes" in mvp_map
    assert "statement coverage target 95%" in mvp_map
    assert "branch coverage" in mvp_map
    assert "target 85%" in mvp_map
    assert "Day 14" in mvp_map


def test_service_boundary_decision_record_identifies_service_sources() -> None:
    decision_record = _read("12_service_boundary_decision_record.md")

    for source_id in (
        "NP-SRC-07",
        "NP-SRC-08",
        "NP-SRC-09",
        "NP-SRC-10",
        "NP-SRC-11",
        "NP-SRC-13",
    ):
        assert source_id in decision_record

    for service_name in (
        "nex-oa",
        "nex-ag",
        "nex-cx",
        "nex-ae-web",
        "nex-ae-api",
        "nex-mo",
    ):
        assert service_name in decision_record

    assert "Freeze Now" in decision_record
    assert "Freeze Candidate" in decision_record
    assert "Boundary Conflict" in decision_record


def test_service_boundary_decision_record_freezes_generation_ownership() -> None:
    decision_record = _read("12_service_boundary_decision_record.md")

    assert "`nex-cx` owns retrieval context" in decision_record
    assert "`nex-ae-api` owns user intent" in decision_record
    assert "`nex-mo` owns generation provider execution" in decision_record
    assert "retrieval context package, not a broad structured draft framework" in (decision_record)
    assert "Direct `nex-ae-api` to `nex-mo` generation is not the default MVP route" in (
        decision_record
    )
    assert "generated artifact metadata" in decision_record


def test_service_boundary_decision_record_keeps_data_authority_separated() -> None:
    decision_record = _read("12_service_boundary_decision_record.md")

    assert "No cross-service database joins" in decision_record
    assert "Only OA writes" in decision_record
    assert "Only CX writes" in decision_record
    assert "Only MO writes" in decision_record
    assert "Generated artifacts remain AE-owned" in decision_record
    assert "AG observes and governs through service APIs" in decision_record
    assert "Provider runtimes are reached through `nex-mo` stable APIs" in (decision_record)


def test_service_boundary_summary_links_decision_record_and_frozen_owners() -> None:
    boundary = _read("01_service_boundary.md")

    assert "12_service_boundary_decision_record.md" in boundary
    assert "Frozen Ownership Summary" in boundary
    assert "`nex-oa` owns identity" in boundary
    assert "`nex-cx` owns original assets" in boundary
    assert "`nex-ae-api` owns intent" in boundary
    assert "`nex-mo` owns provider aliases" in boundary
    assert "`nex-ag` observes and governs through service APIs" in boundary


def test_ae_agent_orchestration_contract_defines_bounded_agent_modes() -> None:
    contract = _read("13_ae_agent_orchestration_contract.md")

    assert "bounded orchestrator" in contract
    assert "`GENERAL_ANSWER`" in contract
    assert "`DOCUMENT_SEARCH`" in contract
    assert "`GROUNDED_ANSWER`" in contract
    assert "`DOCUMENT_SUMMARY`" in contract
    assert "`DOCUMENT_GENERATION`" in contract
    assert "`ARTIFACT_TRANSFORM`" in contract
    assert "Explicit mode can override intent detection" in contract


def test_ae_agent_orchestration_contract_separates_cx_mo_and_ae_ownership() -> None:
    contract = _read("13_ae_agent_orchestration_contract.md")

    assert "`nex-ae-api` is the user-facing agent API" in contract
    assert "`nex-cx` owns retrieval context packages" in contract
    assert "`nex-mo` owns provider execution" in contract
    assert "User-facing generation does not make CX the final answer owner" in contract
    assert "AE -> CX retrieval package -> AE template/output policy" in contract
    assert "CX generation API -> MO generation" in contract
    assert "CX-mediated generation routing" in contract
    assert "AE remains the orchestrator" in contract


def test_ae_agent_orchestration_contract_tracks_packages_jobs_and_guardrails() -> None:
    contract = _read("13_ae_agent_orchestration_contract.md")

    assert "Agent Request Package" in contract
    assert "Retrieval Context Package" in contract
    assert "Generation Policy Package" in contract
    assert "Agent Result Package" in contract
    assert "`INTENT_DETECTED`" in contract
    assert "`PROMPT_POLICY_PACKAGED`" in contract
    assert "`CX_GENERATION_REQUESTED`" in contract
    assert "`GENERATION_RUNNING`" in contract
    assert "These are `current_stage` values, not `job_status` enum values" in contract
    assert "No direct provider URL" in contract
    assert "No ungrounded citation" in contract
    assert "retrieval package hash" in contract


def test_cx_ae_retrieval_context_contract_freezes_call_direction() -> None:
    contract = _read("14_cx_ae_retrieval_context_package_contract.md")

    assert "Direction Decision" in contract
    assert "User prompt intake" in contract
    assert "`nex-ae-web`" in contract
    assert "`nex-ae-api`" in contract
    assert "`nex-cx`" in contract
    assert "`nex-mo`" in contract
    assert "AE requests retrieval from CX" in contract
    assert "CX returns evidence, permission, scoring, and confidence metadata to AE" in contract
    assert "AE calls CX for document-grounded generation by default" in contract
    assert "CX calls MO stable API for provider execution" in contract
    assert "Direct AE-to-MO generation requires a later explicit policy" in contract


def test_cx_ae_retrieval_context_contract_defines_package_fields() -> None:
    contract = _read("14_cx_ae_retrieval_context_package_contract.md")

    assert "Retrieval Context Request" in contract
    assert "Retrieval Context Package" in contract
    assert "Evidence Item Shape" in contract
    assert "`retrieval_package_id`" in contract
    assert "`package_hash`" in contract
    assert "`permission_snapshot`" in contract
    assert "`evidence_items`" in contract
    assert "`source_anchor`" in contract
    assert "`citation_label`" in contract
    assert "`scores`" in contract


def test_cx_ae_retrieval_context_contract_handles_no_answer_and_permissions() -> None:
    contract = _read("14_cx_ae_retrieval_context_package_contract.md")

    assert "`NO_ANSWER`" in contract
    assert "`LOW_CONFIDENCE`" in contract
    assert "`PARTIAL`" in contract
    assert "`FAILED`" in contract
    assert "CX applies permission filtering before scoring output is returned" in contract
    assert "AE does not call CX or MO generation for `NO_ANSWER`" in contract
    assert "no_answer_reason" in contract
    assert "filtered_document_count" in contract
    assert "filtered_chunk_count" in contract


def test_generation_routing_reconciliation_freezes_cx_mediated_document_generation() -> None:
    routing = _read("15_generation_routing_boundary_reconciliation.md")

    assert "Generation Routing Boundary Reconciliation" in routing
    assert "`nex-ae-api` requests generation through `nex-cx`" in routing
    assert "`nex-cx` calls `nex-mo` stable generation API" in routing
    assert "AE document-generation request targets CX, not MO" in routing
    assert "No AE direct provider call for document generation" in routing
    assert "No raw provider URL" in routing


def test_generation_routing_reconciliation_keeps_ae_cx_mo_ownership_separate() -> None:
    routing = _read("15_generation_routing_boundary_reconciliation.md")

    assert "`execution_mode`" in routing
    assert "Intent-analysis prompt policy" in routing
    assert "Provider-facing prompt package" in routing
    assert "Generation provider request" in routing
    assert "Structured draft validation" in routing
    assert "Artifact rendering and links" in routing
    assert "General Answer Or Intent Analysis" in routing
    assert "requires a later explicit policy" in routing


def test_pcx_lessons_seed_covers_retrieval_generation_provider_and_governance() -> None:
    lessons = _read("07_pcx_lessons_learned_seed.md")

    assert "Chunk adjacency" in lessons
    assert "BM25, vector search, hybrid search, and reranking" in lessons
    assert "No-answer and low-confidence" in lessons
    assert "vLLM runtime metrics" in lessons
    assert "prompt-contract alignment" in lessons
    assert "Upload ownership and query scope" in lessons
