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
        "nex-ae-back",
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
    assert "generation requests through `nex-mo`" in boundary
    assert "Embedding provider" in boundary
    assert "reranker provider" in boundary
    assert "generation provider" in boundary
    assert "NeX Open Auth" in boundary
    assert "token/session/API key" in boundary
    assert "permission claims" in boundary
    assert "Admin & governance" in boundary
    assert "agent behavior belongs to `nex-ae-back`" in boundary


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
    assert "Mock-first, live smoke as optional evidence" in matrix


def test_common_modules_focus_on_contracts_not_early_code_sharing() -> None:
    common = _read("06_common_modules_skeleton.md")

    assert "stable cross-service contracts" in common
    assert "Error envelope" in common
    assert "Auth claims" in common
    assert "Provider contract" in common
    assert "Retrieval package" in common
    assert "Do not create a large shared utility package first" in common


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
    assert "P4 roadmap" in inventory


def test_pcx_lessons_seed_covers_retrieval_generation_provider_and_governance() -> None:
    lessons = _read("07_pcx_lessons_learned_seed.md")

    assert "Chunk adjacency" in lessons
    assert "BM25, vector search, hybrid search, and reranking" in lessons
    assert "No-answer and low-confidence" in lessons
    assert "vLLM runtime metrics" in lessons
    assert "prompt-contract alignment" in lessons
    assert "Upload ownership and query scope" in lessons
