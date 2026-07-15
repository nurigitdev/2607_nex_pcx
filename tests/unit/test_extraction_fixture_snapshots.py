from pathlib import Path

import pytest
from fixtures.extraction_corpus import (
    EXTRACTION_FIXTURE_CASES,
    ExtractionFixtureCase,
    extraction_result_snapshot,
)

from app.core.extraction_runtime import ExtractionRuntimeRequest
from app.core.local_extraction import run_local_extraction


@pytest.mark.parametrize(
    "fixture_case",
    EXTRACTION_FIXTURE_CASES,
    ids=lambda fixture_case: fixture_case.case_id,
)
def test_local_extraction_fixture_matches_expected_snapshot(
    fixture_case: ExtractionFixtureCase,
    tmp_path: Path,
) -> None:
    source_path = tmp_path / fixture_case.file_name
    fixture_case.write_source(source_path)

    result = run_local_extraction(
        ExtractionRuntimeRequest(
            file_id=1,
            document_id=2,
            storage_path=str(source_path),
            extraction_profile_name=fixture_case.profile_name,
            mime_type=fixture_case.mime_type,
            detected_file_type=fixture_case.detected_file_type,
        )
    )

    assert extraction_result_snapshot(result) == fixture_case.expected_snapshot
