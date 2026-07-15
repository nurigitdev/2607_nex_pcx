import pytest

from app.core.ingestion_artifacts import get_extraction_profile, list_extraction_profiles

pytestmark = pytest.mark.integration


def test_local_extraction_profiles_are_seeded(migrated_database_url: str) -> None:
    profiles = {
        profile.extraction_profile_name: profile
        for profile in list_extraction_profiles(migrated_database_url, active_only=True)
    }

    assert {
        "local_markdown_default",
        "local_plain_text_default",
        "local_pdf_text_default",
        "local_docx_default",
        "local_pptx_default",
        "local_xlsx_default",
        "local_hwpx_default",
    }.issubset(profiles)
    assert profiles["local_pdf_text_default"].supported_file_types == ("pdf",)
    assert profiles["local_pdf_text_default"].default_options == {
        "text_layer_only": True,
        "ocr_enabled": False,
    }
    assert profiles["local_xlsx_default"].default_options["emit_markdown_tables"] is True


def test_seeded_local_profile_can_be_loaded_by_name(migrated_database_url: str) -> None:
    profile = get_extraction_profile(migrated_database_url, "local_markdown_default")

    assert profile is not None
    assert profile.provider_mode == "local"
    assert profile.extractor_name == "local_markdown"
    assert profile.supported_file_types == ("md",)
