import pytest

from app.core.i18n import InvalidLanguageError, get_translator, normalize_language, resolve_language


def test_translator_uses_korean_default_and_english_catalog() -> None:
    korean = get_translator("ko")
    english = get_translator("en")

    assert korean("nav.dashboard") == "대시보드"
    assert english("nav.dashboard") == "Dashboard"
    assert korean("nav.permissions") == "권한"
    assert english("permissions.title") == "Permission Simulation"


def test_translator_falls_back_to_key_when_missing() -> None:
    translator = get_translator("ko")

    assert translator("missing.translation.key") == "missing.translation.key"


def test_resolve_language_prefers_query_then_cookie_then_default() -> None:
    assert resolve_language(query_language="en", cookie_language="ko") == "en"
    assert resolve_language(query_language=None, cookie_language="en") == "en"
    assert resolve_language(query_language=None, cookie_language=None) == "ko"


def test_normalize_language_rejects_unsupported_language() -> None:
    with pytest.raises(InvalidLanguageError, match="Unsupported language"):
        normalize_language("fr")
