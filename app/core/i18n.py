"""Small JSON-backed UI translation helper."""

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

DEFAULT_LANGUAGE = "ko"
FALLBACK_LANGUAGE = "en"
LANGUAGE_COOKIE_NAME = "nex_pcx_lang"
SUPPORTED_LANGUAGES = ("ko", "en")
LOCALES_DIR = Path(__file__).resolve().parents[1] / "locales"


@dataclass(frozen=True)
class LanguageOption:
    code: str
    label: str


LANGUAGE_OPTIONS = (
    LanguageOption(code="ko", label="한국어"),
    LanguageOption(code="en", label="English"),
)


class InvalidLanguageError(ValueError):
    """Raised when an unsupported UI language is requested."""


def normalize_language(language: str | None) -> str | None:
    if language is None:
        return None
    normalized = language.strip().lower()
    if not normalized:
        return None
    if normalized not in SUPPORTED_LANGUAGES:
        raise InvalidLanguageError(f"Unsupported language: {language}")
    return normalized


def resolve_language(
    *,
    query_language: str | None = None,
    cookie_language: str | None = None,
) -> str:
    for candidate in (query_language, cookie_language):
        try:
            normalized = normalize_language(candidate)
        except InvalidLanguageError:
            continue
        if normalized is not None:
            return normalized
    return DEFAULT_LANGUAGE


@lru_cache(maxsize=1)
def _load_catalogs() -> dict[str, dict[str, Any]]:
    catalogs: dict[str, dict[str, Any]] = {}
    for language in SUPPORTED_LANGUAGES:
        with (LOCALES_DIR / f"{language}.json").open(encoding="utf-8") as locale_file:
            catalogs[language] = json.load(locale_file)
    return catalogs


def _lookup(catalog: dict[str, Any], key: str) -> str | None:
    current: Any = catalog
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current if isinstance(current, str) else None


class Translator:
    def __init__(self, language: str) -> None:
        self.language = resolve_language(query_language=language)
        self._catalogs = _load_catalogs()

    def __call__(self, key: str, **values: object) -> str:
        text = _lookup(self._catalogs[self.language], key)
        if text is None:
            text = _lookup(self._catalogs[FALLBACK_LANGUAGE], key)
        if text is None:
            return key
        if values:
            return text.format(**values)
        return text


def get_translator(language: str) -> Translator:
    return Translator(language)
