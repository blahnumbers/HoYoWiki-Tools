"""Shared glossary loaded from the team's Google Sheets knowledge base.

The sheet is intentionally kept outside the repository so the localization
teams can update it without a new deploy. It is downloaded as a whole
workbook so every sheet (``Locations``, ``Names`` and ``Special Words``) is
read in a single request.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import pandas as pd

KNOWLEDGE_BASE_SHEET_ID = "1OoUh_oioZ66JeVgzSbWagDL1bYEf3uVSyh_OHjxD7Xs"

KNOWLEDGE_BASE_EXPORT_URL = (
    "https://docs.google.com/spreadsheets/d/%s/export?format=xlsx"
    % KNOWLEDGE_BASE_SHEET_ID
)

KNOWLEDGE_BASE_VIEW_URL = (
    "https://docs.google.com/spreadsheets/d/%s" % KNOWLEDGE_BASE_SHEET_ID
)

SECTION_LOCATIONS = "Locations"
SECTION_NAMES = "Names"
SECTION_SPECIAL_WORDS = "Special Words"

SOURCE_COLUMN = "EN"
EMPTY_VALUES = frozenset({"", "nan", "none", "n/a", "-"})


@dataclass
class KnowledgeBase:
    """English key -> per-language translations for each glossary section."""

    locations: dict[str, dict[str, str]] = field(default_factory = dict)
    names: dict[str, dict[str, str]] = field(default_factory = dict)
    special_words: dict[str, dict[str, str]] = field(default_factory = dict)

    @property
    def counts(self) -> dict[str, int]:
        return {
            "locations": len(self.locations),
            "names": len(self.names),
            "special_words": len(self.special_words),
        }

    def maps_for(self, language: str) -> dict[str, dict[str, str]]:
        """Return the maps used by ``classify_dialogue`` for a language."""

        special_words = _column_map(self.special_words, language)

        return {
            # ``Traveler`` lives in the Special Words sheet, so it has to be
            # part of the speaker map for ``{NICKNAME}`` prefills.
            "speakers": {**_column_map(self.names, language), **special_words},
            "locations": _column_map(self.locations, language),
            "labels": special_words,
        }

    def location_keys(self) -> set[str]:
        """English location names, so headers without a comma are recognized."""

        return set(self.locations)


def parse_knowledge_base(file_bytes: bytes) -> KnowledgeBase:
    """Parse the exported ``.xlsx`` workbook into a ``KnowledgeBase``."""

    workbook = pd.ExcelFile(io.BytesIO(file_bytes))

    return KnowledgeBase(
        locations = _parse_sheet(workbook, SECTION_LOCATIONS),
        names = _parse_sheet(workbook, SECTION_NAMES),
        special_words = _parse_sheet(workbook, SECTION_SPECIAL_WORDS),
    )


def _parse_sheet(
    workbook: pd.ExcelFile,
    sheet_name: str,
) -> dict[str, dict[str, str]]:
    if sheet_name not in workbook.sheet_names:
        return {}

    frame = workbook.parse(sheet_name)

    if SOURCE_COLUMN not in frame.columns:
        return {}

    entries: dict[str, dict[str, str]] = {}

    for _, row in frame.iterrows():
        key = _clean(row.get(SOURCE_COLUMN))

        if not key:
            continue

        values = {}

        for column, value in row.items():
            code = str(column).strip().upper()

            if code == SOURCE_COLUMN:
                continue

            cleaned = _clean(value)

            if cleaned:
                values[code] = cleaned

        if values:
            # Keep the first translation when a key appears more than once.
            entries.setdefault(key, values)

    return entries


def _column_map(
    entries: dict[str, dict[str, str]],
    language: str,
) -> dict[str, str]:
    code = (language or "").strip().upper()

    return {
        key: values[code]
        for key, values in entries.items()
        if code in values
    }


def _clean(value) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    text = str(value).strip()

    if text.lower() in EMPTY_VALUES:
        return ""

    return text
