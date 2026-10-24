"""Parsing for quest localization data files (XLSX / CSV).

The expected layout matches the sheets produced by the Quest Team:

* Column A holds the English key (speaker name, section label, ...).
* The first row holds language codes, starting in column B.
* A ``Dialogue`` section starts with a ``Special Speaker`` sub-section that
  lists display names and their translations.

The parser deliberately keeps every language it finds so the page can offer
the user a target language picker.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


LANGUAGE_LABELS = {
    "EN": "English",
    "CHS": "Chinese (Simplified)",
    "CHT": "Chinese (Traditional)",
    "DE": "German",
    "ES": "Spanish",
    "FR": "French",
    "ID": "Indonesian",
    "IT": "Italian",
    "JP": "Japanese",
    "KR": "Korean",
    "PT": "Portuguese",
    "RU": "Russian",
    "TH": "Thai",
    "VI": "Vietnamese",
    "TR": "Turkish",
}

LANGUAGE_CODE_RE = re.compile(r"^[A-Z]{2,3}(?:[-_][A-Z]{2,4})?$")
OBJECTIVE_KEY_RE = re.compile(r"^\d+(?:\.0+)?$")
NAME_WHITESPACE_RE = re.compile(r"\s+")


class QuestDataError(ValueError):
    """Raised when a quest data file cannot be understood."""


def normalize_name(value: str) -> str:
    """Collapse whitespace so keys such as ``"Aderfi "`` match ``"Aderfi"``."""

    return NAME_WHITESPACE_RE.sub(" ", str(value)).strip()


@dataclass
class QuestData:
    file_name: str
    languages: list[str]
    titles: dict[str, str]
    objectives: list[dict[str, str]]
    special_speakers: list[dict[str, str]]
    dialogue_rows: list[dict[str, str]]

    def title_for(self, language: str) -> str:
        return self.titles.get(language, "")

    def objective_dataframe(self, language: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "header": [entry["key"] for entry in self.objectives],
                "text": [entry.get(language, "") for entry in self.objectives],
            }
        )

    def dialogue_dataframe(self, language: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "header": [entry["key"] for entry in self.dialogue_rows],
                "text": [entry.get(language, "") for entry in self.dialogue_rows],
            }
        )

    def speaker_map(self, language: str) -> dict[str, str]:
        """Return ``English name -> localized name`` from ``Special Speaker``."""

        speaker_map: dict[str, str] = {}

        for entry in self.special_speakers:
            english_name = normalize_name(entry.get("EN", ""))
            localized_name = normalize_name(entry.get(language, ""))

            if english_name and localized_name:
                speaker_map.setdefault(english_name, localized_name)

        return speaker_map

    def special_speaker_pairs(self, language: str) -> list[tuple[str, str]]:
        pairs = []

        for entry in self.special_speakers:
            english_name = normalize_name(entry.get("EN", ""))
            localized_name = normalize_name(entry.get(language, ""))

            if english_name:
                pairs.append((english_name, localized_name))

        return pairs

    def missing_translations(self, language: str) -> list[dict[str, str]]:
        """Rows that have English text but no text in ``language``."""

        return [
            entry
            for entry in self.dialogue_rows
            if normalize_name(entry.get("EN", "")) and not normalize_name(entry.get(language, ""))
        ]

    def missing_special_translations(self, language: str) -> list[dict[str, str]]:
        return [
            entry
            for entry in self.special_speakers
            if normalize_name(entry.get("EN", "")) and not normalize_name(entry.get(language, ""))
        ]


def parse_quest_file(file_bytes: bytes, file_name: str) -> QuestData:
    """Parse an uploaded XLSX or CSV quest data file."""

    rows = _read_rows(file_bytes, file_name)

    language_row_index, language_columns = _find_language_row(rows)
    languages = list(language_columns)

    dialogue_header_index = _find_header(rows, "Dialogue")

    if dialogue_header_index is None:
        raise QuestDataError(
            "Could not find the 'Dialogue' section in this file. "
            "Make sure the file follows the Quest Team localization sheet format."
        )

    special_header_index = _find_header(rows, "Special Speaker", dialogue_header_index)

    if special_header_index is None:
        special_speakers: list[dict[str, str]] = []
        dialogue_start_index = dialogue_header_index + 1
    else:
        special_speakers, dialogue_start_index = _parse_special_speakers(
            rows, special_header_index, language_columns
        )

    objectives = _parse_objectives(
        rows, language_columns, dialogue_header_index
    )
    dialogue_rows = _parse_dialogue_rows(
        rows, language_columns, dialogue_start_index
    )
    titles = _parse_title(rows, language_columns, language_row_index)

    return QuestData(
        file_name=file_name,
        languages=languages,
        titles=titles,
        objectives=objectives,
        special_speakers=special_speakers,
        dialogue_rows=dialogue_rows,
    )


def _read_rows(file_bytes: bytes, file_name: str) -> list[list[str]]:
    suffix = Path(file_name).suffix.lower()

    if suffix in {".xlsx", ".xlsm"}:
        frame = pd.read_excel(
            io.BytesIO(file_bytes),
            header=None,
            dtype=str,
            keep_default_na=False,
        )
    elif suffix in {".csv", ".tsv"}:
        text = _decode_text(file_bytes)
        separator = "\t" if suffix == ".tsv" else None

        frame = pd.read_csv(
            io.StringIO(text),
            header=None,
            dtype=str,
            keep_default_na=False,
            sep=separator,
            engine="python" if separator is None else "c",
        )
    else:
        raise QuestDataError(
            "Unsupported file type. Please upload an XLSX or CSV file."
        )

    rows = [[_cell_text(value) for value in row] for row in frame.itertuples(index=False, name=None)]

    # Drop completely blank columns at the right-hand side. The sample sheets
    # contain thousands of empty cells after the last language column.
    last_used_column = -1

    for row in rows:
        for column_index, value in enumerate(row):
            if value.strip():
                last_used_column = max(last_used_column, column_index)

    if last_used_column >= 0:
        rows = [row[: last_used_column + 1] for row in rows]

    while rows and not any(value.strip() for value in rows[-1]):
        rows.pop()

    return rows


def _decode_text(file_bytes: bytes) -> str:
    if file_bytes[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return file_bytes.decode("utf-16")

    try:
        return file_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        return file_bytes.decode("cp1252")


def _cell_text(value: object) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return str(value)


def _find_language_row(rows: list[list[str]]) -> tuple[int, dict[str, int]]:
    for row_index, row in enumerate(rows[:10]):
        language_columns: dict[str, int] = {}

        for column_index in range(1, len(row)):
            code = row[column_index].strip().upper()

            if code and LANGUAGE_CODE_RE.fullmatch(code):
                language_columns.setdefault(code, column_index)

        if "EN" in language_columns and len(language_columns) >= 2:
            return row_index, language_columns

    raise QuestDataError(
        "Could not find the language header row (expected EN, CHS, CHT, ... in the first row)."
    )


def _find_header(rows: list[list[str]], label: str, start_index: int = 0) -> int | None:
    normalized_label = label.strip().casefold()

    for row_index in range(start_index, len(rows)):
        if rows[row_index][0].strip().casefold() == normalized_label:
            return row_index

    return None


def _row_values(row: list[str], language_columns: dict[str, int]) -> dict[str, str]:
    return {
        code: row[column_index] if column_index < len(row) else ""
        for code, column_index in language_columns.items()
    }


def _parse_title(
    rows: list[list[str]], language_columns: dict[str, int], start_index: int
) -> dict[str, str]:
    title_index = _find_header(rows, "Title", start_index)

    if title_index is None:
        return {}

    return _row_values(rows[title_index], language_columns)


def _parse_special_speakers(
    rows: list[list[str]], header_index: int, language_columns: dict[str, int]
) -> tuple[list[dict[str, str]], int]:
    entries: list[dict[str, str]] = []
    row_index = header_index + 1

    while row_index < len(rows):
        key = rows[row_index][0].strip()

        # A non-empty key means the next section has started.
        if key:
            break

        values = _row_values(rows[row_index], language_columns)
        english_name = normalize_name(values.get("EN", ""))

        if english_name:
            entry = {"row": str(row_index + 1), "key": english_name}
            entry.update(values)
            entries.append(entry)
        elif entries and not any(value.strip() for value in rows[row_index]):
            # The blank row between the speaker block and the dialogue rows.
            break

        row_index += 1

    return entries, row_index


def _parse_objectives(
    rows: list[list[str]], language_columns: dict[str, int], dialogue_header_index: int
) -> list[dict[str, str]]:
    objectives_header_index = _find_header(rows, "Quest Objectives")

    if objectives_header_index is None:
        return []

    entries: list[dict[str, str]] = []

    for row_index in range(objectives_header_index + 1, dialogue_header_index):
        key = rows[row_index][0].strip()

        if not OBJECTIVE_KEY_RE.fullmatch(key):
            continue

        values = _row_values(rows[row_index], language_columns)

        if not normalize_name(values.get("EN", "")) and not any(
            normalize_name(value) for value in values.values()
        ):
            continue

        normalized_key = str(int(float(key))) if "." in key else key
        entry = {"row": str(row_index + 1), "key": normalized_key}
        entry.update(values)
        entries.append(entry)

    return entries


def _parse_dialogue_rows(
    rows: list[list[str]], language_columns: dict[str, int], start_index: int
) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []

    while start_index < len(rows) and not any(
        value.strip() for value in rows[start_index]
    ):
        start_index += 1

    for row_index in range(start_index, len(rows)):
        values = _row_values(rows[row_index], language_columns)
        entry = {"row": str(row_index + 1), "key": rows[row_index][0].strip()}
        entry.update(values)
        entries.append(entry)

    while entries and not entries[-1]["key"] and not any(
        normalize_name(entries[-1].get(code, "")) for code in language_columns
    ):
        entries.pop()

    return entries
