"""Parsing for quest localization data files (XLSX / CSV).

The expected layout matches the sheets produced by the Quest Team:

* Column A holds the English key (speaker name, section label, ...).
* The first row holds language codes, starting in column B.
* A ``Dialogue`` section holds the dialogue rows.
* A ``Translation`` section may hold ``Special Speaker``, ``Location`` and
  ``Field Label`` sub-sections. These sub-sections can appear in any order,
  can be missing, and are also accepted directly inside the ``Dialogue``
  section (the older layout).

The parser deliberately keeps every language it finds so the page can offer
the user a target language picker.
"""

from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass, field
from pathlib import Path

import openpyxl
import pandas as pd


# Used by the page to detect a module left behind by a hot reload.
with open(__file__, "rb") as _source_file:
    _SOURCE_HASH = hashlib.sha256(_source_file.read()).hexdigest()

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
CONDITIONAL_HEADER_RE = re.compile(
    r"^\s*[\(（]?\s*(?:If|When|After|Before|Unless|While)\b",
    re.IGNORECASE,
)

TRANSLATION_SEGMENT_KEYS = {
    "special speaker": "special_speakers",
    "location": "locations",
    "field label": "field_labels",
}


class QuestDataError(ValueError):
    """Raised when a quest data file cannot be understood."""


def normalize_name(value: str) -> str:
    """Collapse whitespace so keys such as ``"Aderfi "`` match ``"Aderfi"``."""

    return NAME_WHITESPACE_RE.sub(" ", str(value)).strip()


def _is_description_key(value: str) -> bool:
    return "desc" in str(value).casefold()


def _is_objective_key(value: str) -> bool:
    key = str(value).strip()

    return (
        key.casefold() in {"quest objective", "quest objectives"}
        or bool(OBJECTIVE_KEY_RE.fullmatch(key))
    )


@dataclass
class QuestData:
    file_name: str
    languages: list[str]
    titles: dict[str, str]
    descriptions: list[dict[str, str]]
    objectives: list[dict[str, str]]
    special_speakers: list[dict[str, str]]
    locations: list[dict[str, str]]
    field_labels: list[dict[str, str]]
    dialogue_rows: list[dict[str, str]]
    dialogue_bold: dict[str, set[str]] = field(default_factory=dict)

    def title_for(self, language: str) -> str:
        return self.titles.get(language, "")

    def objective_dataframe(self, language: str) -> pd.DataFrame:
        entries = self.descriptions + self.objectives

        return pd.DataFrame(
            {
                "header": [entry["key"] for entry in entries],
                "text": [entry.get(language, "") for entry in entries],
            },
            dtype = str,
        )

    def dialogue_dataframe(self, language: str) -> pd.DataFrame:
        entries = list(self.dialogue_rows)

        if self.descriptions and not any(
            _is_description_key(entry["key"]) for entry in entries
        ):
            insert_at = next(
                (
                    position
                    for position, entry in enumerate(entries)
                    if _is_objective_key(entry["key"])
                ),
                0,
            )
            entries[insert_at:insert_at] = self.descriptions

        return pd.DataFrame(
            {
                "header": pd.Series(
                    [entry["key"] for entry in entries],
                    dtype = str,
                ),
                "text": pd.Series(
                    [entry.get(language, "") for entry in entries],
                    dtype = str,
                ),
                "bold": pd.Series(
                    [
                        language in self.dialogue_bold.get(entry["row"], set())
                        for entry in entries
                    ],
                    dtype = bool,
                ),
                "conditional": pd.Series(
                    [entry.get("conditional", "") for entry in entries],
                    dtype = str,
                ),
            }
        )

    def speaker_map(self, language: str) -> dict[str, str]:
        """Return ``English name -> localized name`` from ``Special Speaker``."""

        return self._map_from_entries(self.special_speakers, language)

    def location_map(self, language: str) -> dict[str, str]:
        """Return ``English name -> localized name`` from ``Location``."""

        return self._map_from_entries(self.locations, language)

    def location_keys(self) -> set[str]:
        """Return all English location keys, even when a translation is missing."""

        return {
            normalize_name(entry.get("EN", ""))
            for entry in self.locations
            if normalize_name(entry.get("EN", ""))
        }

    def field_label_map(self, language: str) -> dict[str, str]:
        """Return ``English label -> localized label`` from ``Field Label``."""

        return self._map_from_entries(self.field_labels, language)

    def special_speaker_pairs(self, language: str) -> list[tuple[str, str]]:
        return self._translation_pairs(self.special_speakers, language)

    def location_pairs(self, language: str) -> list[tuple[str, str]]:
        return self._translation_pairs(self.locations, language)

    def field_label_pairs(self, language: str) -> list[tuple[str, str]]:
        return self._translation_pairs(self.field_labels, language)

    def missing_translations(self, language: str) -> list[dict[str, str]]:
        """Rows that have English text but no text in ``language``."""

        return [
            entry
            for entry in self.dialogue_rows
            if normalize_name(entry.get("EN", "")) and not normalize_name(entry.get(language, ""))
        ]

    def missing_special_translations(self, language: str) -> list[dict[str, str]]:
        return self._missing_entries(self.special_speakers, language)

    def missing_location_translations(self, language: str) -> list[dict[str, str]]:
        return self._missing_entries(self.locations, language)

    def missing_field_label_translations(self, language: str) -> list[dict[str, str]]:
        return self._missing_entries(self.field_labels, language)

    def _map_from_entries(
        self, entries: list[dict[str, str]], language: str
    ) -> dict[str, str]:
        mapping: dict[str, str] = {}

        for entry in entries:
            english_key = normalize_name(entry.get("EN", ""))
            localized_value = normalize_name(entry.get(language, ""))

            if english_key and localized_value:
                mapping.setdefault(english_key, localized_value)

        return mapping

    def _translation_pairs(
        self, entries: list[dict[str, str]], language: str
    ) -> list[tuple[str, str]]:
        pairs = []

        for entry in entries:
            english_key = normalize_name(entry.get("EN", ""))
            localized_value = normalize_name(entry.get(language, ""))

            if english_key:
                pairs.append((english_key, localized_value))

        return pairs

    def _missing_entries(
        self, entries: list[dict[str, str]], language: str
    ) -> list[dict[str, str]]:
        return [
            entry
            for entry in entries
            if normalize_name(entry.get("EN", "")) and not normalize_name(entry.get(language, ""))
        ]


def parse_quest_file(file_bytes: bytes, file_name: str) -> QuestData:
    """Parse an uploaded XLSX or CSV quest data file."""

    rows = _read_rows(file_bytes, file_name)

    language_row_index, language_columns = _find_language_row(rows)
    languages = list(language_columns)

    dialogue_header_index = _find_header(rows, ("Dialogue", "Dialogues"))

    if dialogue_header_index is None:
        raise QuestDataError(
            "Could not find the 'Dialogue' or 'Dialogues' section in this file. "
            "Make sure the file follows the Quest Team localization sheet format."
        )

    translation_segments, excluded_rows = _parse_translation_segments(
        rows, language_columns
    )

    descriptions = _parse_descriptions(
        rows, language_columns, dialogue_header_index, excluded_rows
    )
    objectives = _parse_objectives(
        rows, language_columns, dialogue_header_index, excluded_rows
    )
    bold_cells = _read_bold_cells(
        file_bytes, file_name, language_columns, len(rows)
    )
    conditional_blocks = _read_conditional_blocks(
        file_bytes, file_name, language_columns, len(rows)
    )
    dialogue_rows, dialogue_bold = _parse_dialogue_rows(
        rows,
        language_columns,
        dialogue_header_index + 1,
        excluded_rows,
        bold_cells,
        conditional_blocks,
    )
    titles = _parse_title(rows, language_columns, language_row_index)

    return QuestData(
        file_name=file_name,
        languages=languages,
        titles=titles,
        descriptions=descriptions,
        objectives=objectives,
        special_speakers=translation_segments["special_speakers"],
        locations=translation_segments["locations"],
        field_labels=translation_segments["field_labels"],
        dialogue_rows=dialogue_rows,
        dialogue_bold=dialogue_bold,
    )


def _read_bold_cells(
    file_bytes: bytes,
    file_name: str,
    language_columns: dict[str, int],
    row_count: int,
) -> set[tuple[int, int]]:
    suffix = Path(file_name).suffix.lower()

    if suffix not in {".xlsx", ".xlsm"}:
        return set()

    try:
        workbook = openpyxl.load_workbook(io.BytesIO(file_bytes), data_only=True)
    except Exception:
        return set()

    try:
        worksheet = workbook.worksheets[0]
        bold_cells: set[tuple[int, int]] = set()

        for row_index in range(row_count):
            for column_index in language_columns.values():
                cell = worksheet.cell(
                    row = row_index + 1,
                    column = column_index + 1,
                )

                if cell.font and cell.font.bold:
                    bold_cells.add((row_index, column_index))

        return bold_cells
    finally:
        workbook.close()


def _read_conditional_blocks(
    file_bytes: bytes,
    file_name: str,
    language_columns: dict[str, int],
    row_count: int,
) -> dict[int, str]:
    """Group rows between condition headers using their cell border range."""

    suffix = Path(file_name).suffix.lower()

    if suffix not in {".xlsx", ".xlsm"}:
        return {}

    try:
        workbook = openpyxl.load_workbook(io.BytesIO(file_bytes), data_only=True)
    except Exception:
        return {}

    try:
        worksheet = workbook.worksheets[0]
        border_column = language_columns.get("EN")

        if border_column is None and language_columns:
            border_column = next(iter(language_columns.values()))

        if border_column is None:
            return {}

        header_rows = [
            row_index
            for row_index in range(row_count)
            if _is_conditional_header(
                worksheet, row_index, language_columns
            )
        ]
        blocks: dict[int, str] = {}
        block_id = 0

        for header_position, header_index in enumerate(header_rows):
            next_header = (
                header_rows[header_position + 1]
                if header_position + 1 < len(header_rows)
                else row_count
            )
            start: int | None = None
            outer_style: str | None = None
            end: int | None = None

            for row_index in range(header_index + 1, next_header):
                cell = worksheet.cell(
                    row = row_index + 1,
                    column = border_column + 1,
                )
                top = cell.border.top.style if cell.border.top else None
                bottom = (
                    cell.border.bottom.style if cell.border.bottom else None
                )

                if start is None and top:
                    start = row_index
                    outer_style = top

                if start is not None and bottom == outer_style:
                    end = row_index
                    break

            if (
                start is None
                or end is None
                or end < start
                or start - header_index > 3
            ):
                continue

            has_content = any(
                _cell_text(
                    worksheet.cell(
                        row = row_index + 1,
                        column = column_index + 1,
                    ).value
                ).strip()
                for row_index in range(start, end + 1)
                for column_index in language_columns.values()
            )

            if not has_content:
                continue

            block_id += 1

            for block_row in range(header_index, end + 1):
                blocks[block_row] = str(block_id)

        return blocks
    finally:
        workbook.close()


def _is_conditional_header(
    worksheet: openpyxl.worksheet.worksheet.Worksheet,
    row_index: int,
    language_columns: dict[str, int],
) -> bool:
    header = _cell_text(
        worksheet.cell(row = row_index + 1, column = 1).value
    ).strip()

    if not header or not CONDITIONAL_HEADER_RE.match(header):
        return False

    for column_index in language_columns.values():
        if _cell_text(
            worksheet.cell(
                row = row_index + 1,
                column = column_index + 1,
            ).value
        ).strip():
            return False

    return True


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


def _first_cell(row: list[str]) -> str:
    return row[0] if row else ""


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


def _find_header(
    rows: list[list[str]], labels: str | tuple[str, ...], start_index: int = 0
) -> int | None:
    if isinstance(labels, str):
        normalized_labels = {labels.strip().casefold()}
    else:
        normalized_labels = {label.strip().casefold() for label in labels}

    for row_index in range(start_index, len(rows)):
        if _first_cell(rows[row_index]).strip().casefold() in normalized_labels:
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


def _parse_translation_segments(
    rows: list[list[str]], language_columns: dict[str, int]
) -> tuple[dict[str, list[dict[str, str]]], set[int]]:
    segments: dict[str, list[dict[str, str]]] = {
        "special_speakers": [],
        "locations": [],
        "field_labels": [],
    }
    excluded_rows: set[int] = set()

    for row_index, row in enumerate(rows):
        header = _first_cell(row).strip().casefold()

        if header == "translation":
            excluded_rows.add(row_index)
            continue

        segment_key = TRANSLATION_SEGMENT_KEYS.get(header)

        if segment_key is None:
            continue

        excluded_rows.add(row_index)

        entries, end_index = _parse_segment_entries(
            rows, row_index, language_columns
        )

        segments[segment_key].extend(entries)
        excluded_rows.update(range(row_index + 1, end_index))

    return segments, excluded_rows


def _parse_segment_entries(
    rows: list[list[str]], header_index: int, language_columns: dict[str, int]
) -> tuple[list[dict[str, str]], int]:
    entries: list[dict[str, str]] = []
    row_index = header_index + 1

    while row_index < len(rows):
        row = rows[row_index]

        # A non-empty key means the next section has started.
        if _first_cell(row).strip():
            break

        # A blank row separates the segment from the next one.
        if not any(value.strip() for value in row):
            break

        values = _row_values(row, language_columns)
        english_key = normalize_name(values.get("EN", ""))

        if english_key:
            entry = {"row": str(row_index + 1), "key": english_key}
            entry.update(values)
            entries.append(entry)

        row_index += 1

    return entries, row_index


def _parse_descriptions(
    rows: list[list[str]],
    language_columns: dict[str, int],
    dialogue_header_index: int,
    excluded_rows: set[int],
) -> list[dict[str, str]]:
    """Read quest description rows from the summary block above Dialogue."""

    entries: list[dict[str, str]] = []

    for row_index in range(0, dialogue_header_index):
        if row_index in excluded_rows:
            continue

        key = _first_cell(rows[row_index]).strip()

        if "desc" not in key.casefold():
            continue

        values = _row_values(rows[row_index], language_columns)

        if not any(normalize_name(value) for value in values.values()):
            continue

        entry = {"row": str(row_index + 1), "key": key}
        entry.update(values)
        entries.append(entry)

    return entries


def _parse_objectives(
    rows: list[list[str]],
    language_columns: dict[str, int],
    dialogue_header_index: int,
    excluded_rows: set[int],
) -> list[dict[str, str]]:
    objectives_header_index = None

    # "Quest Objective" is also used inside the dialogue section as a step
    # marker, so the fallback is only accepted before the dialogue starts.
    for candidate in ("Quest Objectives", "Quest Objective"):
        candidate_index = _find_header(rows, candidate)

        if candidate_index is not None and candidate_index < dialogue_header_index:
            objectives_header_index = candidate_index
            break

    if objectives_header_index is None:
        return []

    entries: list[dict[str, str]] = []

    for row_index in range(objectives_header_index + 1, dialogue_header_index):
        if row_index in excluded_rows:
            continue

        key = _first_cell(rows[row_index]).strip()

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
    rows: list[list[str]],
    language_columns: dict[str, int],
    start_index: int,
    excluded_rows: set[int],
    bold_cells: set[tuple[int, int]],
    conditional_blocks: dict[int, str],
) -> tuple[list[dict[str, str]], dict[str, set[str]]]:
    entries: list[dict[str, str]] = []
    bold_by_row: dict[str, set[str]] = {}

    for row_index in range(start_index, len(rows)):
        if row_index in excluded_rows:
            continue

        row = rows[row_index]
        values = _row_values(row, language_columns)
        entry = {
            "row": str(row_index + 1),
            "key": _first_cell(row).strip(),
            "conditional": conditional_blocks.get(row_index, ""),
        }
        entry.update(values)
        entries.append(entry)

        bold_languages = {
            code
            for code, column_index in language_columns.items()
            if (row_index, column_index) in bold_cells
        }

        if bold_languages:
            bold_by_row[entry["row"]] = bold_languages

    while entries and _is_blank_entry(entries[0], language_columns):
        entries.pop(0)

    while entries and _is_blank_entry(entries[-1], language_columns):
        entries.pop()

    return entries, bold_by_row


def _is_blank_entry(entry: dict[str, str], language_columns: dict[str, int]) -> bool:
    return not entry["key"] and not any(
        normalize_name(entry.get(code, "")) for code in language_columns
    )
