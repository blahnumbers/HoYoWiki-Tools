import io
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from utils.quest_data import QuestDataError, parse_quest_file


FIXTURE_ROWS = [
    [" ", "EN", "CHS", "DE", "RU"],
    ["Title", "Fixture Quest", "测试任务", "Testquest", "Тестовое задание"],
    ["Quest Description", "Overall description", "", "", "Общее описание"],
    ["Quest Objectives"],
    [1.0, "Objective one", "", "", "Задание один"],
    [2.0, "Objective two", "", "", ""],
    ["Dialogue"],
    ["Special Speaker"],
    ["", "Tanit Tribe Scout", "斥候", "Späher", "Разведчик"],
    ["", "???", "???", "???", "???"],
    ["", "Jeht", "婕德", "Jeht", "Джехт"],
    ["", "Messenger", "信使", "Bote", ""],
    ["(If you talked to Katheryne to start the quest)"],
    ["Treasure Street, Sumeru"],
    ["Quest Objective", "Objective line", "", "", "Строка задания"],
    ["Quest Description", "Description line", "", "", "Строка описания"],
    ["Katheryne", "Hello {NICKNAME}", "", "Hallo", "Привет"],
    ["Jeht", "Hi there", "", "Hallo", "Привет"],
    ["Mendas", "Untranslated line", "", "", ""],
    ["Choice"],
    ["1. Traveler", "Choice one", "", "", "Выбор один"],
    ["2.1. Traveler", "Choice two", "", "", "Выбор два"],
    ["", "Narration", "", "", "Повествование"],
]


class QuestDataParsingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.directory = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _write_xlsx(self) -> Path:
        return self._write_rows(FIXTURE_ROWS, "fixture.xlsx")

    def _write_rows(self, rows: list[list[object]], name: str) -> Path:
        path = self.directory / name
        pd.DataFrame(rows).to_excel(path, index=False, header=False)
        return path

    def _write_csv(self) -> Path:
        path = self.directory / "fixture.csv"
        pd.DataFrame(FIXTURE_ROWS).to_csv(path, index=False, header=False)
        return path

    def test_parses_xlsx(self) -> None:
        path = self._write_xlsx()
        quest = parse_quest_file(path.read_bytes(), path.name)

        self.assertEqual(quest.languages, ["EN", "CHS", "DE", "RU"])
        self.assertEqual(quest.title_for("RU"), "Тестовое задание")
        self.assertEqual([entry["key"] for entry in quest.objectives], ["1", "2"])
        self.assertEqual(len(quest.special_speakers), 4)
        self.assertEqual(quest.speaker_map("RU")["Tanit Tribe Scout"], "Разведчик")
        self.assertEqual(quest.speaker_map("RU")["Jeht"], "Джехт")
        self.assertNotIn("Messenger", quest.speaker_map("RU"))
        self.assertEqual(quest.dialogue_rows[0]["key"], "(If you talked to Katheryne to start the quest)")
        self.assertEqual(quest.dialogue_rows[-1]["key"], "")

    def test_parses_csv_like_xlsx(self) -> None:
        xlsx_path = self._write_xlsx()
        csv_path = self._write_csv()

        from_xlsx = parse_quest_file(xlsx_path.read_bytes(), xlsx_path.name)
        from_csv = parse_quest_file(csv_path.read_bytes(), csv_path.name)

        self.assertEqual(from_xlsx.languages, from_csv.languages)
        self.assertEqual(from_xlsx.titles, from_csv.titles)
        self.assertEqual([entry["key"] for entry in from_csv.objectives], ["1", "2"])
        self.assertEqual(from_xlsx.objectives, from_csv.objectives)
        self.assertEqual(from_xlsx.special_speakers, from_csv.special_speakers)
        self.assertEqual(
            [entry["key"] for entry in from_xlsx.dialogue_rows],
            [entry["key"] for entry in from_csv.dialogue_rows],
        )

    def test_reports_missing_translations(self) -> None:
        path = self._write_xlsx()
        quest = parse_quest_file(path.read_bytes(), path.name)

        missing_rows = [entry["key"] for entry in quest.missing_translations("RU")]

        self.assertIn("Mendas", missing_rows)
        self.assertNotIn("Katheryne", missing_rows)

        missing_special = [entry["key"] for entry in quest.missing_special_translations("RU")]
        self.assertEqual(missing_special, ["Messenger"])

    def test_parses_translation_segments_after_dialogue_in_any_order(self) -> None:
        rows = [
            [" ", "EN", "RU", "JP"],
            ["Title", "Test Quest", "Тестовое задание", "テスト"],
            ["Quest Objectives"],
            [1, "Objective one", "Задание один", "目標一"],
            ["Dialogue"],
            ["Katheryne", "Hello", "Привет", "こんにちは"],
            ["Treasures Street, Sumeru"],
            ["Quest Objective", "Objective line", "Строка задания", "目標行"],
            ["Translation"],
            ["Field Label"],
            ["", "Quest Objective", "Цель задания", "目標"],
            ["", "Quest Description", "Описание задания", "説明"],
            ["Location"],
            ["", "Treasures Street, Sumeru", "Улица Сокровищ, Сумеру", "トレジャーストリート"],
            ["Special Speaker"],
            ["", "Katheryne", "Катерина", "キャサリン"],
            ["", "Traveler", "Путешественник", "旅人"],
        ]
        path = self._write_rows(rows, "translation-after.xlsx")

        quest = parse_quest_file(path.read_bytes(), path.name)

        self.assertEqual(quest.languages, ["EN", "RU", "JP"])
        self.assertEqual([entry["key"] for entry in quest.objectives], ["1"])
        self.assertEqual(
            [entry["key"] for entry in quest.dialogue_rows],
            ["Katheryne", "Treasures Street, Sumeru", "Quest Objective"],
        )
        self.assertEqual(
            [entry["key"] for entry in quest.special_speakers],
            ["Katheryne", "Traveler"],
        )
        self.assertEqual(
            [entry["key"] for entry in quest.locations],
            ["Treasures Street, Sumeru"],
        )
        self.assertEqual(
            [entry["key"] for entry in quest.field_labels],
            ["Quest Objective", "Quest Description"],
        )
        self.assertEqual(quest.speaker_map("RU")["Katheryne"], "Катерина")
        self.assertEqual(quest.speaker_map("JP")["Katheryne"], "キャサリン")
        self.assertEqual(
            quest.location_map("RU")["Treasures Street, Sumeru"],
            "Улица Сокровищ, Сумеру",
        )
        self.assertEqual(
            quest.field_label_map("RU")["Quest Objective"],
            "Цель задания",
        )

    def test_parses_translation_segments_inside_dialogue(self) -> None:
        rows = [
            [" ", "EN", "RU"],
            ["Dialogue"],
            ["Location"],
            ["", "Treasures Street, Sumeru", "Улица Сокровищ, Сумеру"],
            ["Field Label"],
            ["", "Quest Objective", "Цель задания"],
            ["Special Speaker"],
            ["", "Katheryne", "Катерина"],
            ["Katheryne", "Hello", "Привет"],
        ]
        path = self._write_rows(rows, "translation-inside.xlsx")

        quest = parse_quest_file(path.read_bytes(), path.name)

        self.assertEqual([entry["key"] for entry in quest.dialogue_rows], ["Katheryne"])
        self.assertEqual([entry["key"] for entry in quest.locations], ["Treasures Street, Sumeru"])
        self.assertEqual([entry["key"] for entry in quest.field_labels], ["Quest Objective"])
        self.assertEqual([entry["key"] for entry in quest.special_speakers], ["Katheryne"])

    def test_translation_segments_are_optional(self) -> None:
        rows = [
            [" ", "EN", "RU"],
            ["Dialogue"],
            ["Katheryne", "Hello", "Привет"],
        ]
        path = self._write_rows(rows, "no-translation.xlsx")

        quest = parse_quest_file(path.read_bytes(), path.name)

        self.assertEqual([entry["key"] for entry in quest.dialogue_rows], ["Katheryne"])
        self.assertEqual(quest.special_speakers, [])
        self.assertEqual(quest.locations, [])
        self.assertEqual(quest.field_labels, [])

    def test_accepts_dialogues_section_header(self) -> None:
        rows = [
            [" ", "EN", "RU"],
            ["Dialogues"],
            ["Katheryne", "Hello", "Привет"],
        ]
        path = self._write_rows(rows, "dialogues-header.xlsx")

        quest = parse_quest_file(path.read_bytes(), path.name)

        self.assertEqual([entry["key"] for entry in quest.dialogue_rows], ["Katheryne"])

    def test_empty_objectives_dataframe_uses_text_columns(self) -> None:
        rows = [
            [" ", "EN", "RU"],
            ["Dialogue"],
            ["Katheryne", "Hello", "Привет"],
        ]
        path = self._write_rows(rows, "no-objectives.xlsx")

        quest = parse_quest_file(path.read_bytes(), path.name)

        objective_df = quest.objective_dataframe("RU")
        dialogue_df = quest.dialogue_dataframe("RU")

        self.assertEqual(objective_df.shape, (0, 2))
        self.assertEqual(objective_df.dtypes.astype(str).tolist(), ["object", "object"])
        self.assertEqual(dialogue_df.dtypes.astype(str).tolist(), ["object", "object"])

    def test_accepts_singular_objectives_header_before_dialogue(self) -> None:
        rows = [
            [" ", "EN", "RU"],
            ["Quest Objective"],
            [1, "Objective one", "Задание один"],
            ["Dialogue"],
            ["Quest Objective", "In-dialogue objective", "Маркер"],
            ["Katheryne", "Hello", "Привет"],
        ]
        path = self._write_rows(rows, "singular-objectives.xlsx")

        quest = parse_quest_file(path.read_bytes(), path.name)

        self.assertEqual([entry["key"] for entry in quest.objectives], ["1"])
        self.assertEqual(
            [entry["key"] for entry in quest.dialogue_rows],
            ["Quest Objective", "Katheryne"],
        )

    def test_ignores_in_dialogue_objective_marker_as_section_header(self) -> None:
        rows = [
            [" ", "EN", "RU"],
            ["Dialogue"],
            ["Quest Objective", "In-dialogue objective", "Маркер"],
            ["Katheryne", "Hello", "Привет"],
        ]
        path = self._write_rows(rows, "in-dialogue-objective.xlsx")

        quest = parse_quest_file(path.read_bytes(), path.name)

        self.assertEqual(quest.objectives, [])
        self.assertEqual(
            [entry["key"] for entry in quest.dialogue_rows],
            ["Quest Objective", "Katheryne"],
        )

    def test_rejects_file_without_language_row(self) -> None:
        rows = [["Title", "Test"], ["Dialogue"], ["Katheryne", "Hello"]]
        path = self.directory / "invalid.xlsx"
        pd.DataFrame(rows).to_excel(path, index=False, header=False)

        with self.assertRaises(QuestDataError):
            parse_quest_file(path.read_bytes(), path.name)

    def test_rejects_file_without_dialogue_section(self) -> None:
        rows = [[" ", "EN", "RU"], ["Title", "Test", "Тест"], ["Quest Objectives"], [1, "One", "Один"]]
        path = self.directory / "no-dialogue.xlsx"
        pd.DataFrame(rows).to_excel(path, index=False, header=False)

        with self.assertRaises(QuestDataError):
            parse_quest_file(path.read_bytes(), path.name)

    def test_rejects_unsupported_extension(self) -> None:
        with self.assertRaises(QuestDataError):
            parse_quest_file(b"", "fixture.txt")

    def test_action_row_after_special_speakers_is_dialogue(self) -> None:
        rows = [
            [" ", "EN", "RU"],
            ["Dialogue"],
            ["Special Speaker"],
            ["", "Jeht", "Джехт"],
            [""],
            ["", "Opening narration", "Вступление"],
            ["Katheryne", "Hello", "Привет"],
        ]
        path = self.directory / "action-row.xlsx"
        pd.DataFrame(rows).to_excel(path, index=False, header=False)

        quest = parse_quest_file(path.read_bytes(), path.name)

        self.assertEqual([entry["key"] for entry in quest.special_speakers], ["Jeht"])
        self.assertEqual(
            [entry["key"] for entry in quest.dialogue_rows],
            ["", "Katheryne"],
        )

    def test_accepts_utf16_csv(self) -> None:
        path = self._write_csv()
        utf16_bytes = io.BytesIO()
        utf16_bytes.write(path.read_text(encoding="utf-8").encode("utf-16"))

        quest = parse_quest_file(utf16_bytes.getvalue(), "fixture.csv")

        self.assertIn("RU", quest.languages)


if __name__ == "__main__":
    unittest.main()
