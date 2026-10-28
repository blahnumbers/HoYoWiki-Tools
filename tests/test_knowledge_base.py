import io
import unittest

import pandas as pd

from utils import knowledge_base


def _workbook_bytes() -> bytes:
    buffer = io.BytesIO()

    with pd.ExcelWriter(buffer, engine = "openpyxl") as writer:
        pd.DataFrame(
            {
                "EN": ["Starfell Valley", "Liyue Harbor", "Starfell Valley", ""],
                "RU": ["Долина звездопада", "Порт Лиюэ", "Дубль", ""],
                "IT": ["Valle Stellare", "Porto di Liyue", "Duplicato", ""],
            }
        ).to_excel(writer, sheet_name = "Locations", index = False)

        pd.DataFrame(
            {
                "EN": ["Paimon", "Vlaicu", "Traveler"],
                "RU": ["Паймон", "Влайку", "Путешественник/ца"],
                "IT": ["Paimon", "Vlaicu", "Viaggiatore"],
            }
        ).to_excel(writer, sheet_name = "Names", index = False)

        pd.DataFrame(
            {
                "EN": ["Quest Objective", "Optional Dialogue", "Traveler"],
                "RU": ["Цель задания", "Необязательный диалог", "Путешественник"],
                "IT": ["Obiettivo missione", "Dialogo opzionale", "Viaggiatore"],
            }
        ).to_excel(writer, sheet_name = "Special Words", index = False)

        pd.DataFrame(
            {
                "EN": ["Down in the Dumps", "Pioneers"],
                "RU": ["По узкому пути", "Первопроходцы"],
                "IT": ["Toccare il fondo", "Pionieri"],
            }
        ).to_excel(writer, sheet_name = "Quests", index = False)

        pd.DataFrame(
            {
                "EN": ["Blue Key", "Metal Control Wheel"],
                "RU": ["Синий ключ", "Металлическое колесо управления"],
                "IT": ["Chiave blu", "Ruota di controllo di metallo"],
            }
        ).to_excel(writer, sheet_name = "Materials", index = False)

    return buffer.getvalue()


class KnowledgeBaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.knowledge_base = knowledge_base.parse_knowledge_base(
            _workbook_bytes()
        )

    def test_parses_every_sheet(self) -> None:
        self.assertEqual(
            set(self.knowledge_base.locations),
            {"Starfell Valley", "Liyue Harbor"},
        )
        self.assertEqual(self.knowledge_base.names["Paimon"]["RU"], "Паймон")
        self.assertEqual(
            self.knowledge_base.special_words["Quest Objective"]["IT"],
            "Obiettivo missione",
        )
        self.assertEqual(
            self.knowledge_base.quests["Down in the Dumps"]["RU"],
            "По узкому пути",
        )
        self.assertEqual(
            self.knowledge_base.materials["Blue Key"]["IT"],
            "Chiave blu",
        )

    def test_keeps_the_first_duplicate(self) -> None:
        self.assertEqual(
            self.knowledge_base.locations["Starfell Valley"]["RU"],
            "Долина звездопада",
        )

    def test_maps_for_language(self) -> None:
        maps = self.knowledge_base.maps_for("RU")

        self.assertEqual(maps["speakers"]["Paimon"], "Паймон")
        self.assertEqual(maps["speakers"]["Traveler"], "Путешественник")
        self.assertEqual(maps["locations"]["Liyue Harbor"], "Порт Лиюэ")
        self.assertEqual(
            maps["labels"]["Optional Dialogue"],
            "Необязательный диалог",
        )
        self.assertEqual(maps["quests"]["Pioneers"], "Первопроходцы")
        self.assertEqual(
            maps["materials"]["Metal Control Wheel"],
            "Металлическое колесо управления",
        )

    def test_unknown_language_returns_empty_maps(self) -> None:
        maps = self.knowledge_base.maps_for("JP")

        for key in ("speakers", "locations", "labels", "quests", "materials"):
            self.assertEqual(maps[key], {})

    def test_location_keys(self) -> None:
        self.assertEqual(
            self.knowledge_base.location_keys(),
            {"Starfell Valley", "Liyue Harbor"},
        )

    def test_counts(self) -> None:
        self.assertEqual(
            self.knowledge_base.counts,
            {
                "locations": 2,
                "names": 3,
                "special_words": 3,
                "quests": 2,
                "materials": 2,
            },
        )


if __name__ == "__main__":
    unittest.main()
