import unittest

import pandas as pd

from utils import quest_format


DIALOGUE_ROWS = {
    "header": [
        "Quest Objectives",
        "Quest Objective",
        "Quest Description",
        "Katheryne",
        "Jeht",
        "Traveler",
        "(If you talked to Katheryne, the quest starts)",
        "Treasure Street, Sumeru",
        "Optional Dialogue: Talk to Jeht again",
        "Choice",
        "1. Traveler",
        "2.1. Traveler",
        "",
        "Missing Translation",
    ],
    "text": [
        "",
        "Objective line",
        "Description line",
        "Hello {NICKNAME}",
        "Hi there",
        "Choice text",
        "",
        "",
        "",
        "",
        "First choice",
        "Nested choice",
        "Narration",
        "",
    ],
}


class ClassifyRowTests(unittest.TestCase):
    def test_recognizes_every_row_type(self) -> None:
        cases = [
            (("Quest Description", "text"), "description"),
            (("Quest Objective", "text"), "objective"),
            (("1.0", "text"), "objective"),
            (("Missing Translation", ""), "missing"),
            (("Optional Text: Something", ""), "addopt"),
            (("Choice", ""), "choice_flag"),
            (("1. Traveler", "text"), "choice_branch"),
            (("2.1. Traveler", "text"), "choice_branch"),
            (("Treasure Street, Sumeru", ""), "location"),
            (("Katheryne", "Hello"), "dialogue"),
            (("(If you talked to Katheryne)", ""), "sub_mission"),
            (("(If you talked to Katheryne, the quest starts)", ""), "sub_mission"),
            (("Quest Objectives", ""), "sub_mission"),
            (("", "Narration"), "action"),
            (("", ""), "blank"),
        ]

        for (header, text), expected in cases:
            with self.subTest(header=header, text=text):
                self.assertEqual(quest_format.classify_row(header, text), expected)


class ClassifyDialogueTests(unittest.TestCase):
    def _classify(self, prompt_variables: bool = True) -> quest_format.ClassificationResult:
        return quest_format.classify_dialogue(
            pd.DataFrame(DIALOGUE_ROWS),
            speaker_map = {"Jeht": "Джехт"},
            prompt_variables = prompt_variables,
            prompt_headers = True,
        )

    def test_assigns_types_and_normalizes_choice_speakers(self) -> None:
        result = self._classify()

        self.assertEqual(
            result.df["type"].tolist(),
            [
                "sub_mission",
                "objective",
                "description",
                "dialogue",
                "dialogue",
                "dialogue",
                "sub_mission",
                "location",
                "addopt",
                "choice_flag",
                "choice_branch",
                "choice_branch",
                "action",
                "missing",
            ],
        )
        self.assertEqual(result.df.iloc[10]["header"], "Traveler")
        self.assertEqual(result.df.iloc[11]["header"], "Traveler")

    def test_groups_prompts_and_auto_localizes_special_speakers(self) -> None:
        result = self._classify()

        self.assertEqual(set(result.prompts["speakers"]), {"Katheryne", "Traveler", "Jeht"})
        self.assertIn("(If you talked to Katheryne, the quest starts)", result.prompts["conditions"])
        self.assertIn("Quest Objectives", result.prompts["conditions"])
        self.assertIn("Optional Dialogue: Talk to Jeht again", result.prompts["optional"])
        self.assertIn("Treasure Street, Sumeru", result.prompts["locations"])
        self.assertEqual(
            result.prompts["labels"],
            {"Quest Objective": 1, "Quest Description": 1},
        )
        self.assertEqual(result.auto_localized, {"Jeht": "Джехт"})
        self.assertEqual(result.defaults["speakers"]["Jeht"], "Джехт")
        self.assertEqual(result.defaults["speakers"]["Katheryne"], "Katheryne")
        self.assertEqual(result.df.iloc[4]["header"], "Jeht")

        quest_format.apply_translations(result.df, {}, result.defaults)
        self.assertEqual(result.df.iloc[4]["header"], "Джехт")

    def test_variables_are_optional(self) -> None:
        result = self._classify(prompt_variables = True)
        self.assertIn("{NICKNAME}", result.prompts["variables"])

        result_without_variables = self._classify(prompt_variables = False)
        self.assertEqual(result_without_variables.prompts["variables"], {})

    def test_gendered_options_and_nickname_are_prompted(self) -> None:
        result = quest_format.classify_dialogue(
            pd.DataFrame(
                {
                    "header": ["Paimon"],
                    "text": ["Hello {NICKNAME}, {M#he}{F#she} is here."],
                }
            ),
            prompt_variables = True,
            prompt_headers = False,
        )

        self.assertIn("{NICKNAME}", result.prompts["variables"])
        self.assertIn("{M#he}{F#she}", result.prompts["variables"])

    def test_variable_context_is_collected(self) -> None:
        result = quest_format.classify_dialogue(
            pd.DataFrame({"header": ["Paimon"], "text": ["прибыл{F#а}"]}),
            prompt_variables = True,
            prompt_headers = False,
        )

        self.assertEqual(result.variable_contexts["{F#а}"], "прибыл")

        no_context = quest_format.classify_dialogue(
            pd.DataFrame({"header": ["Paimon"], "text": ["{NICKNAME}, hello"]}),
            prompt_variables = True,
            prompt_headers = False,
        )

        self.assertEqual(no_context.variable_contexts["{NICKNAME}"], "")

        previous_token = quest_format.classify_dialogue(
            pd.DataFrame(
                {
                    "header": ["Paimon"],
                    "text": ["ты{M#он}{F#она},{F#оставившая}"],
                }
            ),
            prompt_variables = True,
            prompt_headers = False,
        )

        self.assertEqual(
            previous_token.variable_contexts["{M#он}{F#она}"],
            "ты",
        )
        self.assertEqual(previous_token.variable_contexts["{F#оставившая}"], "")

        spaced = quest_format.classify_dialogue(
            pd.DataFrame({"header": ["Paimon"], "text": ["ты {F#а}"]}),
            prompt_variables = True,
            prompt_headers = False,
        )

        self.assertEqual(spaced.variable_contexts["{F#а}"], "")

        attached = quest_format.classify_dialogue(
            pd.DataFrame({"header": ["Paimon"], "text": ["ты{F#а}"]}),
            prompt_variables = True,
            prompt_headers = False,
        )

        self.assertEqual(attached.variable_contexts["{F#а}"], "ты")

    def test_nickname_uses_localized_traveler_name(self) -> None:
        result = quest_format.classify_dialogue(
            pd.DataFrame({"header": ["Paimon"], "text": ["Hello {NICKNAME}!"]}),
            speaker_map = {"Traveler": "Путешественник/ца"},
            prompt_variables = True,
            prompt_headers = False,
        )

        self.assertEqual(
            result.defaults["variables"]["{NICKNAME}"],
            "Путешественник/ца",
        )
        self.assertIn("{NICKNAME}", result.prompts["variables"])

        quest_format.apply_translations(result.df, {}, result.defaults)
        self.assertEqual(
            result.df.iloc[0]["text"],
            "Hello Путешественник/ца!",
        )

    def test_nickname_inside_combined_token_uses_traveler_name(self) -> None:
        result = quest_format.classify_dialogue(
            pd.DataFrame(
                {
                    "header": ["Paimon"],
                    "text": ["{NICKNAME}{M#he}{F#she}"],
                }
            ),
            speaker_map = {"Traveler": "Путешественник/ца"},
            prompt_variables = True,
            prompt_headers = False,
        )

        self.assertEqual(
            result.defaults["variables"]["{NICKNAME}{M#he}{F#she}"],
            "Путешественник/ца{M#he}{F#she}",
        )

    def test_nickname_stays_unchanged_without_traveler_name(self) -> None:
        result = quest_format.classify_dialogue(
            pd.DataFrame({"header": ["Paimon"], "text": ["Hello {NICKNAME}!"]}),
            prompt_variables = True,
            prompt_headers = False,
        )

        self.assertEqual(result.defaults["variables"]["{NICKNAME}"], "{NICKNAME}")

    def test_optional_prefix_uses_field_label_translation(self) -> None:
        result = quest_format.classify_dialogue(
            pd.DataFrame(
                {
                    "header": ["Optional Dialogue: Talk to Anastasya"],
                    "text": [""],
                }
            ),
            field_label_map = {"Optional Dialogue": "Необязательный диалог"},
            prompt_variables = False,
            prompt_headers = True,
        )

        self.assertEqual(
            result.defaults["optional"]["Optional Dialogue: Talk to Anastasya"],
            "Необязательный диалог: Talk to Anastasya",
        )
        self.assertIn(
            "Optional Dialogue: Talk to Anastasya",
            result.prompts["optional"],
        )

        no_prefix = quest_format.classify_dialogue(
            pd.DataFrame(
                {
                    "header": ["Optional Dialogue: Talk to Anastasya"],
                    "text": [""],
                }
            ),
            prompt_variables = False,
            prompt_headers = True,
        )

        self.assertEqual(
            no_prefix.defaults["optional"]["Optional Dialogue: Talk to Anastasya"],
            "Optional Dialogue: Talk to Anastasya",
        )

    def test_prefills_location_and_field_label_translations(self) -> None:
        result = quest_format.classify_dialogue(
            pd.DataFrame(DIALOGUE_ROWS),
            speaker_map = {"Jeht": "Джехт"},
            location_map = {"Treasure Street, Sumeru": "Улица Сокровищ, Сумеру"},
            field_label_map = {
                "Quest Objective": "Цель задания",
                "Quest Description": "Описание задания",
            },
            prompt_variables = False,
            prompt_headers = True,
        )

        self.assertEqual(
            result.defaults["locations"]["Treasure Street, Sumeru"],
            "Улица Сокровищ, Сумеру",
        )
        self.assertEqual(result.defaults["labels"]["Quest Objective"], "Цель задания")
        self.assertEqual(result.defaults["labels"]["Quest Description"], "Описание задания")

        quest_format.apply_translations(result.df, {}, result.defaults)
        headers = set(result.df["header"])

        self.assertIn("Улица Сокровищ, Сумеру", headers)
        self.assertIn("Цель задания", headers)
        self.assertIn("Описание задания", headers)

    def test_location_key_without_comma_is_classified_as_location(self) -> None:
        result = quest_format.classify_dialogue(
            pd.DataFrame({"header": ["Snezhnaya"], "text": [""]}),
            location_map = {"Snezhnaya": "Снежная"},
            location_keys = {"Snezhnaya"},
            prompt_variables = False,
            prompt_headers = True,
        )

        self.assertEqual(result.df.iloc[0]["type"], "location")
        self.assertEqual(result.defaults["locations"]["Snezhnaya"], "Снежная")
        self.assertIn("Snezhnaya", result.prompts["locations"])

        quest_format.apply_translations(result.df, {}, result.defaults)
        self.assertEqual(result.df.iloc[0]["header"], "Снежная")

    def test_location_map_keys_are_used_without_explicit_keys(self) -> None:
        result = quest_format.classify_dialogue(
            pd.DataFrame({"header": ["Snezhnaya"], "text": [""]}),
            location_map = {"Snezhnaya": "Снежная"},
            prompt_variables = False,
            prompt_headers = True,
        )

        self.assertEqual(result.df.iloc[0]["type"], "location")
        self.assertEqual(result.defaults["locations"]["Snezhnaya"], "Снежная")

    def test_classifying_twice_replaces_the_type_column(self) -> None:
        result = self._classify()
        second_result = quest_format.classify_dialogue(result.df, speaker_map = {"Jeht": "Джехт"})

        self.assertIn("type", second_result.df.columns)
        self.assertEqual(list(second_result.df.columns).count("type"), 1)

    def test_disabling_header_prompts_uses_defaults_without_a_dialog(self) -> None:
        result = quest_format.classify_dialogue(
            pd.DataFrame({"header": ["Katheryne", "Jeht"], "text": ["Hello", "Hi"]}),
            speaker_map = {"Jeht": "Джехт"},
            prompt_variables = False,
            prompt_headers = False,
        )

        self.assertFalse(result.has_prompts)
        self.assertEqual(result.df.iloc[0]["header"], "Katheryne")
        self.assertEqual(result.df.iloc[1]["header"], "Jeht")
        self.assertEqual(result.defaults["speakers"]["Jeht"], "Джехт")

        quest_format.apply_translations(result.df, {}, result.defaults)
        self.assertEqual(result.df.iloc[1]["header"], "Джехт")


class ApplyTranslationsTests(unittest.TestCase):
    def test_replaces_headers_without_touching_prose(self) -> None:
        result = quest_format.classify_dialogue(
            pd.DataFrame({"header": ["Paimon"], "text": ["Paimon said hello."]}),
            prompt_variables = False,
        )

        quest_format.apply_translations(
            result.df,
            {"speakers": {"Paimon": "Паймон"}},
        )

        self.assertEqual(result.df.iloc[0]["header"], "Паймон")
        self.assertEqual(result.df.iloc[0]["text"], "Paimon said hello.")

    def test_applies_grouped_headers_and_variables(self) -> None:
        result = quest_format.classify_dialogue(pd.DataFrame(DIALOGUE_ROWS))

        quest_format.apply_translations(
            result.df,
            {
                "speakers": {"Katheryne": "Катерина", "Traveler": "Путешественник"},
                "conditions": {"(If you talked to Katheryne, the quest starts)": "(Если поговорить с Катериной)"},
                "optional": {"Optional Dialogue: Talk to Jeht again": "Необязательный диалог: снова поговорите с Джехт"},
                "locations": {"Treasure Street, Sumeru": "Улица Сокровищ, Сумеру"},
                "labels": {"Quest Objective": "Задание", "Quest Description": "Описание задания"},
                "variables": {"{NICKNAME}": "{ИМЯ}"},
            },
        )

        headers = result.df["header"].tolist()
        self.assertIn("Катерина", headers)
        self.assertIn("Путешественник", headers)
        self.assertIn("Задание", headers)
        self.assertIn("Описание задания", headers)
        self.assertIn("(Если поговорить с Катериной)", headers)
        self.assertIn("Необязательный диалог: снова поговорите с Джехт", headers)
        self.assertIn("Улица Сокровищ, Сумеру", headers)
        self.assertIn("Hello {ИМЯ}", result.df["text"].tolist())

    def test_empty_translation_keeps_original_value(self) -> None:
        result = quest_format.classify_dialogue(
            pd.DataFrame({"header": ["Paimon"], "text": ["Hello"]}),
            prompt_variables = False,
        )

        quest_format.apply_translations(result.df, {"speakers": {"Paimon": "   "}})

        self.assertEqual(result.df.iloc[0]["header"], "Paimon")


class FormattingTests(unittest.TestCase):
    def test_format_objective_ignores_rows_without_header(self) -> None:
        objective_df = pd.DataFrame(
            {"header": ["1", "2", ""], "text": ["One", "Two", "Ignored"]}
        )

        self.assertEqual(
            quest_format.format_objective(objective_df),
            "<ol><li><p>One</p></li><li><p>Two</p></li></ol>",
        )

    def test_format_dialogue_uses_the_selected_preset(self) -> None:
        result = quest_format.classify_dialogue(
            pd.DataFrame({"header": ["Katheryne"], "text": ["Hello"]}),
            prompt_variables = False,
            prompt_headers = False,
        )

        english_output = quest_format.format_dialogue(result.df, "EN")
        russian_output = quest_format.format_dialogue(result.df, "RU")

        self.assertIn("Katheryne: </span>Hello", english_output)
        self.assertIn("<strong>Katheryne: </strong></span><br>Hello</p>", russian_output)
        self.assertNotIn("<p>Hello</p>", russian_output)

    def test_format_dialogue_balances_tables(self) -> None:
        result = quest_format.classify_dialogue(pd.DataFrame(DIALOGUE_ROWS))

        output = quest_format.format_dialogue(result.df, "RU")

        self.assertEqual(output.count("<table>"), output.count("</table>"))

    def test_consecutive_action_rows_share_one_table(self) -> None:
        result = quest_format.classify_dialogue(
            pd.DataFrame(
                {
                    "header": ["", "", "", "Katheryne"],
                    "text": ["Line one", "Line two", "Line three", "Hello"],
                }
            ),
            prompt_variables = False,
            prompt_headers = False,
        )

        output = quest_format.format_dialogue(result.df, "EN")

        self.assertEqual(output.count("<table>"), 1)

        table_start = output.index("<table>")
        table_end = output.index("</table>") + len("</table>")
        action_table = output[table_start:table_end]

        for line in ("Line one", "Line two", "Line three"):
            self.assertIn(line, action_table)

    def test_actions_separated_by_blank_use_separate_tables(self) -> None:
        result = quest_format.classify_dialogue(
            pd.DataFrame(
                {
                    "header": ["", "", "", ""],
                    "text": ["Line one", "Line two", "", "Line three"],
                }
            ),
            prompt_variables = False,
            prompt_headers = False,
        )

        output = quest_format.format_dialogue(result.df, "EN")

        self.assertEqual(output.count("<table>"), 2)

    def test_condition_between_choice_branches_is_outside_previous_table(self) -> None:
        result = quest_format.classify_dialogue(
            pd.DataFrame(
                {
                    "header": [
                        "Choice",
                        "1. Traveler",
                        "Alia",
                        "(If condition)",
                        "2. Traveler",
                        "Paimon",
                    ],
                    "text": [
                        "",
                        "First branch",
                        "First dialogue",
                        "",
                        "Second branch",
                        "Second dialogue",
                    ],
                }
            ),
            prompt_variables = False,
            prompt_headers = False,
        )

        output = quest_format.format_dialogue(result.df, "EN")

        self.assertEqual(output.count("<table>"), 2)

        first_table_start = output.index("<table>")
        first_table_end = output.index("</table>") + len("</table>")
        first_table = output[first_table_start:first_table_end]

        self.assertNotIn("(If condition)", first_table)

        condition_index = output.index("(If condition)")
        second_table_index = output.index("<table>", first_table_end)

        self.assertGreater(condition_index, first_table_end)
        self.assertLess(condition_index, second_table_index)

    def test_consecutive_conditions_before_choice_branch_are_outside(self) -> None:
        result = quest_format.classify_dialogue(
            pd.DataFrame(
                {
                    "header": [
                        "Choice",
                        "1. Traveler",
                        "Alia",
                        "(If condition A)",
                        "(If condition B)",
                        "2. Traveler",
                        "Paimon",
                    ],
                    "text": [
                        "",
                        "First branch",
                        "First dialogue",
                        "",
                        "",
                        "Second branch",
                        "Second dialogue",
                    ],
                }
            ),
            prompt_variables = False,
            prompt_headers = False,
        )

        output = quest_format.format_dialogue(result.df, "EN")

        first_table_end = output.index("</table>") + len("</table>")
        first_table = output[:first_table_end]

        self.assertNotIn("(If condition A)", first_table)
        self.assertNotIn("(If condition B)", first_table)
        self.assertLess(
            output.index("(If condition A)"),
            output.index("(If condition B)"),
        )
        self.assertLess(
            output.index("(If condition B)"),
            output.index("<table>", first_table_end),
        )

    def test_condition_before_dialogue_stays_inside_choice_table(self) -> None:
        result = quest_format.classify_dialogue(
            pd.DataFrame(
                {
                    "header": ["Choice", "1. Traveler", "Alia", "(If condition)", "Alia"],
                    "text": ["", "First branch", "First dialogue", "", "Conditional dialogue"],
                }
            ),
            prompt_variables = False,
            prompt_headers = False,
        )

        output = quest_format.format_dialogue(result.df, "EN")

        self.assertEqual(output.count("<table>"), 1)

        table_start = output.index("<table>")
        table_end = output.index("</table>") + len("</table>")
        choice_table = output[table_start:table_end]

        self.assertIn("(If condition)", choice_table)
        self.assertIn("Conditional dialogue", choice_table)

    def test_bold_action_rows_are_wrapped_in_strong(self) -> None:
        result = quest_format.classify_dialogue(
            pd.DataFrame(
                {
                    "header": ["", "", "Katheryne"],
                    "text": ["Bold action", "Normal action", "Hello"],
                    "bold": [True, False, False],
                }
            ),
            prompt_variables = False,
            prompt_headers = False,
        )

        output = quest_format.format_dialogue(result.df, "EN")

        self.assertIn("<strong>Bold action</strong>", output)
        self.assertIn("Normal action", output)
        self.assertNotIn("<strong>Normal action</strong>", output)
        self.assertNotIn("<strong>Hello</strong>", output)

    def test_prepare_text_handles_non_breaking_spaces_and_ruby(self) -> None:
        self.assertEqual(
            quest_format.prepare_text("A{NON_BREAK_SPACE}B"),
            "A&nbsp;B",
        )

        ruby_output = quest_format.prepare_text("pre{RUBY#[D]ruby text}base")
        self.assertIn('<custom-ruby data-ruby="prebase|ruby text">', ruby_output)


if __name__ == "__main__":
    unittest.main()
