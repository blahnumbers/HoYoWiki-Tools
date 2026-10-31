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

    def test_german_gender_pairs_are_prefilled_in_variable_prompts(self) -> None:
        result = quest_format.classify_dialogue(
            pd.DataFrame(
                {
                    "header": ["Paimon"],
                    "text": ["{F#Die Beraterin}{M#Der Berater}"],
                }
            ),
            target_language = "DE",
            prompt_variables = True,
            prompt_headers = False,
        )

        token = "{F#Die Beraterin}{M#Der Berater}"

        self.assertIn(token, result.prompts["variables"])
        self.assertEqual(
            result.defaults["variables"][token],
            "Der Berater/Die Beraterin",
        )

    def test_german_gender_pairs_are_prompted_for_other_languages(self) -> None:
        result = quest_format.classify_dialogue(
            pd.DataFrame(
                {
                    "header": ["Paimon"],
                    "text": ["{F#Die Beraterin}{M#Der Berater}"],
                }
            ),
            target_language = "KR",
            prompt_variables = True,
            prompt_headers = False,
        )

        token = "{F#Die Beraterin}{M#Der Berater}"

        self.assertIn(token, result.prompts["variables"])
        self.assertEqual(
            result.defaults["variables"][token],
            "Der Berater/Die Beraterin",
        )

    def test_shortened_gender_languages_prefill_pair_prompts(self) -> None:
        for language in ("RU", "FR", "ES"):
            result = quest_format.classify_dialogue(
                pd.DataFrame(
                    {
                        "header": ["Paimon"],
                        "text": ["{M#Путешественник}{F#Путешественница}"],
                    }
                ),
                target_language = language,
                prompt_variables = True,
                prompt_headers = False,
            )

            token = "{M#Путешественник}{F#Путешественница}"

            self.assertIn(token, result.prompts["variables"])
            self.assertEqual(
                result.defaults["variables"][token],
                "Путешественник/ца",
            )

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

    def test_optional_prefix_prefers_the_longest_match(self) -> None:
        result = quest_format.classify_dialogue(
            pd.DataFrame(
                {
                    "header": ["Optional Dialogue: Talk to Anastasya"],
                    "text": [""],
                }
            ),
            field_label_map = {
                "Optional Dialogue": "Необязательный диалог",
                "Optional Dialogue: Talk to": (
                    "Необязательный диалог: Поговорите с"
                ),
            },
            prompt_variables = False,
            prompt_headers = True,
        )

        self.assertEqual(
            result.defaults["optional"]["Optional Dialogue: Talk to Anastasya"],
            "Необязательный диалог: Поговорите с Anastasya",
        )

    def test_obtain_and_submit_conditions_use_materials(self) -> None:
        result = quest_format.classify_dialogue(
            pd.DataFrame(
                {
                    "header": [
                        '(Obtain "Blue Key")',
                        '(Submit "Blue Key" and "White Key")',
                    ],
                    "text": ["", ""],
                }
            ),
            field_label_map = {
                "Obtain": "Получено",
                "Submit": "Используйте",
            },
            material_map = {
                "Blue Key": "Синий ключ",
                "White Key": "Белый ключ",
            },
            prompt_variables = False,
            prompt_headers = True,
        )

        self.assertEqual(
            result.defaults["conditions"]['(Obtain "Blue Key")'],
            '(Получено "Синий ключ")',
        )
        self.assertEqual(
            result.defaults["conditions"][
                '(Submit "Blue Key" and "White Key")'
            ],
            '(Используйте "Синий ключ" and "Белый ключ")',
        )

    def test_material_condition_needs_every_material(self) -> None:
        result = quest_format.classify_dialogue(
            pd.DataFrame(
                {"header": ['(Obtain "Red Key")'], "text": [""]}
            ),
            field_label_map = {"Obtain": "Получено"},
            material_map = {"Blue Key": "Синий ключ"},
            prompt_variables = False,
            prompt_headers = True,
        )

        self.assertEqual(
            result.defaults["conditions"]['(Obtain "Red Key")'],
            '(Obtain "Red Key")',
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

    def test_compound_location_is_translated_part_by_part(self) -> None:
        result = quest_format.classify_dialogue(
            pd.DataFrame({"header": ["Wavechaser Plain, Nod-Krai"], "text": [""]}),
            location_map = {
                "Wavechaser Plain": "Равнина Волногона",
                "Nod-Krai": "Нод-Край",
            },
            prompt_headers = True,
        )

        self.assertEqual(
            result.defaults["locations"]["Wavechaser Plain, Nod-Krai"],
            "Равнина Волногона, Нод-Край",
        )

    def test_compound_location_needs_every_part(self) -> None:
        result = quest_format.classify_dialogue(
            pd.DataFrame({"header": ["Wavechaser Plain, Nod-Krai"], "text": [""]}),
            location_map = {"Wavechaser Plain": "Равнина Волногона"},
            prompt_headers = True,
        )

        self.assertEqual(
            result.defaults["locations"]["Wavechaser Plain, Nod-Krai"],
            "Wavechaser Plain, Nod-Krai",
        )

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
        self.assertIn(
            "<strong>Katheryne: </strong></span><br>Hello</p>",
            russian_output,
        )
        self.assertNotIn("<p>Hello</p>", russian_output)

    def test_markup_preset_follows_the_target_language(self) -> None:
        self.assertEqual(quest_format.markup_preset_for("RU"), "RU")
        self.assertEqual(quest_format.markup_preset_for("JP"), "RU")
        self.assertEqual(quest_format.markup_preset_for("CHS"), "CHS")
        self.assertEqual(quest_format.markup_preset_for("KR"), "KR")
        self.assertEqual(quest_format.markup_preset_for("EN"), "EN")
        self.assertEqual(quest_format.markup_preset_for("DE"), "EN")

    def test_chinese_dialogue_uses_fullwidth_colon_and_bold_speaker(self) -> None:
        result = quest_format.classify_dialogue(
            pd.DataFrame({"header": ["Paimon"], "text": ["你好"]}),
            prompt_variables = False,
            prompt_headers = False,
        )

        output = quest_format.format_dialogue(result.df, "CHS")

        self.assertIn(
            '<span style="color: rgb(255, 255, 255);">'
            "<strong>Paimon：</strong></span><br>你好</p>",
            output,
        )

    def test_chinese_objective_label_and_text_are_white(self) -> None:
        result = quest_format.classify_dialogue(
            pd.DataFrame(
                {
                    "header": ["Quest Objective", "Paimon"],
                    "text": ["找到钥匙", "你好"],
                }
            ),
            field_label_map = {"Quest Objective": "任务目标"},
            target_language = "CHS",
            prompt_variables = False,
            prompt_headers = True,
        )

        quest_format.apply_translations(result.df, {}, result.defaults)
        output = quest_format.format_dialogue(
            result.df,
            "CHS",
            target_language = "CHS",
        )

        self.assertIn(
            '<p><span style="color: rgb(255, 255, 255);">'
            "<strong>任务目标：</strong>找到钥匙</span></p>",
            output,
        )

    def test_chinese_description_label_is_bold_and_text_is_plain(self) -> None:
        result = quest_format.classify_dialogue(
            pd.DataFrame(
                {
                    "header": ["Quest Description", "Paimon"],
                    "text": ["一段描述", "你好"],
                }
            ),
            field_label_map = {"Quest Description": "任务描述"},
            target_language = "CHS",
            prompt_variables = False,
            prompt_headers = True,
        )

        quest_format.apply_translations(result.df, {}, result.defaults)
        output = quest_format.format_dialogue(
            result.df,
            "CHS",
            target_language = "CHS",
        )

        self.assertIn(
            '<p><span style="color: rgb(255, 255, 255);">'
            "<strong>任务描述：</strong></span>一段描述</p>",
            output,
        )

    def test_chinese_optional_header_uses_a_fullwidth_colon(self) -> None:
        result = quest_format.classify_dialogue(
            pd.DataFrame(
                {
                    "header": ["Optional Dialogue: Talk to Anastasya"],
                    "text": [""],
                }
            ),
            field_label_map = {"Optional Dialogue": "可选对话"},
            target_language = "CHS",
            prompt_variables = False,
            prompt_headers = True,
        )

        quest_format.apply_translations(result.df, {}, result.defaults)
        output = quest_format.format_dialogue(
            result.df,
            "CHS",
            target_language = "CHS",
        )

        self.assertIn("<p>可选对话：Talk to Anastasya</p>", output)

    def test_chinese_talk_to_uses_the_name_slot(self) -> None:
        result = quest_format.classify_dialogue(
            pd.DataFrame(
                {
                    "header": ["Optional Dialogue: Talk to Anastasya"],
                    "text": [""],
                }
            ),
            field_label_map = {
                "Optional Dialogue: Talk to": "可选对话：与\u3000对话",
            },
            speaker_map = {"Anastasya": "阿纳斯塔西娅"},
            target_language = "CHS",
            prompt_headers = True,
        )

        self.assertEqual(
            result.defaults["optional"]["Optional Dialogue: Talk to Anastasya"],
            "可选对话：与阿纳斯塔西娅对话",
        )

    def test_japanese_uses_the_russian_speaker_layout(self) -> None:
        result = quest_format.classify_dialogue(
            pd.DataFrame({"header": ["Katheryne"], "text": ["Hello"]}),
            prompt_variables = False,
            prompt_headers = False,
        )

        japanese_output = quest_format.format_dialogue(result.df, "JP")

        self.assertIn(
            "<strong>Katheryne: </strong></span><br>Hello</p>",
            japanese_output,
        )

    def test_repeating_speaker_lines_collapse_into_one_paragraph(self) -> None:
        result = quest_format.classify_dialogue(
            pd.DataFrame(
                {
                    "header": ["Anastasya", "Anastasya", "Paimon"],
                    "text": ["First line.", "Second line.", "Paimon line."],
                }
            ),
            prompt_variables = False,
            prompt_headers = False,
        )

        output = quest_format.format_dialogue(
            result.df,
            "RU",
            collapse_speakers = True,
        )

        self.assertIn(
            "<strong>Anastasya: </strong></span>"
            "<br>First line.<br>Second line.</p>",
            output,
        )
        self.assertEqual(output.count("Anastasya"), 1)
        self.assertIn("Paimon line.", output)

    def test_repeating_speaker_lines_stay_separate_by_default(self) -> None:
        result = quest_format.classify_dialogue(
            pd.DataFrame(
                {
                    "header": ["Anastasya", "Anastasya"],
                    "text": ["First line.", "Second line."],
                }
            ),
            prompt_variables = False,
            prompt_headers = False,
        )

        output = quest_format.format_dialogue(result.df, "RU")

        self.assertEqual(output.count("Anastasya"), 2)
        self.assertIn("First line.</p>", output)
        self.assertIn("Second line.</p>", output)

    def test_repeating_speaker_lines_collapse_with_the_english_preset(self) -> None:
        result = quest_format.classify_dialogue(
            pd.DataFrame(
                {
                    "header": ["Paimon", "Paimon"],
                    "text": ["First line.", "Second line."],
                }
            ),
            prompt_variables = False,
            prompt_headers = False,
        )

        output = quest_format.format_dialogue(
            result.df,
            "EN",
            collapse_speakers = True,
        )

        self.assertIn("Paimon: </span>First line.<br>Second line.</p>", output)
        self.assertEqual(output.count("Paimon"), 1)

    def test_blank_row_prevents_speaker_collapsing(self) -> None:
        result = quest_format.classify_dialogue(
            pd.DataFrame(
                {
                    "header": ["Anastasya", "", "Anastasya"],
                    "text": ["First line.", "", "Second line."],
                }
            ),
            prompt_variables = False,
            prompt_headers = False,
        )

        output = quest_format.format_dialogue(
            result.df,
            "RU",
            collapse_speakers = True,
        )

        self.assertEqual(output.count("Anastasya"), 2)

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

    def test_german_gender_pairs_are_converted_male_first(self) -> None:
        result = quest_format.classify_dialogue(
            pd.DataFrame(
                {
                    "header": ["Paimon"],
                    "text": ["{F#Die Beraterin}{M#Der Berater}"],
                }
            ),
            target_language = "DE",
            prompt_variables = False,
            prompt_headers = False,
        )

        output = quest_format.format_dialogue(
            result.df,
            "EN",
            target_language = "DE",
        )

        self.assertIn("Der Berater/Die Beraterin", output)
        self.assertNotIn("{F#", output)

    def test_german_gender_pairs_reverse_order_still_male_first(self) -> None:
        result = quest_format.classify_dialogue(
            pd.DataFrame(
                {
                    "header": ["Paimon"],
                    "text": ["{M#Der Berater}{F#Die Beraterin}"],
                }
            ),
            target_language = "DE",
            prompt_variables = False,
            prompt_headers = False,
        )

        output = quest_format.format_dialogue(
            result.df,
            "EN",
            target_language = "DE",
        )

        self.assertIn("Der Berater/Die Beraterin", output)

    def test_gender_mode_none_keeps_pairs_untouched(self) -> None:
        result = quest_format.classify_dialogue(
            pd.DataFrame(
                {
                    "header": ["Paimon"],
                    "text": ["{F#Die Beraterin}{M#Der Berater}"],
                }
            ),
            target_language = "JP",
            gender_mode = "None",
            prompt_variables = False,
            prompt_headers = False,
        )

        output = quest_format.format_dialogue(
            result.df,
            "EN",
            target_language = "JP",
            gender_mode = "None",
        )

        self.assertIn("{F#Die Beraterin}{M#Der Berater}", output)

    def test_other_languages_use_male_first_by_default(self) -> None:
        result = quest_format.classify_dialogue(
            pd.DataFrame(
                {
                    "header": ["Paimon"],
                    "text": ["{F#Die Beraterin}{M#Der Berater}"],
                }
            ),
            target_language = "KR",
            prompt_variables = False,
            prompt_headers = False,
        )

        output = quest_format.format_dialogue(
            result.df,
            "EN",
            target_language = "KR",
        )

        self.assertIn("Der Berater/Die Beraterin", output)

    def test_korean_location_is_not_bold(self) -> None:
        result = quest_format.classify_dialogue(
            pd.DataFrame({"header": ["Snezhnaya"], "text": [""]}),
            location_keys = {"Snezhnaya"},
            prompt_variables = False,
            prompt_headers = False,
        )

        output = quest_format.format_dialogue(
            result.df,
            "KR",
            target_language = "KR",
        )

        self.assertIn(
            '<p><span style="color: rgb(236, 229, 216);">Snezhnaya</span></p>',
            output,
        )
        self.assertNotIn("<strong>Snezhnaya</strong>", output)

    def test_korean_description_and_objective_share_one_table(self) -> None:
        result = quest_format.classify_dialogue(
            pd.DataFrame(
                {
                    "header": ["Quest Description", "Quest Objective", "", "Paimon"],
                    "text": ["Desc", "Obj", "", "Hello"],
                }
            ),
            prompt_variables = False,
            prompt_headers = True,
        )

        quest_format.apply_translations(result.df, {}, result.defaults)
        output = quest_format.format_dialogue(
            result.df,
            "KR",
            target_language = "KR",
        )

        self.assertEqual(output.count("<table>"), 1)

        table_start = output.index("<table>")
        table_end = output.index("</table>") + len("</table>")
        meta_table = output[table_start:table_end]

        self.assertIn("Desc", meta_table)
        self.assertIn("Obj", meta_table)

        # A blank line separates the meta table from the dialogue.
        self.assertIn("</table>", output)
        self.assertIn("</table>\n                            <p></p>", output)

    def test_korean_action_text_is_bold_without_a_table(self) -> None:
        result = quest_format.classify_dialogue(
            pd.DataFrame(
                {
                    "header": ["Paimon", "", "Paimon"],
                    "text": ["Hello", "A description", "Bye"],
                }
            ),
            prompt_variables = False,
            prompt_headers = False,
        )

        output = quest_format.format_dialogue(
            result.df,
            "KR",
            target_language = "KR",
        )

        self.assertIn(
            '<p><span style="color: rgb(255, 255, 255);">'
            "<strong>A description</strong></span></p>",
            output,
        )
        self.assertNotIn("<table>", output)

    def test_korean_choice_speaker_is_yellow(self) -> None:
        result = quest_format.classify_dialogue(
            pd.DataFrame(
                {"header": ["Choice", "1. Traveler"], "text": ["", "Option"]}
            ),
            prompt_variables = False,
            prompt_headers = False,
        )

        output = quest_format.format_dialogue(
            result.df,
            "KR",
            target_language = "KR",
        )

        self.assertIn(
            '<span style="color: rgba(255, 214, 128, 1);">Traveler</span>'
            '<span style="color: rgba(255, 255, 255, 0.45);">:</span> '
            '<span style="color: rgba(255, 255, 255, 0.45);">Option</span>',
            output,
        )

    def test_korean_optional_header_is_inside_the_table(self) -> None:
        result = quest_format.classify_dialogue(
            pd.DataFrame(
                {
                    "header": ["Optional Dialogue: Talk to Anastasya", "Anastasya"],
                    "text": ["", "Hello"],
                }
            ),
            prompt_variables = False,
            prompt_headers = False,
        )

        output = quest_format.format_dialogue(
            result.df,
            "KR",
            target_language = "KR",
        )

        table_start = output.index("<table>")
        table_end = output.index("</table>") + len("</table>")
        table = output[table_start:table_end]

        self.assertIn(
            '<span style="color: rgba(255, 214, 128, 1);">'
            "<strong>Optional Dialogue: Talk to Anastasya</strong></span>",
            table,
        )
        self.assertIn(
            '<span style="color: rgba(255, 214, 128, 1);">Anastasya</span>',
            table,
        )

    def test_korean_traveler_speaker_is_white(self) -> None:
        result = quest_format.classify_dialogue(
            pd.DataFrame({"header": ["여행자"], "text": ["Hello"]}),
            prompt_variables = False,
            prompt_headers = False,
        )

        output = quest_format.format_dialogue(
            result.df,
            "KR",
            target_language = "KR",
        )

        self.assertIn(
            '<span style="color: rgb(255, 255, 255);">여행자</span>'
            '<span style="color: rgba(255, 255, 255, 0.45);">:</span> '
            '<span style="color: rgba(255, 255, 255, 0.45);">Hello</span>',
            output,
        )

    def test_japanese_choice_branches_share_one_table(self) -> None:
        result = quest_format.classify_dialogue(
            pd.DataFrame(
                {
                    "header": ["Choice", "1. Traveler", "2. Traveler", "Paimon"],
                    "text": ["", "First option", "Second option", "Response"],
                }
            ),
            prompt_variables = False,
            prompt_headers = False,
        )

        output = quest_format.format_dialogue(
            result.df,
            "RU",
            target_language = "JP",
        )

        self.assertEqual(output.count("<table>"), 1)

        table_start = output.index("<table>")
        table_end = output.index("</table>") + len("</table>")
        choice_table = output[table_start:table_end]

        self.assertEqual(choice_table.count("<tr>"), 2)
        self.assertIn("First option", choice_table)
        self.assertIn("Second option", choice_table)
        self.assertIn("Response", choice_table)

    def test_japanese_choice_response_gets_a_line_break(self) -> None:
        result = quest_format.classify_dialogue(
            pd.DataFrame(
                {
                    "header": ["Choice", "1. Traveler", "Paimon"],
                    "text": ["", "First option", "Response"],
                }
            ),
            prompt_variables = False,
            prompt_headers = False,
        )

        output = quest_format.format_dialogue(
            result.df,
            "RU",
            target_language = "JP",
        )

        self.assertIn("<p></p><p><span", output)

    def test_japanese_gendered_note_is_prepended(self) -> None:
        result = quest_format.classify_dialogue(
            pd.DataFrame({"header": ["Paimon"], "text": ["{M#俺}{F#私}だ"]}),
            target_language = "JP",
            prompt_variables = False,
            prompt_headers = False,
        )

        self.assertTrue(result.has_gendered_content)

        output = quest_format.format_dialogue(
            result.df,
            "RU",
            target_language = "JP",
            show_gendered_note = True,
        )

        self.assertTrue(output.startswith("<p>※主人公に空、蛍"))

        without_note = quest_format.format_dialogue(
            result.df,
            "RU",
            target_language = "JP",
        )

        self.assertNotIn("※主人公に空、蛍", without_note)

    def test_japanese_gender_pairs_use_symbols(self) -> None:
        result = quest_format.classify_dialogue(
            pd.DataFrame({"header": ["Paimon"], "text": ["{M#俺}{F#私}たち"]}),
            target_language = "JP",
            prompt_variables = False,
            prompt_headers = False,
        )

        output = quest_format.format_dialogue(
            result.df,
            "RU",
            target_language = "JP",
        )

        self.assertIn("(♂俺 ♀私)たち", output)

        reversed_output = quest_format.format_dialogue(
            result.df,
            "RU",
            target_language = "JP",
            gender_mode = "F/M",
        )

        self.assertIn("(♀私 ♂俺)たち", reversed_output)

    def test_japanese_nickname_uses_the_player_name_placeholder(self) -> None:
        result = quest_format.classify_dialogue(
            pd.DataFrame({"header": ["Paimon"], "text": ["{NICKNAME}、こんにちは"]}),
            speaker_map = {"Traveler": "旅人"},
            target_language = "JP",
            prompt_variables = True,
            prompt_headers = False,
        )

        self.assertEqual(
            result.defaults["variables"]["{NICKNAME}"],
            "(プレイヤー名)",
        )

    def test_japanese_compound_locations_reverse_the_order(self) -> None:
        result = quest_format.classify_dialogue(
            pd.DataFrame(
                {"header": ["Zapolyarny Palace, Snezhnaya"], "text": [""]}
            ),
            location_map = {
                "Zapolyarny Palace": "ザポリャールヌイ宮殿",
                "Snezhnaya": "スネージナヤ",
            },
            target_language = "JP",
            prompt_headers = True,
        )

        self.assertEqual(
            result.defaults["locations"]["Zapolyarny Palace, Snezhnaya"],
            "スネージナヤ ザポリャールヌイ宮殿",
        )

    def test_prefix_token_collects_the_following_word_as_context(self) -> None:
        result = quest_format.classify_dialogue(
            pd.DataFrame(
                {"header": ["Paimon"], "text": ["{M#俺}{F#私}なら大丈夫。"]}
            ),
            target_language = "JP",
            prompt_variables = True,
            prompt_headers = False,
        )

        self.assertEqual(
            result.variable_suffix_contexts["{M#俺}{F#私}"],
            "なら大丈夫",
        )

    def test_talk_to_header_uses_the_localized_name(self) -> None:
        cases = (
            (
                "DE",
                "Optionaler Dialog: Sprich mit",
                "Anastasia",
                "Optionaler Dialog: Sprich mit Anastasia",
            ),
            (
                "JP",
                "任意の会話：　と話す",
                "アナスタシヤ",
                "任意の会話：アナスタシヤと話す",
            ),
        )

        for language, template, localized_name, expected in cases:
            result = quest_format.classify_dialogue(
                pd.DataFrame(
                    {
                        "header": ["Optional Dialogue: Talk to Anastasya"],
                        "text": [""],
                    }
                ),
                field_label_map = {"Optional Dialogue: Talk to": template},
                speaker_map = {"Anastasya": localized_name},
                target_language = language,
                prompt_headers = True,
            )

            self.assertEqual(
                result.defaults["optional"][
                    "Optional Dialogue: Talk to Anastasya"
                ],
                expected,
            )

    def test_talk_to_header_keeps_trailing_words(self) -> None:
        result = quest_format.classify_dialogue(
            pd.DataFrame(
                {
                    "header": ["Optional Dialogue: Talk to Anastasya again"],
                    "text": [""],
                }
            ),
            field_label_map = {
                "Optional Dialogue: Talk to": "Optionaler Dialog: Sprich mit"
            },
            speaker_map = {"Anastasya": "Anastasia"},
            target_language = "DE",
            prompt_headers = True,
        )

        self.assertEqual(
            result.defaults["optional"][
                "Optional Dialogue: Talk to Anastasya again"
            ],
            "Optionaler Dialog: Sprich mit Anastasia again",
        )

    def test_korean_talk_to_puts_the_name_before_the_particle(self) -> None:
        result = quest_format.classify_dialogue(
            pd.DataFrame(
                {"header": ["Optional Dialogue: Talk to Anastasya"], "text": [""]}
            ),
            field_label_map = {
                "Optional Dialogue: Talk to": "선택: 와/과 대화하기",
            },
            speaker_map = {"Anastasya": "아나스타샤"},
            target_language = "KR",
            prompt_headers = True,
        )

        self.assertEqual(
            result.defaults["optional"]["Optional Dialogue: Talk to Anastasya"],
            "선택: 아나스타샤와/과 대화하기",
        )

    def test_chinese_talk_to_uses_a_space_slot(self) -> None:
        result = quest_format.classify_dialogue(
            pd.DataFrame(
                {"header": ["Optional Dialogue: Talk to Anastasya"], "text": [""]}
            ),
            field_label_map = {
                "Optional Dialogue: Talk to": "可选对话：与 对话",
            },
            speaker_map = {"Anastasya": "安娜丝塔夏"},
            target_language = "CHS",
            prompt_headers = True,
        )

        self.assertEqual(
            result.defaults["optional"]["Optional Dialogue: Talk to Anastasya"],
            "可选对话：与安娜丝塔夏对话",
        )

    def test_talk_to_header_keeps_the_source_name_for_russian(self) -> None:
        result = quest_format.classify_dialogue(
            pd.DataFrame(
                {"header": ["Optional Dialogue: Talk to Anastasya"], "text": [""]}
            ),
            field_label_map = {
                "Optional Dialogue: Talk to": (
                    "Дополнительный диалог: Поговорите с"
                )
            },
            speaker_map = {"Anastasya": "Анастасия"},
            target_language = "RU",
            prompt_headers = True,
        )

        self.assertEqual(
            result.defaults["optional"]["Optional Dialogue: Talk to Anastasya"],
            "Дополнительный диалог: Поговорите с Anastasya",
        )

    def test_gender_modes_change_the_pair_output(self) -> None:
        cases = (
            ("M/F", "Der Berater/Die Beraterin"),
            ("F/M", "Die Beraterin/Der Berater"),
            ("M", "Der Berater"),
            ("F", "Die Beraterin"),
            ("None", "{F#Die Beraterin}{M#Der Berater}"),
        )

        for mode, expected in cases:
            result = quest_format.classify_dialogue(
                pd.DataFrame(
                    {
                        "header": ["Paimon"],
                        "text": ["{F#Die Beraterin}{M#Der Berater}"],
                    }
                ),
                target_language = "DE",
                gender_mode = mode,
                prompt_variables = False,
                prompt_headers = False,
            )

            output = quest_format.format_dialogue(
                result.df,
                "EN",
                target_language = "DE",
                gender_mode = mode,
            )

            self.assertIn(expected, output)

    def test_gender_mode_preferences(self) -> None:
        for language in ("DE", "RU", "FR", "ES", "JP", None):
            self.assertEqual(
                quest_format.gender_mode_for(language),
                "M/F",
            )

        self.assertEqual(quest_format.gender_mode_for("IT"), "M")

    def test_female_first_mode_keeps_full_forms(self) -> None:
        result = quest_format.classify_dialogue(
            pd.DataFrame(
                {
                    "header": ["Paimon"],
                    "text": ["{M#Путешественник}{F#Путешественница}"],
                }
            ),
            target_language = "RU",
            gender_mode = "F/M",
            prompt_variables = False,
            prompt_headers = False,
        )

        output = quest_format.format_dialogue(
            result.df,
            "EN",
            target_language = "RU",
            gender_mode = "F/M",
        )

        self.assertIn("Путешественница/Путешественник", output)

    def test_male_only_mode_drops_the_female_suffix(self) -> None:
        result = quest_format.classify_dialogue(
            pd.DataFrame({"header": ["Paimon"], "text": ["прибыл{F#а}"]}),
            target_language = "RU",
            gender_mode = "M",
            prompt_variables = True,
            prompt_headers = False,
        )

        self.assertEqual(result.defaults["variables"]["{F#а}"], "")

        output = quest_format.format_dialogue(
            result.df,
            "EN",
            target_language = "RU",
            gender_mode = "M",
        )

        self.assertIn("прибыл", output)
        self.assertNotIn("{F#", output)

    def test_female_only_mode_keeps_the_female_suffix(self) -> None:
        result = quest_format.classify_dialogue(
            pd.DataFrame({"header": ["Paimon"], "text": ["прибыл{F#а}"]}),
            target_language = "RU",
            gender_mode = "F",
            prompt_variables = True,
            prompt_headers = False,
        )

        self.assertEqual(result.defaults["variables"]["{F#а}"], "а")

        output = quest_format.format_dialogue(
            result.df,
            "EN",
            target_language = "RU",
            gender_mode = "F",
        )

        self.assertIn("прибыла", output)

    def test_gender_prompts_are_applied_when_unchanged(self) -> None:
        result = quest_format.classify_dialogue(
            pd.DataFrame({"header": ["Paimon"], "text": ["прибыл{F#а}"]}),
            target_language = "RU",
            prompt_variables = True,
            prompt_headers = False,
        )

        quest_format.apply_translations(result.df, {}, result.defaults)

        output = quest_format.format_dialogue(
            result.df,
            "EN",
            target_language = "RU",
        )

        self.assertIn("прибыл/а", output)

    def test_german_multiple_gender_pairs_are_converted(self) -> None:
        result = quest_format.classify_dialogue(
            pd.DataFrame(
                {
                    "header": ["Paimon"],
                    "text": ["{F#A}{M#B}{F#C}{M#D}"],
                }
            ),
            target_language = "DE",
            prompt_variables = True,
            prompt_headers = False,
        )

        self.assertEqual(
            result.defaults["variables"]["{F#A}{M#B}{F#C}{M#D}"],
            "B/A/D/C",
        )

        output = quest_format.format_dialogue(
            result.df,
            "EN",
            target_language = "DE",
        )

        self.assertIn("B/A/D/C", output)

    def test_german_gender_pairs_in_objectives(self) -> None:
        objective_df = pd.DataFrame(
            {
                "header": ["1"],
                "text": ["{F#Die Beraterin}{M#Der Berater}"],
            }
        )

        output = quest_format.format_objective(
            objective_df,
            target_language = "DE",
        )

        self.assertIn("Der Berater/Die Beraterin", output)

    def test_russian_gender_pairs_use_the_shared_base(self) -> None:
        result = quest_format.classify_dialogue(
            pd.DataFrame(
                {
                    "header": ["Paimon"],
                    "text": ["{M#Путешественник}{F#Путешественница}"],
                }
            ),
            target_language = "RU",
            prompt_variables = False,
            prompt_headers = False,
        )

        output = quest_format.format_dialogue(
            result.df,
            "EN",
            target_language = "RU",
        )

        self.assertIn("Путешественник/ца", output)
        self.assertNotIn("{M#", output)
        self.assertNotIn("{F#", output)

    def test_russian_gender_pairs_reverse_order_use_the_shared_base(self) -> None:
        result = quest_format.classify_dialogue(
            pd.DataFrame(
                {
                    "header": ["Paimon"],
                    "text": ["{F#Путешественница}{M#Путешественник}"],
                }
            ),
            target_language = "RU",
            prompt_variables = False,
            prompt_headers = False,
        )

        output = quest_format.format_dialogue(
            result.df,
            "EN",
            target_language = "RU",
        )

        self.assertIn("Путешественник/ца", output)

    def test_russian_gender_pairs_fall_back_without_a_shared_base(self) -> None:
        result = quest_format.classify_dialogue(
            pd.DataFrame(
                {
                    "header": ["Paimon"],
                    "text": [
                        "{M#его}{F#её} и {M#ему}{F#ей}, "
                        "{M#силён}{F#сильна}, {M#него}{F#неё}"
                    ],
                }
            ),
            target_language = "RU",
            prompt_variables = False,
            prompt_headers = False,
        )

        output = quest_format.format_dialogue(
            result.df,
            "EN",
            target_language = "RU",
        )

        self.assertIn("его/её", output)
        self.assertIn("ему/ей", output)
        self.assertIn("силён/сильна", output)
        self.assertIn("него/неё", output)

    def test_russian_short_pronoun_pairs_are_shortened(self) -> None:
        result = quest_format.classify_dialogue(
            pd.DataFrame(
                {"header": ["Paimon"], "text": ["{M#он}{F#она} {M#был}{F#была}"]}
            ),
            target_language = "RU",
            prompt_variables = False,
            prompt_headers = False,
        )

        output = quest_format.format_dialogue(
            result.df,
            "EN",
            target_language = "RU",
        )

        self.assertIn("он/а", output)
        self.assertIn("был/а", output)

    def test_french_gender_pairs_use_the_shared_base(self) -> None:
        result = quest_format.classify_dialogue(
            pd.DataFrame(
                {
                    "header": ["Paimon"],
                    "text": ["{M#acteur}{F#actrice} et {M#un}{F#une}"],
                }
            ),
            target_language = "FR",
            prompt_variables = False,
            prompt_headers = False,
        )

        output = quest_format.format_dialogue(
            result.df,
            "EN",
            target_language = "FR",
        )

        self.assertIn("acteur/rice", output)
        self.assertIn("un/e", output)

    def test_spanish_gender_pairs_use_the_shared_base(self) -> None:
        result = quest_format.classify_dialogue(
            pd.DataFrame(
                {
                    "header": ["Paimon"],
                    "text": ["{M#amigo}{F#amiga} y {M#él}{F#ella}"],
                }
            ),
            target_language = "ES",
            prompt_variables = False,
            prompt_headers = False,
        )

        output = quest_format.format_dialogue(
            result.df,
            "EN",
            target_language = "ES",
        )

        self.assertIn("amigo/a", output)
        self.assertIn("él/ella", output)

    def test_shortened_gender_pairs_in_objectives(self) -> None:
        objective_df = pd.DataFrame(
            {
                "header": ["1"],
                "text": ["{M#Путешественник}{F#Путешественница}"],
            }
        )

        output = quest_format.format_objective(
            objective_df,
            target_language = "RU",
        )

        self.assertIn("Путешественник/ца", output)

    def test_lone_female_suffix_is_merged_into_the_previous_word(self) -> None:
        cases = (
            ("RU", "прибыл{F#а}", "{F#а}", "/а", "прибыл/а"),
            ("FR", "privé{F#e}", "{F#e}", "/e", "privé/e"),
            ("ES", "asesor{F#a}", "{F#a}", "/a", "asesor/a"),
        )

        for language, text, token, default, expected in cases:
            result = quest_format.classify_dialogue(
                pd.DataFrame({"header": ["Paimon"], "text": [text]}),
                target_language = language,
                prompt_variables = True,
                prompt_headers = False,
            )

            self.assertEqual(
                result.defaults["variables"][token],
                default,
            )

            output = quest_format.format_dialogue(
                result.df,
                "EN",
                target_language = language,
            )

            self.assertIn(expected, output)
            self.assertNotIn("{F#", output)

    def test_detached_lone_female_token_stays_a_prompt(self) -> None:
        result = quest_format.classify_dialogue(
            pd.DataFrame({"header": ["Paimon"], "text": ["прибыл {F#а}"]}),
            target_language = "RU",
            prompt_variables = True,
            prompt_headers = False,
        )

        self.assertIn("{F#а}", result.prompts["variables"])

        output = quest_format.format_dialogue(
            result.df,
            "EN",
            target_language = "RU",
        )

        self.assertIn("{F#а}", output)

    def test_long_lone_female_token_stays_a_prompt(self) -> None:
        result = quest_format.classify_dialogue(
            pd.DataFrame(
                {"header": ["Paimon"], "text": ["ты он,{F#оставившая}"]}
            ),
            target_language = "RU",
            prompt_variables = True,
            prompt_headers = False,
        )

        self.assertIn("{F#оставившая}", result.prompts["variables"])

    def test_german_lone_female_token_stays_a_prompt(self) -> None:
        result = quest_format.classify_dialogue(
            pd.DataFrame({"header": ["Paimon"], "text": ["Der {F#Beraterin}"]}),
            target_language = "DE",
            prompt_variables = True,
            prompt_headers = False,
        )

        self.assertIn("{F#Beraterin}", result.prompts["variables"])

    def test_prepare_text_handles_non_breaking_spaces_and_ruby(self) -> None:
        self.assertEqual(
            quest_format.prepare_text("A{NON_BREAK_SPACE}B"),
            "A&nbsp;B",
        )

        ruby_output = quest_format.prepare_text("pre{RUBY#[D]ruby text}base")
        self.assertIn('<custom-ruby data-ruby="prebase|ruby text">', ruby_output)

    def test_color_tags_are_converted_to_html_spans(self) -> None:
        self.assertEqual(
            quest_format.prepare_text("<color=#00E1FFFF>Keys</color>"),
            '<span style="color: rgb(0, 225, 255);">Keys</span>',
        )
        self.assertIn(
            "rgb(255, 0, 0)",
            quest_format.prepare_text("<color=#FF0000>Red</color>"),
        )
        self.assertIn(
            "rgba(255, 0, 0, 0.502)",
            quest_format.prepare_text("<color=#FF000080>Half</color>"),
        )

    def test_color_tags_survive_dialogue_and_objective_formatting(self) -> None:
        result = quest_format.classify_dialogue(
            pd.DataFrame(
                {
                    "header": ["Paimon"],
                    "text": ["<color=#00E1FFFF>Keys</color>"],
                }
            ),
            prompt_variables = False,
            prompt_headers = False,
        )

        dialogue_output = quest_format.format_dialogue(result.df, "EN")
        self.assertIn(
            '<span style="color: rgb(0, 225, 255);">Keys</span>',
            dialogue_output,
        )
        self.assertNotIn("<color=", dialogue_output)

        objective_output = quest_format.format_objective(
            pd.DataFrame(
                {
                    "header": ["1"],
                    "text": ["<color=#00E1FFFF>Keys</color>"],
                }
            )
        )
        self.assertIn(
            '<span style="color: rgb(0, 225, 255);">Keys</span>',
            objective_output,
        )

    def test_text_after_colored_spans_keeps_base_color(self) -> None:
        result = quest_format.classify_dialogue(
            pd.DataFrame(
                {
                    "header": ["", "", "Paimon"],
                    "text": [
                        "Obtain the <color=#00E1FFFF>Keys</color> and go",
                        "<color=#00E1FFFF>Keys</color> and go",
                        "Obtain the <color=#00E1FFFF>Keys</color> and go",
                    ],
                }
            ),
            prompt_variables = False,
            prompt_headers = False,
        )

        english_output = quest_format.format_dialogue(result.df, "EN")
        russian_output = quest_format.format_dialogue(result.df, "RU")

        for output in (english_output, russian_output):
            # Action rows keep the explicit white base color after a color.
            self.assertIn(
                '</span><span style="color: rgb(255, 255, 255);"> and go</span>',
                output,
            )
            # The base color must not wrap the colored span itself.
            self.assertNotIn(
                'color: rgb(255, 255, 255);">Obtain the '
                '<span style="color: rgb(0, 225, 255)',
                output,
            )
            # Dialogue body text keeps the editor default color.
            self.assertIn(
                '<span style="color: rgb(0, 225, 255);">Keys</span> and go</p>',
                output,
            )

    def test_color_tags_and_german_gender_pairs_compose(self) -> None:
        output = quest_format.prepare_text(
            "<color=#00E1FFFF>{F#Die Beraterin}{M#Der Berater}</color>",
            "DE",
        )

        self.assertIn("Der Berater/Die Beraterin", output)
        self.assertIn("rgb(0, 225, 255)", output)


if __name__ == "__main__":
    unittest.main()
