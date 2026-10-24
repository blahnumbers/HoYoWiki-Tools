import hashlib
import json

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from utils import quest_format
from utils.quest_data import LANGUAGE_LABELS, QuestData, QuestDataError, parse_quest_file


PENDING_KEY = "quest_formatter_pending"
OUTPUT_KEY = "quest_formatter_html"
TARGET_LANGUAGE_KEY = "quest_formatter_target_language"
JUST_FORMATTED_KEY = "quest_formatter_just_formatted"


@st.cache_data(show_spinner=False)
def _parse_uploaded(file_bytes: bytes, file_name: str) -> QuestData:
    return parse_quest_file(file_bytes, file_name)


def _translation_key(target: str, group: str, original: str) -> str:
    digest = hashlib.sha1(original.encode("utf-8")).hexdigest()[:16]
    return "qf_translation_%s_%s_%s" % (target, group, digest)


def _set_output(formatted_html: str) -> None:
    st.session_state[OUTPUT_KEY] = formatted_html
    st.session_state[JUST_FORMATTED_KEY] = True


def _handle_dialogue(
    dialogue_df: pd.DataFrame,
    preset: str,
    speaker_map: dict[str, str] | None = None,
    target: str = "EN",
    prompt_variables: bool = True,
    prompt_headers: bool = True,
) -> None:
    result = quest_format.classify_dialogue(
        dialogue_df,
        speaker_map = speaker_map,
        prompt_variables = prompt_variables,
        prompt_headers = prompt_headers,
    )

    if result.has_prompts:
        st.session_state[PENDING_KEY] = {
            "result": result,
            "preset": preset,
            "target": target,
        }
        translation_dialog()
    else:
        _set_output(quest_format.format_dialogue(result.df, preset))


@st.dialog("Content Replacement", width = "large")
def translation_dialog() -> None:
    pending = st.session_state.get(PENDING_KEY)

    if not pending:
        st.warning("Nothing to format. Please try again.")
        return

    result: quest_format.ClassificationResult = pending["result"]
    target = pending["target"]
    preset = pending["preset"]

    st.markdown(
        "Review the fields below. Values are grouped so speaker names stay separate "
        "from conditional strings and other headers. Fields left unchanged keep their "
        "original English value. Press `Format` when you are done."
    )

    if result.auto_localized:
        pairs = ", ".join(
            "%s → %s" % (english_name, localized_name)
            for english_name, localized_name in result.auto_localized.items()
        )
        st.caption("Auto-localized from the file's `Special Speaker` section: %s" % pairs)

    entered_values: dict[str, dict[str, str]] = {}

    for group_key, group_title, group_description in quest_format.PROMPT_GROUPS:
        items = result.prompts.get(group_key) or {}

        if not items:
            continue

        st.subheader("%s (%d)" % (group_title, len(items)))
        st.caption(group_description)

        group_values: dict[str, str] = {}

        for original, count in items.items():
            group_values[original] = st.text_input(
                original,
                value = original,
                key = _translation_key(target, group_key, original),
                help = ("Used in %d rows" % count) if count > 1 else None,
            )

        entered_values[group_key] = group_values

    st.divider()

    if st.button("Format", type = "primary", key = "quest_formatter_apply_translations"):
        with st.spinner("Formatting...", show_time = True):
            quest_format.apply_translations(result.df, entered_values)
            _set_output(quest_format.format_dialogue(result.df, preset))
            st.session_state[PENDING_KEY] = None

        st.rerun()


def _render_import_mode() -> None:
    imported_file = st.file_uploader(
        "Upload Quest Data File",
        type = ["xlsx", "csv"],
        key = "quest_formatter_file",
        help = (
            "The file must follow the Quest Team localization sheet format: keys in "
            "column A, language codes in the first row, and a Dialogue section with a "
            "Special Speaker block."
        ),
    )

    if imported_file is None:
        st.info(
            "Upload the quest's `.xlsx` or `.csv` localization file to get started. "
            "You can also switch back to `Paste Text` above."
        )
        st.stop()

    try:
        quest = _parse_uploaded(imported_file.getvalue(), imported_file.name)
    except QuestDataError as error:
        st.error("Could not read this file: %s" % error)
        st.stop()

    if st.session_state.get(TARGET_LANGUAGE_KEY) not in quest.languages:
        st.session_state[TARGET_LANGUAGE_KEY] = (
            "EN" if "EN" in quest.languages else quest.languages[0]
        )

    target_language = st.selectbox(
        "Target Language:",
        quest.languages,
        format_func = lambda code: "%s — %s" % (code, LANGUAGE_LABELS.get(code, code)),
        key = TARGET_LANGUAGE_KEY,
    )

    title = quest.title_for(target_language)

    if title:
        st.markdown("**%s**" % title)

    st.caption(
        "%d objectives · %d dialogue rows · %d special speaker name(s)"
        % (
            len(quest.objectives),
            len(quest.dialogue_rows),
            len(quest.special_speakers),
        )
    )

    missing_translations = quest.missing_translations(target_language)

    if missing_translations:
        missing_rows = ", ".join(entry["row"] for entry in missing_translations[:10])
        row_suffix = ", ..." if len(missing_translations) > 10 else ""

        st.warning(
            "%d dialogue row(s) have no %s text (sheet rows %s%s). They will be "
            "formatted as empty lines."
            % (
                len(missing_translations),
                target_language,
                missing_rows,
                row_suffix,
            ),
            icon = ":material/warning:",
        )

    missing_special_speakers = quest.missing_special_translations(target_language)

    if missing_special_speakers:
        missing_names = ", ".join(
            entry["key"] for entry in missing_special_speakers[:10]
        )

        st.info(
            "These Special Speaker names have no %s translation and will be requested "
            "manually: %s" % (target_language, missing_names),
            icon = ":material/translate:",
        )

    with st.expander("Preview Imported Data"):
        special_speaker_pairs = quest.special_speaker_pairs(target_language)

        if special_speaker_pairs:
            st.markdown("**Special Speaker**")
            st.dataframe(
                pd.DataFrame(
                    special_speaker_pairs,
                    columns = ["English Key", "Localized Name"],
                ),
                hide_index = True,
            )

        st.markdown("**Objectives**")
        st.dataframe(
            quest.objective_dataframe(target_language).head(25),
            hide_index = True,
        )

        st.markdown("**Dialogue**")
        st.dataframe(
            quest.dialogue_dataframe(target_language).head(50),
            hide_index = True,
            height = 360,
        )

    preset = "RU" if target_language == "RU" else "EN"

    st.caption("Output markup preset: `%s` (auto-selected from the target language)" % preset)

    objective_tab, dialogue_tab = st.tabs(["Objective", "Dialogue"])

    with objective_tab:
        objective_df = quest.objective_dataframe(target_language)

        st.dataframe(objective_df, hide_index = True)

        if st.button(
            "Format Objectives",
            type = "primary",
            key = "format_objective_button_import",
        ):
            _set_output(quest_format.format_objective(objective_df))

    with dialogue_tab:
        dialogue_df = quest.dialogue_dataframe(target_language)

        st.dataframe(dialogue_df, hide_index = True, height = 360)

        if st.button(
            "Format Dialogue",
            type = "primary",
            key = "format_dialogue_button_import",
        ):
            _handle_dialogue(
                dialogue_df,
                preset = preset,
                speaker_map = quest.speaker_map(target_language),
                target = target_language,
                prompt_variables = False,
                prompt_headers = target_language != "EN",
            )


def _render_paste_mode() -> None:
    st.segmented_control(
        "Language Format Preset:",
        ["EN", "RU"],
        default = "EN",
        key = "quest_formatter_preset",
    )

    default_table_data = {"header": [""], "text": [""]}
    default_table_df = pd.DataFrame(default_table_data)

    default_table_config = {
        "header": st.column_config.TextColumn("Row Header", width = "small"),
        "text": st.column_config.TextColumn("Text", width = "large"),
    }

    objective_tab, dialogue_tab = st.tabs(["Objective", "Dialogue"])

    with objective_tab:
        objective_df = st.data_editor(
            default_table_df,
            column_config = default_table_config,
            num_rows = "dynamic",
            key = "objective_data",
        )

        if st.button("Format", type = "primary", key = "format_objective_button"):
            objective_df.fillna("", inplace = True)
            _set_output(quest_format.format_objective(objective_df))

    with dialogue_tab:
        dialogue_df = st.data_editor(
            default_table_df,
            column_config = default_table_config,
            num_rows = "dynamic",
            key = "dialogue_data",
        )

        if st.button("Format", type = "primary", key = "format_dialogue_button"):
            dialogue_df.fillna("", inplace = True)
            preset = st.session_state.get("quest_formatter_preset") or "EN"

            _handle_dialogue(
                dialogue_df,
                preset = preset,
                speaker_map = None,
                target = preset,
                prompt_variables = True,
                prompt_headers = True,
            )


def _render_output() -> None:
    formatted_html = st.session_state.get(OUTPUT_KEY)

    if st.session_state.pop(JUST_FORMATTED_KEY, False):
        st.toast(
            "Formatted text is ready. Scroll down to copy it.",
            icon = ":material/check_circle:",
        )

    if formatted_html is None:
        return

    sanitized_html = json.dumps(formatted_html).replace("</", "<\\/")

    copy_html_button = """
    <button style="font-size: 14px; padding: 10px 10px; border-radius: 8px;" onclick="copyHTML()">Copy Formatted Text</button>
    <script>
    async function copyHTML() {
        const html = %s;
        try {
            await navigator.clipboard.write([
                new ClipboardItem({
                    "text/html": new Blob([html], { type: "text/html" })
                })
            ]);
            alert("Copy successful!");
        } catch (err) {
            alert("Failed to copy: " + err);
        }
    }
    </script>
    """ % sanitized_html

    components.html(copy_html_button, height = 60)


if OUTPUT_KEY not in st.session_state:
    st.session_state[OUTPUT_KEY] = None

if PENDING_KEY not in st.session_state:
    st.session_state[PENDING_KEY] = None

st.title("Quest Formatter")

st.markdown("""
Takes text data from a localization sheet made by the Quest Team and formats it so it is ready to be pasted directly into the WET. You can paste the text columns or import the original `.xlsx` / `.csv` data file.

Tested for: :green-badge[:material/check: Genshin Impact]
""")

with st.expander("How to Use"):
    st.subheader("To Import a Quest Data File")

    st.markdown("""
    1. Set `Input Method` to `Import File`
    2. Upload the quest's localization file (`.xlsx` or `.csv`)
    3. Pick the target language from the languages found in the file
    4. Open the `Dialogue` tab and click on the `Format Dialogue` button
    5. If prompted, fill in the Content Replacement form (speaker names are kept separate from conditional strings)
    6. Click on the `Copy Formatted Text` button that just appeared
    7. Paste the formatted text into the WET
    """)

    st.caption("""
    Speaker names listed in the file's `Special Speaker` section are localized automatically. Conditional strings, optional sections, locations and any names not present there still require manual input.
    """)

    st.subheader("To Format Quest Objective (Paste)")

    st.markdown("""
    1. Open the localization sheet of the quest you're working on
    2. Locate the `Quest Objective` header
    3. Starting from the row just below the header, select and copy the first column of quest objective data as well as your language's one (non-numbered quest objectives will automatically be ignored)
    4. Paste your selection in the page's interactive table, under the `Objective` tab
    5. Click on the `Format` button
    6. Click on the `Copy Formatted Text` button that just appeared
    7. Paste the formatted text into the WET
    """)

    st.subheader("To Format Quest Dialogue (Paste)")

    st.markdown("""
    1. Open the localization sheet of the quest you're working on
    2. Locate the `Dialogue` header
    3. Starting from the row just below it, select and copy the first column of dialogue data as well as your language's one (make sure to fill in the Missing Translation fields beforehand, else exclude them from your selection)
    4. Paste your selection in the page's interactive table, under the `Dialogue` tab
    5. Click on the `Format` button
    6. If prompted, fill in the Content Replacement form
    7. Click on the `Copy Formatted Text` button that just appeared
    8. Paste the formatted text into the WET
    """)

st.divider()

input_method = st.segmented_control(
    "Input Method:",
    ["Paste Text", "Import File"],
    default = "Paste Text",
    key = "quest_formatter_input_method",
)

if input_method == "Import File":
    _render_import_mode()
else:
    _render_paste_mode()

_render_output()
