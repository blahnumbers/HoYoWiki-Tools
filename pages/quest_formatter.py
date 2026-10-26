import hashlib

import pandas as pd
import streamlit as st

from utils import quest_format
from utils.copy_component import copy_formatted_text
from utils.quest_data import LANGUAGE_LABELS, QuestData, QuestDataError, parse_quest_file


PENDING_KEY = "quest_formatter_pending"
OUTPUT_KEY = "quest_formatter_html"
TARGET_LANGUAGE_KEY = "quest_formatter_target_language"
JUST_FORMATTED_KEY = "quest_formatter_just_formatted"
COPY_NONCE_KEY = "quest_formatter_last_copy_nonce"


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
    location_map: dict[str, str] | None = None,
    field_label_map: dict[str, str] | None = None,
    location_keys: set[str] | None = None,
    target: str = "EN",
    prompt_variables: bool = True,
    prompt_headers: bool = True,
) -> None:
    result = quest_format.classify_dialogue(
        dialogue_df,
        speaker_map = speaker_map,
        location_map = location_map,
        field_label_map = field_label_map,
        location_keys = location_keys,
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
        quest_format.apply_translations(result.df, {}, result.defaults)
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
        "from conditional strings and other headers. Fields left unchanged keep the "
        "value shown. Press `Format` when you are done."
    )

    has_prefilled_values = any(
        value and value != key
        for group in ("speakers", "locations", "labels", "variables")
        for key, value in (result.defaults.get(group) or {}).items()
    )

    if has_prefilled_values:
        st.caption(
            "Values provided by the file's `Translation` section are prefilled "
            "below and can be adjusted before formatting."
        )

    entered_values: dict[str, dict[str, str]] = {}

    for group_key, group_title, group_description in quest_format.PROMPT_GROUPS:
        items = result.prompts.get(group_key) or {}

        if not items:
            continue

        st.subheader("%s (%d)" % (group_title, len(items)))
        st.caption(group_description)

        group_values: dict[str, str] = {}
        group_defaults = result.defaults.get(group_key) or {}

        for original, count in items.items():
            label = original
            help_parts = []

            if count > 1:
                help_parts.append("Used in %d rows" % count)

            if group_key == "variables":
                context = result.variable_contexts.get(original, "")

                if context:
                    label = "%s  ·  %s" % (context, original)

                help_parts.append("Only the bracketed placeholder is editable.")

            group_values[original] = st.text_input(
                label,
                value = group_defaults.get(original, original),
                key = _translation_key(target, group_key, original),
                help = " ".join(help_parts) or None,
            )

        entered_values[group_key] = group_values

    st.divider()

    if st.button("Format", type = "primary", key = "quest_formatter_apply_translations"):
        with st.spinner("Formatting...", show_time = True):
            quest_format.apply_translations(result.df, entered_values, result.defaults)
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
            "column A, language codes in the first row, a Dialogue section, and an "
            "optional Translation section with Special Speaker, Location and/or "
            "Field Label blocks."
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

    file_token = hashlib.sha1(imported_file.getvalue()).hexdigest()[:10]

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
        "%d objectives · %d dialogue rows · %d special speaker(s) · "
        "%d location(s) · %d field label(s)"
        % (
            len(quest.objectives),
            len(quest.dialogue_rows),
            len(quest.special_speakers),
            len(quest.locations),
            len(quest.field_labels),
        )
    )

    if not quest.objectives:
        st.warning(
            "No quest objectives were found in this file. This usually means the "
            "`Quest Objectives` header is missing or misspelled (for example, "
            "`Quest objective`). Please check the source sheet — quests are "
            "generally expected to have at least one objective.",
            icon = ":material/warning:",
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

    missing_segments = (
        ("Special Speaker", quest.missing_special_translations(target_language)),
        ("Location", quest.missing_location_translations(target_language)),
        ("Field Label", quest.missing_field_label_translations(target_language)),
    )
    missing_parts = []

    for segment_name, entries in missing_segments:
        if entries:
            missing_keys = ", ".join(entry["key"] for entry in entries[:10])
            missing_parts.append("%s: %s" % (segment_name, missing_keys))

    if missing_parts:
        st.info(
            "These Translation entries have no %s text and will be requested "
            "manually: %s" % (target_language, "; ".join(missing_parts)),
            icon = ":material/translate:",
        )

    with st.expander("Preview Imported Data"):
        translation_tables = (
            ("Special Speaker", quest.special_speaker_pairs(target_language)),
            ("Location", quest.location_pairs(target_language)),
            ("Field Label", quest.field_label_pairs(target_language)),
        )

        for segment_name, pairs in translation_tables:
            if pairs:
                st.markdown("**%s**" % segment_name)
                st.dataframe(
                    pd.DataFrame(
                        pairs,
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
        preview_dialogue_df = (
            quest.dialogue_dataframe(target_language)
            .head(50)
            .rename(columns = {"bold": "Bold"})
        )
        st.dataframe(
            preview_dialogue_df,
            hide_index = True,
            height = 360,
        )

    preset = "RU" if target_language == "RU" else "EN"

    st.caption("Output markup preset: `%s` (auto-selected from the target language)" % preset)

    import_table_config = {
        "header": st.column_config.TextColumn("Row Header", width = "small"),
        "text": st.column_config.TextColumn("Text", width = "large"),
    }

    dialogue_table_config = {
        **import_table_config,
        "bold": None,
    }

    objective_tab, dialogue_tab = st.tabs(["Objective", "Dialogue"])

    with objective_tab:
        st.caption("Imported data is editable. Changes made here are used when formatting.")

        objective_df = st.data_editor(
            quest.objective_dataframe(target_language),
            column_config = import_table_config,
            num_rows = "dynamic",
            hide_index = True,
            key = "import_objective_data_%s_%s" % (file_token, target_language),
        )

        if st.button(
            "Format Objectives",
            type = "primary",
            key = "format_objective_button_import",
        ):
            objective_df.fillna("", inplace = True)
            _set_output(quest_format.format_objective(objective_df))

    with dialogue_tab:
        st.caption("Imported data is editable. Changes made here are used when formatting.")

        dialogue_df = st.data_editor(
            quest.dialogue_dataframe(target_language),
            column_config = dialogue_table_config,
            num_rows = "dynamic",
            hide_index = True,
            height = 360,
            key = "import_dialogue_data_%s_%s" % (file_token, target_language),
        )

        if st.button(
            "Format Dialogue",
            type = "primary",
            key = "format_dialogue_button_import",
        ):
            dialogue_df.fillna("", inplace = True)

            _handle_dialogue(
                dialogue_df,
                preset = preset,
                speaker_map = quest.speaker_map(target_language),
                location_map = quest.location_map(target_language),
                field_label_map = quest.field_label_map(target_language),
                location_keys = quest.location_keys(),
                target = target_language,
                prompt_variables = True,
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

    copy_result = copy_formatted_text(
        html = formatted_html,
        key = "quest_formatter_copy",
    )

    if copy_result:
        nonce = copy_result.get("nonce")

        if nonce and nonce != st.session_state.get(COPY_NONCE_KEY):
            st.session_state[COPY_NONCE_KEY] = nonce

            if copy_result.get("copied"):
                st.toast(
                    "Formatted text copied.",
                    icon = ":material/check_circle:",
                )
            else:
                st.toast(
                    "Could not copy formatted text: %s"
                    % copy_result.get("error", "unknown error"),
                    icon = ":material/error:",
                )


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
    Values provided in the file's `Translation` section (`Special Speaker`, `Location`, `Field Label`) are prefilled automatically and can still be adjusted. Conditional strings, optional sections, and any missing values require manual input.
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
