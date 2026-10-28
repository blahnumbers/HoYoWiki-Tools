import hashlib
import json
import urllib.request

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from utils import quest_format
from utils.copy_component import copy_formatted_text
from utils.knowledge_base import (
    KNOWLEDGE_BASE_EXPORT_URL,
    KNOWLEDGE_BASE_VIEW_URL,
    KnowledgeBase,
    parse_knowledge_base,
)
from utils.quest_data import (
    LANGUAGE_LABELS,
    QuestData,
    QuestDataError,
    normalize_name,
    parse_quest_file,
)


PENDING_KEY = "quest_formatter_pending"
OUTPUT_KEY = "quest_formatter_html"
TARGET_LANGUAGE_KEY = "quest_formatter_target_language"
JUST_FORMATTED_KEY = "quest_formatter_just_formatted"
COPY_NONCE_KEY = "quest_formatter_last_copy_nonce"

DIALOG_WIDTH_CSS = """
<style>
div[data-testid="stDialog"] div[role="dialog"] {
    width: 100% !important;
    max-width: 1000px !important;
}
</style>
"""

INPUT_CONTEXT_SCRIPT = """
<script>
(function() {
    const items = %s;
    const doc = window.parent.document;
    let attempts = 0;

    function applyAll() {
        let pending = 0;

        for (const item of items) {
            const container = doc.querySelector('[class*="st-key-' + item.key + '"]');
            const input = container ? container.querySelector('input') : null;

            if (!input) {
                pending += 1;
                continue;
            }

            const root = input.closest('[data-testid="stTextInputRootElement"]');

            if (!root) {
                pending += 1;
                continue;
            }

            root.querySelectorAll('.input-context').forEach(el => el.remove());

            const base = root.querySelector('[data-baseweb="base-input"]');

            if (!base) {
                pending += 1;
                continue;
            }

            const context = doc.createElement('div');
            context.className = 'input-context';
            context.textContent = item.context;
            context.style.alignSelf = 'stretch';
            context.style.color = 'rgba(250, 250, 250, 0.6)';
            context.style.paddingLeft = '8px';
            context.style.whiteSpace = 'nowrap';
            context.style.display = 'flex';
            context.style.alignItems = 'center';
            context.style.fontSize = '1rem';
            context.style.flex = '0 0 auto';
            context.style.cursor = 'text';
            context.addEventListener('click', () => input.focus());
            input.style.flex = '1';

            base.insertBefore(context, input);
        }

        if (pending > 0 && attempts < 40) {
            attempts += 1;
            setTimeout(applyAll, 50);
        }
    }

    applyAll();
})();
</script>
"""


@st.cache_data(show_spinner=False)
def _parse_uploaded(file_bytes: bytes, file_name: str) -> QuestData:
    return parse_quest_file(file_bytes, file_name)


def _translation_key(target: str, group: str, original: str) -> str:
    digest = hashlib.sha1(original.encode("utf-8")).hexdigest()[:16]
    return "qf_translation_%s_%s_%s" % (target, group, digest)


def _set_output(formatted_html: str) -> None:
    st.session_state[OUTPUT_KEY] = formatted_html
    st.session_state[JUST_FORMATTED_KEY] = True


GENDER_MODE_HELP = (
    "How gendered placeholders such as {M#...}{F#...} are written. "
    "`M/F` keeps the masculine form first, `F/M` the feminine form first, "
    "`M` and `F` keep only one form, and `None` leaves the placeholders for "
    "manual input."
)


def _render_gender_mode(language: str) -> str:
    preferred = quest_format.gender_mode_for(language)

    mode = st.segmented_control(
        "Gendered content handling:",
        list(quest_format.GENDER_MODES),
        default = preferred,
        key = "quest_formatter_gender_mode_%s" % language,
        help = GENDER_MODE_HELP,
    )

    return mode or preferred


@st.cache_data(ttl = 3600, show_spinner = "Loading knowledge base…")
def _fetch_knowledge_base() -> tuple[KnowledgeBase | None, str | None]:
    try:
        with urllib.request.urlopen(
            KNOWLEDGE_BASE_EXPORT_URL,
            timeout = 25,
        ) as response:
            payload = response.read()
    except Exception as error:
        return None, str(error)

    try:
        return parse_knowledge_base(payload), None
    except Exception as error:
        return None, str(error)


def _render_knowledge_base_control() -> KnowledgeBase | None:
    enabled = st.toggle(
        "Use knowledge base",
        value = True,
        key = "quest_formatter_use_knowledge_base",
        help = (
            "Prefill speaker names, locations and field labels from the "
            "shared glossary spreadsheet."
        ),
    )

    if not enabled:
        return None

    knowledge_base, error = _fetch_knowledge_base()

    if knowledge_base is None:
        st.warning(
            "Could not load the knowledge base: %s" % (error or "unknown error"),
            icon = ":material/warning:",
        )

        if st.button(
            "Retry knowledge base",
            key = "quest_formatter_retry_knowledge_base",
        ):
            _fetch_knowledge_base.clear()
            st.rerun()

        return None

    counts = knowledge_base.counts

    caption_column, refresh_column = st.columns([6, 1])

    with caption_column:
        st.caption(
            "Knowledge base: %s locations · %s names · %s special words · "
            "%s quests · %s materials · [open the sheet](%s)"
            % (
                format(counts["locations"], ","),
                format(counts["names"], ","),
                format(counts["special_words"], ","),
                format(counts["quests"], ","),
                format(counts["materials"], ","),
                KNOWLEDGE_BASE_VIEW_URL,
            )
        )

    with refresh_column:
        if st.button(
            "Refresh",
            key = "quest_formatter_refresh_knowledge_base",
            help = "Reload the knowledge base from Google Sheets.",
        ):
            _fetch_knowledge_base.clear()
            st.rerun()

    return knowledge_base


def _handle_dialogue(
    dialogue_df: pd.DataFrame,
    preset: str,
    speaker_map: dict[str, str] | None = None,
    location_map: dict[str, str] | None = None,
    field_label_map: dict[str, str] | None = None,
    location_keys: set[str] | None = None,
    material_map: dict[str, str] | None = None,
    target: str = "EN",
    gender_mode: str | None = None,
    prompt_variables: bool = True,
    prompt_headers: bool = True,
) -> None:
    result = quest_format.classify_dialogue(
        dialogue_df,
        speaker_map = speaker_map,
        location_map = location_map,
        field_label_map = field_label_map,
        location_keys = location_keys,
        material_map = material_map,
        target_language = target,
        gender_mode = gender_mode,
        prompt_variables = prompt_variables,
        prompt_headers = prompt_headers,
    )

    if result.has_prompts:
        st.session_state[PENDING_KEY] = {
            "result": result,
            "preset": preset,
            "target": target,
            "gender_mode": gender_mode,
        }
        translation_dialog()
    else:
        quest_format.apply_translations(result.df, {}, result.defaults)
        _set_output(
            quest_format.format_dialogue(
                result.df,
                preset,
                target,
                gender_mode,
            )
        )


@st.dialog("Content Replacement", width = "large")
def translation_dialog() -> None:
    pending = st.session_state.get(PENDING_KEY)

    if not pending:
        st.warning("Nothing to format. Please try again.")
        return

    result: quest_format.ClassificationResult = pending["result"]
    target = pending["target"]
    preset = pending["preset"]
    gender_mode = pending.get("gender_mode")

    st.markdown(DIALOG_WIDTH_CSS, unsafe_allow_html = True)

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
            "Values from the file's `Translation` section and the selected "
            "`Gendered content handling` mode are prefilled below and can be "
            "adjusted before formatting."
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
        context_items = []

        for original, count in items.items():
            help_parts = []

            if count > 1:
                help_parts.append("Used in %d rows" % count)

            context = ""

            if group_key == "variables":
                context = result.variable_contexts.get(original, "")
                help_parts.append("Only the bracketed placeholder is editable.")

            help_text = " ".join(help_parts) or None
            input_key = _translation_key(target, group_key, original)

            if context:
                group_values[original] = st.text_input(
                    original,
                    value = group_defaults.get(original, original),
                    key = input_key,
                    label_visibility = "collapsed",
                    icon = ":material/translate:",
                )
                context_items.append({"key": input_key, "context": context})
            else:
                group_values[original] = st.text_input(
                    original,
                    value = group_defaults.get(original, original),
                    key = input_key,
                    help = help_text,
                    icon = ":material/translate:",
                )

        if context_items:
            components.html(
                INPUT_CONTEXT_SCRIPT % json.dumps(context_items),
                height = 0,
            )

        entered_values[group_key] = group_values

    st.divider()

    if st.button("Format", type = "primary", key = "quest_formatter_apply_translations"):
        with st.spinner("Formatting...", show_time = True):
            quest_format.apply_translations(result.df, entered_values, result.defaults)
            _set_output(
                quest_format.format_dialogue(
                    result.df,
                    preset,
                    target,
                    gender_mode,
                )
            )
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

    gender_mode = _render_gender_mode(target_language)
    knowledge_base = _render_knowledge_base_control()

    kb_maps = (
        knowledge_base.maps_for(target_language)
        if knowledge_base is not None
        else {
            "speakers": {},
            "locations": {},
            "labels": {},
            "quests": {},
            "materials": {},
        }
    )

    speaker_map = {**kb_maps["speakers"], **quest.speaker_map(target_language)}
    location_map = {**kb_maps["locations"], **quest.location_map(target_language)}
    field_label_map = {**kb_maps["labels"], **quest.field_label_map(target_language)}
    location_keys = quest.location_keys()

    if knowledge_base is not None:
        location_keys = location_keys | knowledge_base.location_keys()

    title = quest.title_for(target_language)

    if not title:
        title = kb_maps["quests"].get(quest.title_for("EN"), "")

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

    def _uncovered_by_knowledge_base(entries, kb_map):
        kb_keys = {normalize_name(key) for key in kb_map}

        return [
            entry
            for entry in entries
            if normalize_name(entry["key"]) not in kb_keys
        ]

    missing_segments = (
        (
            "Special Speaker",
            _uncovered_by_knowledge_base(
                quest.missing_special_translations(target_language),
                kb_maps["speakers"],
            ),
        ),
        (
            "Location",
            _uncovered_by_knowledge_base(
                quest.missing_location_translations(target_language),
                kb_maps["locations"],
            ),
        ),
        (
            "Field Label",
            _uncovered_by_knowledge_base(
                quest.missing_field_label_translations(target_language),
                kb_maps["labels"],
            ),
        ),
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
            _set_output(
                quest_format.format_objective(
                    objective_df,
                    target_language = target_language,
                    gender_mode = gender_mode,
                )
            )

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
                speaker_map = speaker_map,
                location_map = location_map,
                field_label_map = field_label_map,
                location_keys = location_keys,
                material_map = kb_maps["materials"],
                target = target_language,
                gender_mode = gender_mode,
                prompt_variables = True,
                prompt_headers = target_language != "EN",
            )


def _render_paste_mode() -> None:
    preset = st.segmented_control(
        "Language Format Preset:",
        ["EN", "RU"],
        default = "EN",
        key = "quest_formatter_preset",
    ) or "EN"

    gender_mode = _render_gender_mode(preset)
    knowledge_base = _render_knowledge_base_control()

    kb_maps = (
        knowledge_base.maps_for(preset)
        if knowledge_base is not None
        else {
            "speakers": {},
            "locations": {},
            "labels": {},
            "quests": {},
            "materials": {},
        }
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
            _set_output(
                quest_format.format_objective(
                    objective_df,
                    target_language = preset,
                    gender_mode = gender_mode,
                )
            )

    with dialogue_tab:
        dialogue_df = st.data_editor(
            default_table_df,
            column_config = default_table_config,
            num_rows = "dynamic",
            key = "dialogue_data",
        )

        if st.button("Format", type = "primary", key = "format_dialogue_button"):
            dialogue_df.fillna("", inplace = True)

            _handle_dialogue(
                dialogue_df,
                preset = preset,
                speaker_map = kb_maps["speakers"],
                location_map = kb_maps["locations"],
                field_label_map = kb_maps["labels"],
                location_keys = (
                    knowledge_base.location_keys()
                    if knowledge_base is not None
                    else set()
                ),
                material_map = kb_maps["materials"],
                target = preset,
                gender_mode = gender_mode,
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
    1. Make sure `Input Method` is set to `Import File` (it is the default)
    2. Upload the quest's localization file (`.xlsx` or `.csv`)
    3. Pick the target language from the languages found in the file and choose how gendered content should be written under `Gendered content handling`
    4. Open the `Dialogue` tab and click on the `Format Dialogue` button
    5. If prompted, fill in the Content Replacement form (speaker names are kept separate from conditional strings)
    6. Click on the `Copy Formatted Text` button that just appeared
    7. Paste the formatted text into the WET
    """)

    st.caption("""
    Values provided in the file's `Translation` section (`Special Speaker`, `Location`, `Field Label`) are prefilled automatically and can still be adjusted. Missing entries are looked up in the shared knowledge base spreadsheet when `Use knowledge base` is enabled. Conditional strings, optional sections, and any remaining values require manual input.
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
    default = "Import File",
    key = "quest_formatter_input_method",
)

if input_method is None:
    input_method = "Import File"

if input_method == "Import File":
    _render_import_mode()
else:
    _render_paste_mode()

_render_output()
