"""Pure classification and formatting helpers for the Quest Formatter page.

Keeping this logic free of Streamlit makes it straightforward to test and
lets the page focus on widgets, dialogs and session state.
"""

from __future__ import annotations

import html
import re
import time
from dataclasses import dataclass, field

import pandas as pd

from utils.quest_data import normalize_name

# Used by the page to detect a module left behind by a hot reload.
_LOADED_AT = time.time()


SPEAKER_TYPES = frozenset({"dialogue", "choice_branch"})

PROMPT_GROUPS = (
    (
        "speakers",
        "Speaker Names",
        "Speaker display names. Values from the file's Special Speaker list are "
        "prefilled; adjust them if needed.",
    ),
    (
        "conditions",
        "Conditions & Stage Directions",
        "Conditional branches, notes and stage directions. These still need "
        "to be translated manually.",
    ),
    (
        "optional",
        "Optional Sections",
        "Optional dialogue and optional text section headers. Recurring prefixes "
        "from the file's Field Label list are prefilled.",
    ),
    (
        "locations",
        "Locations",
        "Location headers shown before a scene. Values from the file's Location "
        "list are prefilled; adjust them if needed.",
    ),
    (
        "labels",
        "Field Labels",
        "Labels used for quest descriptions and quest objectives in the output. "
        "Values from the file's Field Label list are prefilled.",
    ),
    (
        "variables",
        "Bracket Content",
        "Placeholder content found in the text, such as {NICKNAME} or gendered "
        "options. {NICKNAME} is prefilled with the localized Traveler name when "
        "available (Japanese uses the `(プレイヤー名)` placeholder), and gendered "
        "options are prefilled using the selected `Gendered content handling` "
        "mode. Leave a field as-is when it does not need a change.",
    ),
)

PROMPT_GROUP_KEYS = tuple(group[0] for group in PROMPT_GROUPS)

CHOICE_PREFIX_RE = re.compile(r"^(?:\d+\.)+\s+")
OBJECTIVE_KEY_RE = re.compile(r"^\d+(?:\.0+)?$")
VARIABLE_RE = re.compile(r"(?:\{[^{}]+\})+")
RUBY_RE = re.compile(r"\s*(\S+)\{RUBY#\[.\](.+?)\}(\S+)\s*")
GENDER_PAIR_RE = re.compile(
    r"\{F#([^{}]*)\}\{M#([^{}]*)\}|\{M#([^{}]*)\}\{F#([^{}]*)\}"
)
LONE_FEMALE_RE = re.compile(r"\{F#([^{}]*)\}")
COLOR_TAG_RE = re.compile(
    r"<color=#?([0-9A-Fa-f]{8}|[0-9A-Fa-f]{6})>(.*?)</color>",
    re.IGNORECASE | re.DOTALL,
)
COLOR_SPAN_RE = re.compile(
    r'(<span style="color: [^"]+;">.*?</span>)',
    re.DOTALL,
)
MATERIAL_CONDITION_RE = re.compile(r"^\((Obtain|Submit)\s+(.+)\)$")
QUOTED_NAME_RE = re.compile(r'"([^"]+)"')
TALK_TO_RE = re.compile(
    r"^Optional Dialogue:\s*Talk to\s+(.+)$",
    re.IGNORECASE,
)
TALK_TO_KEY = "Optional Dialogue: Talk to"

# Languages where the feminine form is built from the masculine stem, so a
# lone ``{F#...}`` suffix can be merged into the preceding word.
GENDER_SUFFIX_LANGUAGES = frozenset({"RU", "FR", "ES"})

# How gendered ``{M#...}{F#...}`` content is written in the output.
GENDER_MODE_MALE_FIRST = "M/F"
GENDER_MODE_FEMALE_FIRST = "F/M"
GENDER_MODE_MALE = "M"
GENDER_MODE_FEMALE = "F"
GENDER_MODE_NONE = "None"

GENDER_MODES = (
    GENDER_MODE_MALE_FIRST,
    GENDER_MODE_FEMALE_FIRST,
    GENDER_MODE_MALE,
    GENDER_MODE_FEMALE,
    GENDER_MODE_NONE,
)

DEFAULT_GENDER_MODE = GENDER_MODE_MALE_FIRST

# Korean quest formatting palette.
KR_BASE_COLOR = "rgba(255, 255, 255, 0.45)"
KR_WHITE = "rgb(255, 255, 255)"
KR_YELLOW = "rgba(255, 214, 128, 1)"

# Languages where consecutive lines from the same speaker are merged by default.
COLLAPSE_SPEAKER_LANGUAGES = frozenset({"RU", "JP"})

# Languages with a dedicated convention keep it; everything else uses M/F.
GENDER_LANGUAGE_PREFERENCES = {
    "DE": GENDER_MODE_MALE_FIRST,
    "RU": GENDER_MODE_MALE_FIRST,
    "FR": GENDER_MODE_MALE_FIRST,
    "ES": GENDER_MODE_MALE_FIRST,
    "IT": GENDER_MODE_MALE,
}

# A lone ``{F#...}`` token is treated as a suffix when its content is short.
# Longer content is usually a full replacement word and stays a prompt.
GENDER_SUFFIX_MAX_LENGTH = 4

# Minimum shared prefix length before the masculine/feminine forms are
# shortened to ``base + male suffix / female suffix``.
GENDER_BASE_MIN_LENGTH = 2

# When both forms add a suffix after the shared prefix, the prefix needs to be
# a real stem; otherwise the result reads poorly (``него/ё``).
GENDER_STEM_MIN_LENGTH = 3

# Suffixes that start with these characters split inside the stem
# (``силён/ьна``), so those pairs keep their full forms.
GENDER_BAD_SUFFIX_STARTS = ("ь", "ъ")


@dataclass
class ClassificationResult:
    df: pd.DataFrame
    prompts: dict[str, dict[str, int]] = field(default_factory=dict)
    auto_localized: dict[str, str] = field(default_factory=dict)
    defaults: dict[str, dict[str, str]] = field(default_factory=dict)
    variable_contexts: dict[str, str] = field(default_factory=dict)
    variable_suffix_contexts: dict[str, str] = field(default_factory=dict)
    has_gendered_content: bool = False
    traveler_name: str = ""

    @property
    def has_prompts(self) -> bool:
        return any(self.prompts.values())


def classify_row(header: str, text: str) -> str:
    """Return the row type for a header/text pair."""

    header = header.strip()

    if "Desc" in header:
        return "description"

    if re.search(r"\bObjective\b", header, re.IGNORECASE) or OBJECTIVE_KEY_RE.fullmatch(header):
        return "objective"

    if "Missing" in header:
        return "missing"

    if any(keyword in header for keyword in ("Additional", "Alternative", "Optional")):
        return "addopt"

    if header == "Choice":
        return "choice_flag"

    if CHOICE_PREFIX_RE.search(header):
        return "choice_branch"

    # Parenthesised headers are conditional notes / stage directions. Check
    # this before the comma rule so "(If you talked to X, ...)" is not
    # mistaken for a location.
    if header.startswith(("(", "（")):
        return "sub_mission"

    if "," in header:
        return "location"

    if header and text:
        return "dialogue"

    if header:
        return "sub_mission"

    if text:
        return "action"

    return "blank"


def classify_dialogue(
    dialogue_df: pd.DataFrame,
    speaker_map: dict[str, str] | None = None,
    location_map: dict[str, str] | None = None,
    field_label_map: dict[str, str] | None = None,
    location_keys: set[str] | None = None,
    material_map: dict[str, str] | None = None,
    target_language: str | None = None,
    gender_mode: str | None = None,
    prompt_variables: bool = True,
    prompt_headers: bool = True,
) -> ClassificationResult:
    """Classify dialogue rows and collect the values that need user input.

    ``speaker_map``, ``location_map``, ``field_label_map`` and ``material_map``
    map English keys to localized values (usually taken from the file's
    ``Translation`` section or the shared knowledge base). Matching values are
    prefilled in the prompt groups so the user can still adjust them.
    ``location_keys`` contains every English location key, which lets location
    headers without a comma be recognized even when their target translation is
    missing. Gendered content is prefilled with the value produced by
    ``gender_mode`` (the language's preferred mode by default), so editors can
    still adjust it. Every other header is collected into a prompt group as
    well.
    """

    mode = gender_mode or gender_mode_for(target_language)

    df = _prepare_frame(dialogue_df)
    has_gendered_content = any(
        "{M#" in str(value) or "{F#" in str(value)
        for column in ("header", "text")
        for value in df[column]
    )
    normalized_speaker_map = {
        normalize_name(key): str(value).strip()
        for key, value in (speaker_map or {}).items()
        if normalize_name(key) and str(value).strip()
    }
    normalized_location_map = {
        normalize_name(key): str(value).strip()
        for key, value in (location_map or {}).items()
        if normalize_name(key) and str(value).strip()
    }
    normalized_field_label_map = {
        normalize_name(key): str(value).strip()
        for key, value in (field_label_map or {}).items()
        if normalize_name(key) and str(value).strip()
    }
    normalized_material_map = {
        normalize_name(key): str(value).strip()
        for key, value in (material_map or {}).items()
        if normalize_name(key) and str(value).strip()
    }
    normalized_location_keys = {
        normalize_name(key) for key in (location_keys or set()) if normalize_name(key)
    }

    if not normalized_location_keys:
        normalized_location_keys = set(normalized_location_map)

    localized_traveler_name = normalized_speaker_map.get(
        normalize_name("Traveler"),
        "",
    )
    traveler_name = localized_traveler_name

    if target_language == "JP":
        traveler_name = "(プレイヤー名)"

    prompts: dict[str, dict[str, int]] = {key: {} for key in PROMPT_GROUP_KEYS}
    defaults: dict[str, dict[str, str]] = {key: {} for key in PROMPT_GROUP_KEYS}
    auto_localized: dict[str, str] = {}
    variable_contexts: dict[str, str] = {}
    variable_suffix_contexts: dict[str, str] = {}

    def add_prompt(group: str, value: str, default: str | None = None) -> None:
        if value:
            prompts[group][value] = prompts[group].get(value, 0) + 1
            defaults[group].setdefault(value, value if default is None else default)

    def resolve_speaker(row_index: int, speaker: str) -> None:
        df.at[row_index, "header"] = speaker

        localized_name = normalized_speaker_map.get(speaker)

        if localized_name:
            defaults["speakers"][speaker] = localized_name

            if localized_name != speaker:
                auto_localized[speaker] = localized_name

            if prompt_headers:
                add_prompt("speakers", speaker, localized_name)
        elif prompt_headers:
            add_prompt("speakers", speaker, speaker)

    for row_index, row in df.iterrows():
        raw_header = str(row["header"]).strip()
        text = str(row["text"])
        normalized_raw_header = normalize_name(raw_header)
        row_type = classify_row(raw_header, text)

        if (
            row_type == "sub_mission"
            and normalized_raw_header in normalized_location_keys
        ):
            row_type = "location"

        df.at[row_index, "type"] = row_type

        if row_type == "description":
            label = "Quest Description"
            df.at[row_index, "header"] = label
            localized_label = normalized_field_label_map.get(label)

            if localized_label:
                defaults["labels"][label] = localized_label

            if prompt_headers:
                add_prompt("labels", label, localized_label or label)
        elif row_type == "objective":
            label = "Quest Objective"
            df.at[row_index, "header"] = label
            localized_label = normalized_field_label_map.get(label)

            if localized_label:
                defaults["labels"][label] = localized_label

            if prompt_headers:
                add_prompt("labels", label, localized_label or label)
        elif row_type == "addopt":
            normalized_header = normalize_name(raw_header)
            df.at[row_index, "header"] = normalized_header
            localized_optional = _localize_talk_to_header(
                normalized_header,
                normalized_field_label_map,
                normalized_speaker_map,
                target_language,
            )

            if not localized_optional:
                localized_optional = _localize_header_prefix(
                    normalized_header, normalized_field_label_map
                )

            if localized_optional:
                defaults["optional"][normalized_header] = localized_optional

            if prompt_headers:
                add_prompt(
                    "optional",
                    normalized_header,
                    localized_optional or normalized_header,
                )
        elif row_type == "choice_branch":
            speaker = normalize_name(CHOICE_PREFIX_RE.sub("", raw_header))
            resolve_speaker(row_index, speaker)
        elif row_type == "location":
            normalized_header = normalize_name(raw_header)
            df.at[row_index, "header"] = normalized_header
            localized_location = normalized_location_map.get(normalized_header)

            if not localized_location:
                localized_location = _localize_compound_location(
                    normalized_header,
                    normalized_location_map,
                    target_language,
                )

            if localized_location:
                defaults["locations"][normalized_header] = localized_location

            if prompt_headers:
                add_prompt(
                    "locations",
                    normalized_header,
                    localized_location or normalized_header,
                )
        elif row_type == "sub_mission":
            normalized_header = normalize_name(raw_header)
            df.at[row_index, "header"] = normalized_header
            localized_condition = _localize_material_condition(
                normalized_header,
                normalized_field_label_map,
                normalized_material_map,
            )

            if localized_condition:
                defaults["conditions"][normalized_header] = localized_condition

            if prompt_headers:
                add_prompt(
                    "conditions",
                    normalized_header,
                    localized_condition or normalized_header,
                )
        else:
            normalized_header = normalize_name(raw_header)

            if row_type == "dialogue":
                resolve_speaker(row_index, normalized_header)
            else:
                df.at[row_index, "header"] = normalized_header

        if prompt_variables:
            for source in (raw_header, text):
                for match in VARIABLE_RE.finditer(source):
                    token = match.group(0)

                    if "{RUBY#" in token or "{NON_BREAK_SPACE}" in token:
                        continue

                    auto_value = _auto_gender_value(
                        source,
                        match,
                        target_language,
                        mode,
                    )

                    if auto_value is not None:
                        variable_default = auto_value
                    else:
                        variable_default = token

                        if traveler_name:
                            variable_default = token.replace(
                                "{NICKNAME}", traveler_name
                            )

                    add_prompt("variables", token, variable_default)

                    if token not in variable_contexts:
                        context = _variable_context(source, match.start())
                        variable_contexts[token] = context

                        if not context and (
                            "{M#" in token or "{F#" in token
                        ):
                            variable_suffix_contexts[token] = (
                                _variable_suffix_context(source, match.end())
                            )

    return ClassificationResult(
        df=df,
        prompts=prompts,
        auto_localized=auto_localized,
        defaults=defaults,
        variable_contexts=variable_contexts,
        variable_suffix_contexts=variable_suffix_contexts,
        has_gendered_content=has_gendered_content,
        traveler_name=localized_traveler_name,
    )


def apply_translations(
    classified_df: pd.DataFrame,
    translations: dict[str, dict[str, str]],
    defaults: dict[str, dict[str, str]] | None = None,
) -> None:
    """Apply grouped prompt values to headers and text.

    Values provided in ``defaults`` (for example the auto-localized Special
    Speaker names) are used when a group is not present in ``translations``.
    Replacements are intentionally scoped: speaker names only touch the
    speaker cell, so translating ``Jeht`` cannot accidentally rewrite prose.
    """

    defaults = defaults or {}

    groups = {}

    for key in PROMPT_GROUP_KEYS:
        group_values = {
            original: translated.strip() or original
            for original, translated in defaults.get(key, {}).items()
        }
        group_values.update(
            {
                original: translated.strip() or original
                for original, translated in translations.get(key, {}).items()
            }
        )
        groups[key] = group_values

    variables = sorted(groups["variables"].items(), key=lambda item: len(item[0]), reverse=True)

    for row_index, row in classified_df.iterrows():
        row_type = row["type"]
        header = str(row["header"])
        text = str(row["text"])

        if row_type in SPEAKER_TYPES:
            header = groups["speakers"].get(header, header)
        elif row_type == "addopt":
            header = groups["optional"].get(header, header)
        elif row_type == "sub_mission":
            header = groups["conditions"].get(header, header)
        elif row_type == "location":
            header = groups["locations"].get(header, header)
        elif row_type in {"description", "objective"}:
            header = groups["labels"].get(header, header)

        for original, translated in variables:
            if translated != original:
                header = header.replace(original, translated)
                text = text.replace(original, translated)

        classified_df.at[row_index, "header"] = header
        classified_df.at[row_index, "text"] = text


def format_objective(
    objective_df: pd.DataFrame,
    target_language: str | None = None,
    gender_mode: str | None = None,
) -> str:
    html_data = "<ol>"

    for _, row in objective_df.iterrows():
        header = "" if row.get("header") is None else str(row["header"]).strip()

        if header:
            text = "" if row.get("text") is None else str(row["text"])
            text = _replace_color_tags(text)

            if target_language:
                text = _replace_gender_pairs(
                    text,
                    target_language,
                    gender_mode or gender_mode_for(target_language),
                )

            html_data += "<li><p>%s</p></li>" % text

    html_data += "</ol>"

    return html_data


def markup_preset_for(language: str | None) -> str:
    """Return the dialogue markup preset used for a target language."""

    if language == "CHS":
        return "CHS"

    if language == "KR":
        return "KR"

    return "RU" if language in {"RU", "JP"} else "EN"


def format_dialogue(
    classified_dialogue_df: pd.DataFrame,
    preset: str = "EN",
    target_language: str | None = None,
    gender_mode: str | None = None,
    collapse_speakers: bool = False,
    show_gendered_note: bool = False,
    traveler_name: str | None = None,
) -> str:
    if preset == "CHS":
        language_preset = "CHS"
    elif preset == "KR":
        language_preset = "KR"
    elif preset in {"RU", "JP"}:
        language_preset = "RU"
    else:
        language_preset = "EN"

    is_japanese = target_language == "JP"
    is_korean = target_language == "KR"
    kr_meta_table_open = False

    html_data = ""
    opened_tags: list[str] = []

    if is_japanese and show_gendered_note:
        html_data += (
            "<p>※主人公に空、蛍どちらを選んだかによって会話に差異が生じる。"
            "空を選んだ場合は♂、蛍を選んだ場合は♀として、以下のように記載。<br>"
            "例: (♂俺 ♀私)には見えて、ティレルには見えない魔物もいる"
            "(♂みたいだ ♀みたい)…</p>"
            "<p></p>"
        )

    for row_index, row in classified_dialogue_df.iterrows():
        last_row_type = (
            classified_dialogue_df.at[row_index - 1, "type"] if row_index > 0 else None
        )
        previous_header = (
            str(classified_dialogue_df.at[row_index - 1, "header"])
            if row_index > 0
            else ""
        )
        next_row_type = (
            classified_dialogue_df.at[row_index + 1, "type"]
            if row_index < (len(classified_dialogue_df) - 1)
            else None
        )
        next_content_type = next_row_type
        lookahead_index = row_index + 1

        while (
            lookahead_index < len(classified_dialogue_df)
            and classified_dialogue_df.at[lookahead_index, "type"] == "blank"
        ):
            lookahead_index += 1

        if lookahead_index < len(classified_dialogue_df):
            next_content_type = classified_dialogue_df.at[lookahead_index, "type"]

        next_note_target_type = next_content_type
        note_lookahead_index = row_index + 1

        while (
            note_lookahead_index < len(classified_dialogue_df)
            and classified_dialogue_df.at[note_lookahead_index, "type"]
            in {"blank", "sub_mission"}
        ):
            note_lookahead_index += 1

        if note_lookahead_index < len(classified_dialogue_df):
            next_note_target_type = classified_dialogue_df.at[note_lookahead_index, "type"]

        if (
            is_korean
            and kr_meta_table_open
            and row["type"] not in {"description", "objective", "blank"}
        ):
            html_data += """
                                        </td>
                                    </tr>
                                </tbody>
                            </table>
                            """
            kr_meta_table_open = False

        text = prepare_text(str(row["text"]), target_language, gender_mode)
        header = _replace_color_tags(str(row["header"]))

        if target_language:
            header = _replace_gender_pairs(
                header,
                target_language,
                gender_mode or gender_mode_for(target_language),
            )

        match row["type"]:
            case "description":
                match language_preset:
                    case "EN":
                        description_content = _wrap_with_base_color(
                            "<strong>%s: </strong> %s" % (header, text),
                            "rgb(255, 255, 255)",
                        )
                        html_data += """
                        <table>
                            <tbody>
                                <tr>
                                    <td colspan="1" rowspan="1">
                                        <p>%s</p>
                                    </td>
                                </tr>
                            </tbody>
                        </table>
                        """ % description_content
                    case "RU":
                        description_header = _wrap_with_base_color(
                            "<strong>%s: </strong>" % header,
                            "rgb(255, 255, 255)",
                        )
                        html_data += """
                        <table>
                            <tbody>
                                <tr>
                                    <td colspan="1" rowspan="1">
                                        <p>%s%s</p>
                                    </td>
                                </tr>
                            </tbody>
                        </table>
                        """ % (description_header, text)
                    case "CHS":
                        description_header = _wrap_with_base_color(
                            "<strong>%s：</strong>" % header,
                            "rgb(255, 255, 255)",
                        )
                        html_data += """
                        <table>
                            <tbody>
                                <tr>
                                    <td colspan="1" rowspan="1">
                                        <p>%s%s</p>
                                    </td>
                                </tr>
                            </tbody>
                        </table>
                        """ % (description_header, text)
                    case "KR":
                        if not kr_meta_table_open:
                            html_data += """
                            <table>
                                <tbody>
                                    <tr>
                                        <td colspan="1" rowspan="1">
                            """
                            kr_meta_table_open = True

                        description_content = _wrap_with_base_color(
                            "<strong>%s: </strong>%s" % (header, text),
                            "rgb(255, 255, 255)",
                        )
                        html_data += "<p>%s</p>" % description_content
            case "objective":
                match language_preset:
                    case "EN":
                        html_data += """
                        <table>
                            <tbody>
                                <tr>
                                    <td colspan="1" rowspan="1">
                                        <p><strong>%s: </strong>%s</p>
                                    </td>
                                </tr>
                            </tbody>
                        </table>
                        """ % (header, text)
                    case "RU":
                        objective_content = _wrap_with_base_color(
                            "<strong>%s: </strong>%s" % (header, text),
                            "rgb(255, 255, 255)",
                        )
                        html_data += """
                        <table>
                            <tbody>
                                <tr>
                                    <td colspan="1" rowspan="1">
                                        <p>%s</p>
                                    </td>
                                </tr>
                            </tbody>
                        </table>
                        """ % objective_content
                    case "CHS":
                        objective_content = _wrap_with_base_color(
                            "<strong>%s：</strong>%s" % (header, text),
                            "rgb(255, 255, 255)",
                        )
                        html_data += """
                        <table>
                            <tbody>
                                <tr>
                                    <td colspan="1" rowspan="1">
                                        <p>%s</p>
                                    </td>
                                </tr>
                            </tbody>
                        </table>
                        """ % objective_content
                    case "KR":
                        if not kr_meta_table_open:
                            html_data += """
                            <table>
                                <tbody>
                                    <tr>
                                        <td colspan="1" rowspan="1">
                            """
                            kr_meta_table_open = True

                        html_data += "<p><strong>%s: </strong>%s</p>" % (
                            header,
                            text,
                        )
            case "missing":
                continue
            case "addopt":
                optional_header = header

                if language_preset == "CHS":
                    optional_header = _use_fullwidth_colon(optional_header)

                if language_preset == "KR":
                    html_data += """
                    <table>
                        <tbody>
                            <tr>
                                <td colspan="1" rowspan="1">
                                    <p><strong style="color: %s;">%s</strong></p>
                    """ % (KR_YELLOW, optional_header)
                else:
                    html_data += """
                    <p>%s</p>
                    <table>
                        <tbody>
                            <tr>
                                <td colspan="1" rowspan="1">
                    """ % optional_header

                opened_tags.append("addopt")
            case "choice_flag":
                continue
            case "choice_branch":
                match language_preset:
                    case "CHS":
                        choice_content = _wrap_with_base_color(
                            "<strong>%s：</strong>%s" % (header, text),
                            "rgb(255, 255, 255)",
                        )
                    case "KR":
                        choice_content = (
                            '<span style="color: %s;">%s</span>'
                            '<span style="color: %s;">:</span> '
                            '<span style="color: %s;">%s</span>'
                            % (
                                KR_YELLOW,
                                header,
                                KR_BASE_COLOR,
                                KR_BASE_COLOR,
                                text,
                            )
                        )
                    case "EN":
                        choice_content = _wrap_with_base_color(
                            "%s: %s" % (header, text),
                            "rgb(255, 255, 255)",
                        )
                    case _:
                        choice_content = _wrap_with_base_color(
                            "<strong>%s: </strong>%s" % (header, text),
                            "rgb(255, 255, 255)",
                        )

                if (
                    is_japanese
                    and opened_tags
                    and opened_tags[-1] == "choice_branch"
                    and last_row_type == "choice_branch"
                ):
                    # Add another option row to the open choice group.
                    html_data += """
                                    </td>
                                </tr>
                                <tr>
                                    <td colspan="1" rowspan="1">
                                        <p>%s</p>
                    """ % choice_content
                else:
                    if (
                        last_row_type != "choice_flag"
                        and "choice_branch" in opened_tags
                    ):
                        opened_tags.reverse()

                        html_data += """
                                    </td>
                                </tr>
                            </tbody>
                        </table>
                        """

                        opened_tags.remove("choice_branch")
                        opened_tags.reverse()

                    html_data += """
                        <table>
                            <tbody>
                                <tr>
                                    <td colspan="1" rowspan="1">
                                        <p>%s</p>
                        """ % choice_content

                    opened_tags.append("choice_branch")
            case "location":
                match language_preset:
                    case "EN":
                        location_content = _wrap_with_base_color(
                            "<strong>%s</strong>" % header,
                            "rgb(236, 229, 216)",
                        )
                        html_data += """
                        <p>%s</p>
                        """ % location_content
                    case "RU":
                        location_content = _wrap_with_base_color(
                            "<strong>%s</strong>" % header,
                            "rgb(255, 255, 255)",
                        )
                        html_data += """
                        <p>%s</p>
                        """ % location_content
                    case "CHS":
                        location_content = _wrap_with_base_color(
                            "<strong>%s</strong>" % header,
                            "rgb(255, 255, 255)",
                        )
                        html_data += """
                        <p>%s</p>
                        """ % location_content
                    case "KR":
                        html_data += """
                        <p><span style="color: rgb(236, 229, 216);">%s</span></p>
                        """ % header
            case "dialogue":
                if last_row_type == "objective" and not is_korean:
                    html_data += "<p></p>"

                if (
                    is_japanese
                    and opened_tags
                    and opened_tags[-1] in {"choice_branch", "addopt", "action"}
                    and last_row_type
                    in {"dialogue", "choice_branch", "addopt", "action"}
                    and previous_header != header
                ):
                    html_data += "<p></p>"

                if (
                    collapse_speakers
                    and last_row_type == "dialogue"
                    and previous_header == header
                    and html_data.endswith("</p>")
                    and text.strip()
                ):
                    html_data = html_data[:-4] + "<br>%s</p>" % text
                else:
                    match language_preset:
                        case "EN":
                            dialogue_header = _wrap_with_base_color(
                                "%s: " % header,
                                "rgb(255, 255, 255)",
                            )
                            html_data += """<p>%s%s</p>""" % (
                                dialogue_header,
                                text,
                            )
                        case "RU":
                            dialogue_header = _wrap_with_base_color(
                                "<strong>%s: </strong>" % header,
                                "rgb(255, 255, 255)",
                            )
                            html_data += """<p>%s<br>%s</p>""" % (
                                dialogue_header,
                                text,
                            )
                        case "CHS":
                            dialogue_header = _wrap_with_base_color(
                                "<strong>%s：</strong>" % header,
                                "rgb(255, 255, 255)",
                            )
                            html_data += """<p>%s<br>%s</p>""" % (
                                dialogue_header,
                                text,
                            )
                        case "KR":
                            inside_addopt = "addopt" in opened_tags
                            is_traveler = bool(
                                traveler_name and traveler_name in header
                            ) or "Traveler" in header

                            if inside_addopt:
                                speaker_color = KR_YELLOW
                            elif is_traveler:
                                speaker_color = KR_BASE_COLOR
                            else:
                                speaker_color = KR_WHITE

                            text_color = KR_BASE_COLOR

                            dialogue_header = (
                                '<span style="color: %s;">%s</span>'
                                '<span style="color: %s;">:</span>'
                                % (speaker_color, header, KR_BASE_COLOR)
                            )
                            html_data += (
                                '<p>%s <span style="color: %s;">%s</span></p>'
                                % (dialogue_header, text_color, text)
                            )
            case "sub_mission":
                if (
                    "choice_branch" in opened_tags
                    and next_note_target_type in {"choice_branch", "choice_flag"}
                ):
                    opened_tags.reverse()

                    html_data += """
                                </td>
                            </tr>
                        </tbody>
                    </table>
                    """

                    opened_tags.remove("choice_branch")
                    opened_tags.reverse()

                if last_row_type == "objective":
                    html_data += "<p></p>"

                html_data += "<p><em>%s</em></p>" % header
            case "action":
                action_text = (
                    "<strong>%s</strong>" % text
                    if bool(row.get("bold", False))
                    else text
                )
                action_html = _wrap_with_base_color(
                    action_text,
                    "rgb(255, 255, 255)",
                )

                if is_korean:
                    html_data += (
                        '<p><strong style="color: %s;">%s</strong></p>'
                        % (KR_WHITE, text)
                    )
                elif opened_tags and opened_tags[-1] != "action":
                    # Already inside an optional/choice table; keep the
                    # phrase in that table instead of nesting another one.
                    html_data += """<p>%s</p>""" % action_html
                elif last_row_type == "action":
                    # Continue the current group of speaker-less phrases.
                    html_data += """<p>%s</p>""" % action_html
                else:
                    html_data += """
                    <p></p>
                    <table>
                        <tbody>
                            <tr>
                                <td colspan="1" rowspan="1">
                                    <p>%s</p>
                    """ % action_html

                    opened_tags.append("action")

                if opened_tags and opened_tags[-1] == "action" and next_row_type != "action":
                    html_data += """
                                </td>
                            </tr>
                        </tbody>
                    </table>
                    <p></p>
                    """

                    opened_tags.pop()
            case "blank":
                if opened_tags and next_row_type != "choice_flag":
                    match opened_tags[-1]:
                        case "addopt" | "choice_branch" | "action":
                            html_data += """
                                        </td>
                                    </tr>
                                </tbody>
                            </table>
                            """

                            opened_tags.pop()

                if next_row_type != "choice_branch" and not is_korean:
                    html_data += "<p></p>"

    if kr_meta_table_open:
        html_data += """
                                        </td>
                                    </tr>
                                </tbody>
                            </table>
                            """

    if opened_tags:
        for tag in reversed(opened_tags):
            match tag:
                case "addopt" | "choice_branch" | "action":
                    html_data += """
                                </td>
                            </tr>
                        </tbody>
                    </table>
                    """

                    opened_tags.pop()

    return html_data


def prepare_text(
    text: str,
    target_language: str | None = None,
    gender_mode: str | None = None,
) -> str:
    text = _replace_color_tags(text)

    if target_language:
        text = _replace_gender_pairs(
            text,
            target_language,
            gender_mode or gender_mode_for(target_language),
        )

    text = RUBY_RE.sub(_replace_ruby, text)
    return text.replace("{NON_BREAK_SPACE}", "&nbsp;")


def _replace_color_tags(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        hex_value = match.group(1)
        inner_text = match.group(2)

        red = int(hex_value[0:2], 16)
        green = int(hex_value[2:4], 16)
        blue = int(hex_value[4:6], 16)

        if len(hex_value) == 8:
            alpha = int(hex_value[6:8], 16)
        else:
            alpha = 255

        if alpha >= 255:
            color = "rgb(%d, %d, %d)" % (red, green, blue)
        else:
            color = "rgba(%d, %d, %d, %s)" % (
                red,
                green,
                blue,
                round(alpha / 255, 3),
            )

        return '<span style="color: %s;">%s</span>' % (color, inner_text)

    return COLOR_TAG_RE.sub(replace, text)


def _wrap_with_base_color(content: str, base_color: str) -> str:
    """Wrap content in a base color without nesting colored spans.

    The WET editor can lose the outer color after a nested colored span, so
    split the base span around any inline color spans instead.
    """

    parts = COLOR_SPAN_RE.split(content)
    wrapped_parts = []

    for part in parts:
        if not part:
            continue

        if COLOR_SPAN_RE.fullmatch(part):
            wrapped_parts.append(part)
        else:
            wrapped_parts.append(
                '<span style="color: %s;">%s</span>' % (base_color, part)
            )

    return "".join(wrapped_parts)


def _is_gender_pair_token(token: str) -> bool:
    return bool(token) and GENDER_PAIR_RE.sub("", token) == ""


def _is_lone_female_suffix_token(
    source: str,
    start: int,
    token: str,
) -> bool:
    match = LONE_FEMALE_RE.fullmatch(token)

    if match is None:
        return False

    return _is_lone_female_suffix(source, start, match.group(1))


def _is_lone_female_suffix(source: str, start: int, content: str) -> bool:
    if len(content) > GENDER_SUFFIX_MAX_LENGTH:
        return False

    return start > 0 and not source[start - 1].isspace()


def _replace_lone_female(match: re.Match[str]) -> str:
    if not _is_lone_female_suffix(
        match.string,
        match.start(),
        match.group(1),
    ):
        return match.group(0)

    return "/%s" % match.group(1)


def _drop_lone_female(match: re.Match[str]) -> str:
    if not _is_lone_female_suffix(
        match.string,
        match.start(),
        match.group(1),
    ):
        return match.group(0)

    return ""


def _keep_lone_female(match: re.Match[str]) -> str:
    if not _is_lone_female_suffix(
        match.string,
        match.start(),
        match.group(1),
    ):
        return match.group(0)

    return match.group(1)


def gender_mode_for(language: str | None) -> str:
    """Return the preferred gendered content mode for a language."""

    return GENDER_LANGUAGE_PREFERENCES.get(language or "", DEFAULT_GENDER_MODE)


def _auto_gender_value(
    source: str,
    match: re.Match[str],
    language: str,
    mode: str,
) -> str | None:
    """Return the automatic value for a gendered token, if there is one."""

    if not language or mode == GENDER_MODE_NONE:
        return None

    token = match.group(0)

    if _is_gender_pair_token(token):
        return _replace_gender_pairs(token, language, mode)

    if (
        language in GENDER_SUFFIX_LANGUAGES
        and _is_lone_female_suffix_token(source, match.start(), token)
    ):
        content = LONE_FEMALE_RE.fullmatch(token).group(1)

        if mode == GENDER_MODE_MALE:
            return ""
        if mode == GENDER_MODE_FEMALE:
            return content
        return "/%s" % content

    return None


def _replace_gender_pairs(text: str, language: str, mode: str) -> str:
    if mode == GENDER_MODE_NONE:
        return text

    def replace(match: re.Match[str]) -> str:
        if match.group(1) is not None:
            female = match.group(1)
            male = match.group(2)
        else:
            male = match.group(3)
            female = match.group(4)

        replacement = _format_gender_pair(male, female, language, mode)

        if GENDER_PAIR_RE.match(match.string, match.end()):
            replacement += "/"

        return replacement

    text = GENDER_PAIR_RE.sub(replace, text)

    if language in GENDER_SUFFIX_LANGUAGES:
        if mode == GENDER_MODE_MALE:
            text = LONE_FEMALE_RE.sub(_drop_lone_female, text)
        elif mode == GENDER_MODE_FEMALE:
            text = LONE_FEMALE_RE.sub(_keep_lone_female, text)
        else:
            text = LONE_FEMALE_RE.sub(_replace_lone_female, text)

    return text


def _format_gender_pair(
    male: str,
    female: str,
    language: str,
    mode: str,
) -> str:
    if mode == GENDER_MODE_MALE:
        return male

    if mode == GENDER_MODE_FEMALE:
        return female

    if language == "JP":
        # Japanese gendered words are prefixes to the rest of the phrase, so
        # they are written with explicit symbols instead of a slash.
        if mode == GENDER_MODE_FEMALE_FIRST:
            return "(♀%s ♂%s)" % (female, male)

        return "(♂%s ♀%s)" % (male, female)

    if mode == GENDER_MODE_FEMALE_FIRST:
        return "%s/%s" % (female, male)

    if language not in GENDER_SUFFIX_LANGUAGES:
        return "%s/%s" % (male, female)

    base = _common_prefix(male, female)
    male_suffix = male[len(base):]
    female_suffix = female[len(base):]

    if len(base) < GENDER_BASE_MIN_LENGTH:
        return "%s/%s" % (male, female)

    if (
        male_suffix
        and female_suffix
        and len(base) < GENDER_STEM_MIN_LENGTH
    ):
        return "%s/%s" % (male, female)

    if " " in male or " " in female:
        return "%s/%s" % (male, female)

    if female_suffix.startswith(GENDER_BAD_SUFFIX_STARTS):
        return "%s/%s" % (male, female)

    if not male_suffix and not female_suffix:
        return base

    if not male_suffix:
        return "%s/%s" % (base, female_suffix)

    if not female_suffix:
        return "%s/%s" % (male, female)

    return "%s%s/%s" % (base, male_suffix, female_suffix)


def _common_prefix(first: str, second: str) -> str:
    length = 0

    for first_char, second_char in zip(first, second):
        if first_char != second_char:
            break

        length += 1

    return first[:length]


def _replace_ruby(match: re.Match[str]) -> str:
    base_text = html.escape(match.group(1) + match.group(3))
    ruby_text = html.escape(match.group(2))

    return """<custom-ruby data-ruby="%s|%s"></custom-ruby>""" % (base_text, ruby_text)


def _variable_context(source: str, start: int) -> str:
    raw_before = source[:start]

    if not raw_before:
        return ""

    # Only show context when the placeholder is attached directly to the
    # preceding text (no space) and is not directly after another placeholder.
    if raw_before[-1].isspace() or raw_before.endswith("}"):
        return ""

    before = VARIABLE_RE.sub(" ", raw_before).rstrip()

    if not before:
        return ""

    match = re.search(r"(\S+)$", before)

    if match is None:
        return ""

    context = re.sub(r"^[\W_]+|[\W_]+$", "", match.group(1))

    if len(context) > 30:
        context = "…" + context[-30:]

    return context


def _variable_suffix_context(source: str, end: int) -> str:
    """Return the word attached after a placeholder, for prefix-style tokens."""

    raw_after = source[end:]

    if not raw_after or raw_after[0].isspace():
        return ""

    after = VARIABLE_RE.sub(" ", raw_after).lstrip()

    if not after:
        return ""

    match = re.match(r"(\S+)", after)

    if match is None:
        return ""

    context = re.sub(r"^[\W_]+|[\W_]+$", "", match.group(1))

    if len(context) > 30:
        context = context[:30] + "…"

    return context


def _localize_material_condition(
    header: str,
    field_label_map: dict[str, str],
    material_map: dict[str, str],
) -> str:
    """Prefill ``(Obtain "X")`` / ``(Submit "X")`` headers from the glossary."""

    match = MATERIAL_CONDITION_RE.match(header)

    if match is None:
        return ""

    localized_verb = field_label_map.get(match.group(1))

    if not localized_verb:
        return ""

    body = match.group(2)
    names = QUOTED_NAME_RE.findall(body)

    if not names or any(name not in material_map for name in names):
        return ""

    localized_body = QUOTED_NAME_RE.sub(
        lambda quoted: '"%s"' % material_map[quoted.group(1)],
        body,
    )

    return "(%s %s)" % (localized_verb, localized_body)


def _localize_compound_location(
    header: str,
    location_map: dict[str, str],
    language: str | None = None,
) -> str:
    """Translate ``Region, Subarea`` headers part by part when all parts match."""

    parts = [part.strip() for part in header.split(",")]

    if len(parts) < 2:
        return ""

    localized_parts = []

    for part in parts:
        localized = location_map.get(part)

        if not localized:
            return ""

        localized_parts.append(localized)

    if language == "JP":
        # Japanese location names put the broader area first and are separated
        # by a space rather than a comma.
        localized_parts.reverse()
        return " ".join(localized_parts)

    return ", ".join(localized_parts)


def _use_fullwidth_colon(text: str) -> str:
    """Replace the first colon and its trailing spaces with a full-width colon."""

    return re.sub(r"[:：]\s*", "：", text, count=1)


def _localize_talk_to_header(
    header: str,
    field_label_map: dict[str, str],
    speaker_map: dict[str, str],
    language: str | None,
) -> str:
    """Localize ``Optional Dialogue: Talk to <name>`` with the name translated."""

    match = TALK_TO_RE.match(header)

    if match is None:
        return ""

    name = match.group(1).strip()
    template = field_label_map.get(TALK_TO_KEY)

    if not template:
        return ""

    matched_name, name_suffix = _match_speaker_name(name, speaker_map)
    localized_name = speaker_map.get(matched_name, "")

    if language == "RU" or not localized_name:
        # Russian names need a different grammatical case, so the source name
        # is left for the editor to adjust.
        return "%s %s" % (template, name)

    if "{name}" in template:
        localized = template.replace("{name}", localized_name)
    elif "\u3000" in template:
        # Templates can use an ideographic space as the name slot.
        localized = template.replace("\u3000", localized_name, 1)
    elif language == "JP" and " " in template:
        head, tail = template.rsplit(" ", 1)
        localized = "%s%s%s" % (head, localized_name, tail)
    else:
        localized = "%s %s" % (template, localized_name)

    if name_suffix:
        localized = "%s %s" % (localized, name_suffix)

    return localized


def _match_speaker_name(
    name: str,
    speaker_map: dict[str, str],
) -> tuple[str, str]:
    """Split ``Anastasya again`` into a known name and its trailing words."""

    normalized = normalize_name(name)

    if normalized in speaker_map:
        return normalized, ""

    best = ""

    for candidate in speaker_map:
        if normalized.startswith(candidate + " ") and len(candidate) > len(best):
            best = candidate

    if not best:
        return "", normalized

    return best, normalized[len(best):].strip()


def _localize_header_prefix(
    header: str, field_label_map: dict[str, str]
) -> str:
    """Replace the longest known English prefix with its translation.

    The glossary can hold both ``Optional Dialogue`` and the more specific
    ``Optional Dialogue: Talk to``, so the longest matching entry wins.
    """

    best_key = ""
    best_value = ""

    for key, localized in field_label_map.items():
        if not key or not localized:
            continue

        if header == key:
            return localized

        if not header.startswith(key):
            continue

        if header[len(key)] not in ":： ":
            continue

        if len(key) > len(best_key):
            best_key = key
            best_value = localized

    if not best_key:
        return ""

    remainder = header[len(best_key):]

    if remainder[:1] in {":", "："}:
        if best_value.endswith((":", "：")):
            remainder = remainder[1:]
        elif remainder[1:2] not in {" ", ""}:
            remainder = remainder[0] + " " + remainder[1:]

    return "%s%s" % (best_value, remainder)


def _prepare_frame(dialogue_df: pd.DataFrame) -> pd.DataFrame:
    frame = dialogue_df.copy()

    if "type" in frame.columns:
        frame = frame.drop(columns=["type"])

    for column in ("header", "text"):
        if column not in frame.columns:
            frame[column] = ""

    if "bold" in frame.columns:
        bold_values = frame["bold"].fillna(False).astype(bool).reset_index(drop=True)
    else:
        bold_values = pd.Series(False, index=range(len(frame)), dtype=bool)

    frame = frame[["header", "text"]].fillna("").astype(str)
    frame = frame.reset_index(drop=True)
    frame["bold"] = bold_values
    frame.insert(0, "type", "")

    return frame
