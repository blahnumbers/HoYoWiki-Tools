"""Pure classification and formatting helpers for the Quest Formatter page.

Keeping this logic free of Streamlit makes it straightforward to test and
lets the page focus on widgets, dialogs and session state.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field

import pandas as pd

from utils.quest_data import normalize_name


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
        "available. Leave a field as-is when it does not need a change.",
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


@dataclass
class ClassificationResult:
    df: pd.DataFrame
    prompts: dict[str, dict[str, int]] = field(default_factory=dict)
    auto_localized: dict[str, str] = field(default_factory=dict)
    defaults: dict[str, dict[str, str]] = field(default_factory=dict)
    variable_contexts: dict[str, str] = field(default_factory=dict)

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
    target_language: str | None = None,
    prompt_variables: bool = True,
    prompt_headers: bool = True,
) -> ClassificationResult:
    """Classify dialogue rows and collect the values that need user input.

    ``speaker_map``, ``location_map`` and ``field_label_map`` map English keys
    to localized values (usually taken from the file's ``Translation``
    section). Matching values are prefilled in the prompt groups so the user
    can still adjust them. ``location_keys`` contains every English location
    key, which lets location headers without a comma be recognized even when
    their target translation is missing. For German, complete gendered pairs
    are auto-handled instead of being added to the bracket prompts. Every
    other header is collected into a prompt group as well.
    """

    df = _prepare_frame(dialogue_df)
    normalized_speaker_map = {
        normalize_name(key): normalize_name(value)
        for key, value in (speaker_map or {}).items()
        if normalize_name(key) and normalize_name(value)
    }
    normalized_location_map = {
        normalize_name(key): normalize_name(value)
        for key, value in (location_map or {}).items()
        if normalize_name(key) and normalize_name(value)
    }
    normalized_field_label_map = {
        normalize_name(key): normalize_name(value)
        for key, value in (field_label_map or {}).items()
        if normalize_name(key) and normalize_name(value)
    }
    normalized_location_keys = {
        normalize_name(key) for key in (location_keys or set()) if normalize_name(key)
    }

    if not normalized_location_keys:
        normalized_location_keys = set(normalized_location_map)

    traveler_name = normalized_speaker_map.get(normalize_name("Traveler"), "")

    prompts: dict[str, dict[str, int]] = {key: {} for key in PROMPT_GROUP_KEYS}
    defaults: dict[str, dict[str, str]] = {key: {} for key in PROMPT_GROUP_KEYS}
    auto_localized: dict[str, str] = {}
    variable_contexts: dict[str, str] = {}

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

            if prompt_headers:
                add_prompt("conditions", normalized_header, normalized_header)
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

                    if (
                        target_language == "DE"
                        and _is_german_gender_pair_token(token)
                    ):
                        continue

                    variable_default = token

                    if traveler_name:
                        variable_default = token.replace(
                            "{NICKNAME}", traveler_name
                        )

                    add_prompt("variables", token, variable_default)

                    if token not in variable_contexts:
                        variable_contexts[token] = _variable_context(
                            source, match.start()
                        )

    return ClassificationResult(
        df=df,
        prompts=prompts,
        auto_localized=auto_localized,
        defaults=defaults,
        variable_contexts=variable_contexts,
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
) -> str:
    html_data = "<ol>"

    for _, row in objective_df.iterrows():
        header = "" if row.get("header") is None else str(row["header"]).strip()

        if header:
            text = "" if row.get("text") is None else str(row["text"])

            if target_language == "DE":
                text = _replace_german_gender_pairs(text)

            html_data += "<li><p>%s</p></li>" % text

    html_data += "</ol>"

    return html_data


def format_dialogue(
    classified_dialogue_df: pd.DataFrame,
    preset: str = "EN",
    target_language: str | None = None,
) -> str:
    language_preset = preset if preset in {"EN", "RU"} else "EN"

    html_data = ""
    opened_tags: list[str] = []

    for row_index, row in classified_dialogue_df.iterrows():
        last_row_type = (
            classified_dialogue_df.at[row_index - 1, "type"] if row_index > 0 else None
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

        text = prepare_text(str(row["text"]), target_language)
        header = str(row["header"])

        if target_language == "DE":
            header = _replace_german_gender_pairs(header)

        match row["type"]:
            case "description":
                match language_preset:
                    case "EN":
                        html_data += """
                        <table>
                            <tbody>
                                <tr>
                                    <td colspan="1" rowspan="1">
                                        <p><span style="color: rgb(255, 255, 255);"><strong>%s: </strong> %s</span></p>
                                    </td>
                                </tr>
                            </tbody>
                        </table>
                        """ % (header, text)
                    case "RU":
                        html_data += """
                        <table>
                            <tbody>
                                <tr>
                                    <td colspan="1" rowspan="1">
                                        <p><span style="color: rgb(255, 255, 255);"><strong>%s: </strong></span>%s</p>
                                    </td>
                                </tr>
                            </tbody>
                        </table>
                        """ % (header, text)
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
                        html_data += """
                        <table>
                            <tbody>
                                <tr>
                                    <td colspan="1" rowspan="1">
                                        <p><span style="color: rgb(255, 255, 255);"><strong>%s: </strong>%s</span></p>
                                    </td>
                                </tr>
                            </tbody>
                        </table>
                        """ % (header, text)
            case "missing":
                continue
            case "addopt":
                html_data += """
                <p>%s</p>
                <table>
                    <tbody>
                        <tr>
                            <td colspan="1" rowspan="1">
                """ % header

                opened_tags.append("addopt")
            case "choice_flag":
                continue
            case "choice_branch":
                if last_row_type != "choice_flag" and "choice_branch" in opened_tags:
                    opened_tags.reverse()

                    html_data += """
                                </td>
                            </tr>
                        </tbody>
                    </table>
                    """

                    opened_tags.remove("choice_branch")
                    opened_tags.reverse()

                match language_preset:
                    case "EN":
                        html_data += """
                        <table>
                            <tbody>
                                <tr>
                                    <td colspan="1" rowspan="1">
                                        <p><span style="color: rgb(255, 255, 255);">%s: %s</span></p>
                        """ % (header, text)
                    case "RU":
                        html_data += """
                        <table>
                            <tbody>
                                <tr>
                                    <td colspan="1" rowspan="1">
                                        <p><span style="color: rgb(255, 255, 255);"><strong>%s: </strong>%s</span></p>
                        """ % (header, text)

                opened_tags.append("choice_branch")
            case "location":
                match language_preset:
                    case "EN":
                        html_data += """
                        <p><span style="color: rgb(236, 229, 216);"><strong>%s</strong></span></p>
                        """ % header
                    case "RU":
                        html_data += """
                        <p><span style="color: rgb(255, 255, 255);"><strong>%s</strong></span></p>
                        """ % header
            case "dialogue":
                if last_row_type == "objective":
                    html_data += "<p></p>"

                match language_preset:
                    case "EN":
                        html_data += """<p><span style="color: rgb(255, 255, 255);">%s: </span>%s</p>""" % (
                            header,
                            text,
                        )
                    case "RU":
                        html_data += """<p><span style="color: rgb(255, 255, 255);"><strong>%s: </strong></span><br>%s</p>""" % (
                            header,
                            text,
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

                if opened_tags and opened_tags[-1] != "action":
                    # Already inside an optional/choice table; keep the
                    # phrase in that table instead of nesting another one.
                    html_data += """<p><span style="color: rgb(255, 255, 255);">%s</span></p>""" % action_text
                elif last_row_type == "action":
                    # Continue the current group of speaker-less phrases.
                    html_data += """<p><span style="color: rgb(255, 255, 255);">%s</span></p>""" % action_text
                else:
                    html_data += """
                    <p></p>
                    <table>
                        <tbody>
                            <tr>
                                <td colspan="1" rowspan="1">
                                    <p><span style="color: rgb(255, 255, 255);">%s</span></p>
                    """ % action_text

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

                if next_row_type != "choice_branch":
                    html_data += "<p></p>"

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


def prepare_text(text: str, target_language: str | None = None) -> str:
    if target_language == "DE":
        text = _replace_german_gender_pairs(text)

    text = RUBY_RE.sub(_replace_ruby, text)
    return text.replace("{NON_BREAK_SPACE}", "&nbsp;")


def _is_german_gender_pair_token(token: str) -> bool:
    return bool(token) and GENDER_PAIR_RE.sub("", token) == ""


def _replace_german_gender_pairs(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        if match.group(1) is not None:
            female = match.group(1)
            male = match.group(2)
        else:
            male = match.group(3)
            female = match.group(4)

        replacement = "%s/%s" % (male, female)

        if GENDER_PAIR_RE.match(match.string, match.end()):
            replacement += "/"

        return replacement

    return GENDER_PAIR_RE.sub(replace, text)


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


def _localize_header_prefix(
    header: str, field_label_map: dict[str, str]
) -> str:
    for separator in (":", "："):
        if separator not in header:
            continue

        prefix, rest = header.split(separator, 1)
        localized_prefix = field_label_map.get(normalize_name(prefix))

        if not localized_prefix:
            continue

        localized_prefix = localized_prefix.rstrip(":：").strip()
        rest = rest.strip()

        if rest:
            return "%s%s %s" % (localized_prefix, separator, rest)

        return "%s%s" % (localized_prefix, separator)

    return ""


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
