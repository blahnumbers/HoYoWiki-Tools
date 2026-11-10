"""Streamlit component used to copy the formatted rich text."""

import hashlib
from pathlib import Path

import streamlit.components.v1 as components


COMPONENT_PATH = Path(__file__).resolve().parents[1] / "components" / "copy_formatted_text"

# Used by the page to detect a module left behind by a hot reload.
with open(__file__, "rb") as _source_file:
    _SOURCE_HASH = hashlib.sha256(_source_file.read()).hexdigest()

copy_formatted_text = components.declare_component(
    "copy_formatted_text",
    path = str(COMPONENT_PATH),
)
