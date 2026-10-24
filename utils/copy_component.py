"""Streamlit component used to copy the formatted rich text."""

from pathlib import Path

import streamlit.components.v1 as components


COMPONENT_PATH = Path(__file__).resolve().parents[1] / "components" / "copy_formatted_text"

copy_formatted_text = components.declare_component(
    "copy_formatted_text",
    path = str(COMPONENT_PATH),
)
