"""Standard notices shown on pages with individual risk (scope §3, §4)."""

import streamlit as st

INTENDED_USE = (
    "Use risk scores only to plan supportive conversations, such as stay interviews. A high "
    "score is a prompt to talk, never a basis for disciplinary action, pay decisions or "
    "dismissal."
)


def intended_use() -> None:
    """Intended-use notice for every page that shows individual risk."""
    st.info(INTENDED_USE, icon=":material/volunteer_activism:")
