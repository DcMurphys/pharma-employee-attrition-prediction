"""Static tables whose cells stay on one line; on a narrow screen they scroll sideways.

Used where short status labels would otherwise break into a column of single words on a
375 px screen (scope §9). Tables with long sentences should keep wrapping instead.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Literal

import streamlit as st

SWIPE_HINT = "On a small screen, swipe the table sideways to see every column."


def nowrap_css(key: str) -> str:
    """CSS keeping the cells of tables inside container ``key`` on one line.

    Streamlit renders each cell as Markdown (<p>) with its own "white-space: normal" rule; the
    table test id makes this selector more specific than Streamlit's.
    """
    cells = ", ".join(f".st-key-{key} [data-testid='stTable'] {cell} p" for cell in ("th", "td"))
    return f"<style>{cells} {{ white-space: nowrap; }}</style>"


@contextmanager
def one_line(
    key: str, height: int | Literal["content"] = "content", *, hint: bool = True
) -> Iterator[None]:
    """Container for ``st.table`` calls whose cells must not wrap, with an optional hint."""
    st.html(nowrap_css(key))
    with st.container(height=height, key=key):
        yield
    if hint:
        st.caption(SWIPE_HINT)
