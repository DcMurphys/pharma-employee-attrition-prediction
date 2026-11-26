"""Shared page frame: sidebar note, page header, disclaimer, and plain-language errors."""

from datetime import date

import streamlit as st

# Plain messages for ArtifactError.kind; technical details go to the log only (scope §4).
ARTIFACT_MESSAGES = {
    "missing": "Some information this page needs is not available right now.",
    "unreadable": "Some information this page needs could not be opened.",
    "invalid": "Some information this page needs did not pass its quality check.",
}
OTHER_PAGES_HINT = "The other pages may still work. Please contact the People Analytics team."

MIN_TARGET_PX = 44  # scope §7.4: buttons, tabs and other controls at least 44 px tall
# Streamlit draws most controls 36-40 px tall; this raises them to the minimum target size.
_TARGET_SELECTORS = ", ".join(
    [
        "[data-testid^='stBaseButton']",  # buttons, download buttons, segmented control
        "[data-testid='stTab']",
        "[data-baseweb='select'] > div",
        "[data-testid='stNumberInputContainer']",
        "[data-testid='stTextInputRootElement']",
        "[data-testid='stExpander'] summary",
        "[data-testid='stSidebarNavLink']",
    ]
)
TARGET_SIZE_CSS = f"<style>{_TARGET_SELECTORS} {{ min-height: {MIN_TARGET_PX}px; }}</style>"


def apply_target_size() -> None:
    """Raise interactive controls to the minimum touch-target height."""
    st.html(TARGET_SIZE_CSS)


def render_sidebar(
    employees: int | None,
    trained_on_rows: int | None,
    exported_on: date | None,
    disclaimer: str,
) -> None:
    """Snapshot note and disclaimer under the page navigation; unknown values are left out."""
    lines = ["**Snapshot data**"]
    if employees is not None:
        lines.append(f"{employees:,} employees in this snapshot.")
    if trained_on_rows is not None:
        lines.append(f"Model learned from {trained_on_rows:,} employees with a known outcome.")
    if exported_on is not None:
        lines.append(f"Model prepared on {exported_on.day} {exported_on:%b %Y}.")
    with st.sidebar:
        st.markdown("  \n".join(lines))
        st.caption(disclaimer)


def render_page_header(title: str, intro: str) -> None:
    """Page title and a one-sentence purpose."""
    st.title(title)
    st.markdown(intro)


def render_coming_soon() -> None:
    """Placeholder body for a page that is not built yet."""
    st.info("This page is being prepared and will be available soon.", icon=":material/schedule:")


def render_footer(disclaimer: str) -> None:
    """Disclaimer at the end of every page (also visible when the sidebar is collapsed)."""
    st.divider()
    st.caption(f":material/info: {disclaimer}")


def render_artifact_error(kind: str) -> None:
    """Plain message when a file the page needs is missing or invalid."""
    message = ARTIFACT_MESSAGES.get(kind, ARTIFACT_MESSAGES["unreadable"])
    st.error(f"{message} {OTHER_PAGES_HINT}", icon=":material/error:")


def render_unexpected_error() -> None:
    """Plain message for any other failure while showing a page."""
    st.error(
        f"Something went wrong while showing this page. {OTHER_PAGES_HINT}",
        icon=":material/error:",
    )
