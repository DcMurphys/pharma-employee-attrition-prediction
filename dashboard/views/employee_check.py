"""Check an Employee: result card, cut-off note and what-if comparison (scope §4.3)."""

import math

import streamlit as st

from dashboard.views.components.format import count, rate
from dashboard.views.components.status_badge import badge_html, cutoff_status

WHAT_IF_CAPTION = (
    "The model shows an association, not a cause; use the change as a topic for the "
    "conversation, not as a promise."
)


def render_form_intro() -> None:
    """Heading and hint above the 22 inputs."""
    st.subheader("Employee details")
    st.caption(
        "Change any input and the result above updates straight away. Number ranges are the "
        "values seen in the data the model learned from."
    )


def render_group_title(title: str) -> None:
    """Heading of one input group (Job, Pay, Career history, Satisfaction)."""
    st.markdown(f"**{title}**")


def render_result(
    probability: float, above: bool, cutoff: float, share_below: float, scored: int
) -> None:
    """Result card: risk, status against the cut-off, and position among scored employees."""
    with st.container(border=True):
        st.markdown("#### Result")
        risk_col, status_col = st.columns([1, 2])
        risk_col.metric("Estimated risk", rate(probability, 0))
        with status_col:
            st.markdown(badge_html(cutoff_status(above), above), unsafe_allow_html=True)
            st.caption(
                f"Review cut-off: {rate(cutoff)}. This risk is higher than "
                # Rounded down, so the top employee reads "99%", never "100%".
                f"{math.floor(share_below * 100)}% of the {count(scored)} employees awaiting "
                "an outcome."
            )


def render_cutoff_note(
    cutoff: float, share: float, ranks: list[int], min_probability: float, max_probability: float
) -> None:
    """One sentence on the two flag rules (Q1)."""
    span = f"ranks {ranks[0]}–{ranks[-1]}" if len(ranks) > 1 else f"rank {ranks[0]}"
    st.caption(
        f":material/info: This check compares one employee with the {rate(cutoff)} cut-off. "
        f"The Retention Review List instead fills the top {rate(share, 0)} of review slots by "
        f"rank, so {span} there ({rate(min_probability)}–{rate(max_probability)}) are flagged "
        "although they are below this cut-off."
    )


def render_what_if_intro() -> None:
    """Heading and hint for the what-if section."""
    st.subheader("What if one thing changed?")
    st.caption("Pick one input and a new value to compare the risk before and after.")


def render_comparison(
    change: str,
    before_probability: float,
    before_above: bool,
    after_probability: float,
    after_above: bool,
) -> None:
    """Current and what-if results side by side, with the change in percentage points."""
    points = round(after_probability * 100) - round(before_probability * 100)
    current, changed = st.columns(2)
    for column, title, probability, above, delta in (
        (current, "Current", before_probability, before_above, None),
        (changed, f"What if: {change}", after_probability, after_above, f"{points:+d} points"),
    ):
        with column, st.container(border=True):
            st.markdown(f"**{title}**")
            st.metric(
                "Estimated risk",
                rate(probability, 0),
                delta=delta,
                delta_color="off",  # neutral: no green "good" signal (scope §7.3)
            )
            st.markdown(badge_html(cutoff_status(above), above), unsafe_allow_html=True)
    if points == 0:
        st.caption("This change makes no visible difference to the estimated risk.")
    st.caption(f":material/lightbulb: {WHAT_IF_CAPTION}")
