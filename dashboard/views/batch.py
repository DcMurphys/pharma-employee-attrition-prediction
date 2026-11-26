"""Batch scoring tab: upload hint, validation report, summary and ranked results (§4.3)."""

from dataclasses import dataclass

import pandas as pd
import streamlit as st

from dashboard.views.components import ranked_table
from dashboard.views.components.feature_labels import label
from dashboard.views.components.format import count, rate

UPLOAD_HINT = "Use sample or synthetic data only. Do not upload real personal data."
FILE_ERRORS = {
    "too_large": "The file is larger than 5 MB. Please split it into smaller files.",
    "empty": "The file has no rows to score.",
    "unreadable": "The file could not be read as a CSV table. Please check its format.",
}


@dataclass(frozen=True)
class IssueText:
    """One validation finding, already resolved to plain values for display."""

    column: str
    kind: str
    rows: int
    low: float | None = None
    high: float | None = None
    allowed: str = ""


def render_intro() -> None:
    """What the tab does and the privacy hint."""
    st.markdown(
        "Score a file of employees at once. The file needs one row per employee and the same "
        "22 inputs as the single check; an `EmployeeId` column is optional."
    )
    st.caption(
        f":material/privacy_tip: {UPLOAD_HINT} Files are processed in memory and never stored."
    )


def render_file_error(kind: str) -> None:
    """Plain message when the upload is not a readable table."""
    st.error(FILE_ERRORS.get(kind, FILE_ERRORS["unreadable"]), icon=":material/error:")


def _message(issue: IssueText) -> str:
    name = f"**{label(issue.column)}** (`{issue.column}`)"
    rows = f"{count(issue.rows)} row{'s' if issue.rows != 1 else ''}"
    match issue.kind:
        case "missing_column":
            return f"{name}: the column is missing."
        case "empty":
            return f"{name}: {rows} with an empty value."
        case "not_numeric":
            return f"{name}: {rows} with a value that is not a number."
        case "out_of_scale":
            return (
                f"{name}: {rows} outside the scale; use whole numbers from "
                f"{issue.low:,.0f} to {issue.high:,.0f}."
            )
        case "unknown_category":
            return (
                f"{name}: {rows} with an unknown value (known values: {issue.allowed}). "
                "The model treats it as unknown."
            )
        case "out_of_range":
            return (
                f"{name}: {rows} outside the range seen in the training data "
                f"({issue.low:,.0f} to {issue.high:,.0f}). Scores for these rows are less reliable."
            )
    return f"{name}: {rows} with a problem."


def render_issues(blocking: list[IssueText], warnings: list[IssueText]) -> None:
    """Validation report: blocking issues stop scoring, warnings allow it with a notice."""
    if blocking:
        lines = "\n".join(f"- {_message(issue)}" for issue in blocking)
        st.error(
            f"The file cannot be scored yet. Please fix these problems and upload it again:\n\n"
            f"{lines}",
            icon=":material/block:",
        )
    if warnings:
        lines = "\n".join(f"- {_message(issue)}" for issue in warnings)
        heading = (
            "Please also check these values (they do not stop scoring on their own):"
            if blocking
            else "Scored with these notices:"
        )
        st.warning(f"{heading}\n\n{lines}", icon=":material/warning:")


def render_summary(rows: int, flagged: int, mean_probability: float, share: float) -> None:
    """Rows, flagged rows (top share) and mean risk."""
    columns = st.columns(3)
    columns[0].metric("Rows scored", count(rows), border=True)
    columns[1].metric(f"Flagged (top {rate(share, 0)})", count(flagged), border=True)
    columns[2].metric("Average risk", rate(mean_probability, 0), border=True)


def render_small_batch_note(min_rows: int) -> None:
    """Note for very small files (fewer than ``min_rows`` rows)."""
    st.info(
        f"This file has fewer than {min_rows} rows. Flagging the top 20% says little at this "
        "size; look at the risk of each row instead.",
        icon=":material/info:",
    )


def results_table(scored: pd.DataFrame, share: float, id_label: str) -> pd.DataFrame:
    """Ranked results for display: rank, identifier, risk, status and job details."""
    return pd.concat(
        [
            ranked_table.ranked_columns(scored, share, id_label, scope="file"),
            ranked_table.job_columns(scored),
        ],
        axis=1,
    )


def render_results(table: pd.DataFrame) -> None:
    """Ranked results (1 = highest risk in this file)."""
    st.caption(
        "Ranked within this file (1 = highest risk); the top 20% are flagged. Monthly income is "
        "in monetary units."
    )
    ranked_table.render(table)
