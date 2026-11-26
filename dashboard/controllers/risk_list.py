"""Retention Review List controller: filters, ranked list, download, team counts (§4.2)."""

import logging

import streamlit as st

from dashboard.controllers import data
from dashboard.controllers.shell import page_header
from dashboard.models import export
from dashboard.models import risk_list as rl
from dashboard.models.artifacts import ArtifactError
from dashboard.views import risk_list as view
from dashboard.views.components import notices
from dashboard.views.components.download import csv_download
from employee_attrition.config import DATA_DISCLAIMER

logger = logging.getLogger(__name__)

KEY_DEPARTMENTS = "risk_departments"
KEY_ROLES = "risk_roles"
KEY_OVERTIME = "risk_overtime"
KEY_FLAGGED_ONLY = "risk_flagged_only"
OVERTIME_CHOICES = {"All": None, "Works overtime": "Yes", "No overtime": "No"}
DOWNLOAD_STEM = "retention_review_list"


def render() -> None:
    """Draw the Retention Review List page."""
    page_header("risk_list")
    notices.intended_use()
    scores = data.risk_scores()
    summary = rl.round_summary(scores)
    view.render_headline(summary.flagged, summary.employees, summary.capacity_share)

    filters = _filters(scores)
    rows = rl.filter_list(scores, **filters)
    table = view.review_table(rows, summary.capacity_share)
    view.render_table(table, summary.employees)
    _rank_note(scores, summary.flagged)
    csv_download(
        "Download this list (CSV)",
        export.csv_with_notes(table, [DATA_DISCLAIMER, notices.INTENDED_USE]),
        export.demo_file_name(DOWNLOAD_STEM),
        key="risk_download",
    )

    team = rl.filter_list(scores, **{**filters, "flagged_only": True})
    view.render_team_counts(
        rl.flagged_counts(team, "Department"), rl.flagged_counts(team, "JobRole")
    )


def _filters(scores) -> dict:
    """Draw the filter widgets and return the keyword arguments for ``filter_list``."""
    department_col, role_col, overtime_col, flagged_col = st.columns([3, 3, 3, 2])
    departments = department_col.multiselect(
        "Department", rl.options(scores, "Department"), key=KEY_DEPARTMENTS, placeholder="All"
    )
    within = scores["Department"].isin(departments) if departments else None
    roles = rl.options(scores, "JobRole", within)
    # Drop job roles that no longer belong to the selected departments.
    if KEY_ROLES in st.session_state:
        st.session_state[KEY_ROLES] = [r for r in st.session_state[KEY_ROLES] if r in roles]
    job_roles = role_col.multiselect("Job role", roles, key=KEY_ROLES, placeholder="All")
    overtime = overtime_col.segmented_control(
        "Overtime", list(OVERTIME_CHOICES), key=KEY_OVERTIME, default="All", required=True
    )
    flagged_col.markdown("")  # aligns the toggle with the other inputs
    flagged_only = flagged_col.toggle("Flagged only", value=True, key=KEY_FLAGGED_ONLY)
    return {
        "departments": departments,
        "job_roles": job_roles,
        "overtime": OVERTIME_CHOICES[overtime],
        "flagged_only": flagged_only,
    }


def _rank_note(scores, flagged: int) -> None:
    """Q1 note; left out (and logged) if the model details are unavailable."""
    try:
        cutoff = float(data.model_metadata()["decision_threshold"])
    except ArtifactError as err:
        logger.warning("Rank note left out: %s", err)
        return
    gap = rl.flagged_below_cutoff(scores, cutoff)
    if gap is not None:
        view.render_rank_note(
            flagged, gap.ranks, gap.min_probability, gap.max_probability, gap.cutoff
        )
