"""Workforce Insights controller: department filter, workforce statistics, rendering (§4.1)."""

from dataclasses import asdict

import streamlit as st

from dashboard.controllers import data
from dashboard.controllers.shell import page_header
from dashboard.models import workforce as wf
from dashboard.views import overview as view

ALL_DEPARTMENTS = "All departments"
DEPARTMENT_KEY = "overview_department"


def render() -> None:
    """Draw the Workforce Insights page."""
    page_header("overview")
    workforce = data.workforce()

    choice = st.selectbox(
        "Department",
        [ALL_DEPARTMENTS, *wf.departments(workforce)],
        key=DEPARTMENT_KEY,
        width=320,
    )
    department = None if choice == ALL_DEPARTMENTS else choice
    selected = wf.filter_department(workforce, department)

    kpis = wf.kpis(selected)
    view.render_kpis(
        kpis.employees,
        kpis.known_outcome,
        kpis.left,
        kpis.attrition_rate,
        kpis.awaiting_outcome,
        scope="all departments" if department is None else f"the {department} department",
    )
    if kpis.known_outcome == 0:
        st.info("No employees with a known outcome in this selection.")
        return

    view.render_tabs(
        segments=[
            view.Segment(title, wf.attrition_by(selected, column))
            for column, (title, _labels) in wf.SEGMENTS.items()
        ],
        reference_rate=kpis.attrition_rate,
        reference_label="Company rate" if department is None else "Department rate",
        small_group_min=wf.SMALL_GROUP_MIN,
        pay_bands=wf.attrition_by_pay_band(selected),
        income_by_level=wf.median_income_by_level(selected),
        level1_leaver_share=wf.leaver_share_at_level(selected, 1),
        findings=view.findings_text(**asdict(wf.finding_values(selected))),
    )
