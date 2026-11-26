"""Ranked risk table shared by the Retention Review List and batch scoring (scope §4.2-4.3).

A static table (not ``st.dataframe``): it has no built-in download button that would bypass the
disclaimer in the app's own CSV downloads. Cells stay on one line (``one_line``); on a narrow
screen the table scrolls sideways instead of breaking a status into a column of single words.
"""

import pandas as pd
import streamlit as st

from dashboard.views.components.format import rate
from dashboard.views.components.one_line import one_line
from dashboard.views.components.palette import SLATE, STATUS_FLAG
from dashboard.views.components.status_badge import FLAG_ICON, rank_status
from employee_attrition.config import ORDINAL_LABELS

STATUS = "Review status"
RISK = "Risk (%)"  # whole percentages in the CSV; "Risk" with a % sign on screen
TABLE_HEIGHT = 420  # px; long tables scroll inside the page
KEY = "ranked_table"


def ranked_columns(
    rows: pd.DataFrame, share: float, id_label: str, scope: str = "round"
) -> pd.DataFrame:
    """Rank, identifier, risk and review status of scored ``rows`` (index = identifier)."""
    return pd.DataFrame(
        {
            "Rank": rows["risk_rank"].to_numpy(),
            id_label: rows.index.to_numpy(),
            RISK: (rows["attrition_probability"] * 100).round().astype(int).to_numpy(),
            STATUS: [
                rank_status(bool(flag), rate(share, 0), scope) for flag in rows["attrition_flag"]
            ],
        }
    )


def job_columns(rows: pd.DataFrame) -> pd.DataFrame:
    """Job role, job level, overtime and monthly income, as shown next to the ranking."""
    return pd.DataFrame(
        {
            "Job role": rows["JobRole"].to_numpy(),
            "Job level": rows["JobLevel"].map(ORDINAL_LABELS["JobLevel"]).to_numpy(),
            "Overtime": rows["OverTime"].to_numpy(),
            "Monthly income": rows["MonthlyIncome"].to_numpy(),
        }
    )


def render(table: pd.DataFrame) -> None:
    """Draw a ranked table; flagged status in the ``status-flag`` colour with its icon."""

    def status_style(value: str) -> str:
        flagged = value.startswith(FLAG_ICON)
        return f"color: {STATUS_FLAG}; font-weight: 600" if flagged else f"color: {SLATE}"

    styled = (
        table.rename(columns={RISK: "Risk"})
        .style.map(status_style, subset=[STATUS])
        .format({"Risk": "{}%", "Monthly income": "{:,}"})
        .hide(axis="index")
    )
    with one_line(KEY, TABLE_HEIGHT if len(table) > 10 else "content"):
        st.table(styled)
