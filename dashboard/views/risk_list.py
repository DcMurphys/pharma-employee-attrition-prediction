"""Retention Review List: headline, ranked table, rank-rule note, team counts (scope §4.2)."""

import pandas as pd
import streamlit as st

from dashboard.views.components import charts, ranked_table
from dashboard.views.components.format import count, rate
from dashboard.views.components.palette import BRAND_TEAL

FLAGGED_LEGEND = [charts.LegendItem("Flagged employees", BRAND_TEAL)]
TEAM_TOOLTIP = [(charts.TOOLTIP_HEADING, "group"), ("Flagged for review", "tip_flagged")]


def team_tooltips(counts: pd.DataFrame, team: str) -> pd.DataFrame:
    """Tooltip sentence per team: count and share of all flagged employees in the selection."""
    total = int(counts["count"].sum())
    return counts.assign(
        tip_flagged=[
            f"{count(n)} employee{'s' if n != 1 else ''}, {rate(n / total, 0)} of the "
            f"{count(total)} flagged employees in this selection ({team.lower()} view)"
            for n in counts["count"]
        ]
    )


def render_headline(flagged: int, employees: int, share: float) -> None:
    """ "83 of 412 employees flagged for review this round (top 20%)." """
    st.markdown(
        f"#### {count(flagged)} of {count(employees)} employees flagged for review this round "
        f"(top {rate(share, 0)})"
    )


def review_table(rows: pd.DataFrame, share: float) -> pd.DataFrame:
    """Display columns for the ranked list (also used for the CSV download)."""
    ranked = ranked_table.ranked_columns(rows, share, "Employee ID")
    job = ranked_table.job_columns(rows)
    return pd.concat(
        [ranked, pd.DataFrame({"Department": rows["Department"].to_numpy()}), job], axis=1
    )


def render_table(table: pd.DataFrame, employees: int) -> None:
    """Ranked list in rank order (1 = highest risk)."""
    st.caption(
        f"Showing {count(len(table))} of {count(employees)} employees, in rank order "
        "(1 = highest risk). Monthly income is in monetary units."
    )
    if table.empty:
        st.info("No employees match these filters.", icon=":material/filter_alt_off:")
        return
    ranked_table.render(table)


def render_rank_note(
    flagged: int, ranks: list[int], min_probability: float, max_probability: float, cutoff: float
) -> None:
    """One sentence on why the last flagged ranks sit below the single-check cut-off (Q1)."""
    span = f"Ranks {ranks[0]}–{ranks[-1]}" if len(ranks) > 1 else f"Rank {ranks[0]}"
    st.caption(
        f":material/info: This list flags by rank: the {count(flagged)} highest-risk employees "
        f"fill this round's review slots. {span} ({rate(min_probability)}–"
        f"{rate(max_probability)}) are just below the {rate(cutoff)} cut-off used on Check an "
        "Employee, so they are flagged here but would show as below the cut-off there."
    )


def render_team_counts(by_department: pd.DataFrame, by_role: pd.DataFrame) -> None:
    """Flagged employees per department and per job role, side by side on wide screens."""
    st.subheader("Flagged employees by team")
    if by_department.empty:
        st.info("No flagged employees in this selection.")
        return
    st.caption("Counts follow the department, job role and overtime filters above.")
    for column, (title, table) in zip(
        st.columns(2),
        [("Department", by_department), ("Job role", by_role)],
        strict=True,
    ):
        with column:
            st.markdown(f"**{title}**")
            data = table.rename(columns={"flagged": "count"})
            charts.show(
                charts.count_bars(
                    team_tooltips(data, title),
                    value_title="Flagged employees",
                    tooltip=TEAM_TOOLTIP,
                ),
                pd.DataFrame({title: data["group"], "Flagged employees": data["count"]}),
                description=f"Flagged employees by {title.lower()}.",
                legend=FLAGGED_LEGEND,
            )
