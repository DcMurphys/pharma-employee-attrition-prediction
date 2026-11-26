"""Workforce Insights: KPI row and three tabs (scope §4.1). Receives computed tables only."""

from dataclasses import dataclass

import pandas as pd
import streamlit as st

from dashboard.views.components import charts
from dashboard.views.components.format import count, money, rate
from dashboard.views.components.kpi_card import Kpi, kpi_row

SMALL_LABEL = "Small group"
SMALL_MARK = "small"  # short marker printed after a small group's bar, so it fits at 375 px


@dataclass(frozen=True)
class Segment:
    """One breakdown on "Where attrition concentrates"."""

    title: str
    table: pd.DataFrame  # group, employees, leavers, rate, small_group


def render_kpis(
    employees: int,
    known_outcome: int,
    left: int,
    attrition_rate: float,
    awaiting_outcome: int,
    scope: str,
) -> None:
    """Headline numbers and the caption naming the population and the active filter."""
    kpi_row(
        [
            Kpi("Employees", count(employees)),
            Kpi("Known outcome", count(known_outcome), "Employees who either left or stayed."),
            Kpi("Left", count(left)),
            Kpi("Attrition rate", rate(attrition_rate), "Leavers among known outcomes."),
            Kpi(
                "Awaiting outcome",
                count(awaiting_outcome),
                "Employees without a recorded outcome; they appear on the Retention Review List.",
            ),
        ]
    )
    st.caption(
        f"Showing {scope}. Rates use only the {count(known_outcome)} employees with a known "
        "outcome."
    )


def render_tabs(
    segments: list[Segment],
    reference_rate: float,
    reference_label: str,
    small_group_min: int,
    pay_bands: pd.DataFrame,
    income_by_level: pd.DataFrame,
    level1_leaver_share: float | None,
    findings: list[str],
) -> None:
    """The three tabs: concentration, pay, key findings."""
    concentration, pay, key_findings = st.tabs(
        ["Where attrition concentrates", "Pay", "Key findings"]
    )
    with concentration:
        _render_concentration(segments, reference_rate, reference_label, small_group_min)
    with pay:
        _render_pay(
            pay_bands, income_by_level, level1_leaver_share, reference_rate, reference_label
        )
    with key_findings:
        _render_findings(findings)


def _render_concentration(
    segments: list[Segment], reference_rate: float, reference_label: str, small_group_min: int
) -> None:
    st.caption(
        "Bars are sorted by attrition rate; the number in brackets is the group size, and the "
        f"dashed line is the {reference_label.lower()} ({rate(reference_rate)}). "
        f'Grey bars marked "{SMALL_MARK}" are small groups: they have fewer than {small_group_min} '
        "employees with a known outcome, so one person moves their rate by 2 percentage "
        "points or more."
    )
    for segment in segments:
        st.subheader(segment.title)
        _rate_chart(segment.table, segment.title, reference_rate, reference_label)


RATE_TOOLTIP = [
    (charts.TOOLTIP_HEADING, "tip_group"),
    ("Who left", "tip_left"),
    ("Compared with the average", "tip_compare"),
    ("Group size", "tip_size"),
]


def rate_tooltips(table: pd.DataFrame, reference: float, reference_label: str) -> pd.DataFrame:
    """Tooltip sentences for attrition-rate bars (columns named in ``RATE_TOOLTIP``)."""
    group, left, compare, size = [], [], [], []
    for name, r, n, leavers, small in zip(
        table["group"],
        table["rate"],
        table["employees"],
        table["leavers"],
        table["small_group"],
        strict=True,
    ):
        group.append(name.replace("\n", " "))
        left.append(f"{rate(r)}, that is {count(leavers)} of {count(n)} employees")
        gap = round((r - reference) * 100, 1)
        average = f"the {reference_label.lower()} ({rate(reference)})"
        if gap == 0:
            compare.append(f"Same as {average}")
        else:
            direction = "above" if gap > 0 else "below"
            compare.append(f"{abs(gap):.1f} percentage points {direction} {average}")
        note = f"{count(n)} employees with a known outcome"
        if small:
            note += (
                f"; small group, so one person moves the rate by {100 / n:.1f} percentage "
                "points: read with care"
            )
        size.append(note)
    return table.assign(tip_group=group, tip_left=left, tip_compare=compare, tip_size=size)


def _rate_chart(
    table: pd.DataFrame,
    title: str,
    reference_rate: float,
    reference_label: str,
    row_height: int = charts.BAR_ROW_HEIGHT,
) -> None:
    """Attrition-rate bars for one breakdown, with its table alternative."""
    labelled = table.assign(
        label=[
            f"{rate(r)} ({count(n)}{', ' + SMALL_MARK if small else ''})"
            for r, n, small in zip(
                table["rate"], table["employees"], table["small_group"], strict=True
            )
        ]
    )
    chart = charts.rate_bars(
        rate_tooltips(labelled, reference_rate, reference_label),
        reference=reference_rate,
        reference_label=f"{reference_label} {rate(reference_rate)}",
        small_label=SMALL_LABEL,
        row_height=row_height,
        tooltip=RATE_TOOLTIP,
        reference_tooltip=[
            (charts.TOOLTIP_HEADING, reference_label),
            (
                "Who left",
                f"{rate(reference_rate)} of employees with a known outcome; bars to the "
                "right of this line lose more people than average",
            ),
        ],
    )
    charts.show(
        chart,
        pd.DataFrame(
            {
                title: table["group"].str.replace("\n", " "),
                "Employees": table["employees"].map(count),
                "Left": table["leavers"].map(count),
                "Attrition rate": table["rate"].map(rate),
                "Note": table["small_group"].map({True: SMALL_LABEL, False: ""}),
            }
        ),
        description=f"Attrition rate by {title.lower()}.",
        legend=charts.rate_legend(
            has_small=bool(table["small_group"].any()),
            small_label=SMALL_LABEL,
            reference_label=f"{reference_label} {rate(reference_rate)}",
        ),
    )


def _render_pay(
    pay_bands: pd.DataFrame,
    by_level: pd.DataFrame,
    level1_leaver_share: float | None,
    reference_rate: float,
    reference_label: str,
) -> None:
    st.subheader("Attrition rate by pay band")
    st.caption(
        f"Employees are split into {len(pay_bands)} equal-sized groups by monthly income, from "
        "the lowest-paid to the highest-paid. The range in brackets is the monthly income in "
        "monetary units."
    )
    _rate_chart(
        pay_bands.assign(group=pay_band_labels(pay_bands)),
        "Pay band",
        reference_rate,
        reference_label,
        row_height=2 * charts.BAR_ROW_HEIGHT,  # two-line labels
    )

    st.subheader("Median monthly income by job level")
    labelled = by_level.assign(
        label=[
            f"{money(value)}{' (' + count(n) + ', ' + SMALL_MARK + ')' if small else ''}"
            for value, n, small in zip(
                by_level["median_income"],
                by_level["employees"],
                by_level["small_group"],
                strict=True,
            )
        ]
    )
    charts.show(
        charts.grouped_income_bars(income_tooltips(labelled), tooltip=INCOME_TOOLTIP),
        pd.DataFrame(
            {
                "Job level": by_level["level"],
                "Status": by_level["status"],
                "Employees": by_level["employees"].map(count),
                "Median monthly income": by_level["median_income"].map(money),
                "Note": by_level["small_group"].map({True: SMALL_LABEL, False: ""}),
            }
        ),
        description="Median monthly income by job level for employees who stayed and who left.",
        legend=charts.STATUS_LEGEND,
    )
    caption = "Pay rises steeply with job level, so pay differences largely reflect seniority."
    if level1_leaver_share is not None:
        caption += (
            f" {rate(level1_leaver_share, 0)} of leavers were at Level 1, the lowest-paid level."
        )
    st.caption(caption)


INCOME_TOOLTIP = [
    (charts.TOOLTIP_HEADING, "tip_title"),
    ("Median monthly income", "tip_income"),
    ("Employees", "tip_size"),
]


def income_tooltips(by_level: pd.DataFrame) -> pd.DataFrame:
    """Tooltip sentences for median income by job level (columns named in ``INCOME_TOOLTIP``)."""
    titles, incomes, sizes = [], [], []
    for level, status, value, n, small in zip(
        by_level["level"],
        by_level["status"],
        by_level["median_income"],
        by_level["employees"],
        by_level["small_group"],
        strict=True,
    ):
        who = "employees who left" if status == "Left" else "employees who stayed"
        titles.append(f"{level}, {who}")
        incomes.append(
            f"{money(value)} monetary units: half of these employees earn less, half earn more"
        )
        note = f"{count(n)} {who}"
        if small:
            note += "; small group, so the median can shift a lot with one person"
        sizes.append(note)
    return by_level.assign(tip_title=titles, tip_income=incomes, tip_size=sizes)


def pay_band_labels(pay_bands: pd.DataFrame) -> list[str]:
    """Band names with the income range on a second line ("Lowest pay", "(1,009–2,899)")."""
    last = len(pay_bands)
    names = {1: "Lowest pay", last: "Highest pay"}
    if last == 4:
        names |= {2: "Lower-middle pay", 3: "Upper-middle pay"}
    return [
        f"{names.get(band, f'Pay band {band}')}\n({money(low)}–{money(high)})"
        for band, low, high in zip(
            pay_bands["band"], pay_bands["min_income"], pay_bands["max_income"], strict=True
        )
    ]


def _render_findings(findings: list[str]) -> None:
    st.caption(
        "These are associations in past data, not causes. Use them to choose topics for "
        "conversations, not to judge individuals."
    )
    if not findings:
        st.info("There are too few employees in this selection for key findings.")
    for number, finding in enumerate(findings, start=1):
        st.markdown(f"{number}. {finding}")


def findings_text(
    overtime_rate: float | None,
    no_overtime_rate: float | None,
    level1_rate: float | None,
    higher_level_min_rate: float | None,
    higher_level_max_rate: float | None,
    leaver_median_income: float | None,
    stayer_median_income: float | None,
    no_stock_rate: float | None,
    with_stock_rate: float | None,
    top_department: str | None,
    top_department_rate: float | None,
) -> list[str]:
    """Plain-language findings (NB §4.6, §9.1); a finding is left out if a value is missing."""
    findings = []
    if overtime_rate is not None and no_overtime_rate:
        times = round(overtime_rate / no_overtime_rate * 2) / 2
        findings.append(
            f"Employees who work overtime leave about {times:g} times as often as those who "
            f"do not: **{rate(overtime_rate)}** vs. **{rate(no_overtime_rate)}**."
        )
    if None not in (level1_rate, higher_level_min_rate, higher_level_max_rate):
        findings.append(
            f"Entry-level employees (Level 1) left at **{rate(level1_rate)}**, compared with "
            f"{rate(higher_level_min_rate)} to {rate(higher_level_max_rate)} at higher job "
            "levels."
        )
    if leaver_median_income and stayer_median_income:
        gap = 1 - leaver_median_income / stayer_median_income
        direction = "less" if gap > 0 else "more"
        findings.append(
            f"Leavers earned about {rate(abs(gap), 0)} {direction} than stayers: median monthly "
            f"income **{money(leaver_median_income)}** vs. **{money(stayer_median_income)}** "
            "monetary units."
        )
    if no_stock_rate is not None and with_stock_rate is not None:
        findings.append(
            f"Employees without stock options left at **{rate(no_stock_rate)}**, compared with "
            f"{rate(with_stock_rate)} for employees with stock options."
        )
    if top_department is not None and top_department_rate is not None:
        findings.append(
            f"Among departments, {top_department} has the highest attrition rate: "
            f"**{rate(top_department_rate)}**."
        )
    return findings
