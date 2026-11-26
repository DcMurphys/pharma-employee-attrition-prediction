"""Descriptive workforce statistics for Workforce Insights (B1; NB §4.4-4.6, §9.1).

Rates use only employees with a known outcome (``AttritionStatus`` "Yes" or "No"). Gender,
marital status and age are deliberately not offered here (scope Q3). This module must not
import streamlit.
"""

from dataclasses import dataclass

import pandas as pd

from employee_attrition.config import ORDINAL_LABELS, STATUS_COL

LEFT, STAYED, UNKNOWN = "Yes", "No", "Unknown"
STATUS_LABELS = {LEFT: "Left", STAYED: "Stayed"}
# Groups below this many employees with a known outcome get a "small group" note: below 50,
# one employee moves the rate by 2 percentage points or more (NB §4.4, Insight 1).
SMALL_GROUP_MIN = 50
INCOME_COL = "MonthlyIncome"
PAY_BANDS = 4  # quarters of the workforce by monthly income
DEPARTMENT_COL = "Department"

# Breakdowns on "Where attrition concentrates": column -> (title, value labels or None).
SEGMENTS: dict[str, tuple[str, dict | None]] = {
    "JobRole": ("Job role", None),
    "JobLevel": ("Job level", ORDINAL_LABELS["JobLevel"]),
    "OverTime": ("Overtime", {"Yes": "Works overtime", "No": "No overtime"}),
    "WorkLifeBalance": ("Work-life balance", ORDINAL_LABELS["WorkLifeBalance"]),
    "BusinessTravel": (
        "Business travel",
        {
            "Non-Travel": "Does not travel",
            "Travel_Rarely": "Travels rarely",
            "Travel_Frequently": "Travels frequently",
        },
    ),
    "StockOptionLevel": (
        "Stock option level",
        {0: "No stock options", 1: "Level 1", 2: "Level 2", 3: "Level 3"},
    ),
}


@dataclass(frozen=True)
class Kpis:
    """Headline counts for the selected employees."""

    employees: int
    known_outcome: int
    left: int
    awaiting_outcome: int

    @property
    def attrition_rate(self) -> float:
        """Share of leavers among employees with a known outcome (0 if there are none)."""
        return self.left / self.known_outcome if self.known_outcome else 0.0


@dataclass(frozen=True)
class FindingValues:
    """Computed values behind the key findings; ``None`` when a group is absent."""

    overtime_rate: float | None
    no_overtime_rate: float | None
    level1_rate: float | None
    higher_level_min_rate: float | None
    higher_level_max_rate: float | None
    leaver_median_income: float | None
    stayer_median_income: float | None
    no_stock_rate: float | None
    with_stock_rate: float | None
    top_department: str | None
    top_department_rate: float | None


def departments(workforce: pd.DataFrame) -> list[str]:
    """Department names, sorted."""
    return sorted(workforce[DEPARTMENT_COL].dropna().unique())


def filter_department(workforce: pd.DataFrame, department: str | None) -> pd.DataFrame:
    """Employees of one department, or everyone when ``department`` is ``None``."""
    if department is None:
        return workforce
    return workforce[workforce[DEPARTMENT_COL] == department]


def known_outcome(workforce: pd.DataFrame) -> pd.DataFrame:
    """Employees whose outcome (left or stayed) is recorded."""
    return workforce[workforce[STATUS_COL].isin([LEFT, STAYED])]


def _left(frame: pd.DataFrame) -> pd.Series:
    return frame[STATUS_COL].eq(LEFT)


def kpis(workforce: pd.DataFrame) -> Kpis:
    """Headline counts: all employees, known outcome, leavers, awaiting an outcome."""
    status = workforce[STATUS_COL]
    return Kpis(
        employees=len(workforce),
        known_outcome=int(status.isin([LEFT, STAYED]).sum()),
        left=int(status.eq(LEFT).sum()),
        awaiting_outcome=int(status.eq(UNKNOWN).sum()),
    )


def attrition_by(workforce: pd.DataFrame, column: str) -> pd.DataFrame:
    """Attrition rate per group of ``column``, highest rate first.

    Returns:
        Columns ``group`` (display label), ``employees`` (known outcome), ``leavers``, ``rate``
        and ``small_group`` (fewer than ``SMALL_GROUP_MIN`` employees).
    """
    known = known_outcome(workforce)
    table = (
        _left(known)
        .groupby(known[column], observed=True)
        .agg(employees="size", leavers="sum")
        .reset_index(names="value")
    )
    labels = SEGMENTS.get(column, ("", None))[1] or {}
    table["group"] = table["value"].map(lambda value: labels.get(value, str(value)))
    table["rate"] = table["leavers"] / table["employees"]
    table["small_group"] = table["employees"] < SMALL_GROUP_MIN
    table = table.sort_values(["rate", "employees"], ascending=[False, False])
    return table[["group", "employees", "leavers", "rate", "small_group"]].reset_index(drop=True)


def rate_of(workforce: pd.DataFrame, column: str, value: object) -> float | None:
    """Attrition rate of one group (``None`` if nobody with a known outcome is in it)."""
    known = known_outcome(workforce)
    members = known[known[column] == value]
    return float(_left(members).mean()) if len(members) else None


def attrition_by_pay_band(workforce: pd.DataFrame, bands: int = PAY_BANDS) -> pd.DataFrame:
    """Attrition rate for equal-sized monthly income bands, lowest pay first.

    Employees with a known outcome are split into ``bands`` groups of (nearly) equal size by
    monthly income, so every band is large enough to compare.

    Returns:
        Columns ``band`` (1 = lowest pay), ``min_income``, ``max_income``, ``employees``,
        ``leavers``, ``rate`` and ``small_group``.
    """
    known = known_outcome(workforce)
    band = pd.qcut(known[INCOME_COL], bands, labels=False, duplicates="drop") + 1
    table = (
        pd.DataFrame({"band": band, "income": known[INCOME_COL], "left": _left(known)})
        .groupby("band")
        .agg(
            min_income=("income", "min"),
            max_income=("income", "max"),
            employees=("left", "size"),
            leavers=("left", "sum"),
        )
        .reset_index()
    )
    table["rate"] = table["leavers"] / table["employees"]
    table["small_group"] = table["employees"] < SMALL_GROUP_MIN
    return table


def median_income(workforce: pd.DataFrame, status: str) -> float | None:
    """Median monthly income of leavers (``"Yes"``) or stayers (``"No"``)."""
    incomes = workforce.loc[workforce[STATUS_COL] == status, INCOME_COL]
    return float(incomes.median()) if len(incomes) else None


def median_income_by_level(workforce: pd.DataFrame) -> pd.DataFrame:
    """Median monthly income by job level and status, with group sizes.

    Returns:
        Columns ``level`` (label), ``status`` ("Left"/"Stayed"), ``employees``,
        ``median_income`` and ``small_group``; ordered by level, then status.
    """
    known = known_outcome(workforce)
    table = (
        known.groupby(["JobLevel", STATUS_COL], observed=True)[INCOME_COL]
        .agg(employees="size", median_income="median")
        .reset_index()
        .sort_values(["JobLevel", STATUS_COL])
    )
    table["level"] = table["JobLevel"].map(ORDINAL_LABELS["JobLevel"])
    table["status"] = table[STATUS_COL].map(STATUS_LABELS)
    table["small_group"] = table["employees"] < SMALL_GROUP_MIN
    return table[["level", "status", "employees", "median_income", "small_group"]].reset_index(
        drop=True
    )


def leaver_share_at_level(workforce: pd.DataFrame, level: int = 1) -> float | None:
    """Share of all leavers who are at ``level``."""
    leavers = workforce[_left(workforce)]
    return float(leavers["JobLevel"].eq(level).mean()) if len(leavers) else None


def finding_values(workforce: pd.DataFrame) -> FindingValues:
    """Values for the key findings (NB §4.6, §9.1), computed for the selected employees."""
    level_rates = attrition_by(workforce, "JobLevel").set_index("group")["rate"]
    level1 = ORDINAL_LABELS["JobLevel"][1]
    higher = level_rates.drop(level1, errors="ignore")
    known = known_outcome(workforce)
    stock = known["StockOptionLevel"] > 0
    with_stock = known[stock]

    top_department = top_rate = None
    if workforce[DEPARTMENT_COL].nunique() > 1:
        by_department = attrition_by(workforce, DEPARTMENT_COL)
        top_department = str(by_department.loc[0, "group"])
        top_rate = float(by_department.loc[0, "rate"])

    return FindingValues(
        overtime_rate=rate_of(workforce, "OverTime", "Yes"),
        no_overtime_rate=rate_of(workforce, "OverTime", "No"),
        level1_rate=float(level_rates[level1]) if level1 in level_rates else None,
        higher_level_min_rate=float(higher.min()) if len(higher) else None,
        higher_level_max_rate=float(higher.max()) if len(higher) else None,
        leaver_median_income=median_income(workforce, LEFT),
        stayer_median_income=median_income(workforce, STAYED),
        no_stock_rate=rate_of(workforce, "StockOptionLevel", 0),
        with_stock_rate=float(_left(with_stock).mean()) if len(with_stock) else None,
        top_department=top_department,
        top_department_rate=top_rate,
    )
