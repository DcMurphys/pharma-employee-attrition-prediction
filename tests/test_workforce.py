"""Workforce statistics reproduce the B1 values of scope §1 (NB §4.4-4.6)."""

import pandas as pd
import pytest

from dashboard.models import workforce as wf


def _rate(table: pd.DataFrame, group: str) -> float:
    return float(table.set_index("group").loc[group, "rate"])


def test_kpis_match_notebook(workforce) -> None:
    kpis = wf.kpis(workforce)
    assert (kpis.employees, kpis.known_outcome, kpis.left, kpis.awaiting_outcome) == (
        1470,
        1058,
        179,
        412,
    )
    assert round(kpis.attrition_rate, 3) == 0.169


@pytest.mark.parametrize(
    ("column", "group", "expected"),
    [
        ("Department", "Sales", 0.207),
        ("OverTime", "Works overtime", 0.319),
        ("OverTime", "No overtime", 0.108),
        ("JobLevel", "Level 1", 0.274),
        ("WorkLifeBalance", "Bad", 0.321),
        ("StockOptionLevel", "No stock options", 0.257),
        ("BusinessTravel", "Travels frequently", 0.249),
    ],
)
def test_attrition_rates_match_notebook(workforce, column, group, expected) -> None:
    assert round(_rate(wf.attrition_by(workforce, column), group), 3) == expected


def test_attrition_table_is_sorted_and_marks_small_groups(workforce) -> None:
    table = wf.attrition_by(workforce, "JobRole")
    assert table["rate"].is_monotonic_decreasing
    assert table["employees"].sum() == 1058
    assert table.loc[0, "group"] == "Sales Representative"
    assert set(table.loc[table["small_group"], "employees"]) <= set(range(wf.SMALL_GROUP_MIN))


def test_median_income_matches_notebook(workforce) -> None:
    assert wf.median_income(workforce, wf.LEFT) == 3388
    assert wf.median_income(workforce, wf.STAYED) == 5210


def test_income_by_level_covers_known_outcomes(workforce) -> None:
    table = wf.median_income_by_level(workforce)
    assert table["employees"].sum() == 1058
    assert list(table["level"].unique()) == [f"Level {n}" for n in range(1, 6)]
    level4_left = table[(table["level"] == "Level 4") & (table["status"] == "Left")]
    assert bool(level4_left["small_group"].iloc[0])


def test_finding_values_for_all_employees(workforce) -> None:
    values = wf.finding_values(workforce)
    assert round(values.overtime_rate, 3) == 0.319
    assert round(values.no_overtime_rate, 3) == 0.108
    assert round(values.level1_rate, 3) == 0.274
    assert round(values.higher_level_min_rate, 3) == 0.050
    assert round(values.higher_level_max_rate, 3) == 0.152
    assert (values.leaver_median_income, values.stayer_median_income) == (3388, 5210)
    assert round(values.no_stock_rate, 3) == 0.257
    assert (values.top_department, round(values.top_department_rate, 3)) == ("Sales", 0.207)


def test_department_filter(workforce) -> None:
    sales = wf.filter_department(workforce, "Sales")
    kpis = wf.kpis(sales)
    assert (kpis.known_outcome, kpis.left) == (319, 66)
    assert wf.finding_values(sales).top_department is None  # one department: no comparison
    assert wf.filter_department(workforce, None) is workforce
    assert wf.departments(workforce) == ["Human Resources", "Research & Development", "Sales"]


def test_no_sensitive_breakdowns() -> None:
    assert not {"Gender", "MaritalStatus", "Age"} & set(wf.SEGMENTS)


def test_attrition_by_pay_band(workforce) -> None:
    table = wf.attrition_by_pay_band(workforce)
    assert table["band"].tolist() == [1, 2, 3, 4]
    assert table["employees"].sum() == 1058 and table["employees"].between(264, 265).all()
    assert table["max_income"].is_monotonic_increasing
    assert (table.loc[0, "min_income"], table.loc[0, "max_income"]) == (1009, 2899)
    assert [round(r, 3) for r in table["rate"]] == [0.291, 0.167, 0.121, 0.098]


def test_small_group_threshold(workforce) -> None:
    departments = wf.attrition_by(workforce, "Department").set_index("group")
    assert bool(departments.loc["Human Resources", "small_group"])  # 38 employees
    assert not departments.loc["Sales", "small_group"].item()
    roles = wf.attrition_by(workforce, "JobRole").set_index("group")
    assert bool(roles.loc["Human Resources", "small_group"])  # 30 employees
    assert not roles.loc["Sales Representative", "small_group"].item()  # 58 employees
