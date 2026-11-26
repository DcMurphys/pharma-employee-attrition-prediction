"""Chart tooltips: plain sentences with a heading, built from the same numbers as the chart."""

import pandas as pd

from dashboard.views import overview, risk_list
from dashboard.views.components import charts

RATES = pd.DataFrame(
    {
        "group": ["Sales Representative", "Human Resources", "Manager"],
        "rate": [0.431, 0.2, 0.169],
        "employees": [58, 30, 79],
        "leavers": [25, 6, 13],
        "small_group": [False, True, False],
        "label": ["43.1% (58)", "20.0% (30, small)", "16.9% (79)"],
    }
)


def test_rate_tooltips_explain_the_numbers() -> None:
    tips = overview.rate_tooltips(RATES, 0.169, "Company rate").set_index("tip_group")
    assert tips.at["Sales Representative", "tip_left"] == "43.1%, that is 25 of 58 employees"
    assert tips.at["Sales Representative", "tip_compare"] == (
        "26.2 percentage points above the company rate (16.9%)"
    )
    assert tips.at["Manager", "tip_compare"] == "Same as the company rate (16.9%)"
    assert "small group" in tips.at["Human Resources", "tip_size"]
    assert "3.3 percentage points" in tips.at["Human Resources", "tip_size"]
    assert "small group" not in tips.at["Manager", "tip_size"]


def test_rate_chart_uses_heading_and_reference_tooltip() -> None:
    data = overview.rate_tooltips(RATES, 0.169, "Company rate")
    spec = charts.rate_bars(
        data,
        reference=0.169,
        reference_label="Company rate 16.9%",
        small_label="Small group",
        tooltip=overview.RATE_TOOLTIP,
        reference_tooltip=[(charts.TOOLTIP_HEADING, "Company rate"), ("Who left", "16.9%")],
    ).to_dict()
    bars, line = spec["layer"][0], spec["layer"][1]
    titles = [t["title"] for t in bars["encoding"]["tooltip"]]
    assert titles == [charts.TOOLTIP_HEADING, "Who left", "Compared with the average", "Group size"]
    assert [t["title"] for t in line["encoding"]["tooltip"]] == [charts.TOOLTIP_HEADING, "Who left"]


def test_income_tooltips_name_the_group() -> None:
    by_level = pd.DataFrame(
        {
            "level": ["Level 2"],
            "status": ["Left"],
            "median_income": [5304.0],
            "employees": [37],
            "small_group": [True],
        }
    )
    row = overview.income_tooltips(by_level).iloc[0]
    assert row["tip_title"] == "Level 2, employees who left"
    assert row["tip_income"].startswith("5,304 monetary units: half")
    assert row["tip_size"].startswith("37 employees who left; small group")


def test_team_tooltips_give_share_of_selection() -> None:
    counts = pd.DataFrame({"group": ["Sales", "Human Resources"], "count": [3, 1]})
    tips = risk_list.team_tooltips(counts, "Department")["tip_flagged"].tolist()
    assert (
        tips[0] == "3 employees, 75% of the 4 flagged employees in this selection (department view)"
    )
    assert tips[1].startswith("1 employee, 25%")


def test_no_tooltip_leaves_encoding_out() -> None:
    table = pd.DataFrame({"group": ["A"], "value": [1.0], "label": ["1"]})
    bars = charts.value_bars(table, value_title="Value").to_dict()["layer"][0]
    assert "tooltip" not in bars["encoding"]


def test_numbers_next_to_bars_have_no_tooltip() -> None:
    data = overview.rate_tooltips(RATES, 0.169, "Company rate")
    spec = charts.rate_bars(
        data,
        reference=0.169,
        reference_label="Company rate 16.9%",
        small_label="Small group",
        tooltip=overview.RATE_TOOLTIP,
    ).to_dict()
    text_layers = [
        layer
        for item in spec["layer"]
        for layer in item.get("layer", [item])
        if (layer["mark"]["type"] if isinstance(layer["mark"], dict) else layer["mark"]) == "text"
    ]
    assert len(text_layers) == 2  # halo and number
    assert all(layer["encoding"]["tooltip"] == {"value": None} for layer in text_layers)
