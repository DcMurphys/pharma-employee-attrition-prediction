"""Workforce Insights page (M2): KPIs, department filter, tabs and findings."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

from dashboard.models import workforce as wf
from dashboard.views.overview import pay_band_labels

APP = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"


def _app() -> AppTest:
    app = AppTest.from_file(str(APP), default_timeout=60)
    app.run()
    return app


def _metrics(app: AppTest) -> dict[str, str]:
    return {metric.label: metric.value for metric in app.metric}


def test_kpis_for_all_departments() -> None:
    app = _app()
    assert not app.exception and not app.error
    assert _metrics(app) == {
        "Employees": "1,470",
        "Known outcome": "1,058",
        "Left": "179",
        "Attrition rate": "16.9%",
        "Awaiting outcome": "412",
    }
    assert [tab.label for tab in app.tabs] == [
        "Where attrition concentrates",
        "Pay",
        "Key findings",
    ]


def test_findings_carry_computed_values() -> None:
    findings = [m.value for m in _app().markdown if m.value[:2] in {f"{n}." for n in range(1, 10)}]
    assert len(findings) == 5
    assert "**31.9%** vs. **10.8%**" in findings[0]
    assert "**27.4%**" in findings[1]
    assert "**3,388** vs. **5,210**" in findings[2]
    assert "**25.7%**" in findings[3]
    assert "Sales" in findings[4] and "**20.7%**" in findings[4]


def test_department_filter_updates_kpis() -> None:
    app = _app()
    app.selectbox(key="overview_department").select("Sales").run()
    assert not app.exception
    metrics = _metrics(app)
    assert (metrics["Known outcome"], metrics["Left"], metrics["Attrition rate"]) == (
        "319",
        "66",
        "20.7%",
    )
    assert any("the Sales department" in caption.value for caption in app.caption)


def test_no_sensitive_attributes_on_page() -> None:
    app = _app()
    text = " ".join(
        [s.value for s in app.subheader] + [str(sb.options) for sb in app.selectbox]
    ).lower()
    for word in ("gender", "marital", "age band"):
        assert word not in text


def test_pay_tab_shows_pay_bands_not_box_plot() -> None:
    subheaders = [s.value for s in _app().subheader]
    assert "Attrition rate by pay band" in subheaders
    assert "Monthly income of leavers and stayers" not in subheaders


def test_pay_band_labels(workforce) -> None:
    labels = pay_band_labels(wf.attrition_by_pay_band(workforce))
    assert labels[0] == "Lowest pay\n(1,009–2,899)"
    assert labels[-1] == "Highest pay\n(8,740–19,999)"
    assert labels[1].startswith("Lower-middle pay")


def test_rate_legend_lists_only_what_is_drawn() -> None:
    from dashboard.views.components import charts

    with_small = charts.rate_legend(
        has_small=True, small_label="Small group", reference_label="Company rate 16.9%"
    )
    assert [item.label for item in with_small] == [
        "Share who left",
        "Small group",
        "Company rate 16.9%",
    ]
    assert with_small[-1].kind == "dash"
    without = charts.rate_legend(
        has_small=False, small_label="Small group", reference_label="Company rate 16.9%"
    )
    assert "Small group" not in [item.label for item in without]


def test_every_chart_has_a_legend() -> None:
    app = _app()
    legends = [m for m in app.markdown if 'aria-label="Legend"' in m.value]
    charts_drawn = sum(e.label == "Show as table" for e in app.expander)  # one per chart
    assert charts_drawn == 8  # six breakdowns, pay bands, income by level
    assert len(legends) == charts_drawn
    assert any("Stayed" in m.value and "Left" in m.value for m in legends)
