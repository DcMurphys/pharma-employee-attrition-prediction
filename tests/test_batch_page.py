"""Batch scoring tab (M5): the three sample scenarios end to end in the app."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

from dashboard.controllers.batch import SOURCE_SAMPLE

APP = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"


def _app(sample: str | None = None) -> AppTest:
    app = AppTest.from_file(str(APP), default_timeout=60)
    app.run()
    app.switch_page("page_scripts/employee_check.py").run()
    if sample is not None:
        app.radio(key="batch_source").set_value(SOURCE_SAMPLE).run()
        app.selectbox(key="batch_sample").set_value(sample).run()
    assert not app.exception
    return app


def _metrics(app: AppTest) -> dict[str, str]:
    return {m.label: m.value for m in app.metric}


def _downloads(app: AppTest) -> list:
    return app.get("download_button")


def test_upload_mode_waits_for_a_file() -> None:
    app = _app()
    assert "Rows scored" not in _metrics(app)
    assert not app.error
    assert len(_downloads(app)) == 3  # the three sample files only


def test_valid_sample_is_scored() -> None:
    app = _app("valid")
    assert not app.error
    metrics = _metrics(app)
    assert (metrics["Rows scored"], metrics["Flagged (top 20%)"]) == ("20", "4")
    table = app.table[0].value
    assert table["Rank"].tolist() == list(range(1, 21))
    assert table["Review status"].iloc[0] == "▲ In this file's top 20%"
    assert len(_downloads(app)) == 4  # samples + scored file


def test_error_sample_lists_blocking_issues() -> None:
    app = _app("errors")
    assert "Rows scored" not in _metrics(app)
    report = app.error[0].value
    assert "cannot be scored" in report
    for column in ("DailyRate", "TotalWorkingYears", "JobLevel", "MonthlyIncome"):
        assert f"`{column}`" in report
    notices = app.warning[0].value
    assert "do not stop scoring" in notices and "Scored with" not in notices
    assert "`BusinessTravel`" in notices and "`DistanceFromHome`" in notices


def test_high_risk_sample_flags_top_five(risk_scores) -> None:
    app = _app("high_risk")
    assert _metrics(app)["Flagged (top 20%)"] == "5"
    table = app.table[0].value
    top_five = risk_scores.sort_values("risk_rank").index[:5].tolist()
    flagged = table.loc[table["Review status"].str.startswith("▲"), "Employee ID"].tolist()
    assert flagged == top_five
