"""Retention Review List page (M3): headline, filters, rank note, download, team counts."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

from dashboard.views.components.notices import INTENDED_USE

APP = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"


def _app() -> AppTest:
    app = AppTest.from_file(str(APP), default_timeout=60)
    app.run()
    app.switch_page("page_scripts/risk_list.py").run()
    assert not app.exception
    return app


def _table(app: AppTest):
    return app.table[0].value


def test_headline_and_default_list() -> None:
    app = _app()
    assert any(
        "83 of 412 employees flagged for review this round (top 20%)" in m.value
        for m in app.markdown
    )
    table = _table(app)
    assert len(table) == 83
    assert table["Rank"].tolist() == list(range(1, 84))
    assert set(table["Review status"]) == {"▲ In this round's top 20%"}
    assert any(INTENDED_USE in info.value for info in app.info)


def test_rank_note_explains_borderline_ranks() -> None:
    notes = [c.value for c in _app().caption if "flags by rank" in c.value]
    assert len(notes) == 1
    assert "Ranks 81–83 (25.3%–26.1%)" in notes[0] and "26.2% cut-off" in notes[0]


def test_filters_hide_rows_but_keep_ranks() -> None:
    app = _app()
    app.toggle(key="risk_flagged_only").set_value(False).run()
    assert len(_table(app)) == 412
    app.multiselect(key="risk_departments").select("Sales").run()
    table = _table(app)
    assert set(table["Department"]) == {"Sales"} and len(table) == 127
    assert table["Rank"].is_monotonic_increasing and table["Rank"].iloc[0] == 1
    assert "▲ In this round's top 20%" in set(table["Review status"])
    assert "● Not in top 20%" in set(table["Review status"])


def test_job_role_options_follow_department() -> None:
    app = _app()
    app.multiselect(key="risk_departments").select("Human Resources").run()
    roles = app.multiselect(key="risk_roles").options
    assert set(roles) <= {"Human Resources", "Manager"}


def test_download_is_offered() -> None:
    buttons = _app().get("download_button")
    assert len(buttons) == 1
