"""App shell (M1): every page renders with the shared frame; missing files never crash the app."""

import tomllib
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from dashboard.controllers import data, overview
from dashboard.controllers.shell import PAGES
from dashboard.models.artifacts import ArtifactError
from dashboard.views.components import palette
from employee_attrition.config import DATA_DISCLAIMER, MODEL_METADATA_PATH, WORKFORCE_PATH

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "dashboard" / "app.py"
THEME = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text())["theme"]


def _run(page_id: str) -> AppTest:
    app = AppTest.from_file(str(APP), default_timeout=60)
    app.run()
    if page_id != PAGES[0].id:
        app.switch_page(f"page_scripts/{page_id}.py").run()
    return app


def _texts(app: AppTest) -> str:
    """All markdown and caption text, main area and sidebar."""
    elements = [*app.markdown, *app.caption, *app.sidebar.markdown, *app.sidebar.caption]
    return "\n".join(element.value for element in elements)


def test_palette_matches_theme() -> None:
    for key, token in palette.THEME_TOKENS.items():
        assert THEME[key].upper() == token.upper(), key


@pytest.mark.parametrize("page", PAGES, ids=lambda page: page.id)
def test_page_renders_with_frame(page) -> None:
    app = _run(page.id)
    assert not app.exception and not app.error
    assert app.title[0].value == page.title
    texts = _texts(app)
    assert DATA_DISCLAIMER in texts
    assert "1,470 employees in this snapshot." in texts
    assert "1,058 employees with a known outcome" in texts


def test_missing_data_folder_keeps_app_running(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(data.DATA_DIR_ENV, str(tmp_path))
    app = _run(PAGES[0].id)  # Workforce Insights needs employee.csv
    assert not app.exception
    assert "not available right now" in app.error[0].value
    texts = _texts(app)
    assert "employees in this snapshot" not in texts
    assert "1,058 employees with a known outcome" in texts  # model details still shown
    assert DATA_DISCLAIMER in texts


def test_redirect_moves_only_its_own_folder(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(data.ARTIFACTS_DIR_ENV, str(tmp_path))
    assert data.artifact_path(MODEL_METADATA_PATH) == (
        tmp_path / "trained_models" / MODEL_METADATA_PATH.name
    )
    assert data.artifact_path(WORKFORCE_PATH) == WORKFORCE_PATH


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        (ArtifactError("workforce", "missing", Path("employee.csv")), "not available right now"),
        (RuntimeError("boom"), "Something went wrong"),
    ],
    ids=["artifact_error", "unexpected_error"],
)
def test_page_failure_shows_plain_message(monkeypatch, failure, message) -> None:
    def broken_page(page_id: str) -> None:
        raise failure

    monkeypatch.setattr(overview, "page_header", broken_page)
    app = _run(PAGES[0].id)
    assert not app.exception
    assert message in app.error[0].value
    assert "boom" not in app.error[0].value and "employee.csv" not in app.error[0].value
    assert DATA_DISCLAIMER in _texts(app)
