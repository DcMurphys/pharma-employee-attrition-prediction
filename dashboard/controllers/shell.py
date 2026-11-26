"""App frame: page config, logo, navigation, sidebar, and fatal-error handling (scope §4, §6)."""

import logging
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import streamlit as st

from dashboard.controllers import data
from dashboard.models.artifacts import ArtifactError
from dashboard.views import shell as view
from employee_attrition.config import COMPANY_NAME, DATA_DISCLAIMER

logger = logging.getLogger(__name__)

DASHBOARD_DIR = Path(__file__).resolve().parents[1]
BRAND_DIR = DASHBOARD_DIR / "assets" / "brand"
PAGE_SCRIPTS_DIR = DASHBOARD_DIR / "page_scripts"
LOGO = BRAND_DIR / "murphysbio_logo.svg"
ICON = BRAND_DIR / "murphysbio_icon.svg"


@dataclass(frozen=True)
class PageSpec:
    """One page: internal id (also the URL path and script name), display name, icon, intro."""

    id: str
    title: str
    icon: str
    intro: str


PAGES = (
    PageSpec(
        "overview",
        "Workforce Insights",
        ":material/insights:",
        "Where attrition concentrates at MurphysBio, based on employees with a known outcome.",
    ),
    PageSpec(
        "risk_list",
        "Retention Review List",
        ":material/checklist:",
        "Employees without a recorded outcome, ranked by risk, to plan supportive stay "
        "conversations.",
    ),
    PageSpec(
        "employee_check",
        "Check an Employee",
        ":material/person_search:",
        "Estimate the risk for one employee or a file of employees and explore what-if changes.",
    ),
    PageSpec(
        "model_card",
        "How the Model Works",
        ":material/fact_check:",
        "How reliable the model is, what it may be used for, and where its limits are.",
    ),
)
PAGES_BY_ID = {page.id: page for page in PAGES}


def run() -> None:
    """Configure the app, draw the shared frame, and run the selected page."""
    st.set_page_config(page_title=f"{COMPANY_NAME} Attrition", page_icon=str(ICON), layout="wide")
    st.logo(str(LOGO), icon_image=str(ICON), size="large")
    view.apply_target_size()
    page = st.navigation(
        [
            st.Page(
                PAGE_SCRIPTS_DIR / f"{spec.id}.py",
                title=spec.title,
                icon=spec.icon,
                url_path=spec.id,
                default=index == 0,
            )
            for index, spec in enumerate(PAGES)
        ]
    )
    view.render_sidebar(*_snapshot(), DATA_DISCLAIMER)
    try:
        page.run()
    except ArtifactError as err:
        logger.error("Page %r stopped: %s", page.url_path, err)
        view.render_artifact_error(err.kind)
    except Exception:  # last resort: plain message for users, traceback in the log
        logger.exception("Page %r failed", page.url_path)
        view.render_unexpected_error()
    view.render_footer(DATA_DISCLAIMER)


def page_header(page_id: str) -> None:
    """Title and intro of a page, from ``PAGES``."""
    spec = PAGES_BY_ID[page_id]
    view.render_page_header(spec.title, spec.intro)


def placeholder_page(page_id: str) -> None:
    """Stub body for a page that a later milestone builds."""
    page_header(page_id)
    view.render_coming_soon()


def _snapshot() -> tuple[int | None, int | None, date | None]:
    """Sidebar values: employees in the snapshot, training rows, model export date.

    A missing artifact leaves its value out instead of stopping the app.
    """
    employees = trained_on_rows = exported_on = None
    try:
        employees = len(data.workforce())
    except ArtifactError as err:
        logger.warning("Snapshot note without employee count: %s", err)
    try:
        metadata = data.model_metadata()
        trained_on_rows = int(metadata["trained_on_rows"])
        exported_on = datetime.fromisoformat(metadata["created_at_utc"]).date()
    except (ArtifactError, KeyError, ValueError) as err:
        logger.warning("Snapshot note without model details: %s", err)
    return employees, trained_on_rows, exported_on
