"""Accessibility rules of scope §7.3 and §9: contrast, colour-independent charts, target size."""

import pytest

from dashboard.views import shell
from dashboard.views.components import palette as p
from dashboard.views.components.charts import STATUS_LEGEND, edge
from dashboard.views.components.one_line import nowrap_css


def luminance(colour: str) -> float:
    """WCAG relative luminance of a "#RRGGBB" colour."""
    channels = [int(colour[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    r, g, b = (c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a: str, b: str) -> float:
    """WCAG contrast ratio between two colours."""
    high, low = sorted([luminance(a), luminance(b)], reverse=True)
    return (high + 0.05) / (low + 0.05)


@pytest.mark.parametrize("text", [p.INK, p.SLATE, p.BRAND_TEAL_DARK, p.STATUS_FLAG, p.INFO])
@pytest.mark.parametrize("background", [p.SURFACE, p.SURFACE_TINT])
def test_text_tokens_meet_aa(text: str, background: str) -> None:
    assert contrast(text, background) >= 4.5


def test_white_on_primary_meets_aa() -> None:
    assert contrast("#FFFFFF", p.BRAND_TEAL_DARK) >= 4.5


@pytest.mark.parametrize("fill", [p.BRAND_TEAL, p.CONTEXT_GREY])
def test_chart_fills_have_outlines_of_3_to_1(fill: str) -> None:
    assert contrast(edge(fill), p.SURFACE) >= 3


def test_chart_series_differ_without_colour_vision() -> None:
    stayed, left = (item.colour for item in STATUS_LEGEND)
    assert contrast(stayed, left) >= 1.5


def test_controls_reach_44_px() -> None:
    assert shell.MIN_TARGET_PX >= 44
    for selector in ("stBaseButton", "stTab", "select", "stExpander", "stSidebarNavLink"):
        assert selector in shell.TARGET_SIZE_CSS
    assert f"min-height: {shell.MIN_TARGET_PX}px" in shell.TARGET_SIZE_CSS


def test_ranked_table_cells_stay_on_one_line() -> None:
    css = nowrap_css("ranked_table")
    assert (
        "white-space: nowrap" in css and ".st-key-ranked_table [data-testid='stTable'] td p" in css
    )
