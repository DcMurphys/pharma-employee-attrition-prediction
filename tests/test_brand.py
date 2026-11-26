"""Brand assets and theme: files exist, SVGs are self-contained, and theme colours pass WCAG AA."""

import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BRAND_DIR = ROOT / "dashboard" / "assets" / "brand"
STATIC_DIR = ROOT / "dashboard" / "static"
THEME = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text())["theme"]
SVG_NS = "{http://www.w3.org/2000/svg}"
SVGS = [
    "murphysbio_logo.svg",
    "murphysbio_logo_white.svg",
    "murphysbio_logo_tagline.svg",
    "murphysbio_logo_tagline_white.svg",
    "murphysbio_icon.svg",
    "murphysbio_symbol.svg",
]


def _luminance(hex_color: str) -> float:
    channels = [int(hex_color.lstrip("#")[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast(a: str, b: str) -> float:
    """WCAG 2.2 contrast ratio between two hex colours."""
    high, low = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (high + 0.05) / (low + 0.05)


@pytest.mark.parametrize("name", SVGS)
def test_svg_is_self_contained(name: str) -> None:
    root = ET.parse(BRAND_DIR / name).getroot()
    assert root.tag == f"{SVG_NS}svg" and root.get("role") == "img" and root.get("aria-label")
    assert root.find(f"{SVG_NS}title") is not None
    # Text is outlined: no <text> (font dependency) and no external references.
    assert not list(root.iter(f"{SVG_NS}text"))
    assert "href" not in (BRAND_DIR / name).read_text()


def test_png_icons_exist() -> None:
    for size in (64, 192):
        assert (BRAND_DIR / f"murphysbio_icon_{size}.png").stat().st_size > 0


def test_theme_fonts_are_bundled() -> None:
    for face in THEME["fontFaces"]:
        relative = face["url"].removeprefix("app/static/")
        assert (STATIC_DIR / relative).is_file(), face["url"]
    assert (STATIC_DIR / "fonts" / "Inter-LICENSE.txt").is_file()


def test_theme_colours_pass_wcag_aa() -> None:
    assert contrast("#FFFFFF", THEME["primaryColor"]) >= 4.5  # white button text
    assert contrast(THEME["textColor"], THEME["backgroundColor"]) >= 4.5
    assert contrast(THEME["textColor"], THEME["secondaryBackgroundColor"]) >= 4.5
    assert contrast(THEME["linkColor"], THEME["secondaryBackgroundColor"]) >= 4.5
