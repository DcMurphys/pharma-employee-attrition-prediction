"""MurphysBio colour tokens (scope §7.3), the single place for colours in code.

The theme values are mirrored in ``.streamlit/config.toml``; ``tests/test_app_shell.py`` checks
that both stay identical.
"""

BRAND_TEAL_DARK = "#0F766E"  # primaryColor: buttons (white text), links, focus, teal text
BRAND_TEAL = "#14B8A6"  # logo atoms and chart series in focus; never behind white text
INK = "#1B1B1F"  # body text and headings
SLATE = "#475467"  # secondary text, axis labels, "Not flagged" badge
SURFACE = "#FFFFFF"  # main background
SURFACE_TINT = "#EEF8F7"  # sidebar, cards, info panels
BORDER = "#E7E7EA"  # dividers and card borders (non-text only)
STATUS_FLAG = "#B42318"  # "Flagged for review" badge and review cut-off lines
INFO = "#2E5AAC"  # intended-use and model notes
# Chart context series (scope §7.3 "neutral grey"); non-text only. Light enough to differ in
# lightness from BRAND_TEAL (1.7:1), so the two series stay apart without colour vision.
CONTEXT_GREY = "#D0D5DD"

# Outline drawn around each chart fill: the fills alone are below 3:1 on white, their outlines
# reach 3:1 or more (WCAG 2.2, 1.4.11 non-text contrast).
CHART_EDGE = {BRAND_TEAL: BRAND_TEAL_DARK, CONTEXT_GREY: SLATE}

# Theme keys in .streamlit/config.toml and the token each one must equal.
THEME_TOKENS = {
    "primaryColor": BRAND_TEAL_DARK,
    "backgroundColor": SURFACE,
    "secondaryBackgroundColor": SURFACE_TINT,
    "textColor": INK,
    "linkColor": BRAND_TEAL_DARK,
    "borderColor": BORDER,
}
