"""Review-status labels: always an icon plus words, never colour alone (scope §4, §7.3).

There is deliberately no green "safe" state: a low score does not mean an employee is
satisfied.
"""

import html

from dashboard.views.components.palette import SLATE, STATUS_FLAG

FLAG_ICON, NEUTRAL_ICON = "▲", "●"

# Rank rule (Retention Review List) and cut-off rule (Check an Employee); scope Q1.
RANK_FLAGGED = f"{FLAG_ICON} In this {{scope}}'s top {{share}}"
RANK_NOT_FLAGGED = f"{NEUTRAL_ICON} Not in top {{share}}"
CUTOFF_ABOVE = f"{FLAG_ICON} Above the review cut-off"
CUTOFF_BELOW = f"{NEUTRAL_ICON} Below the review cut-off"


def rank_status(flagged: bool, share: str, scope: str = "round") -> str:
    """Rank-rule status, e.g. "▲ In this round's top 20%" (``scope`` "file" for batches)."""
    return (RANK_FLAGGED if flagged else RANK_NOT_FLAGGED).format(share=share, scope=scope)


def cutoff_status(above: bool) -> str:
    """Status for a single-employee check against the review cut-off."""
    return CUTOFF_ABOVE if above else CUTOFF_BELOW


def badge_html(text: str, flagged: bool) -> str:
    """Status badge as HTML: icon and words, in ``status-flag`` or neutral ``slate``."""
    colour = STATUS_FLAG if flagged else SLATE
    return (
        '<span style="display:inline-block;padding:4px 10px;border-radius:999px;'
        f'border:1.5px solid {colour};color:{colour};font-weight:600">{html.escape(text)}</span>'
    )
