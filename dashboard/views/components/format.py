"""Number formats shared by all views (scope §4: "," separator, one decimal for rates)."""


def count(value: int | float) -> str:
    """Whole number with a thousands separator: ``1,058``."""
    return f"{value:,.0f}"


def rate(value: float, digits: int = 1) -> str:
    """Share as a percentage: ``0.169`` -> ``16.9%``."""
    return f"{value:.{digits}%}"


def money(value: float) -> str:
    """Monetary amount in whole units (currency undocumented, NB §9.6): ``3,388``."""
    return f"{value:,.0f}"
