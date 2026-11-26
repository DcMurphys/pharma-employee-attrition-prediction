"""Row of headline numbers."""

from collections.abc import Sequence
from dataclasses import dataclass

import streamlit as st


@dataclass(frozen=True)
class Kpi:
    """One headline number, already formatted."""

    label: str
    value: str
    help: str | None = None


def kpi_row(kpis: Sequence[Kpi]) -> None:
    """Bordered metric cards side by side (they stack on narrow screens)."""
    for column, kpi in zip(st.columns(len(kpis)), kpis, strict=True):
        column.metric(kpi.label, kpi.value, help=kpi.help, border=True)
