"""Altair charts in the MurphysBio palette; every chart comes with a legend and a table
alternative (scope §9).

Colour rules (scope §7.3): ``brand-teal`` for the series in focus, a neutral grey for context,
``status-flag`` only for the review cut-off. Every fill gets a darker outline (``CHART_EDGE``) so
marks reach 3:1 against white. Meaning never depends on colour alone: bars carry text labels and
small groups are named in the label.

Legends are HTML above the chart instead of Vega legends, so they wrap on a 375 px screen and
are read as text by screen readers.

Tooltips are written by the views as short sentences: a heading (the bar's name) and a few
labelled lines that say what the number means. Charts only place the prepared text columns.
"""

import html
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import altair as alt
import pandas as pd
import streamlit as st

from dashboard.views.components.palette import (
    BRAND_TEAL,
    BRAND_TEAL_DARK,
    CHART_EDGE,
    CONTEXT_GREY,
    INK,
    SLATE,
)

# A tooltip: (line title, text column) pairs. The title TOOLTIP_HEADING makes that line the
# tooltip's heading (vega-tooltip convention).
Tooltip = Sequence[tuple[str, str]]
TOOLTIP_HEADING = "title"

BAR_ROW_HEIGHT = 30  # px per bar, so labels stay readable
LEFT_LABEL = "Share who left"  # legend entry for the attrition-rate bars


def _tooltip(tooltip: Tooltip) -> dict:
    """Encoding keyword for a prepared tooltip (empty: no tooltip)."""
    if not tooltip:
        return {}
    return {"tooltip": [alt.Tooltip(f"{column}:N", title=title) for title, column in tooltip]}


def tooltip_frame(lines: Sequence[tuple[str, str]]) -> tuple[pd.DataFrame, Tooltip]:
    """One-row tooltip data for a single mark (such as a reference line) from (title, text)."""
    columns = [f"tip_{i}" for i in range(len(lines))]
    frame = pd.DataFrame({c: [text] for c, (_, text) in zip(columns, lines, strict=True)})
    return frame, [(title, c) for c, (title, _) in zip(columns, lines, strict=True)]


def _value_labels(
    data: pd.DataFrame, y: alt.Y, x: alt.X, *, y_offset: alt.YOffset | None = None, size: int = 12
) -> alt.LayerChart:
    """Text after each bar, on a white halo so a reference line never cuts through it.

    The numbers carry no tooltip (``tooltip`` set to null); the bars hold the tooltips.
    """
    encoding = {"y": y, "x": x, "text": "label:N", "tooltip": alt.value(None)} | (
        {"yOffset": y_offset} if y_offset else {}
    )
    base = alt.Chart(data)
    halo = base.mark_text(
        align="left", dx=4, fontSize=size, stroke="white", strokeWidth=4, strokeJoin="round"
    ).encode(**encoding)
    text = base.mark_text(align="left", dx=4, fontSize=size, color=INK).encode(**encoding)
    return halo + text


@dataclass(frozen=True)
class LegendItem:
    """One legend entry: a colour swatch (box) or a dashed line, and its label."""

    label: str
    colour: str
    kind: Literal["box", "dash"] = "box"


def show(
    chart: alt.TopLevelMixin,
    table: pd.DataFrame,
    *,
    description: str,
    legend: Sequence[LegendItem] = (),
) -> None:
    """Draw ``legend``, then ``chart`` full width, with ``table`` as its text alternative."""
    if legend:
        render_legend(legend)
    st.altair_chart(chart.properties(description=description), width="stretch")
    with st.expander("Show as table", icon=":material/table:"):
        st.table(table, hide_index=True)


def render_legend(items: Sequence[LegendItem]) -> None:
    """Legend as wrapping HTML; the swatches are decorative, the labels carry the meaning."""
    entries = []
    for item in items:
        if item.kind == "dash":
            style = f"width:18px;height:0;border-top:2px dashed {item.colour}"
        else:
            style = (
                f"width:12px;height:12px;border-radius:2px;background:{item.colour};"
                f"box-sizing:border-box;border:1px solid {edge(item.colour)}"
            )
        entries.append(
            '<span role="listitem" style="display:inline-flex;align-items:center;gap:6px;'
            'white-space:nowrap">'
            f'<span aria-hidden="true" style="display:inline-block;{style}"></span>'
            f"{html.escape(item.label)}</span>"
        )
    st.markdown(
        f'<div role="list" aria-label="Legend" style="display:flex;flex-wrap:wrap;'
        f'column-gap:16px;row-gap:4px;font-size:0.875rem;color:{SLATE}">'
        f"{''.join(entries)}</div>",
        unsafe_allow_html=True,
    )


def edge(colour: str) -> str:
    """Outline colour for a chart fill (the fill itself if it needs none)."""
    return CHART_EDGE.get(colour, colour)


def rate_legend(*, has_small: bool, small_label: str, reference_label: str) -> list[LegendItem]:
    """Legend for ``rate_bars``: teal bars, grey small groups (if any), dashed reference."""
    items = [LegendItem(LEFT_LABEL, BRAND_TEAL)]
    if has_small:
        items.append(LegendItem(small_label, CONTEXT_GREY))
    items.append(LegendItem(reference_label, SLATE, "dash"))
    return items


STATUS_LEGEND = [LegendItem("Stayed", CONTEXT_GREY), LegendItem("Left", BRAND_TEAL)]


def rate_bars(
    table: pd.DataFrame,
    *,
    reference: float,
    reference_label: str,
    small_label: str,
    row_height: int = BAR_ROW_HEIGHT,
    tooltip: Tooltip = (),
    reference_tooltip: Sequence[tuple[str, str]] = (),
) -> alt.LayerChart:
    """Horizontal attrition-rate bars with value labels and a dashed reference line.

    ``table`` needs ``group``, ``rate``, ``employees``, ``small_group`` and ``label`` (the text
    printed after each bar), plus the columns named in ``tooltip``; a line break in ``group``
    gives a two-line axis label. Small groups are drawn in grey; ``rate_legend`` gives the
    matching legend. ``reference_tooltip`` holds (title, text) lines for the dashed line.
    """
    data = table.assign(kind=table["small_group"].map({True: small_label, False: LEFT_LABEL}))
    top = max(float(data["rate"].max()), reference) * 1.6  # room for labels on narrow screens
    # A line break in a group name ("Lowest pay\n(1,009–2,899)") becomes a two-line label.
    y = alt.Y(
        "group:N",
        sort=None,
        title=None,
        axis=alt.Axis(labelLimit=150, labelExpr="split(datum.label, '\\n')"),
    )
    x = alt.X(
        "rate:Q",
        title="Attrition rate",
        axis=alt.Axis(format="%", tickCount=4),
        scale=alt.Scale(domain=[0, top]),
    )
    domain, fills = [LEFT_LABEL, small_label], [BRAND_TEAL, CONTEXT_GREY]
    bars = (
        alt.Chart(data)
        .mark_bar(cornerRadiusEnd=3, strokeWidth=1)
        .encode(
            y=y,
            x=x,
            color=alt.Color("kind:N", scale=alt.Scale(domain=domain, range=fills), legend=None),
            stroke=alt.Stroke(
                "kind:N",
                scale=alt.Scale(domain=domain, range=[edge(c) for c in fills]),
                legend=None,
            ),
            **_tooltip(tooltip),
        )
    )
    tip_data, tip = tooltip_frame(reference_tooltip or [(TOOLTIP_HEADING, reference_label)])
    line = (
        alt.Chart(tip_data.assign(rate=reference))
        .mark_rule(color=SLATE, strokeDash=[4, 4], strokeWidth=1.5)
        .encode(x="rate:Q", **_tooltip(tip))
    )
    # Line before labels, so the labels' halo sits on top of it.
    return (bars + line + _value_labels(data, y, x)).properties(height=alt.Step(row_height))


def grouped_bars(
    table: pd.DataFrame,
    *,
    legend: Sequence[LegendItem],
    value_title: str,
    axis_format: str = "",
    tooltip: Tooltip = (),
) -> alt.LayerChart:
    """Horizontal bars per ``category``, one bar per ``series`` side by side, with labels.

    ``table`` needs ``category``, ``series``, ``value``, ``label`` and the ``tooltip`` columns;
    ``legend`` names the series in display order with their colours.
    """
    order = [item.label for item in legend]
    y = alt.Y("category:N", title=None, sort=None, axis=alt.Axis(labelLimit=150))
    y_offset = alt.YOffset("series:N", sort=order)
    x = alt.X(
        "value:Q",
        title=value_title,
        axis=alt.Axis(format=axis_format, tickCount=5) if axis_format else alt.Axis(tickCount=5),
        scale=alt.Scale(domain=[0, float(table["value"].max()) * 1.45]),
    )
    fills = [item.colour for item in legend]
    bars = (
        alt.Chart(table)
        .mark_bar(cornerRadiusEnd=3, strokeWidth=1)
        .encode(
            y=y,
            yOffset=y_offset,
            x=x,
            color=alt.Color("series:N", scale=alt.Scale(domain=order, range=fills), legend=None),
            stroke=alt.Stroke(
                "series:N",
                scale=alt.Scale(domain=order, range=[edge(c) for c in fills]),
                legend=None,
            ),
            **_tooltip(tooltip),
        )
    )
    labels = _value_labels(table, y, x, y_offset=y_offset, size=11)
    # With yOffset the step sizes each sub-bar, not the whole category.
    return (bars + labels).properties(height=alt.Step(24))


def grouped_income_bars(table: pd.DataFrame, *, tooltip: Tooltip = ()) -> alt.LayerChart:
    """Median income by job level, stayers and leavers side by side (``STATUS_LEGEND``).

    ``table`` needs ``level``, ``status``, ``median_income``, ``label`` and the ``tooltip``
    columns.
    """
    data = table.rename(columns={"level": "category", "status": "series", "median_income": "value"})
    return grouped_bars(
        data,
        legend=STATUS_LEGEND,
        value_title="Median monthly income (monetary units)",
        tooltip=tooltip,
    )


def value_bars(
    table: pd.DataFrame,
    *,
    value_title: str,
    axis_format: str = "",
    colour: str = BRAND_TEAL,
    tooltip: Tooltip = (),
) -> alt.LayerChart:
    """Horizontal bars with value labels; ``table`` needs ``group``, ``value``, ``label`` and
    the ``tooltip`` columns."""
    y = alt.Y("group:N", sort=None, title=None, axis=alt.Axis(labelLimit=220))
    top = max(float(table["value"].max()), 0.0) * 1.3 or 1.0
    # Title aligned to the right end, so a long title is not cut off at 375 px.
    axis = {"titleAnchor": "end"} | ({"format": axis_format} if axis_format else {})
    x = alt.X(
        "value:Q",
        title=value_title,
        axis=alt.Axis(**axis),
        scale=alt.Scale(domain=[min(0.0, float(table["value"].min())), top]),
    )
    bars = (
        alt.Chart(table)
        .mark_bar(cornerRadiusEnd=3, color=colour, stroke=edge(colour), strokeWidth=1)
        .encode(y=y, x=x, **_tooltip(tooltip))
    )
    return (bars + _value_labels(table, y, x)).properties(height=alt.Step(BAR_ROW_HEIGHT))


def count_bars(table: pd.DataFrame, *, value_title: str, tooltip: Tooltip = ()) -> alt.LayerChart:
    """Horizontal count bars with value labels; ``table`` needs ``group``, ``count`` and the
    ``tooltip`` columns."""
    data = table.assign(value=table["count"], label=table["count"].map("{:,}".format))
    return value_bars(data, value_title=value_title, axis_format="d", tooltip=tooltip)


def best_so_far(
    history: pd.DataFrame, *, value_title: str, tooltip: Tooltip = ()
) -> alt.LayerChart:
    """Each trial's value (grey points) and the best value so far (dark teal step line; a thin
    line in the light teal would fall below 3:1 on white).

    ``history`` needs ``trial``, ``value``, ``best`` and the ``tooltip`` columns (shown on the
    points); ``TUNING_LEGEND`` is the legend.
    """
    x = alt.X("trial:Q", title="Trial", axis=alt.Axis(tickMinStep=1, format="d"))
    points = (
        alt.Chart(history)
        .mark_circle(size=28, color=CONTEXT_GREY, stroke=edge(CONTEXT_GREY), strokeWidth=1)
        .encode(
            x=x,
            y=alt.Y("value:Q", title=value_title, scale=alt.Scale(zero=False)),
            **_tooltip(tooltip),
        )
    )
    line = (
        alt.Chart(history)
        .mark_line(interpolate="step-after", color=BRAND_TEAL_DARK, strokeWidth=2.5)
        .encode(x=x, y="best:Q")
    )
    return (points + line).properties(height=260)


TUNING_LEGEND = [LegendItem("Best so far", BRAND_TEAL_DARK), LegendItem("Each trial", CONTEXT_GREY)]
