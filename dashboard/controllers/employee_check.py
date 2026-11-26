"""Check an Employee controller: start values, 22-input form, result and what-if (§4.3).

Form values live in ``st.session_state`` under ``ec_<feature>``. They are (re)loaded only when
the starting point changes, so edits survive reruns.
"""

from typing import Any

import streamlit as st

from dashboard.controllers import batch, data
from dashboard.controllers.shell import page_header
from dashboard.models import employee_check as ec
from dashboard.models import risk_list as rl
from dashboard.models.scoring import score_one
from dashboard.models.workforce import SEGMENTS
from dashboard.views import employee_check as view
from dashboard.views.components import notices
from dashboard.views.components.feature_labels import FEATURE_GROUPS, label
from employee_attrition.config import ORDINAL_LABELS, REVIEW_CAPACITY_SHARE

PREFIX = "ec_"
KEY_START = f"{PREFIX}start"
KEY_EMPLOYEE = f"{PREFIX}employee"
KEY_LOADED = f"{PREFIX}loaded"
KEY_WHAT_IF_FEATURE = f"{PREFIX}what_if_feature"
KEY_WHAT_IF_LOADED = f"{PREFIX}what_if_loaded"
START_TYPICAL = "A typical employee"
START_EXISTING = "An employee awaiting an outcome"
DEFAULT_WHAT_IF = "OverTime"

# Display labels of category values: travel and stock options from Workforce Insights,
# ordinal scales from the data dictionary.
VALUE_LABELS: dict[str, dict] = {
    column: labels for column, (_title, labels) in SEGMENTS.items() if labels
} | ORDINAL_LABELS


def render() -> None:
    """Draw the Check an Employee page."""
    page_header("employee_check")
    notices.intended_use()
    single_tab, batch_tab = st.tabs(["Single employee what-if", "Batch scoring"])
    with single_tab:
        _single_employee()
    with batch_tab:
        batch.render()


def _single_employee() -> None:
    schema, features = data.feature_schema(), data.features()
    model, workforce, scores = data.model(), data.workforce(), data.risk_scores()
    cutoff = float(data.model_metadata()["decision_threshold"])

    signature, start_record = _starting_point(schema, workforce, features, scores)
    _load(signature, start_record)

    # Result and what-if come first so they stay visible; widget values from the last
    # interaction are already in session state, so the record can be read before the form.
    fields = {field["feature"]: field for field in schema}
    record = {name: st.session_state[f"{PREFIX}{name}"] for name in fields}

    result = score_one(model, record, features)
    probability, above = result["attrition_probability"], bool(result["attrition_flag"])
    view.render_result(probability, above, cutoff, ec.share_below(probability, scores), len(scores))
    gap = rl.flagged_below_cutoff(scores, cutoff)
    if gap is not None:
        view.render_cutoff_note(
            cutoff, REVIEW_CAPACITY_SHARE, gap.ranks, gap.min_probability, gap.max_probability
        )

    view.render_what_if_intro()
    feature, value = _what_if_inputs(fields, record, signature)
    before, after = ec.what_if(model, record, feature, value, features)
    view.render_comparison(
        f"{label(feature)} = {_value_label(feature, value)}",
        before["attrition_probability"],
        bool(before["attrition_flag"]),
        after["attrition_probability"],
        bool(after["attrition_flag"]),
    )

    view.render_form_intro()
    _form(fields)


def _starting_point(schema, workforce, features, scores) -> tuple[tuple, dict[str, Any]]:
    """Radio for the starting point (and the employee); returns its signature and values."""
    start = st.radio("Start from", [START_TYPICAL, START_EXISTING], key=KEY_START, horizontal=True)
    if start == START_TYPICAL:
        return (START_TYPICAL,), ec.default_record(schema, workforce)
    ranks = scores["risk_rank"].astype(int)
    employee_id = st.selectbox(
        "Employee ID (only employees awaiting an outcome)",
        [int(i) for i in scores.sort_values("risk_rank").index],  # Q2: the 412 only
        key=KEY_EMPLOYEE,
        format_func=lambda i: f"{i} · rank {ranks[i]} on the review list",
        width=360,
    )
    return (START_EXISTING, employee_id), ec.prefill_record(workforce, employee_id, features)


def _load(signature: tuple, record: dict[str, Any]) -> None:
    """Write ``record`` into the form when the starting point has changed."""
    if st.session_state.get(KEY_LOADED) == signature:
        return
    for name, value in record.items():
        st.session_state[f"{PREFIX}{name}"] = value
    st.session_state[KEY_LOADED] = signature


def _form(fields: dict[str, dict]) -> None:
    """The 22 inputs in four groups (two columns on wide screens)."""
    groups = list(FEATURE_GROUPS.items())
    for column, half in zip(st.columns(2), (groups[:2], groups[2:]), strict=True):
        with column:
            for title, names in half:
                with st.container(border=True):
                    view.render_group_title(title)
                    for name in names:
                        _input(fields[name], key=f"{PREFIX}{name}", text=label(name))


def _input(field: dict, *, key: str, text: str) -> Any:
    """One input built from ``feature_schema.json``: select box or bounded number input."""
    name = field["feature"]
    if field["kind"] in {"nominal", "ordinal"}:
        return st.selectbox(
            text, ec.options(field), key=key, format_func=lambda v: _value_label(name, v)
        )
    low, high = int(field["min"]), int(field["max"])
    return st.number_input(
        text, min_value=low, max_value=high, step=1, key=key, help=f"From {low:,} to {high:,}."
    )


def _what_if_inputs(fields, record, signature) -> tuple[str, Any]:
    """Which input to change and its new value (default: overtime -> No)."""
    names = [name for group in FEATURE_GROUPS.values() for name in group]
    st.session_state.setdefault(KEY_WHAT_IF_FEATURE, DEFAULT_WHAT_IF)
    feature_col, value_col = st.columns(2)
    feature = feature_col.selectbox(
        "Input to change", names, key=KEY_WHAT_IF_FEATURE, format_func=label
    )
    value_key = f"{PREFIX}what_if_value_{feature}"  # one key per input type
    current = record[feature]
    loaded = (signature, feature, current)
    if st.session_state.get(KEY_WHAT_IF_LOADED) != loaded:
        st.session_state[value_key] = ec.suggested_value(fields[feature], current)
        st.session_state[KEY_WHAT_IF_LOADED] = loaded
    with value_col:
        value = _input(fields[feature], key=value_key, text="New value")
    return feature, value


def _value_label(feature: str, value: Any) -> str:
    """Readable category value, e.g. WorkLifeBalance 1 -> "Bad"; numbers with separators."""
    labels = VALUE_LABELS.get(feature, {})
    if value in labels:
        return str(labels[value])
    return f"{value:,}" if isinstance(value, int) else str(value)
