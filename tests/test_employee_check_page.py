"""Check an Employee page (M4): form from the schema, Q2 prefill, parity, what-if."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from dashboard.controllers.employee_check import START_EXISTING
from dashboard.models import employee_check as ec
from dashboard.models.scoring import score_one

APP = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"


def _app() -> AppTest:
    app = AppTest.from_file(str(APP), default_timeout=60)
    app.run()
    app.switch_page("page_scripts/employee_check.py").run()
    assert not app.exception
    return app


def _risks(app: AppTest) -> list[str]:
    return [m.value for m in app.metric if m.label == "Estimated risk"]


def test_form_has_22_inputs_bounded_by_schema(schema) -> None:
    app = _app()
    for field in schema:
        widget = app.get("number_input" if field["kind"] == "numeric" else "selectbox")
        keys = {w.key for w in widget}
        assert f"ec_{field['feature']}" in keys
        if field["kind"] == "numeric":
            element = app.number_input(key=f"ec_{field['feature']}")
            assert (element.proto.min, element.proto.max) == (field["min"], field["max"])
        else:
            element = app.selectbox(key=f"ec_{field['feature']}")
            assert len(element.options) == len(ec.options(field))


def test_typical_employee_matches_score_one(model, schema, workforce, features) -> None:
    app = _app()
    expected = score_one(model, ec.default_record(schema, workforce), features)
    assert _risks(app)[0] == f"{expected['attrition_probability']:.0%}"


def test_prefill_only_offers_employees_awaiting_outcome(risk_scores) -> None:
    app = _app()
    app.radio(key="ec_start").set_value(START_EXISTING).run()
    options = app.selectbox(key="ec_employee").options
    assert len(options) == 412
    assert options[0].startswith("698 · rank 1")
    assert _risks(app)[0] == f"{risk_scores.at[698, 'attrition_probability']:.0%}"
    assert any("Above the review cut-off" in m.value for m in app.markdown)


def test_what_if_defaults_to_overtime_no() -> None:
    app = _app()
    app.radio(key="ec_start").set_value(START_EXISTING).run()  # rank 1 works overtime
    assert app.selectbox(key="ec_what_if_feature").value == "OverTime"
    assert app.selectbox(key="ec_what_if_value_OverTime").value == "No"
    result, current, what_if = _risks(app)  # result card, then the comparison
    assert current == result and what_if != current
    assert any("association, not a cause" in c.value for c in app.caption)


def test_cutoff_note_on_page() -> None:
    notes = [c.value for c in _app().caption if "fills the top 20%" in c.value]
    assert len(notes) == 1 and "26.2% cut-off" in notes[0] and "ranks 81–83" in notes[0]


@pytest.mark.parametrize("feature", ["MonthlyIncome", "WorkLifeBalance"])
def test_editing_an_input_rescores(feature) -> None:
    app = _app()
    before = _risks(app)[0]
    if feature == "MonthlyIncome":
        app.number_input(key="ec_MonthlyIncome").set_value(1009).run()
    else:
        app.selectbox(key="ec_WorkLifeBalance").set_value(1).run()
    assert not app.exception
    assert _risks(app)[0] != before


def test_position_never_reads_100_percent() -> None:
    app = _app()
    app.radio(key="ec_start").set_value(START_EXISTING).run()  # rank 1 of 412
    assert any("higher than 99% of the 412 employees" in c.value for c in app.caption)
