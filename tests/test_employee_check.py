"""Check an Employee (single record): defaults, Q2 prefill, parity with score_one, what-if."""

import pytest

from dashboard.models import employee_check as ec
from dashboard.models.scoring import score_one


def test_default_record_is_typical_and_valid(schema, workforce) -> None:
    record = ec.default_record(schema, workforce)
    assert len(record) == 22
    assert record["OverTime"] == "No" and record["JobRole"] == "Sales Executive"
    assert record["MonthlyIncome"] == 4904 and record["JobLevel"] == 2
    for field in schema:
        value = record[field["feature"]]
        if field["kind"] == "nominal":
            assert value in ec.options(field)
        else:
            assert field["min"] <= value <= field["max"]


def test_prefill_only_from_employees_awaiting_outcome(workforce, features, risk_scores) -> None:
    ids = ec.awaiting_ids(workforce)
    assert len(ids) == 412 and set(ids) == set(risk_scores.index)
    known_id = int(workforce.index[workforce["AttritionStatus"] == "Yes"][0])
    with pytest.raises(ValueError):
        ec.prefill_record(workforce, known_id, features)


def test_prefill_scores_match_export(model, workforce, features, risk_scores) -> None:
    for employee_id in risk_scores.index[:5]:
        record = ec.prefill_record(workforce, int(employee_id), features)
        result = score_one(model, record, features)
        assert result["attrition_probability"] == pytest.approx(
            risk_scores.at[employee_id, "attrition_probability"], abs=1e-12
        )


def test_prefilled_values_fit_form_bounds(schema, workforce, features) -> None:
    for employee_id in ec.awaiting_ids(workforce):
        record = ec.prefill_record(workforce, employee_id, features)
        for field in schema:
            value = record[field["feature"]]
            if field["kind"] == "nominal":
                assert value in ec.options(field)
            else:
                assert field["min"] <= value <= field["max"], (employee_id, field["feature"])


def test_share_below(risk_scores) -> None:
    top = risk_scores["attrition_probability"].max()
    assert ec.share_below(top, risk_scores) == pytest.approx(411 / 412)
    assert ec.share_below(0.0, risk_scores) == 0.0


def test_what_if_changes_one_input(model, workforce, features, risk_scores) -> None:
    employee_id = int(risk_scores.index[0])  # rank 1, works overtime
    record = ec.prefill_record(workforce, employee_id, features)
    assert record["OverTime"] == "Yes"
    before, after = ec.what_if(model, record, "OverTime", "No", features)
    assert before == score_one(model, record, features)
    assert after["attrition_probability"] < before["attrition_probability"]
    assert record["OverTime"] == "Yes"  # the original record is unchanged


def test_suggested_value(schema) -> None:
    fields = {field["feature"]: field for field in schema}
    assert ec.suggested_value(fields["OverTime"], "Yes") == "No"
    assert ec.suggested_value(fields["WorkLifeBalance"], 4) == 1
    assert ec.suggested_value(fields["MonthlyIncome"], 3000) == 3000
