"""Batch scoring: file reading, the three sample scenarios, and parity with the export."""

import numpy as np
import pandas as pd
import pytest

from dashboard.models import batch, export
from dashboard.models.scoring import score_batch, validate_records


@pytest.fixture(scope="module")
def samples(workforce, risk_scores, features) -> dict[str, pd.DataFrame]:
    return batch.sample_frames(workforce, risk_scores, features)


def _roundtrip(frame: pd.DataFrame) -> pd.DataFrame:
    """Write a sample as the app offers it (with the disclaimer) and read it back."""
    return batch.read_csv_bytes(export.csv_with_notes(frame, ["Disclaimer.", "Use note."]))


def test_valid_sample_scores_like_the_export(samples, schema, features, model, risk_scores):
    records = _roundtrip(samples["valid"])
    assert records.index.name == "EmployeeId" and len(records) == 20
    issues = validate_records(records, schema)
    assert issues == []
    scored = score_batch(model, batch.prepare(records, schema), features)
    np.testing.assert_allclose(
        scored["attrition_probability"],
        risk_scores.loc[scored.index, "attrition_probability"],
        atol=1e-12,
    )
    assert batch.summary(scored).flagged == 4  # ceil(20% of 20)
    assert scored["risk_rank"].tolist() == list(range(1, 21))


def test_error_sample_lists_blocking_issues_and_warnings(samples, schema):
    issues = validate_records(_roundtrip(samples["errors"]), schema)
    blocking = {(i.column, i.kind) for i in issues if i.blocking}
    warnings = {(i.column, i.kind) for i in issues if not i.blocking}
    assert blocking == {
        ("DailyRate", "missing_column"),
        ("TotalWorkingYears", "not_numeric"),
        ("JobLevel", "out_of_scale"),
        ("MonthlyIncome", "empty"),
    }
    assert warnings == {
        ("BusinessTravel", "unknown_category"),
        ("DistanceFromHome", "out_of_range"),
    }


def test_high_risk_sample_flags_the_top_five(samples, schema, features, model, risk_scores):
    records = _roundtrip(samples["high_risk"])
    assert not [i for i in validate_records(records, schema) if i.blocking]
    scored = score_batch(model, batch.prepare(records, schema), features)
    flagged = set(scored.index[scored["attrition_flag"] == 1])
    top_five = set(risk_scores.sort_values("risk_rank").index[:5])
    assert flagged == top_five


def test_file_without_unique_ids_is_numbered():
    records = batch.read_csv_bytes(b"OverTime,JobLevel\nYes,1\nNo,2\n")
    assert records.index.name == "Row" and records.index.tolist() == [1, 2]


@pytest.mark.parametrize(
    ("data", "kind"),
    [
        (b"", "empty"),
        (b"# only a comment\n", "empty"),
        (b"\xff\xfe\x00bad", "unreadable"),
        (b"a,b\n1,2\n3,4,5\n", "unreadable"),
        (b"x" * (batch.MAX_BYTES + 1), "too_large"),
    ],
)
def test_unreadable_files(data, kind):
    with pytest.raises(batch.BatchFileError) as err:
        batch.read_csv_bytes(data)
    assert err.value.kind == kind
