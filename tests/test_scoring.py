"""Scoring parity with the notebook export, plus record validation."""

import numpy as np
import pandas as pd
import pytest

from dashboard.models.scoring import score_batch, score_one, validate_records
from employee_attrition.scoring import flag_rate_ratios, rank_and_flag, review_slots


@pytest.fixture(scope="module")
def unlabeled(workforce: pd.DataFrame) -> pd.DataFrame:
    return workforce[workforce["AttritionStatus"] == "Unknown"]


def test_review_slots_rounds_up() -> None:
    assert review_slots(412) == 83
    assert review_slots(0) == 0
    with pytest.raises(ValueError):
        review_slots(10, 0)


def test_rank_and_flag_reproduces_export(risk_scores: pd.DataFrame) -> None:
    shuffled = risk_scores.sample(frac=1, random_state=0)
    ranked = rank_and_flag(shuffled["attrition_probability"].to_numpy())
    ranked.index = shuffled.index
    pd.testing.assert_series_equal(
        ranked["attrition_flag"].sort_index(), risk_scores["attrition_flag"].sort_index()
    )


def test_batch_scoring_reproduces_export(model, features, unlabeled, risk_scores) -> None:
    scored = score_batch(model, unlabeled, features)
    assert scored.index.tolist() == risk_scores.index.tolist()
    np.testing.assert_allclose(
        scored["attrition_probability"], risk_scores["attrition_probability"], atol=1e-12
    )
    assert (scored["attrition_flag"] == risk_scores["attrition_flag"]).all()


def test_single_scoring_uses_tuned_threshold(model, features, metadata, unlabeled) -> None:
    record = unlabeled.iloc[0][features].to_dict()
    result = score_one(model, record, features)
    expected = int(result["attrition_probability"] >= metadata["decision_threshold"])
    assert result["attrition_flag"] == expected


def test_valid_records_have_no_blocking_issues(schema, unlabeled) -> None:
    assert not [issue for issue in validate_records(unlabeled, schema) if issue.blocking]


def test_validation_reports_problems(schema, features, unlabeled) -> None:
    broken = unlabeled[features].head(3).copy()
    broken = broken.drop(columns="OverTime").astype({"JobLevel": float})
    broken.loc[broken.index[0], "JobLevel"] = 2.5
    broken.loc[broken.index[1], "BusinessTravel"] = "Remote"
    kinds = {(issue.column, issue.kind) for issue in validate_records(broken, schema)}
    assert ("OverTime", "missing_column") in kinds
    assert ("JobLevel", "out_of_scale") in kinds
    assert ("BusinessTravel", "unknown_category") in kinds


def test_flag_rate_ratios() -> None:
    flags = pd.Series([1, 0, 1, 1, 0, 0])
    groups = pd.Series(["a", "a", "a", "b", "b", "b"])
    table = flag_rate_ratios(flags, groups)
    assert table.loc["b", "ratio_to_highest"] == pytest.approx(0.5)
    assert bool(table.loc["b", "below_four_fifths"])
