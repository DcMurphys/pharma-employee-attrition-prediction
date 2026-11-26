"""Validate new employee records and score them with the saved pipeline (NB §8.5).

The pipeline encodes inputs itself and the final feature set has no engineered features, so raw
records go straight to the model. Validation returns language-neutral issue codes; views turn
them into messages.
"""

from dataclasses import dataclass
from typing import Any

import pandas as pd

from employee_attrition.config import REVIEW_CAPACITY_SHARE
from employee_attrition.scoring import rank_and_flag


@dataclass(frozen=True)
class Issue:
    """One validation problem.

    ``kind`` is ``"missing_column"`` or ``"empty"`` (blocking), ``"not_numeric"`` or
    ``"out_of_scale"`` (blocking), or ``"unknown_category"`` / ``"out_of_range"`` (warnings:
    the encoder maps unseen categories to zeros, and trees extrapolate flat).
    """

    column: str
    kind: str
    n_rows: int = 0

    @property
    def blocking(self) -> bool:
        return self.kind in {"missing_column", "empty", "not_numeric", "out_of_scale"}


def validate_records(records: pd.DataFrame, schema: list[dict]) -> list[Issue]:
    """Check ``records`` against ``feature_schema.json``; return every issue found."""
    issues = [
        Issue(record["feature"], "missing_column")
        for record in schema
        if record["feature"] not in records.columns
    ]
    for record in schema:
        column = record["feature"]
        if column not in records.columns:
            continue
        values = records[column]
        if (n_empty := int(values.isna().sum())) > 0:
            issues.append(Issue(column, "empty", n_empty))
        values = values.dropna()
        if record["kind"] == "nominal":
            allowed = set(record["categories"].split("|"))
            if n_unknown := int((~values.astype(str).isin(allowed)).sum()):
                issues.append(Issue(column, "unknown_category", n_unknown))
            continue
        numbers = pd.to_numeric(values, errors="coerce")
        if n_bad := int(numbers.isna().sum()):
            issues.append(Issue(column, "not_numeric", n_bad))
        numbers = numbers.dropna()
        outside = (numbers < record["min"]) | (numbers > record["max"])
        if record["kind"] == "ordinal":
            outside |= numbers != numbers.round()
            if n_out := int(outside.sum()):
                issues.append(Issue(column, "out_of_scale", n_out))
        elif n_out := int(outside.sum()):
            issues.append(Issue(column, "out_of_range", n_out))
    return issues


def score_one(model: Any, record: dict[str, Any], features: list[str]) -> dict[str, float | int]:
    """Probability and flag for one employee; the flag uses the model's tuned threshold."""
    row = pd.DataFrame([record])[features]
    return {
        "attrition_probability": float(model.predict_proba(row)[0, 1]),
        "attrition_flag": int(model.predict(row)[0]),
    }


def score_batch(
    model: Any,
    records: pd.DataFrame,
    features: list[str],
    capacity_share: float = REVIEW_CAPACITY_SHARE,
) -> pd.DataFrame:
    """Score a batch and flag its top ``capacity_share`` by rank (NB §8.2), highest risk first."""
    probabilities = model.predict_proba(records[features])[:, 1]
    ranked = rank_and_flag(probabilities, capacity_share).set_index(records.index)
    return records.join(ranked).sort_values("risk_rank")
