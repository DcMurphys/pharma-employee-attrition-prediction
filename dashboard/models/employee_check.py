"""Check an Employee, single record: form defaults, prefill, position and what-if (scope §4.3).

Prefill is allowed only for employees without a recorded outcome (Q2), so a known leaver or
stayer is never "predicted". Scoring itself is ``dashboard.models.scoring.score_one``. This
module must not import streamlit.
"""

from typing import Any

import pandas as pd

from dashboard.models.scoring import score_one
from dashboard.models.workforce import known_outcome
from employee_attrition.config import STATUS_COL

UNKNOWN = "Unknown"


def options(record: dict) -> list:
    """Allowed values of a nominal or ordinal schema record (ordinal: whole numbers)."""
    if record["kind"] == "nominal":
        return record["categories"].split("|")
    return list(range(int(record["min"]), int(record["max"]) + 1))


def default_record(schema: list[dict], workforce: pd.DataFrame) -> dict[str, Any]:
    """A typical employee: schema medians (whole numbers) and the most frequent category.

    Most frequent categories come from the employees with a known outcome, the population the
    model learned from.
    """
    known = known_outcome(workforce)
    record: dict[str, Any] = {}
    for field in schema:
        name = field["feature"]
        if field["kind"] == "nominal":
            record[name] = str(known[name].mode().iloc[0])
        else:
            record[name] = int(round(field["median"]))
    return record


def awaiting_ids(workforce: pd.DataFrame) -> list[int]:
    """Employee IDs without a recorded outcome (the only ones that may prefill the form)."""
    return sorted(int(i) for i in workforce.index[workforce[STATUS_COL] == UNKNOWN])


def prefill_record(
    workforce: pd.DataFrame, employee_id: int, features: list[str]
) -> dict[str, Any]:
    """Model inputs of one employee awaiting an outcome.

    Raises:
        ValueError: if the employee does not exist or already has a recorded outcome (Q2).
    """
    if employee_id not in workforce.index or workforce.at[employee_id, STATUS_COL] != UNKNOWN:
        raise ValueError(f"Employee {employee_id} is not awaiting an outcome.")
    row = workforce.loc[employee_id, features]
    return {
        name: (value.item() if hasattr(value, "item") else value) for name, value in row.items()
    }


def share_below(probability: float, scores: pd.DataFrame) -> float:
    """Share of scored employees awaiting an outcome with a risk below ``probability``."""
    return float((scores["attrition_probability"] < probability).mean())


def what_if(
    model: Any, record: dict[str, Any], feature: str, value: Any, features: list[str]
) -> tuple[dict[str, float | int], dict[str, float | int]]:
    """Score ``record`` before and after setting ``feature`` to ``value`` (one change only)."""
    changed = {**record, feature: value}
    return score_one(model, record, features), score_one(model, changed, features)


def suggested_value(field: dict, current: Any) -> Any:
    """Default what-if value: overtime "No" (scope §4.3), otherwise the next allowed value."""
    if field["feature"] == "OverTime":
        return "No" if current != "No" else "Yes"
    if field["kind"] == "numeric":
        return current
    allowed = options(field)
    position = allowed.index(current) if current in allowed else 0
    return allowed[(position + 1) % len(allowed)]
