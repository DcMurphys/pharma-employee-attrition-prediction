"""Batch scoring: read an uploaded CSV, prepare it for the model, and build demo samples
(scope §4.3, Tab 2).

Uploads are read from memory and never stored. Validation and scoring themselves are
``validate_records`` and ``score_batch`` in ``dashboard.models.scoring``. This module must not
import streamlit.
"""

import io
from dataclasses import dataclass

import pandas as pd

from employee_attrition.config import ID_COL, RANDOM_STATE

MAX_BYTES = 5 * 1024 * 1024  # 5 MB, as server.maxUploadSize in .streamlit/config.toml
MIN_MEANINGFUL_ROWS = 5  # below this a top-20% rule says little (scope §4.3)
ROW_LABEL = "Row"  # identifier when the file has no unique EmployeeId column


class BatchFileError(Exception):
    """The upload cannot be read as a table. ``kind``: too_large, empty or unreadable."""

    def __init__(self, kind: str, detail: str = "") -> None:
        self.kind, self.detail = kind, detail
        super().__init__(f"{kind}: {detail}".rstrip(": "))


@dataclass(frozen=True)
class BatchSummary:
    """Headline numbers of a scored batch."""

    rows: int
    flagged: int
    mean_probability: float


def read_csv_bytes(data: bytes) -> pd.DataFrame:
    """Read CSV bytes, skipping leading ``#`` comment lines (e.g. the disclaimer).

    Rows are indexed by ``EmployeeId`` when that column exists and is unique, otherwise by
    row number starting at 1.
    """
    if len(data) > MAX_BYTES:
        raise BatchFileError("too_large", f"{len(data)} bytes")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise BatchFileError("unreadable", str(exc)) from exc
    lines = text.splitlines()
    while lines and lines[0].lstrip().startswith("#"):
        lines.pop(0)
    if not any(line.strip() for line in lines):
        raise BatchFileError("empty")
    try:
        records = pd.read_csv(io.StringIO("\n".join(lines)), skipinitialspace=True)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, ValueError) as exc:
        raise BatchFileError("unreadable", f"{type(exc).__name__}: {exc}") from exc
    if records.empty:
        raise BatchFileError("empty")
    if ID_COL in records.columns and records[ID_COL].notna().all() and records[ID_COL].is_unique:
        return records.set_index(ID_COL)
    records.index = pd.RangeIndex(1, len(records) + 1, name=ROW_LABEL)
    return records


def prepare(records: pd.DataFrame, schema: list[dict]) -> pd.DataFrame:
    """Model inputs with the schema's types (call only when validation found no blocking issue)."""
    prepared = pd.DataFrame(index=records.index)
    for field in schema:
        column = records[field["feature"]]
        if field["kind"] == "nominal":
            prepared[field["feature"]] = column.astype(str).str.strip()
        else:
            prepared[field["feature"]] = pd.to_numeric(column)
    return prepared


def summary(scored: pd.DataFrame) -> BatchSummary:
    """Rows, flagged rows and mean probability of a ``score_batch`` result."""
    return BatchSummary(
        rows=len(scored),
        flagged=int(scored["attrition_flag"].sum()),
        mean_probability=float(scored["attrition_probability"].mean()),
    )


# ===== Demo samples, built from the employees awaiting an outcome =====
SAMPLES = {
    "valid": "Valid file (20 employees)",
    "errors": "File with errors (8 employees)",
    "high_risk": "High-risk file (25 employees)",
}


def sample_frames(
    workforce: pd.DataFrame, risk_scores: pd.DataFrame, features: list[str]
) -> dict[str, pd.DataFrame]:
    """The three demo files, keyed as ``SAMPLES``; each has ``EmployeeId`` plus the inputs.

    - ``valid``: 20 random employees awaiting an outcome; scores without issues.
    - ``errors``: the first 8 of them with deliberate problems: a missing column, a text in a
      number column, an empty value, a job level off the scale (all blocking), an unknown
      travel category and a distance outside the training range (warnings).
    - ``high_risk``: the 25 highest-ranked employees of the review list; the batch flags the
      top 5 of them.
    """

    def inputs(ids: pd.Index) -> pd.DataFrame:
        return workforce.loc[ids, features].rename_axis(ID_COL).reset_index()

    valid = inputs(risk_scores.sample(20, random_state=RANDOM_STATE).index.sort_values())

    errors = valid.head(8).drop(columns="DailyRate").astype(object)
    errors.loc[0, "TotalWorkingYears"] = "ten"
    errors.loc[1, "JobLevel"] = 7
    errors.loc[2, "MonthlyIncome"] = None
    errors.loc[3, "BusinessTravel"] = "Remote"
    errors.loc[4, "DistanceFromHome"] = 45

    high_risk = inputs(risk_scores.sort_values("risk_rank").index[:25])
    return {"valid": valid, "errors": errors, "high_risk": high_risk}
