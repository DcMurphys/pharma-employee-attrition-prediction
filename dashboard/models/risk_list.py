"""Retention Review List: round summary, filters and team counts (B2; NB §8.2, scope §4.2).

The list flags by rank: the top ``REVIEW_CAPACITY_SHARE`` of a scoring round (rounded up) fill
the review slots. Filters only hide rows; they never re-rank. This module must not import
streamlit.
"""

from collections.abc import Collection
from dataclasses import dataclass

import pandas as pd

from employee_attrition.config import REVIEW_CAPACITY_SHARE

PROBABILITY = "attrition_probability"
RANK = "risk_rank"
FLAG = "attrition_flag"


@dataclass(frozen=True)
class RoundSummary:
    """Size of the scoring round and how many employees fill its review slots."""

    employees: int
    flagged: int
    capacity_share: float = REVIEW_CAPACITY_SHARE


@dataclass(frozen=True)
class CutoffGap:
    """Flagged employees whose probability is below the single-record cut-off (Q1)."""

    ranks: list[int]
    min_probability: float
    max_probability: float
    cutoff: float


def round_summary(scores: pd.DataFrame) -> RoundSummary:
    """Employees scored in this round and how many are flagged."""
    return RoundSummary(employees=len(scores), flagged=int(scores[FLAG].sum()))


def flagged_below_cutoff(scores: pd.DataFrame, cutoff: float) -> CutoffGap | None:
    """Flagged employees below ``cutoff`` (the model's single-record threshold), if any."""
    gap = scores[(scores[FLAG] == 1) & (scores[PROBABILITY] < cutoff)]
    if gap.empty:
        return None
    return CutoffGap(
        ranks=sorted(gap[RANK].astype(int).tolist()),
        min_probability=float(gap[PROBABILITY].min()),
        max_probability=float(gap[PROBABILITY].max()),
        cutoff=cutoff,
    )


def options(scores: pd.DataFrame, column: str, within: pd.Series | None = None) -> list[str]:
    """Sorted distinct values of ``column``, optionally limited to rows where ``within``."""
    values = scores.loc[within, column] if within is not None else scores[column]
    return sorted(values.dropna().unique())


def filter_list(
    scores: pd.DataFrame,
    *,
    departments: Collection[str] = (),
    job_roles: Collection[str] = (),
    overtime: str | None = None,
    flagged_only: bool = True,
) -> pd.DataFrame:
    """Rows matching the filters, in rank order; an empty collection means "all"."""
    keep = pd.Series(True, index=scores.index)
    if departments:
        keep &= scores["Department"].isin(departments)
    if job_roles:
        keep &= scores["JobRole"].isin(job_roles)
    if overtime is not None:
        keep &= scores["OverTime"] == overtime
    if flagged_only:
        keep &= scores[FLAG] == 1
    return scores[keep].sort_values(RANK)


def flagged_counts(scores: pd.DataFrame, column: str) -> pd.DataFrame:
    """Flagged employees per value of ``column``, most first (``group``, ``flagged``)."""
    flagged = scores[scores[FLAG] == 1]
    counts = flagged[column].value_counts().rename_axis("group").reset_index(name="flagged")
    return counts.sort_values(["flagged", "group"], ascending=[False, True]).reset_index(drop=True)
