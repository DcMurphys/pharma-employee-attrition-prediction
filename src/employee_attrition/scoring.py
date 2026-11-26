"""Decision rules shared by the notebook and the app (NB §8.2, §8.3).

These functions work on plain probabilities and labels, so they need neither scikit-learn nor
the trained model.
"""

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

from employee_attrition.config import FAIRNESS_MIN_RATIO, REVIEW_CAPACITY_SHARE


def review_slots(n_employees: int, capacity_share: float = REVIEW_CAPACITY_SHARE) -> int:
    """Number of employees HR can review in one scoring round (rounded up)."""
    if n_employees < 0 or not 0 < capacity_share <= 1:
        raise ValueError("n_employees must be >= 0 and capacity_share in (0, 1].")
    return math.ceil(capacity_share * n_employees)


def rank_and_flag(
    probabilities: Sequence[float] | np.ndarray | pd.Series,
    capacity_share: float = REVIEW_CAPACITY_SHARE,
) -> pd.DataFrame:
    """Rank employees by attrition probability and flag the top ``capacity_share``.

    Ties keep their input order (``method="first"``), as in the notebook export.

    Returns:
        Frame with ``attrition_probability``, ``risk_rank`` (1 = highest risk) and
        ``attrition_flag`` (0/1), in the input order and with the input index.
    """
    scores = pd.Series(probabilities, name="attrition_probability", dtype=float)
    ranks = scores.rank(ascending=False, method="first").astype(int)
    slots = review_slots(len(scores), capacity_share)
    return pd.DataFrame(
        {
            "attrition_probability": scores,
            "risk_rank": ranks,
            "attrition_flag": (ranks <= slots).astype(int),
        }
    )


def flag_rate_ratios(flags: pd.Series, groups: pd.Series) -> pd.DataFrame:
    """Compare flag rates across groups using the four-fifths rule of thumb.

    Args:
        flags: 0/1 model flags indexed by employee.
        groups: Group label of each employee, with the same index.

    Returns:
        One row per group with its size, flag rate, ratio to the highest flag rate and
        whether that ratio is below ``FAIRNESS_MIN_RATIO``.
    """
    table = flags.groupby(groups, observed=True).agg(employees="size", flag_rate="mean")
    table["ratio_to_highest"] = table["flag_rate"] / table["flag_rate"].max()
    table["below_four_fifths"] = table["ratio_to_highest"] < FAIRNESS_MIN_RATIO
    return table
