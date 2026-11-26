"""How the Model Works: criteria, benchmarks, inputs, fairness and tuning history (scope §4.4).

Every value comes from ``model_metadata.json``, ``model_card.json`` or the Optuna records.
This module must not import streamlit.
"""

from dataclasses import dataclass

import pandas as pd

from dashboard.models.artifacts import feature_list
from employee_attrition.config import ENGINEERED_COLS, SENSITIVE_FEATURES

DEPLOYED_TEST = "Gradient Boosting (iteration 2)"
DEPLOYED_EXTERNAL = "Gradient Boosting (deployed)"
DEPLOYED_INTERVALS = "Iteration 2"
OVERTIME_RULE = "Rule: OverTime == 'Yes'"
BENCHMARK_COLUMNS = ["model", "flagged_share", "recall", "precision", "f2", "roc_auc"]
FAIRNESS_ORDER = ["Gender", "MaritalStatus", "AgeBand"]


@dataclass(frozen=True)
class DecisionRule:
    """The review rule and its test-set result (NB §7.9, §8.2)."""

    capacity_share: float
    recall: float  # share of leavers found by the review list
    precision: float  # share of reviews that concern someone who left
    threshold: float  # single-record cut-off
    test_employees: int
    test_leavers: int


def decision_rule(metadata: dict, card: dict) -> DecisionRule:
    """Review share, test recall and precision, and the single-record cut-off."""
    return DecisionRule(
        capacity_share=float(card["review_capacity_share"]),
        recall=float(metadata["test_recall"]),
        precision=float(metadata["test_precision"]),
        threshold=float(metadata["decision_threshold"]),
        test_employees=int(card["test_employees"]),
        test_leavers=int(card["test_leavers"]),
    )


def criteria(card: dict) -> pd.DataFrame:
    """Success criteria D1-D4 (``id``, ``criterion``, ``value``, ``met``)."""
    return pd.DataFrame(card["criteria"])[["id", "criterion", "value", "met"]]


def criteria_from_metadata(metadata: dict) -> dict[str, float]:
    """The D1, D3 and D4 values recomputed from ``model_metadata.json`` (consistency check)."""
    return {
        "D1": float(metadata["test_roc_auc"]),
        "D2": float(metadata["test_f2"]),
        "D3": float(metadata["test_average_precision"]),
        "D4": abs(float(metadata["cv_roc_auc"]) - float(metadata["test_roc_auc"])),
    }


def benchmarks(card: dict, table: str) -> pd.DataFrame:
    """Test (``test_benchmarks``) or external (``external_validation``) comparison table."""
    return pd.DataFrame(card[table])[BENCHMARK_COLUMNS].reset_index(drop=True)


def deployed_row(card: dict, table: str) -> pd.Series:
    """The deployed model's row in a comparison table."""
    name = DEPLOYED_TEST if table == "test_benchmarks" else DEPLOYED_EXTERNAL
    frame = benchmarks(card, table)
    return frame[frame["model"] == name].iloc[0]


def overtime_rule_row(card: dict, table: str) -> pd.Series:
    """The simple overtime rule's row in a comparison table."""
    frame = benchmarks(card, table)
    return frame[frame["model"] == OVERTIME_RULE].iloc[0]


def intervals(card: dict) -> pd.DataFrame:
    """95% bootstrap intervals of the deployed setup (``metric``, ``ci_low``, ``ci_high``)."""
    frame = pd.DataFrame(card["test_intervals"])
    return frame[frame["model"] == DEPLOYED_INTERVALS][["metric", "ci_low", "ci_high"]]


def inputs(metadata: dict) -> tuple[list[str], pd.DataFrame]:
    """Model inputs, and the excluded columns with a reason code.

    Reasons: ``sensitive`` (fairness), ``engineered`` (no measurable gain, NB Insight 20) or
    ``no_association`` (removed by the chi-square screen, NB §5.6).
    """
    used = feature_list(metadata)
    excluded = [n.strip() for n in metadata["excluded_features"].split(",") if n.strip()]

    def reason(name: str) -> str:
        if name in SENSITIVE_FEATURES:
            return "sensitive"
        if name in ENGINEERED_COLS:
            return "engineered"
        return "no_association"

    table = pd.DataFrame({"feature": excluded, "reason": [reason(n) for n in excluded]})
    return used, table


def importance(card: dict) -> pd.DataFrame:
    """Permutation importance, largest first (``feature``, ``mean_drop_ap``, ``std``)."""
    frame = pd.DataFrame(card["permutation_importance"])[["feature", "mean_drop_ap", "std"]]
    return frame.sort_values("mean_drop_ap", ascending=False).reset_index(drop=True)


def fairness(card: dict) -> pd.DataFrame:
    """Four-fifths check on the external review list, grouped by attribute."""
    frame = pd.DataFrame(card["fairness"])
    frame["below_four_fifths"] = frame["ratio_to_highest"] < float(card["fairness_min_ratio"])
    order = {name: i for i, name in enumerate(FAIRNESS_ORDER)}
    position = frame["attribute"].map(order).fillna(len(order))
    frame = frame.assign(_order=position).sort_values(["_order", "group"]).drop(columns="_order")
    return frame[
        ["attribute", "group", "employees", "flag_rate", "ratio_to_highest", "below_four_fifths"]
    ].reset_index(drop=True)


def tuning_history(trials: pd.DataFrame) -> pd.DataFrame:
    """Completed Optuna trials with the best value found so far (``trial``, ``value``, ``best``)."""
    done = trials[trials["state"] == "COMPLETE"].sort_index()
    return pd.DataFrame({"trial": done.index.to_numpy(), "value": done["value"].to_numpy()}).assign(
        best=lambda f: f["value"].cummax()
    )
