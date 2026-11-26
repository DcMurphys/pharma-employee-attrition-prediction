"""How the Model Works: every table matches model_metadata.json, model_card.json and Optuna."""

import pytest

from dashboard.models import artifacts
from dashboard.models import model_card as mc


@pytest.fixture(scope="module")
def card(features) -> dict:
    return artifacts.load_model_card(features=features)


def test_criteria_match_metadata(card, metadata) -> None:
    table = mc.criteria(card).set_index("id")
    assert table.index.tolist() == ["D1", "D2", "D3", "D4"]
    assert table["met"].all()
    for key, value in mc.criteria_from_metadata(metadata).items():
        assert table.at[key, "value"] == pytest.approx(value, abs=1e-6), key
    assert round(table.at["D1", "value"], 3) == 0.852
    assert round(table.at["D3", "value"], 3) == 0.663
    assert round(table.at["D4", "value"], 3) == 0.048


def test_decision_rule(card, metadata) -> None:
    rule = mc.decision_rule(metadata, card)
    assert rule.capacity_share == 0.2
    assert round(rule.recall, 3) == round(rule.precision, 3) == 0.593
    assert round(rule.threshold, 3) == 0.262
    assert (rule.test_employees, rule.test_leavers) == (159, 27)


def test_benchmarks_match_metadata(card, metadata) -> None:
    deployed = mc.deployed_row(card, "test_benchmarks")
    assert deployed["roc_auc"] == pytest.approx(metadata["test_roc_auc"])
    assert deployed["recall"] == pytest.approx(metadata["test_recall"])
    rule = mc.benchmarks(card, "test_benchmarks").set_index("model").loc["Rule: OverTime == 'Yes'"]
    assert round(rule["f2"], 3) == 0.523


def test_external_check_matches_scope(card) -> None:
    deployed = mc.deployed_row(card, "external_validation")
    assert [round(deployed[k], 3) for k in ("roc_auc", "precision", "recall")] == [
        0.828,
        0.470,
        0.672,
    ]
    assert card["external_employees"] == 412


def test_inputs_used_and_excluded(metadata) -> None:
    used, excluded = mc.inputs(metadata)
    assert len(used) == 22 and len(excluded) == 11
    reasons = excluded.set_index("feature")["reason"]
    assert set(reasons[reasons == "sensitive"].index) == {"Age", "Gender", "MaritalStatus"}
    assert (reasons == "engineered").sum() == 3
    assert (reasons == "no_association").sum() == 5


def test_importance_and_fairness(card) -> None:
    importance = mc.importance(card)
    assert len(importance) == 22 and importance.loc[0, "feature"] == "OverTime"
    fairness = mc.fairness(card)
    assert fairness["attribute"].unique().tolist() == ["Gender", "MaritalStatus", "AgeBand"]
    below = fairness[fairness["below_four_fifths"]]
    assert set(below["attribute"]) == {"MaritalStatus", "AgeBand"}


def test_tuning_history(card) -> None:
    best = artifacts.load_tuning_best()
    history = mc.tuning_history(artifacts.load_tuning_trials())
    assert len(history) == best["n_trials"] == 100
    assert history["best"].is_monotonic_increasing
    assert history["best"].iloc[-1] == pytest.approx(best["cv_average_precision"])
    assert int(history.loc[history["value"].idxmax(), "trial"]) == best["best_trial_number"] == 66


def test_overtime_rule_row(card) -> None:
    row = mc.overtime_rule_row(card, "external_validation")
    assert [round(row[k], 3) for k in ("recall", "flagged_share")] == [0.5, 0.265]
