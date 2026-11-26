"""How the Model Works controller: loads the evaluation records and fills four tabs (§4.4)."""

from datetime import datetime

import streamlit as st

from dashboard.controllers import data
from dashboard.controllers.shell import BRAND_DIR, page_header
from dashboard.models import model_card as mc
from dashboard.views import model_card as view

SYMBOL = BRAND_DIR / "murphysbio_symbol.svg"


def render() -> None:
    """Draw the How the Model Works page."""
    page_header("model_card")
    metadata, card = data.model_metadata(), data.model_card()
    best, trials = data.tuning_best(), data.tuning_trials()
    rule = mc.decision_rule(metadata, card)
    external = mc.deployed_row(card, "external_validation")
    external_rule = mc.overtime_rule_row(card, "external_validation")

    view.render_in_short(
        rule.capacity_share,
        int(card["external_employees"]),
        float(external["recall"]),
        float(external["flagged_share"]),
        float(external_rule["recall"]),
        float(external_rule["flagged_share"]),
    )
    reliability, inputs, development, use = st.tabs(
        ["Reliability", "Inputs and fairness", "Development", "Use, limits and About"]
    )
    with reliability:
        view.render_decision_rule(
            rule.capacity_share,
            rule.recall,
            rule.precision,
            rule.test_employees,
            rule.test_leavers,
            int(card["external_employees"]),
            float(external["recall"]),
            float(external["precision"]),
        )
        view.render_criteria(mc.criteria(card))
        view.render_test_comparison(
            mc.benchmarks(card, "test_benchmarks"), mc.intervals(card), rule.test_leavers
        )
        view.render_external_check(
            mc.benchmarks(card, "external_validation"),
            int(card["external_employees"]),
            int(card["external_leavers"]),
        )
    with inputs:
        used, excluded = mc.inputs(metadata)
        view.render_inputs(used, excluded)
        view.render_importance(mc.importance(card))
        view.render_fairness(
            mc.fairness(card), float(card["fairness_min_ratio"]), int(card["external_employees"])
        )
    with development:
        view.render_development_note()
        test = mc.benchmarks(card, "test_benchmarks").set_index("model")
        view.render_versions_history(
            test.loc["Gradient Boosting (iteration 1)"], test.loc[mc.DEPLOYED_TEST]
        )
        view.render_tuning(
            mc.tuning_history(trials),
            int(best["n_trials"]),
            int(best["best_trial_number"]),
            float(best["cv_average_precision"]),
            str(best["objective_metric"]),
        )
    with use:
        view.render_use_and_limits(rule.threshold, rule.capacity_share, rule.test_leavers)
        view.render_about(
            SYMBOL,
            view.Versions(
                exported_on=datetime.fromisoformat(metadata["created_at_utc"]).date(),
                iteration=int(metadata.get("iteration", 2)),
                sklearn=str(metadata["sklearn_version"]),
                python=str(metadata["python_version"]),
                optuna=str(best.get("optuna_version", "—")),
            ),
            withheld=int(card["external_employees"]),
        )
