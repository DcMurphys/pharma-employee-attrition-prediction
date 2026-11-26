"""How the Model Works: model card and About page in four tabs (scope §4.4).

Numbers arrive from the controller (artifacts); the fixed text summarises NB §8.3-8.4 and
§9.5-9.6. Gender, marital status and age appear only in the fairness section (Q3).
"""

from dataclasses import dataclass
from datetime import date
from fractions import Fraction
from pathlib import Path

import pandas as pd
import streamlit as st

from dashboard.views.components import charts
from dashboard.views.components.feature_labels import FEATURE_GROUPS, label
from dashboard.views.components.format import count, rate
from dashboard.views.components.notices import INTENDED_USE
from dashboard.views.components.one_line import one_line
from dashboard.views.components.palette import BRAND_TEAL, CONTEXT_GREY
from dashboard.views.components.status_badge import FLAG_ICON, NEUTRAL_ICON

NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}

CRITERIA_NAMES = {
    "D1": "Ranks leavers above stayers",
    "D2": "Better than the simple overtime rule",
    "D3": "Finds leavers well beyond chance",
    "D4": "Stable on new data",
}
APPROACH_NAMES = {
    "Majority class (nobody leaves)": "No list (assume nobody leaves)",
    "Rule: OverTime == 'Yes'": "Overtime rule (flag everyone who works overtime)",
    "Gradient Boosting (iteration 1)": "First model version (not used)",
    "Gradient Boosting (iteration 2)": "This model",
    "Gradient Boosting (deployed)": "This model",
}
METRIC_NAMES = {
    "flagged_share": "Employees flagged",
    "recall": "Leavers found",
    "precision": "Flags that were leavers",
    "roc_auc": "Ranking quality (ROC AUC)",
    "average_precision": "Average precision",
}
EXCLUSION_REASONS = {
    "sensitive": "Sensitive personal attribute; left out to reduce the risk of unfair treatment",
    "engineered": "Derived input; added nothing measurable to the ranking",
    "no_association": "No significant link with leaving in the data",
}
ATTRIBUTE_NAMES = {"Gender": "Gender", "MaritalStatus": "Marital status", "AgeBand": "Age band"}
MODEL_LEGEND = [
    charts.LegendItem("This model", BRAND_TEAL),
    charts.LegendItem("Overtime rule", CONTEXT_GREY),
]
# Plain-language glossary for readers without a data background (one sentence per term).
GLOSSARY = [
    (
        "Held-out test data",
        "Employees set aside before training, so the model is checked on "
        "people it did not learn from.",
    ),
    ("Leavers found", "Of everyone who actually left, the share the review list flagged."),
    (
        "Flags that were leavers",
        "Of everyone the list flagged, the share who actually left; "
        "the rest were reviews of people who stayed.",
    ),
    (
        "Ranking quality (ROC AUC)",
        "How often the model places a leaver above a stayer: 0.5 is a coin toss, 1 is perfect.",
    ),
    (
        "Average precision and F2",
        "Single scores for how well the list balances finding "
        "leavers against unnecessary reviews; higher is better.",
    ),
    ("95% range", "The span the true result probably lies in; wide ranges mean few examples."),
    (
        "Four-fifths rule",
        "A fairness check: no group's flag rate should fall below 80% of the "
        "highest group's rate without a review.",
    ),
    (
        "What the model relies on",
        "How much worse the ranking gets when one input is scrambled; "
        "it shows importance, not cause.",
    ),
    (
        "Tuning trial",
        "One automatic attempt with different model settings; the best attempt was kept.",
    ),
]
MONITORING = [
    "Retrain every quarter and compare the new model with the current one on the same "
    "criteria (D1–D4).",
    "Run the four-fifths check on every scoring round, and compare flag rates with actual "
    "leaving rates per group.",
    "Review with HR and legal whether stock-option eligibility should stay a model input.",
    "Record the reason for leaving and the exit date for every departure, so future versions "
    "can learn why and when people leave.",
]
LIMITATIONS = [
    "**Fictional data and company.** The patterns cannot be transferred to a real company "
    "without retraining on its own data.",
    "**One snapshot, no dates.** The model cannot say when someone will leave.",
    "**Few leavers.** {leavers} leavers in the test data limit how precisely the results "
    "can be measured.",
    "**Undocumented currency.** Pay is in monetary units, so pay findings are relative.",
    "**Associations, not causes.** Changes suggested by the model should be piloted and measured.",
    "**No reason for leaving.** The data do not say whether a departure was voluntary.",
]
REFERENCES = [
    "Dicoding Academy. *Employee data* (`employee_data.csv`), copy of the IBM data with an "
    "`EmployeeId` column and {withheld} outcomes removed.",
    "IBM. *HR Analytics Employee Attrition & Performance* (fictional sample data); supplies "
    "the {withheld} withheld outcomes for the external check only.",
    "McFeely, S., & Wigert, B. (2019). *This fixable problem costs U.S. businesses $1 "
    "trillion*. Gallup.",
    "Uniform Guidelines on Employee Selection Procedures, 29 C.F.R. § 1607.4(D) (four-fifths "
    "rule).",
    "Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. "
    "*The Annals of Statistics, 29*(5), 1189–1232.",
    "Akiba, T., et al. (2019). Optuna: A next-generation hyperparameter optimization "
    "framework. *Proc. KDD 2019*, 2623–2631.",
]


@dataclass(frozen=True)
class Versions:
    """Build details shown in the About section."""

    exported_on: date
    iteration: int
    sklearn: str
    python: str
    optuna: str


def in_words(share: float) -> str:
    """Nearest simple fraction in words: 0.593 -> "three in five"."""
    fraction = Fraction(share).limit_denominator(5)
    return f"{NUMBER_WORDS.get(fraction.numerator, fraction.numerator)} in " + (
        NUMBER_WORDS.get(fraction.denominator, str(fraction.denominator))
    )


def _percent_or_dash(value: float) -> str:
    return "—" if pd.isna(value) else rate(value, 0)


def _score_or_dash(value: float) -> str:
    return "—" if pd.isna(value) else f"{value:.3f}"


# ===== Above the tabs: summary for readers without a data background =====
def render_in_short(
    share: float,
    unseen_employees: int,
    model_recall: float,
    model_flagged: float,
    rule_recall: float,
    rule_flagged: float,
) -> None:
    """Three plain sentences: what the model does, how well it works, and its limits."""
    with st.container(border=True):
        st.markdown("**In short**")
        st.markdown(
            "- **What it does.** It ranks employees without a recorded outcome by their risk of "
            f"leaving and flags the top {rate(share, 0)}, so HR can plan supportive stay "
            "conversations.\n"
            f"- **How well it works.** On {count(unseen_employees)} employees it never saw, it "
            f"found {rate(model_recall, 0)} of those who left (about {in_words(model_recall)}) "
            f"while flagging {rate(model_flagged, 0)} of employees. Flagging everyone who works "
            f"overtime found {rate(rule_recall, 0)} and flagged {rate(rule_flagged, 0)}.\n"
            "- **Its limits.** It learned from fictional data in a single snapshot, so it cannot "
            "say when someone will leave, and it is never a basis for disciplinary action, pay "
            "decisions or dismissal."
        )
    with st.expander("What these terms mean", icon=":material/menu_book:"):
        st.markdown("\n".join(f"- **{term}:** {meaning}" for term, meaning in GLOSSARY))


# ===== Tab 1: Reliability =====
def render_decision_rule(
    share: float,
    test_recall: float,
    test_precision: float,
    test_employees: int,
    test_leavers: int,
    external_employees: int,
    external_recall: float,
    external_precision: float,
) -> None:
    """The decision rule in one paragraph, with its tested results."""
    st.subheader("The rule in one paragraph")
    st.markdown(
        f"HR reviews the **{rate(share, 0)} of employees with the highest risk** each round. On "
        f"held-out test data ({count(test_employees)} employees, {count(test_leavers)} of whom "
        f"left), this list found **{rate(test_recall, 0)} of leavers** (about "
        f"{in_words(test_recall)}), and **{rate(test_precision, 0)} of reviews** concerned "
        f"someone who later left (about {in_words(test_precision)}). On "
        f"{count(external_employees)} employees the model never saw, it found "
        f"{rate(external_recall, 0)} of leavers (about {in_words(external_recall)}), and "
        f"{rate(external_precision, 0)} of reviews concerned a leaver (about "
        f"{in_words(external_precision)})."
    )


def render_criteria(table: pd.DataFrame) -> None:
    """D1-D4 with the technical target, the result and the status (icon and words)."""
    st.subheader("Does it meet its targets?")
    st.table(
        pd.DataFrame(
            {
                "Criterion": [f"{i} · {CRITERIA_NAMES.get(i, i)}" for i in table["id"]],
                "Target": table["criterion"],
                "Result": table["value"].map("{:.3f}".format),
                "Status": table["met"].map({True: "✓ Met", False: "✗ Not met"}),
            }
        ),
        hide_index=True,
    )
    st.caption(
        "ROC AUC: how often the model ranks a leaver above a stayer (0.5 = chance, 1 = "
        "perfect). F2 and average precision: how well the list balances finding leavers and "
        "avoiding unnecessary reviews."
    )


def _comparison_table(table: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Approach": table["model"].map(lambda m: APPROACH_NAMES.get(m, m)),
            METRIC_NAMES["flagged_share"]: table["flagged_share"].map(_percent_or_dash),
            METRIC_NAMES["recall"]: table["recall"].map(_percent_or_dash),
            METRIC_NAMES["precision"]: table["precision"].map(_percent_or_dash),
            METRIC_NAMES["roc_auc"]: table["roc_auc"].map(_score_or_dash),
        }
    )


def render_test_comparison(table: pd.DataFrame, intervals: pd.DataFrame, leavers: int) -> None:
    """Held-out test results against simple alternatives, with uncertainty ranges."""
    st.subheader("Compared with simple alternatives")
    st.caption("Held-out test data the model did not learn from.")
    st.table(_comparison_table(table), hide_index=True)
    ranges = intervals.set_index("metric")
    parts = []
    for metric in ("recall", "precision"):
        if metric in ranges.index:
            low, high = ranges.at[metric, "ci_low"], ranges.at[metric, "ci_high"]
            parts.append(f"{METRIC_NAMES[metric].lower()} {rate(low, 0)}–{rate(high, 0)}")
    if "roc_auc" in ranges.index:
        low, high = ranges.at["roc_auc", "ci_low"], ranges.at["roc_auc", "ci_high"]
        parts.append(f"ROC AUC {low:.2f}–{high:.2f}")
    st.caption(
        f"With only {count(leavers)} leavers in the test data, these results are uncertain. "
        f"95% ranges: {'; '.join(parts)}."
    )


def render_external_check(table: pd.DataFrame, employees: int, leavers: int) -> None:
    """External check on employees the model never saw: chart and table."""
    st.subheader("Check on employees the model never saw")
    st.caption(
        f"{count(employees)} employees whose outcome ({count(leavers)} leavers) became known "
        "only from the original IBM file; it was never used to build the model."
    )
    rows = table[table["model"].isin(["Rule: OverTime == 'Yes'", "Gradient Boosting (deployed)"])]
    series = rows["model"].map(
        {"Rule: OverTime == 'Yes'": "Overtime rule", "Gradient Boosting (deployed)": "This model"}
    )
    meaning = {
        "recall": lambda v: f"{rate(v, 0)} of the {count(leavers)} employees who left were flagged",
        "precision": lambda v: (
            f"{rate(v, 0)} of flagged employees had actually left; the rest "
            "were reviews of people who stayed"
        ),
        "flagged_share": lambda v: (
            f"{rate(v, 0)} of the {count(employees)} employees were flagged for review"
        ),
    }
    long = pd.DataFrame(
        [
            {
                "category": METRIC_NAMES[m],
                "series": s,
                "value": v,
                "label": rate(v, 0),
                "tip_title": f"{s}: {METRIC_NAMES[m].lower()}",
                "tip_meaning": meaning[m](v),
            }
            for m in ("recall", "precision", "flagged_share")
            for s, v in zip(series, rows[m], strict=True)
        ]
    )
    charts.show(
        charts.grouped_bars(
            long,
            legend=MODEL_LEGEND,
            value_title="Share",
            axis_format="%",
            tooltip=[(charts.TOOLTIP_HEADING, "tip_title"), ("What this means", "tip_meaning")],
        ),
        _comparison_table(table),
        description="Leavers found, flags that were leavers and employees flagged: this model "
        "versus the overtime rule on employees the model never saw.",
        legend=MODEL_LEGEND,
    )


# ===== Tab 2: Inputs and fairness =====
def render_inputs(used: list[str], excluded: pd.DataFrame) -> None:
    """The model inputs by group, and the excluded columns with reasons."""
    st.subheader("What the model looks at")
    st.markdown(f"The model uses **{len(used)} inputs** from the HR record:")
    used_set = set(used)
    columns = st.columns(2)
    for index, (group, names) in enumerate(FEATURE_GROUPS.items()):
        shown = [label(n) for n in names if n in used_set]
        columns[index % 2].markdown(f"**{group}**  \n" + "  \n".join(f"- {n}" for n in shown))
    st.markdown(f"**Not used ({len(excluded)} columns)**")
    st.table(
        pd.DataFrame(
            {
                "Column": excluded["feature"].map(label),
                "Reason": excluded["reason"].map(EXCLUSION_REASONS),
            }
        ),
        hide_index=True,
    )


def render_importance(table: pd.DataFrame, top: int = 10) -> None:
    """What the model relies on most (whole model, never per employee; Q4)."""
    st.subheader("What the model relies on most")
    st.caption(
        "How much the ranking weakens when one input is shuffled at random (drop in average "
        "precision on test data). This shows what the model relies on, not what causes "
        "leaving, and it describes the whole model, not any one employee."
    )
    head = table.head(top)
    data = pd.DataFrame(
        {
            "group": head["feature"].map(label),
            "value": head["mean_drop_ap"],
            "label": head["mean_drop_ap"].map("{:.3f}".format),
            "tip_drop": [
                f"Ranking quality (average precision) falls by {drop:.3f} when this input is "
                f"shuffled; repeats vary by ±{spread:.3f}"
                for drop, spread in zip(head["mean_drop_ap"], head["std"], strict=True)
            ],
            "tip_rank": [f"{i} of {len(table)} inputs" for i in range(1, len(head) + 1)],
        }
    )
    legend = [charts.LegendItem("Drop in ranking quality without this input", BRAND_TEAL)]
    charts.show(
        charts.value_bars(
            data,
            value_title="Drop in average precision",
            axis_format=".2f",
            tooltip=[
                (charts.TOOLTIP_HEADING, "group"),
                ("Without this input", "tip_drop"),
                ("Importance rank", "tip_rank"),
            ],
        ),
        pd.DataFrame(
            {
                "Input": table["feature"].map(label),
                "Drop in average precision": table["mean_drop_ap"].map("{:.3f}".format),
                "Spread (±)": table["std"].map("{:.3f}".format),
            }
        ),
        description=f"The {top} inputs the model relies on most.",
        legend=legend,
    )


def render_fairness(table: pd.DataFrame, min_ratio: float, employees: int) -> None:
    """Four-fifths check by gender, marital status and age band, with the review note."""
    st.subheader("Fairness check")
    st.markdown(
        f"Flag rates on the {count(employees)} employees the model never saw, by group. If a "
        f"group's flag rate is below {rate(min_ratio, 0)} of the highest group's rate (the "
        "*four-fifths rule*), HR and legal review the model before wider use."
    )
    below = f"{FLAG_ICON} Below four-fifths: HR/legal review"
    within = f"{NEUTRAL_ICON} Within four-fifths"
    with one_line("fairness_table"):
        st.table(
            pd.DataFrame(
                {
                    "Group type": table["attribute"].map(lambda a: ATTRIBUTE_NAMES.get(a, a)),
                    "Group": table["group"],
                    "Employees": table["employees"].map(count),
                    "Flagged": table["flag_rate"].map(lambda r: rate(r)),
                    "Ratio to highest": table["ratio_to_highest"].map("{:.2f}".format),
                    "Check": table["below_four_fifths"].map({True: below, False: within}),
                }
            ),
            hide_index=True,
        )
    st.caption(
        "Gender, marital status and age are not model inputs. Correlated inputs, such as "
        "stock options and career length, still carry part of their information, which "
        "explains the gaps by marital status and age. They trigger the HR and legal review; "
        "a gap can also reflect genuine differences in leaving rates."
    )


# ===== Tab 3: Development =====
def render_development_note() -> None:
    """Tells readers without a data background that this tab is optional."""
    st.caption(
        "For data reviewers: how the model was built and tuned. You can skip this tab to "
        "use the app."
    )


def render_tuning(
    history: pd.DataFrame, n_trials: int, best_trial: int, best_value: float, metric: str
) -> None:
    """Optuna search summary and the best value found so far by trial."""
    st.subheader("How the settings were tuned")
    columns = st.columns(3)
    columns[0].metric("Trials", count(n_trials), border=True)
    columns[1].metric("Best trial", f"#{best_trial}", border=True)
    columns[2].metric(f"Best {metric.replace('_', ' ')}", f"{best_value:.3f}", border=True)
    st.caption(
        "An automated search (Optuna) tried different model settings and kept the best by "
        "cross-validated average precision."
    )
    tips = history.assign(
        tip_trial=[f"Trial {t}" for t in history["trial"]],
        tip_value=[f"{v:.3f} average precision for these settings" for v in history["value"]],
        tip_best=[
            "New best so far" if v >= b else f"{b:.3f}, from an earlier trial"
            for v, b in zip(history["value"], history["best"], strict=True)
        ],
    )
    charts.show(
        charts.best_so_far(
            tips,
            value_title="Average precision (cross-validated)",
            tooltip=[
                (charts.TOOLTIP_HEADING, "tip_trial"),
                ("Result", "tip_value"),
                ("Best so far", "tip_best"),
            ],
        ),
        pd.DataFrame(
            {
                "Trial": history["trial"],
                "Average precision": history["value"].map("{:.3f}".format),
                "Best so far": history["best"].map("{:.3f}".format),
            }
        ),
        description="Average precision of each tuning trial and the best value so far.",
        legend=charts.TUNING_LEGEND,
    )


def render_versions_history(first: pd.Series, deployed: pd.Series) -> None:
    """From the first model version to the deployed one, in two sentences."""
    st.subheader("From the first version to this model")
    st.markdown(
        f"The first version flagged {rate(first['flagged_share'], 0)} of test employees and "
        f"found {rate(first['recall'], 0)} of leavers, a list too long for HR to review. This "
        f"model flags {rate(deployed['flagged_share'], 0)}, finds {rate(deployed['recall'], 0)} "
        f"of leavers, and {rate(deployed['precision'], 0)} of its flags concern a leaver "
        f"(first version: {rate(first['precision'], 0)}). It also uses fewer inputs and no "
        "sensitive attributes."
    )


# ===== Tab 4: Use, limits and About =====
def render_use_and_limits(threshold: float, share: float, test_leavers: int) -> None:
    """Intended use, monitoring and limitations (NB §8.4, §9.5-9.6)."""
    st.subheader("Intended use")
    st.info(INTENDED_USE, icon=":material/volunteer_activism:")
    st.markdown(
        f"- The **Retention Review List** ranks each round and flags the top {rate(share, 0)}.\n"
        f"- **Check an Employee** compares one employee with the {rate(threshold)} cut-off "
        f"(the top-{rate(share, 0)} level on the data the model learned from).\n"
        "- The model is approved for a one-quarter pilot as decision support only."
    )
    st.subheader("Monitoring")
    st.markdown("\n".join(f"- {item}" for item in MONITORING))
    st.subheader("Limitations")
    st.markdown("\n".join(f"- {item.format(leavers=count(test_leavers))}" for item in LIMITATIONS))


def render_about(symbol: Path, versions: Versions, withheld: int) -> None:
    """Data sources, fictional company, references and versions."""
    st.subheader("About")
    logo, text = st.columns([1, 4])
    logo.image(str(symbol), width=96)
    text.markdown(
        "MurphysBio is a **fictional** pharmaceutical and life-sciences company. The data are "
        "IBM's fictional HR Analytics sample, so the findings illustrate the method and say "
        "nothing about any real company."
    )
    st.markdown("**Sources and references**")
    st.markdown("\n".join(f"- {ref.format(withheld=count(withheld))}" for ref in REFERENCES))
    st.markdown("**Versions**")
    st.markdown(
        f"- Model: Gradient Boosting, iteration {versions.iteration}, prepared on "
        f"{versions.exported_on.day} {versions.exported_on:%b %Y}\n"
        f"- scikit-learn {versions.sklearn}, Python {versions.python}, Optuna {versions.optuna}"
    )
