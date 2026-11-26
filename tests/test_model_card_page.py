"""How the Model Works page (M6): every shown number comes from the artifacts."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from dashboard.views.model_card import in_words

APP = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"


@pytest.fixture(scope="module")
def app() -> AppTest:
    app = AppTest.from_file(str(APP), default_timeout=60)
    app.run()
    app.switch_page("page_scripts/model_card.py").run()
    assert not app.exception and not app.error
    return app


def _table(app: AppTest, column: str):
    return next(t.value for t in app.table if column in t.value.columns)


@pytest.mark.parametrize(
    ("share", "words"), [(0.593, "three in five"), (0.470, "one in two"), (0.672, "two in three")]
)
def test_in_words(share, words) -> None:
    assert in_words(share) == words


def test_tabs(app) -> None:
    assert [t.label for t in app.tabs] == [
        "Reliability",
        "Inputs and fairness",
        "Development",
        "Use, limits and About",
    ]


def test_decision_rule_uses_metadata(app) -> None:
    text = next(m.value for m in app.markdown if "of employees with the highest risk" in m.value)
    assert "**20% of employees with the highest risk**" in text
    assert "159 employees, 27 of whom left" in text
    assert "**59% of leavers** (about three in five)" in text
    assert "found 67% of leavers (about two in three)" in text
    assert "47% of reviews concerned a leaver (about one in two)" in text


def test_criteria_table(app) -> None:
    table = _table(app, "Criterion")
    assert table["Result"].tolist() == ["0.852", "0.593", "0.663", "0.048"]
    assert set(table["Status"]) == {"✓ Met"}


def test_comparison_tables(app) -> None:
    test = _table(app, "Approach")
    model = test[test["Approach"] == "This model"].iloc[0]
    assert (model["Leavers found"], model["Flags that were leavers"]) == ("59%", "59%")
    assert model["Ranking quality (ROC AUC)"] == "0.852"
    external = [t.value for t in app.table if "Approach" in t.value.columns][1]
    deployed = external[external["Approach"] == "This model"].iloc[0]
    assert (deployed["Leavers found"], deployed["Flags that were leavers"]) == ("67%", "47%")
    assert deployed["Ranking quality (ROC AUC)"] == "0.828"


def test_fairness_and_inputs(app) -> None:
    fairness = _table(app, "Group type")
    below = fairness[fairness["Check"].str.startswith("▲")]
    assert set(below["Group type"]) == {"Marital status", "Age band"}
    excluded = _table(app, "Reason")
    assert len(excluded) == 11


def test_tuning_summary(app) -> None:
    metrics = {m.label: m.value for m in app.metric}
    assert metrics["Trials"] == "100" and metrics["Best trial"] == "#66"
    assert metrics["Best average precision"] == "0.604"


def test_in_short_summary_uses_external_check(app) -> None:
    text = next(m.value for m in app.markdown if "**What it does.**" in m.value)
    assert "flags the top 20%" in text
    assert "On 412 employees it never saw, it found 67% of those who left" in text
    assert "(about two in three) while flagging 20% of employees" in text
    assert "Flagging everyone who works overtime found 50% and flagged 26%" in text


def test_glossary_and_development_note(app) -> None:
    assert any(e.label == "What these terms mean" for e in app.expander)
    glossary = next(m.value for m in app.markdown if "**Four-fifths rule:**" in m.value)
    assert "**Ranking quality (ROC AUC):**" in glossary
    assert any("For data reviewers" in c.value for c in app.caption)
