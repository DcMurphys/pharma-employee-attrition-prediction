"""Retention Review List logic: 83 of 412 flagged, filters never re-rank, Q1 gap (scope §4.2)."""

import io

import pandas as pd

from dashboard.models import export
from dashboard.models import risk_list as rl


def test_round_summary(risk_scores) -> None:
    summary = rl.round_summary(risk_scores)
    assert (summary.employees, summary.flagged, summary.capacity_share) == (412, 83, 0.2)


def test_flagged_below_cutoff_matches_scope(risk_scores, metadata) -> None:
    gap = rl.flagged_below_cutoff(risk_scores, metadata["decision_threshold"])
    assert gap.ranks == [81, 82, 83]
    assert (round(gap.min_probability, 3), round(gap.max_probability, 3)) == (0.253, 0.261)
    assert round(gap.cutoff, 3) == 0.262
    assert rl.flagged_below_cutoff(risk_scores, 0.0) is None


def test_filters_never_rerank(risk_scores) -> None:
    rows = rl.filter_list(risk_scores, departments=["Sales"], overtime="Yes", flagged_only=False)
    assert rows["risk_rank"].is_monotonic_increasing
    assert (rows["risk_rank"] == risk_scores.loc[rows.index, "risk_rank"]).all()
    assert set(rows["Department"]) == {"Sales"} and set(rows["OverTime"]) == {"Yes"}


def test_flagged_only_default(risk_scores) -> None:
    assert len(rl.filter_list(risk_scores)) == 83
    assert len(rl.filter_list(risk_scores, flagged_only=False)) == 412
    sales = rl.filter_list(risk_scores, departments=["Sales"])
    assert len(sales) == 27


def test_job_role_options_follow_departments(risk_scores) -> None:
    roles = rl.options(risk_scores, "JobRole", risk_scores["Department"].isin(["Sales"]))
    assert "Sales Executive" in roles and "Laboratory Technician" not in roles


def test_flagged_counts(risk_scores) -> None:
    counts = rl.flagged_counts(risk_scores, "Department").set_index("group")["flagged"]
    assert counts.to_dict() == {"Research & Development": 50, "Sales": 27, "Human Resources": 6}


def test_csv_starts_with_disclaimer() -> None:
    data = export.csv_with_notes(pd.DataFrame({"a": [1]}), ["Disclaimer.", "Use note."])
    text = data.decode("utf-8")
    assert text.splitlines()[:2] == ["# Disclaimer.", "# Use note."]
    assert pd.read_csv(io.StringIO(text), comment="#")["a"].tolist() == [1]
    assert export.demo_file_name("retention_review_list") == "retention_review_list_demo.csv"
