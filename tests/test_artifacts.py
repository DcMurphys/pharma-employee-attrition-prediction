"""Artifact loaders accept the real exports and reject broken files."""

import json
from pathlib import Path

import pytest

from dashboard.models import artifacts
from employee_attrition.config import ATTRITION_STATUSES


def test_real_artifacts_load(metadata, schema, model, workforce, risk_scores) -> None:
    assert len(schema) == metadata["n_features"]
    assert set(workforce["AttritionStatus"]) == set(ATTRITION_STATUSES)
    unknown = workforce.index[workforce["AttritionStatus"] == "Unknown"]
    assert set(risk_scores.index) == set(unknown)


def test_tuning_records_load() -> None:
    best = artifacts.load_tuning_best()
    trials = artifacts.load_tuning_trials()
    assert len(trials) == best["n_trials"]
    assert trials["value"].max() == pytest.approx(best["cv_average_precision"])


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(artifacts.ArtifactError) as err:
        artifacts.load_model_metadata(tmp_path / "absent.json")
    assert err.value.kind == "missing"


def test_unreadable_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{not json")
    with pytest.raises(artifacts.ArtifactError) as err:
        artifacts.load_model_metadata(path)
    assert err.value.kind == "unreadable"


def test_schema_out_of_order_is_invalid(tmp_path: Path, schema, features) -> None:
    path = tmp_path / "feature_schema.json"
    path.write_text(json.dumps(schema[::-1]))
    with pytest.raises(artifacts.ArtifactError) as err:
        artifacts.load_feature_schema(path, features=features)
    assert err.value.kind == "invalid"


# ===== model_card.json (written by the NB §8.4 export cell) =====
def _model_card(features: list[str]) -> dict:
    metrics = {"precision": 0.5, "recall": 0.6, "f2": 0.58, "flagged_share": 0.2}
    return {
        "company": "MurphysBio",
        "criteria": [
            {"id": f"D{n}", "criterion": "...", "value": 0.9, "met": True} for n in range(1, 5)
        ],
        "test_benchmarks": [{"model": "Gradient Boosting (iteration 2)", **metrics}],
        "test_intervals": [
            {"model": "Iteration 2", "metric": "roc_auc", "ci_low": 0.7, "ci_high": 0.9}
        ],
        "external_validation": [{"model": "Gradient Boosting (deployed)", **metrics}],
        "fairness": [
            {
                "attribute": "Gender",
                "group": "Female",
                "employees": 150,
                "flag_rate": 0.2,
                "ratio_to_highest": 0.99,
            }
        ],
        "permutation_importance": [
            {"feature": feature, "mean_drop_ap": 0.01, "std": 0.0} for feature in features
        ],
    }


def test_model_card_contract(tmp_path: Path, features) -> None:
    path = tmp_path / "model_card.json"
    card = _model_card(features)
    path.write_text(json.dumps(card))
    assert artifacts.load_model_card(path, features=features)["company"] == "MurphysBio"

    card["criteria"] = card["criteria"][:1]
    path.write_text(json.dumps(card))
    with pytest.raises(artifacts.ArtifactError) as err:
        artifacts.load_model_card(path, features=features)
    assert err.value.kind == "invalid"


@pytest.mark.skipif(
    not artifacts.MODEL_CARD_PATH.is_file(), reason="model_card.json not exported yet (NB §8.4)"
)
def test_real_model_card_matches_metadata(metadata, features) -> None:
    card = artifacts.load_model_card(features=features)
    deployed = card["test_benchmarks"][-1]
    assert deployed["precision"] == pytest.approx(metadata["test_precision"])
    assert deployed["recall"] == pytest.approx(metadata["test_recall"])
