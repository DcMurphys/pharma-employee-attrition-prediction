"""The shared config agrees with the values recorded in the exported artifacts."""

import employee_attrition
from employee_attrition import config


def test_package_is_importable() -> None:
    assert employee_attrition.__version__ == "0.1.0"


def test_paths_point_into_the_repository() -> None:
    assert (config.PROJECT_ROOT / "pyproject.toml").is_file()
    assert config.MODEL_PATH.is_file()


def test_branding_matches_metadata(metadata: dict) -> None:
    assert metadata["company"] == config.COMPANY_NAME
    assert metadata["disclaimer"] == config.DATA_DISCLAIMER


def test_decision_rule_matches_metadata(metadata: dict) -> None:
    assert (
        metadata["threshold_objective"] == f"top {config.REVIEW_CAPACITY_SHARE:.0%} review capacity"
    )
    assert metadata["calibration_method"] == config.CALIBRATION_METHOD


def test_final_features_exclude_sensitive_and_engineered(features: list[str]) -> None:
    assert not set(features) & set(config.SENSITIVE_FEATURES)
    assert not set(features) & set(config.ENGINEERED_COLS)


def test_schema_kinds_follow_config(schema: list[dict]) -> None:
    for record in schema:
        if record["feature"] in config.NOMINAL_COLS:
            assert record["kind"] == "nominal"
        elif record["feature"] in config.ORDINAL_COLS:
            assert record["kind"] == "ordinal"
