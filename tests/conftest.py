"""Shared fixtures: the real notebook artifacts, loaded read-only once per test session."""

import pandas as pd
import pytest

from dashboard.models import artifacts


@pytest.fixture(scope="session")
def metadata() -> dict:
    return artifacts.load_model_metadata()


@pytest.fixture(scope="session")
def features(metadata: dict) -> list[str]:
    return artifacts.feature_list(metadata)


@pytest.fixture(scope="session")
def schema(features: list[str]) -> list[dict]:
    return artifacts.load_feature_schema(features=features)


@pytest.fixture(scope="session")
def model(metadata: dict):
    return artifacts.load_model(metadata=metadata)


@pytest.fixture(scope="session")
def workforce() -> pd.DataFrame:
    return artifacts.load_workforce()


@pytest.fixture(scope="session")
def risk_scores() -> pd.DataFrame:
    return artifacts.load_risk_scores()
