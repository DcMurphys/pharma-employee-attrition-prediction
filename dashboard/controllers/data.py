"""Cached access to the notebook artifacts (scope §6).

The model is loaded once per process with ``st.cache_resource``; frames and small records use
``st.cache_data``. Every cache key includes the file's modification time, so a re-exported
artifact is picked up without restarting the app. Loaders raise ``ArtifactError``, which is
never cached.

For tests, the environment variables ``MURPHYSBIO_ARTIFACTS_DIR`` and ``MURPHYSBIO_DATA_DIR``
redirect the ``artifacts/`` and ``data/processed/`` folders (scope §10).
"""

import os
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from dashboard.models import artifacts
from employee_attrition import config

ARTIFACTS_DIR_ENV = "MURPHYSBIO_ARTIFACTS_DIR"
DATA_DIR_ENV = "MURPHYSBIO_DATA_DIR"
_REDIRECTS = ((ARTIFACTS_DIR_ENV, config.ARTIFACTS_DIR), (DATA_DIR_ENV, config.PROCESSED_DIR))


def artifact_path(default: Path) -> Path:
    """Return ``default``, moved to a redirected folder when its environment variable is set."""
    for env_var, base in _REDIRECTS:
        override = os.environ.get(env_var)
        if override and default.is_relative_to(base):
            return Path(override) / default.relative_to(base)
    return default


def _key(default: Path) -> tuple[str, float]:
    """Cache key for one file: its resolved path and modification time (0 if missing)."""
    path = artifact_path(default)
    try:
        return str(path), path.stat().st_mtime
    except OSError:
        return str(path), 0.0  # the loader reports the missing file


# ===== Cached loaders =====
# Every argument is part of the cache key (Streamlit skips only names starting with "_"),
# so the modification times must keep their plain names even though the bodies ignore them.
@st.cache_data(show_spinner=False)
def _metadata(path: str, mtime: float) -> dict:
    return artifacts.load_model_metadata(Path(path))


@st.cache_data(show_spinner=False)
def _schema(path: str, mtime: float, features: tuple[str, ...]) -> list[dict]:
    return artifacts.load_feature_schema(Path(path), features=list(features))


@st.cache_resource(show_spinner=False)
def _model(path: str, mtime: float, metadata_path: str, metadata_mtime: float) -> Any:
    metadata = artifacts.load_model_metadata(Path(metadata_path))
    return artifacts.load_model(Path(path), metadata=metadata)


@st.cache_data(show_spinner=False)
def _workforce(path: str, mtime: float) -> pd.DataFrame:
    return artifacts.load_workforce(Path(path))


@st.cache_data(show_spinner=False)
def _risk_scores(path: str, mtime: float) -> pd.DataFrame:
    return artifacts.load_risk_scores(Path(path))


@st.cache_data(show_spinner=False)
def _model_card(path: str, mtime: float, features: tuple[str, ...]) -> dict:
    return artifacts.load_model_card(Path(path), features=list(features))


@st.cache_data(show_spinner=False)
def _tuning_best(path: str, mtime: float) -> dict:
    return artifacts.load_tuning_best(Path(path))


@st.cache_data(show_spinner=False)
def _tuning_trials(path: str, mtime: float) -> pd.DataFrame:
    return artifacts.load_tuning_trials(Path(path))


# ===== Public accessors =====
def model_metadata() -> dict:
    """``model_metadata.json``."""
    return _metadata(*_key(config.MODEL_METADATA_PATH))


def features() -> list[str]:
    """Model input columns, in training order."""
    return artifacts.feature_list(model_metadata())


def feature_schema() -> list[dict]:
    """``feature_schema.json``, one record per model input."""
    return _schema(*_key(config.FEATURE_SCHEMA_PATH), tuple(features()))


def model() -> Any:
    """The fitted pipeline, checked against ``model_metadata.json``."""
    return _model(*_key(config.MODEL_PATH), *_key(config.MODEL_METADATA_PATH))


def workforce() -> pd.DataFrame:
    """``employee.csv``: all employees, indexed by Employee ID."""
    return _workforce(*_key(config.WORKFORCE_PATH))


def risk_scores() -> pd.DataFrame:
    """``employee_risk_scores.csv``: employees without an outcome, ranked."""
    return _risk_scores(*_key(config.RISK_SCORES_PATH))


def model_card() -> dict:
    """``model_card.json``: evaluation evidence for How the Model Works."""
    return _model_card(*_key(config.MODEL_CARD_PATH), tuple(features()))


def tuning_best() -> dict:
    """``optuna_best_params.json``."""
    return _tuning_best(*_key(config.TUNING_BEST_PATH))


def tuning_trials() -> pd.DataFrame:
    """``optuna_tuning_results.json``, one row per trial."""
    return _tuning_trials(*_key(config.TUNING_RESULTS_PATH))
