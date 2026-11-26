"""Load the notebook artifacts and validate them against their data contracts (NB §5.4, §8).

Every loader raises ``ArtifactError`` when a file is missing, unreadable, or violates its
contract. The error carries a language-neutral ``kind`` code that views can map to a message.
This module must not import streamlit; caching belongs to the controllers.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from employee_attrition.config import (
    ATTRITION_STATUSES,
    COMPANY_NAME,
    FEATURE_SCHEMA_PATH,
    ID_COL,
    MODEL_CARD_PATH,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    REVIEW_CAPACITY_SHARE,
    RISK_SCORE_COLUMNS,
    RISK_SCORES_PATH,
    STATUS_COL,
    TUNING_BEST_PATH,
    TUNING_RESULTS_PATH,
    WORKFORCE_PATH,
)
from employee_attrition.scoring import review_slots

METADATA_KEYS = [
    "company",
    "disclaimer",
    "model",
    "decision_threshold",
    "threshold_objective",
    "n_features",
    "features",
    "trained_on_rows",
    "cv_average_precision",
    "cv_roc_auc",
    "test_roc_auc",
    "test_average_precision",
    "test_precision",
    "test_recall",
    "sklearn_version",
]
FEATURE_KINDS = {"nominal", "ordinal", "numeric", "engineered"}
TUNING_BEST_KEYS = ["best_trial_number", "objective_metric", "n_trials", "cv_average_precision"]
BENCHMARK_COLUMNS = {"model", "precision", "recall", "f2", "flagged_share"}
MODEL_CARD_TABLES: dict[str, set[str]] = {
    "criteria": {"id", "criterion", "value", "met"},
    "test_benchmarks": BENCHMARK_COLUMNS,
    "test_intervals": {"model", "metric", "ci_low", "ci_high"},
    "external_validation": BENCHMARK_COLUMNS,
    "fairness": {"attribute", "group", "employees", "flag_rate", "ratio_to_highest"},
    "permutation_importance": {"feature", "mean_drop_ap", "std"},
}
CRITERIA_IDS = ["D1", "D2", "D3", "D4"]


class ArtifactError(Exception):
    """An artifact is missing, unreadable, or violates its data contract.

    ``kind`` is one of ``"missing"``, ``"unreadable"``, or ``"invalid"``; ``detail`` is an
    English message for the log.
    """

    def __init__(self, artifact: str, kind: str, path: Path, detail: str = "") -> None:
        self.artifact, self.kind, self.path, self.detail = artifact, kind, Path(path), detail
        super().__init__(f"{artifact} ({kind}) at {path}: {detail}".rstrip(": "))


# ===== Reading =====
def _read(artifact: str, path: Path, reader: Callable[[Path], Any]) -> Any:
    """Read a file, mapping I/O and parse failures to ``ArtifactError``."""
    path = Path(path)
    if not path.is_file():
        raise ArtifactError(artifact, "missing", path)
    try:
        return reader(path)
    except Exception as exc:  # any parse failure means the file cannot be used
        raise ArtifactError(artifact, "unreadable", path, f"{type(exc).__name__}: {exc}") from exc


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _require(condition: bool, artifact: str, path: Path, detail: str) -> None:
    if not condition:
        raise ArtifactError(artifact, "invalid", path, detail)


def feature_list(metadata: dict) -> list[str]:
    """Model input columns, in training order, from ``model_metadata.json``."""
    return [name.strip() for name in metadata["features"].split(",") if name.strip()]


# ===== Loaders =====
def load_model_metadata(path: Path = MODEL_METADATA_PATH) -> dict:
    """Load ``model_metadata.json`` (NB §8.4)."""
    name = "model_metadata"
    metadata = _read(name, path, _read_json)
    _require(isinstance(metadata, dict), name, path, "top level is not an object")
    missing = [key for key in METADATA_KEYS if key not in metadata]
    _require(not missing, name, path, f"missing keys {missing}")
    _require(metadata["company"] == COMPANY_NAME, name, path, f"company is not {COMPANY_NAME}")
    features = feature_list(metadata)
    _require(
        len(features) == metadata["n_features"] == len(set(features)),
        name,
        path,
        "feature list does not match n_features or has duplicates",
    )
    _require(0 < metadata["decision_threshold"] < 1, name, path, "threshold outside (0, 1)")
    return metadata


def load_feature_schema(path: Path = FEATURE_SCHEMA_PATH, *, features: list[str]) -> list[dict]:
    """Load ``feature_schema.json``: one record per model input, in ``features`` order."""
    name = "feature_schema"
    schema = _read(name, path, _read_json)
    _require(isinstance(schema, list), name, path, "top level is not a list")
    _require(
        [record.get("feature") for record in schema] == features,
        name,
        path,
        "features differ from model_metadata.json",
    )
    for record in schema:
        kind = record.get("kind")
        _require(kind in FEATURE_KINDS, name, path, f"unknown kind {kind!r}")
        if kind == "nominal":
            _require(bool(record.get("categories")), name, path, f"{record['feature']}: no levels")
        else:
            ordered = record.get("min", 1) <= record.get("median", 0) <= record.get("max", -1)
            _require(ordered, name, path, f"{record['feature']}: min <= median <= max fails")
    return schema


def load_model(path: Path = MODEL_PATH, *, metadata: dict) -> Any:
    """Load the fitted pipeline and check it against ``model_metadata.json``."""
    name = "model"
    model = _read(name, path, joblib.load)
    _require(
        hasattr(model, "predict_proba") and hasattr(model, "feature_names_in_"),
        name,
        path,
        "object is not a fitted classifier",
    )
    _require(
        list(model.feature_names_in_) == feature_list(metadata),
        name,
        path,
        "input columns differ from model_metadata.json",
    )
    _require(
        abs(float(model.threshold) - metadata["decision_threshold"]) < 1e-12,
        name,
        path,
        "decision threshold differs from model_metadata.json",
    )
    return model


def load_workforce(path: Path = WORKFORCE_PATH) -> pd.DataFrame:
    """Load ``employee.csv`` (all employees, original units; NB §5.4) indexed by ``ID_COL``."""
    name = "workforce"
    df = _read(name, path, pd.read_csv)
    _require({ID_COL, STATUS_COL} <= set(df.columns), name, path, "missing id or status column")
    _require(df[ID_COL].is_unique, name, path, f"duplicate {ID_COL}")
    unknown = set(df[STATUS_COL].unique()) - set(ATTRITION_STATUSES)
    _require(not unknown, name, path, f"unknown statuses {sorted(unknown)}")
    return df.set_index(ID_COL)


def load_risk_scores(path: Path = RISK_SCORES_PATH) -> pd.DataFrame:
    """Load ``employee_risk_scores.csv`` (unlabeled employees, ranked; NB §8.2)."""
    name = "risk_scores"
    df = _read(name, path, pd.read_csv)
    _require(list(df.columns) == RISK_SCORE_COLUMNS, name, path, "columns differ from contract")
    _require(df[ID_COL].is_unique, name, path, f"duplicate {ID_COL}")
    _require(
        df["attrition_probability"].between(0, 1).all(), name, path, "probability not in [0, 1]"
    )
    _require(
        df["risk_rank"].tolist() == list(range(1, len(df) + 1)),
        name,
        path,
        "rows are not sorted by a complete 1..n rank",
    )
    _require(
        df["attrition_probability"].is_monotonic_decreasing,
        name,
        path,
        "probability does not decrease with rank",
    )
    slots = review_slots(len(df), REVIEW_CAPACITY_SHARE)
    _require(
        int(df["attrition_flag"].sum()) == slots,
        name,
        path,
        f"expected {slots} flagged employees (top {REVIEW_CAPACITY_SHARE:.0%})",
    )
    return df.set_index(ID_COL)


def load_tuning_best(path: Path = TUNING_BEST_PATH) -> dict:
    """Load ``optuna_best_params.json`` (iteration 1 search summary; NB §6.4)."""
    name = "tuning_best"
    best = _read(name, path, _read_json)
    _require(isinstance(best, dict), name, path, "top level is not an object")
    missing = [key for key in TUNING_BEST_KEYS if key not in best]
    _require(not missing, name, path, f"missing keys {missing}")
    return best


def load_tuning_trials(path: Path = TUNING_RESULTS_PATH) -> pd.DataFrame:
    """Load ``optuna_tuning_results.json`` (one row per Optuna trial)."""
    name = "tuning_trials"
    trials = _read(name, path, lambda p: pd.DataFrame(_read_json(p)))
    _require(
        {"trial_number", "state", "value"} <= set(trials.columns),
        name,
        path,
        "missing trial_number, state or value",
    )
    _require(trials["trial_number"].is_unique, name, path, "duplicate trial numbers")
    return trials.set_index("trial_number")


def load_model_card(path: Path = MODEL_CARD_PATH, *, features: list[str]) -> dict:
    """Load ``model_card.json`` (evaluation evidence exported in NB §8.4).

    Checks the company, the six tables with their required columns, the criteria D1-D4, and
    that permutation importance covers exactly the model ``features``.
    """
    name = "model_card"
    card = _read(name, path, _read_json)
    _require(isinstance(card, dict), name, path, "top level is not an object")
    _require(card.get("company") == COMPANY_NAME, name, path, f"company is not {COMPANY_NAME}")
    for table, columns in MODEL_CARD_TABLES.items():
        rows = card.get(table)
        _require(isinstance(rows, list) and bool(rows), name, path, f"{table}: missing or empty")
        _require(
            all(isinstance(row, dict) and columns <= set(row) for row in rows),
            name,
            path,
            f"{table}: rows lack columns {sorted(columns)}",
        )
    _require(
        [row["id"] for row in card["criteria"]] == CRITERIA_IDS,
        name,
        path,
        f"criteria are not {CRITERIA_IDS}",
    )
    importance = [row["feature"] for row in card["permutation_importance"]]
    _require(
        sorted(importance) == sorted(features) and len(set(importance)) == len(importance),
        name,
        path,
        "permutation importance does not cover exactly the model features",
    )
    return card
