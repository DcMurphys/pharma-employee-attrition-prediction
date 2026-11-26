"""Shared constants for the notebook and the Streamlit app.

Values are copied from the notebook configuration cell (NB §3.2). The notebook stays the source
of truth: when a value changes there, update it here as well (``tests/test_config.py`` checks
the values that are recorded in the exported artifacts).
"""

from pathlib import Path

# ===== Reproducibility =====
RANDOM_STATE = 126

# ===== Branding (fictional company, NB §1) =====
COMPANY_NAME = "MurphysBio"
DATA_DISCLAIMER = (
    f"{COMPANY_NAME} is a fictional company. Data: IBM HR Analytics Employee "
    "Attrition sample (fictional)."
)

# ===== Repository paths (editable install: src/employee_attrition -> repository root) =====
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_DIR = ARTIFACTS_DIR / "trained_models"  # files loaded by the Streamlit app
TUNING_DIR = ARTIFACTS_DIR / "tuning"  # Optuna trial logs (experiment records)
FIG_DIR = PROJECT_ROOT / "reports" / "figures"

MODEL_PATH = MODEL_DIR / "gradient_boosting_pipeline.joblib"
MODEL_METADATA_PATH = MODEL_DIR / "model_metadata.json"
FEATURE_SCHEMA_PATH = MODEL_DIR / "feature_schema.json"
MODEL_CARD_PATH = MODEL_DIR / "model_card.json"  # evaluation evidence (NB §8.4)
TUNING_RESULTS_PATH = TUNING_DIR / "optuna_tuning_results.json"
TUNING_BEST_PATH = TUNING_DIR / "optuna_best_params.json"
WORKFORCE_PATH = PROCESSED_DIR / "employee.csv"  # dashboard input (all employees)
RISK_SCORES_PATH = PROCESSED_DIR / "employee_risk_scores.csv"  # unlabeled employees, ranked

# ===== Data schema =====
TARGET = "Attrition"
ID_COL = "EmployeeId"
STATUS_COL = "AttritionStatus"  # exported status: "Yes" (left), "No" (stayed), "Unknown"
ATTRITION_STATUSES = ("Yes", "No", "Unknown")
NOMINAL_COLS = [
    "BusinessTravel",
    "Department",
    "EducationField",
    "Gender",
    "JobRole",
    "MaritalStatus",
    "OverTime",
]
# Ordered survey/grade scales stay numeric so tree splits respect their order.
ORDINAL_COLS = [
    "Education",
    "EnvironmentSatisfaction",
    "JobInvolvement",
    "JobLevel",
    "JobSatisfaction",
    "PerformanceRating",
    "RelationshipSatisfaction",
    "StockOptionLevel",
    "WorkLifeBalance",
]
# Official labels from the IBM HR Analytics data dictionary.
SATISFACTION_LABELS = {1: "Low", 2: "Medium", 3: "High", 4: "Very High"}
ORDINAL_LABELS: dict[str, dict[int, str]] = {
    "JobSatisfaction": SATISFACTION_LABELS,
    "EnvironmentSatisfaction": SATISFACTION_LABELS,
    "WorkLifeBalance": {1: "Bad", 2: "Good", 3: "Better", 4: "Best"},
    "JobLevel": {level: f"Level {level}" for level in range(1, 6)},
}
# Exported in employee.csv for the dashboard; not used by the final model (NB §7.9.1).
ENGINEERED_COLS = [
    "MaxYearsSinceChange",
    "ExperienceOutsideCompany",
    "TrainingsPerYearAtCompany",
]
SENSITIVE_FEATURES = ("Gender", "MaritalStatus", "Age")
RISK_SCORE_COLUMNS = [
    ID_COL,
    "risk_rank",
    "attrition_probability",
    "attrition_flag",
    "Department",
    "JobRole",
    "JobLevel",
    "OverTime",
    "MonthlyIncome",
]

# ===== Decision rule and success criteria (NB §2.4, §7.9) =====
REVIEW_CAPACITY_SHARE = 0.20  # HR reviews the top 20% of each scoring round
CALIBRATION_METHOD = "sigmoid"
F_BETA = 2.0  # recall weighted twice as much as precision
MIN_TEST_ROC_AUC = 0.75  # D1
AP_LIFT_TARGET = 2.0  # D3: average precision >= 2 x prevalence
MAX_CV_TEST_GAP = 0.05  # D4: tolerated |CV ROC AUC - test ROC AUC|

# ===== Fairness monitoring (NB §8.3) =====
FAIRNESS_MIN_RATIO = 0.8  # four-fifths rule of thumb (29 C.F.R. 1607.4(D))
AGE_BANDS = [17, 29, 39, 49, 60]  # bin edges -> 18-29, 30-39, 40-49, 50-60
