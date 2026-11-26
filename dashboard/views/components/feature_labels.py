"""Plain-language names and form groups for the 22 model inputs (scope §4.3)."""

import re

FEATURE_LABELS = {
    "BusinessTravel": "Business travel",
    "DailyRate": "Daily rate",
    "DistanceFromHome": "Distance from home",
    "EnvironmentSatisfaction": "Environment satisfaction",
    "HourlyRate": "Hourly rate",
    "JobInvolvement": "Job involvement (1 = low, 4 = very high)",
    "JobLevel": "Job level",
    "JobRole": "Job role",
    "JobSatisfaction": "Job satisfaction",
    "MonthlyIncome": "Monthly income (monetary units)",
    "MonthlyRate": "Monthly rate",
    "NumCompaniesWorked": "Previous employers",
    "OverTime": "Overtime",
    "PercentSalaryHike": "Last salary increase (%)",
    "StockOptionLevel": "Stock option level (0 = none)",
    "TotalWorkingYears": "Total working years (whole career)",
    "TrainingTimesLastYear": "Trainings last year",
    "WorkLifeBalance": "Work-life balance",
    "YearsAtCompany": "Years at MurphysBio",
    "YearsInCurrentRole": "Years in current role",
    "YearsSinceLastPromotion": "Years since last promotion",
    "YearsWithCurrManager": "Years with current manager",
}

FEATURE_GROUPS = {
    "Job": ["JobRole", "JobLevel", "OverTime", "BusinessTravel", "DistanceFromHome"],
    "Pay": [
        "MonthlyIncome",
        "PercentSalaryHike",
        "StockOptionLevel",
        "DailyRate",
        "HourlyRate",
        "MonthlyRate",
    ],
    "Career history": [
        "TotalWorkingYears",
        "NumCompaniesWorked",
        "YearsAtCompany",
        "YearsInCurrentRole",
        "YearsSinceLastPromotion",
        "YearsWithCurrManager",
        "TrainingTimesLastYear",
    ],
    "Satisfaction": [
        "JobSatisfaction",
        "EnvironmentSatisfaction",
        "WorkLifeBalance",
        "JobInvolvement",
    ],
}


def label(feature: str) -> str:
    """Display name of a column: a plain name for model inputs, otherwise the column name
    split into words ("EducationField" -> "Education field")."""
    if feature in FEATURE_LABELS:
        return FEATURE_LABELS[feature]
    words = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", feature).split()
    return " ".join([words[0], *(w.lower() for w in words[1:])]) if words else feature
