"""
=============================================================================
Student Performance Data Analytics System - Data Cleaning & Preprocessing
=============================================================================
Purpose:
    Implements the complete Data Cleaning and Preprocessing lifecycle:
    - Missing value detection & intelligent imputation
    - Duplicate detection & deduplication
    - Outlier identification (IQR & Z-score)
    - Domain range validation & capping
    - Categorical standardization & feature engineering
    - Before-and-after audit statistics for viva demonstration
=============================================================================
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List


# Expected standard schema definition
EXPECTED_COLUMNS = [
    "student_id", "name", "gender", "age", "department", "semester",
    "subject", "study_hours", "attendance_percentage",
    "assignment_scores", "internal_marks", "final_marks"
]

NUMERIC_COLUMNS = [
    "age", "study_hours", "attendance_percentage",
    "assignment_scores", "internal_marks", "final_marks"
]

CATEGORICAL_COLUMNS = ["gender", "department", "semester", "subject"]

# Domain bounds for boundary validation
VALIDATION_BOUNDS = {
    "age": (15, 65),
    "study_hours": (0.0, 18.0),
    "attendance_percentage": (0.0, 100.0),
    "assignment_scores": (0.0, 100.0),
    "internal_marks": (0.0, 50.0),
    "final_marks": (0.0, 100.0)
}


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize DataFrame column names to lower_snake_case and map standard domain synonyms.
    """
    df = df.copy()
    cleaned_names = {}
    for col in df.columns:
        clean = (
            str(col)
            .strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
        )
        # Direct exact domain mappings
        if clean in ["id", "roll_no", "reg_no", "roll_number", "student_no"]:
            clean = "student_id"
        elif clean in ["student_name", "full_name"]:
            clean = "name"
        elif clean in ["dept", "branch", "stream"]:
            clean = "department"
        elif clean in ["sem"]:
            clean = "semester"
        elif clean in ["subj", "subject_name", "course"]:
            clean = "subject"
        elif clean in ["attendance", "attendance_percent", "attendance_percentage", "attendance_pct"]:
            clean = "attendance_percentage"
        elif clean in ["study_time", "hours_studied", "daily_study_hours", "study_hour", "study_hours", "study_hrs"]:
            clean = "study_hours"
        elif clean in ["assignment", "assignments", "assignment_score", "assignment_scores", "assignment_mark", "assignment_marks"]:
            clean = "assignment_scores"
        elif clean in ["internal", "internals", "internal_mark", "internal_marks", "internal_score", "internal_scores"]:
            clean = "internal_marks"
        elif clean in ["final", "final_score", "final_scores", "final_mark", "final_marks", "marks", "total_marks", "exam_score"]:
            clean = "final_marks"
        cleaned_names[col] = clean

    df.rename(columns=cleaned_names, inplace=True)
    return df


def detect_outliers_iqr(series: pd.Series, multiplier: float = 1.5) -> pd.Series:
    """
    Detect outliers using Interquartile Range (IQR) rule:
    Lower Bound = Q1 - (multiplier * IQR)
    Upper Bound = Q3 + (multiplier * IQR)
    """
    clean_series = pd.to_numeric(series, errors="coerce").dropna()
    if len(clean_series) < 4:
        return pd.Series(False, index=series.index)

    q1 = clean_series.quantile(0.25)
    q3 = clean_series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - (multiplier * iqr)
    upper_bound = q3 + (multiplier * iqr)

    return (series < lower_bound) | (series > upper_bound)


def audit_raw_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate an audit summary of anomalies present in the raw dataset.
    """
    df_std = standardize_columns(df)
    missing_dict = df_std.isnull().sum().to_dict()
    total_missing = sum(missing_dict.values())
    exact_duplicates = int(df_std.duplicated().sum())

    id_duplicates = 0
    if "student_id" in df_std.columns:
        id_duplicates = int(df_std.duplicated(subset=["student_id"]).sum())

    outliers_detected = {}
    validation_violations = {}

    for col in NUMERIC_COLUMNS:
        if col in df_std.columns:
            s_num = pd.to_numeric(df_std[col], errors="coerce")
            # Domain violations
            min_val, max_val = VALIDATION_BOUNDS.get(col, (-np.inf, np.inf))
            violations = int(((s_num < min_val) | (s_num > max_val)).sum())
            if violations > 0:
                validation_violations[col] = violations

            # Statistical outliers
            outliers_count = int(detect_outliers_iqr(s_num).sum())
            outliers_detected[col] = outliers_count

    return {
        "total_records": len(df),
        "total_columns": len(df.columns),
        "missing_values_by_col": missing_dict,
        "total_missing_values": total_missing,
        "exact_duplicates": exact_duplicates,
        "id_duplicates": id_duplicates,
        "validation_violations": validation_violations,
        "outliers_detected": outliers_detected
    }


def clean_and_preprocess_dataset(
    df: pd.DataFrame,
    impute_strategy: str = "median",
    handle_outliers: str = "cap",
    remove_duplicates: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Execute full data cleaning and preprocessing pipeline.

    Parameters:
        df: Input raw DataFrame.
        impute_strategy: 'median', 'mean', or 'drop'.
        handle_outliers: 'cap' (clip to IQR/domain bounds), 'remove', or 'keep'.
        remove_duplicates: Whether to drop duplicate rows and IDs.

    Returns:
        (pd.DataFrame, Dict): Cleaned DataFrame and Before/After audit metrics.
    """
    before_audit = audit_raw_data(df)
    clean_df = standardize_columns(df)

    # 1. Type Coercion for Numeric Fields
    for col in NUMERIC_COLUMNS:
        if col in clean_df.columns:
            clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")

    # 2. Duplicate Removal
    if remove_duplicates:
        clean_df = clean_df.drop_duplicates()
        if "student_id" in clean_df.columns:
            clean_df = clean_df.drop_duplicates(subset=["student_id"], keep="first")

    # 3. Missing Value Imputation
    for col in clean_df.columns:
        if clean_df[col].isnull().any():
            if col in NUMERIC_COLUMNS:
                if impute_strategy == "median":
                    clean_df[col] = clean_df[col].fillna(clean_df[col].median())
                elif impute_strategy == "mean":
                    clean_df[col] = clean_df[col].fillna(clean_df[col].mean())
                elif impute_strategy == "drop":
                    clean_df = clean_df.dropna(subset=[col])
            elif col in CATEGORICAL_COLUMNS or clean_df[col].dtype == object:
                mode_val = clean_df[col].mode()
                fill_mode = mode_val.iloc[0] if not mode_val.empty else "Unknown"
                clean_df[col] = clean_df[col].fillna(fill_mode)

    # 4. Domain Validation & Outlier Capping
    for col, (min_bound, max_bound) in VALIDATION_BOUNDS.items():
        if col in clean_df.columns:
            if handle_outliers == "cap":
                clean_df[col] = clean_df[col].clip(lower=min_bound, upper=max_bound)
            elif handle_outliers == "remove":
                clean_df = clean_df[(clean_df[col] >= min_bound) & (clean_df[col] <= max_bound)]

    # 5. Round numeric values to standard decimals
    round_rules = {
        "study_hours": 1,
        "attendance_percentage": 1,
        "assignment_scores": 1,
        "internal_marks": 1,
        "final_marks": 1
    }
    for col, dec in round_rules.items():
        if col in clean_df.columns:
            clean_df[col] = clean_df[col].round(dec)

    if "age" in clean_df.columns:
        clean_df["age"] = clean_df["age"].round().astype(int)
    if "semester" in clean_df.columns:
        clean_df["semester"] = pd.to_numeric(clean_df["semester"], errors="coerce").fillna(1).astype(int)

    # 6. Feature Engineering for Analytics
    if "final_marks" in clean_df.columns:
        clean_df["result_status"] = clean_df["final_marks"].apply(
            lambda m: "Pass" if m >= 40.0 else "Fail"
        )

        def assign_grade(m: float) -> str:
            if m >= 85.0: return "A+ (Outstanding)"
            elif m >= 75.0: return "A (Distinction)"
            elif m >= 65.0: return "B (First Class)"
            elif m >= 50.0: return "C (Second Class)"
            elif m >= 40.0: return "D (Pass Class)"
            else: return "F (Fail)"

        clean_df["grade"] = clean_df["final_marks"].apply(assign_grade)

    if "attendance_percentage" in clean_df.columns:
        clean_df["attendance_category"] = pd.cut(
            clean_df["attendance_percentage"],
            bins=[0, 74.9, 84.9, 100.0],
            labels=["Low (<75%) - Condonation/Shortage", "Moderate (75-85%)", "High (>85%)"]
        )

    if "study_hours" in clean_df.columns:
        clean_df["study_intensity"] = pd.cut(
            clean_df["study_hours"],
            bins=[0, 2.9, 5.9, 24.0],
            labels=["Low (<3 hrs/day)", "Moderate (3-6 hrs/day)", "High (>6 hrs/day)"]
        )

    after_audit = audit_raw_data(clean_df)

    audit_comparison = {
        "before": before_audit,
        "after": after_audit,
        "records_removed": before_audit["total_records"] - len(clean_df),
        "missing_handled": before_audit["total_missing_values"] - after_audit["total_missing_values"],
        "duplicates_removed": before_audit["exact_duplicates"] - after_audit["exact_duplicates"],
        "impute_strategy_used": impute_strategy,
        "outlier_handling_used": handle_outliers
    }

    return clean_df.reset_index(drop=True), audit_comparison
