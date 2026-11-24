"""
=============================================================================
Student Performance Data Analytics System - Advanced Analysis & Insights
=============================================================================
Purpose:
    Carries out advanced academic performance analytics:
    - Subject-wise comparative performance
    - Department-wise benchmarks & disparity analysis
    - Gender parity evaluation
    - Behavioral drivers: Attendance & Study Hours vs Marks
    - Pass/Fail distributions & At-Risk student diagnostics
    - Correlation matrix & Key impact driver ranking
    - Automated real-data insight engine (genuine calculated findings)
=============================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple


def analyze_subject_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes subject-wise benchmarks: student count, mean, median, min, max, std dev, and pass rate.
    """
    if "subject" not in df.columns or "final_marks" not in df.columns:
        return pd.DataFrame()

    def calc_pass_rate(s):
        return round((s >= 40.0).mean() * 100, 2)

    grouped = df.groupby("subject").agg(
        student_count=("final_marks", "count"),
        avg_marks=("final_marks", lambda x: round(x.mean(), 2)),
        median_marks=("final_marks", lambda x: round(x.median(), 2)),
        highest_marks=("final_marks", "max"),
        lowest_marks=("final_marks", "min"),
        std_dev=("final_marks", lambda x: round(x.std(), 2)),
        pass_rate_pct=("final_marks", calc_pass_rate),
        avg_attendance=("attendance_percentage", lambda x: round(x.mean(), 1)) if "attendance_percentage" in df.columns else ("final_marks", lambda x: 0.0),
        avg_study_hours=("study_hours", lambda x: round(x.mean(), 1)) if "study_hours" in df.columns else ("final_marks", lambda x: 0.0)
    ).reset_index()

    grouped.rename(columns={
        "subject": "Subject",
        "student_count": "Total Students",
        "avg_marks": "Average Marks",
        "median_marks": "Median Marks",
        "highest_marks": "Highest Marks",
        "lowest_marks": "Lowest Marks",
        "std_dev": "Std Dev",
        "pass_rate_pct": "Pass Rate (%)",
        "avg_attendance": "Average Attendance (%)",
        "avg_study_hours": "Average Study Hours"
    }, inplace=True)

    return grouped.sort_values(by="Average Marks", ascending=False).reset_index(drop=True)


def analyze_department_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Departmental performance comparison.
    """
    if "department" not in df.columns or "final_marks" not in df.columns:
        return pd.DataFrame()

    def calc_pass_rate(s):
        return round((s >= 40.0).mean() * 100, 2)

    grouped = df.groupby("department").agg(
        student_count=("final_marks", "count"),
        avg_marks=("final_marks", lambda x: round(x.mean(), 2)),
        median_marks=("final_marks", lambda x: round(x.median(), 2)),
        pass_rate_pct=("final_marks", calc_pass_rate),
        avg_attendance=("attendance_percentage", lambda x: round(x.mean(), 1)),
        avg_study_hours=("study_hours", lambda x: round(x.mean(), 1))
    ).reset_index()

    grouped.rename(columns={
        "department": "Department",
        "student_count": "Total Students",
        "avg_marks": "Average Marks",
        "median_marks": "Median Marks",
        "pass_rate_pct": "Pass Rate (%)",
        "avg_attendance": "Average Attendance (%)",
        "avg_study_hours": "Average Study Hours (hrs/day)"
    }, inplace=True)

    return grouped.sort_values(by="Average Marks", ascending=False)


def analyze_gender_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gender performance comparison (for unbiased descriptive analytics).
    """
    if "gender" not in df.columns or "final_marks" not in df.columns:
        return pd.DataFrame()

    grouped = df.groupby("gender").agg(
        count=("final_marks", "count"),
        avg_final_marks=("final_marks", lambda x: round(x.mean(), 2)),
        median_marks=("final_marks", lambda x: round(x.median(), 2)),
        pass_rate=("final_marks", lambda x: round((x >= 40.0).mean() * 100, 2)),
        avg_attendance=("attendance_percentage", lambda x: round(x.mean(), 1)),
        avg_study_hours=("study_hours", lambda x: round(x.mean(), 1))
    ).reset_index()

    grouped.rename(columns={
        "gender": "Gender",
        "count": "Total Students",
        "avg_final_marks": "Average Marks",
        "median_marks": "Median Marks",
        "pass_rate": "Pass Rate (%)",
        "avg_attendance": "Average Attendance (%)",
        "avg_study_hours": "Average Study Hours"
    }, inplace=True)

    return grouped


def analyze_attendance_vs_marks(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Attendance versus marks analysis and threshold impact (<75% vs >=75%).
    """
    if "attendance_percentage" not in df.columns or "final_marks" not in df.columns:
        return {}

    corr = round(float(df["attendance_percentage"].corr(df["final_marks"])), 3)

    # Binning by attendance thresholds
    bins = [0, 60, 75, 85, 100]
    labels = ["<60% (Critical)", "60-74% (Shortage)", "75-84% (Good)", "85-100% (High)"]
    df_temp = df.copy()
    df_temp["attendance_tier"] = pd.cut(df_temp["attendance_percentage"], bins=bins, labels=labels)

    tier_summary = df_temp.groupby("attendance_tier", observed=False).agg(
        student_count=("final_marks", "count"),
        avg_marks=("final_marks", lambda x: round(x.mean(), 2)),
        pass_rate=("final_marks", lambda x: round((x >= 40.0).mean() * 100, 2))
    ).reset_index()

    # Statutory threshold comparison: Eligible (>=75%) vs Shortage (<75%)
    eligible = df[df["attendance_percentage"] >= 75.0]["final_marks"]
    shortage = df[df["attendance_percentage"] < 75.0]["final_marks"]

    avg_eligible = round(float(eligible.mean()), 2) if len(eligible) > 0 else 0.0
    avg_shortage = round(float(shortage.mean()), 2) if len(shortage) > 0 else 0.0
    delta = round(avg_eligible - avg_shortage, 2)

    return {
        "correlation": corr,
        "tier_summary": tier_summary,
        "avg_marks_eligible_att": avg_eligible,
        "avg_marks_shortage_att": avg_shortage,
        "performance_premium_eligible": delta
    }


def analyze_study_hours_vs_marks(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Study hours versus marks impact analysis.
    """
    if "study_hours" not in df.columns or "final_marks" not in df.columns:
        return {}

    corr = round(float(df["study_hours"].corr(df["final_marks"])), 3)

    bins = [0, 2.9, 5.0, 8.0, 24.0]
    labels = ["<3 hrs (Light)", "3-5 hrs (Moderate)", "5.1-8 hrs (High)", ">8 hrs (Intensive)"]
    df_temp = df.copy()
    df_temp["study_tier"] = pd.cut(df_temp["study_hours"], bins=bins, labels=labels)

    tier_summary = df_temp.groupby("study_tier", observed=False).agg(
        student_count=("final_marks", "count"),
        avg_marks=("final_marks", lambda x: round(x.mean(), 2)),
        pass_rate=("final_marks", lambda x: round((x >= 40.0).mean() * 100, 2))
    ).reset_index()

    return {
        "correlation": corr,
        "tier_summary": tier_summary
    }


def get_top_and_bottom_performers(df: pd.DataFrame, n: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Retrieve top N and bottom N performing students with key metadata.
    """
    cols_pref = [
        "student_id", "name", "department", "subject",
        "study_hours", "attendance_percentage", "internal_marks", "final_marks", "grade"
    ]
    cols = [c for c in cols_pref if c in df.columns]

    top_n = df.sort_values(by="final_marks", ascending=False).head(n)[cols].reset_index(drop=True)
    bottom_n = df.sort_values(by="final_marks", ascending=True).head(n)[cols].reset_index(drop=True)

    return top_n, bottom_n


def compute_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Pearson correlation matrix across all numerical variables.
    """
    num_cols = ["study_hours", "attendance_percentage", "assignment_scores", "internal_marks", "final_marks"]
    valid_cols = [c for c in num_cols if c in df.columns]
    if len(valid_cols) < 2:
        return pd.DataFrame()
    return df[valid_cols].corr().round(3)


def identify_performance_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank factors affecting final academic marks by correlation magnitude.
    """
    if "final_marks" not in df.columns:
        return pd.DataFrame()

    num_cols = ["internal_marks", "study_hours", "attendance_percentage", "assignment_scores", "age"]
    factors = []
    for col in num_cols:
        if col in df.columns:
            r = df[col].corr(df["final_marks"])
            impact_level = "Very High" if abs(r) >= 0.7 else "High" if abs(r) >= 0.5 else "Moderate" if abs(r) >= 0.3 else "Low"
            factors.append({
                "Factor / Driver": col.replace("_", " ").title(),
                "Correlation (r)": round(float(r), 3),
                "Impact Direction": "Positive" if r > 0 else "Negative",
                "Relative Strength": impact_level
            })

    return pd.DataFrame(factors).sort_values(by="Correlation (r)", ascending=False).reset_index(drop=True)


def identify_at_risk_students(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify students who require immediate academic intervention:
    - Failing (<40 final marks) OR
    - Low Attendance (<75%) with Below-Average Internal Marks
    """
    df_copy = df.copy()
    avg_internal = df_copy["internal_marks"].mean() if "internal_marks" in df_copy.columns else 25.0

    at_risk_mask = (
        (df_copy["final_marks"] < 40.0) |
        ((df_copy["attendance_percentage"] < 75.0) & (df_copy["internal_marks"] < avg_internal))
    )

    at_risk_df = df_copy[at_risk_mask].copy()

    # Determine risk category
    def assign_risk_reason(row):
        reasons = []
        if row.get("final_marks", 100) < 40.0:
            reasons.append("Failing Final Marks")
        if row.get("attendance_percentage", 100) < 75.0:
            reasons.append("Severe Attendance Shortage (<75%)")
        if row.get("study_hours", 10) < 2.5:
            reasons.append("Inadequate Study Hours")
        if row.get("internal_marks", 50) < 20.0:
            reasons.append("Weak Internal Assessment")
        return "; ".join(reasons) if reasons else "Academic Warning"

    at_risk_df["Risk Diagnosis"] = at_risk_df.apply(assign_risk_reason, axis=1)

    display_cols = [
        "student_id", "name", "department", "subject",
        "attendance_percentage", "study_hours", "internal_marks", "final_marks", "Risk Diagnosis"
    ]
    cols = [c for c in display_cols if c in at_risk_df.columns]
    return at_risk_df[cols].sort_values(by="final_marks", ascending=True).reset_index(drop=True)


def generate_automated_insights(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Extract strictly calculated domain insights from the dataset.
    Never uses fabricated or placeholder values.
    """
    total_students = len(df)
    avg_marks = round(float(df["final_marks"].mean()), 2)
    highest_marks = round(float(df["final_marks"].max()), 2)
    lowest_marks = round(float(df["final_marks"].min()), 2)
    pass_count = int((df["final_marks"] >= 40.0).sum())
    pass_pct = round((pass_count / total_students * 100), 2) if total_students > 0 else 0.0

    # Best & lowest department
    dept_marks = df.groupby("department")["final_marks"].mean()
    best_dept = str(dept_marks.idxmax())
    best_dept_avg = round(float(dept_marks.max()), 2)
    lowest_dept = str(dept_marks.idxmin())
    lowest_dept_avg = round(float(dept_marks.min()), 2)

    # Best & hardest subject
    sub_marks = df.groupby("subject")["final_marks"].mean()
    best_sub = str(sub_marks.idxmax())
    best_sub_avg = round(float(sub_marks.max()), 2)
    hardest_sub = str(sub_marks.idxmin())
    hardest_sub_avg = round(float(sub_marks.min()), 2)

    # Attendance impact
    eligible_att = df[df["attendance_percentage"] >= 75.0]["final_marks"]
    shortage_att = df[df["attendance_percentage"] < 75.0]["final_marks"]
    eligible_avg = round(float(eligible_att.mean()), 2) if len(eligible_att) > 0 else 0.0
    shortage_avg = round(float(shortage_att.mean()), 2) if len(shortage_att) > 0 else 0.0
    att_delta = round(eligible_avg - shortage_avg, 2)
    att_corr = round(float(df["attendance_percentage"].corr(df["final_marks"])), 3)

    # Study hours impact
    study_corr = round(float(df["study_hours"].corr(df["final_marks"])), 3)
    high_study = df[df["study_hours"] >= 6.0]["final_marks"]
    low_study = df[df["study_hours"] < 3.0]["final_marks"]
    high_study_avg = round(float(high_study.mean()), 2) if len(high_study) > 0 else 0.0
    low_study_avg = round(float(low_study.mean()), 2) if len(low_study) > 0 else 0.0
    study_delta = round(high_study_avg - low_study_avg, 2)

    # Internal marks correlation
    internal_corr = round(float(df["internal_marks"].corr(df["final_marks"])), 3) if "internal_marks" in df.columns else 0.0

    # Identify dominant driver
    driver_dict = {
        "Internal Assessments": internal_corr,
        "Study Hours": study_corr,
        "Class Attendance": att_corr
    }
    strongest_driver = max(driver_dict, key=driver_dict.get)
    strongest_r = driver_dict[strongest_driver]

    # At risk count
    at_risk_df = identify_at_risk_students(df)
    at_risk_count = len(at_risk_df)
    at_risk_pct = round((at_risk_count / total_students * 100), 2)

    narratives = [
        f"**Subject Performance**: The highest scoring subject is **{best_sub}** (average **{best_sub_avg} marks**), while **{hardest_sub}** presents the greatest academic rigor with the lowest average (**{hardest_sub_avg} marks**).",
        f"**Departmental Benchmark**: The **{best_dept}** department leads academic achievement with an average of **{best_dept_avg} marks**, outperforming **{lowest_dept}** by **{round(best_dept_avg - lowest_dept_avg, 2)} marks**.",
        f"**Attendance Influence**: Class attendance strongly correlates with achievement (r = **{att_corr}**). Students meeting the 75% attendance threshold averaged **{eligible_avg} marks**, a massive **+{att_delta} marks advantage** over peers with attendance shortages (**{shortage_avg} marks**).",
        f"**Study Habit Impact**: Dedicated study time positively drives outcomes (r = **{study_corr}**). Students studying 6+ hours daily outperformed those studying under 3 hours daily by **+{study_delta} marks**.",
        f"**Primary Determinant**: **{strongest_driver}** was calculated as the single strongest statistical driver of final exam scores with a correlation of **r = {strongest_r}**.",
        f"**Academic Health**: Overall pass percentage stands at **{pass_pct}%** ({pass_count} out of {total_students} students passed).",
        f"**Intervention Need**: **{at_risk_count} students ({at_risk_pct}%)** are flagged as at-risk due to either failing marks or severe attendance deficits combined with low internal scores."
    ]

    return {
        "total_students": total_students,
        "avg_marks": avg_marks,
        "highest_marks": highest_marks,
        "lowest_marks": lowest_marks,
        "pass_count": pass_count,
        "pass_percentage": pass_pct,
        "best_department": best_dept,
        "best_dept_avg": best_dept_avg,
        "lowest_department": lowest_dept,
        "lowest_dept_avg": lowest_dept_avg,
        "best_subject": best_sub,
        "best_subject_avg": best_sub_avg,
        "hardest_subject": hardest_sub,
        "hardest_subject_avg": hardest_sub_avg,
        "attendance_correlation": att_corr,
        "attendance_eligible_avg": eligible_avg,
        "attendance_shortage_avg": shortage_avg,
        "attendance_delta": att_delta,
        "study_hours_correlation": study_corr,
        "study_hours_delta": study_delta,
        "strongest_driver": strongest_driver,
        "strongest_driver_corr": strongest_r,
        "at_risk_count": at_risk_count,
        "at_risk_percentage": at_risk_pct,
        "narratives": narratives
    }
