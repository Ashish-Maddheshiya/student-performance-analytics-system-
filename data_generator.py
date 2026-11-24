"""
=============================================================================
Student Performance Data Analytics System - Data Generator Module
=============================================================================
Purpose:
    Generates a statistically realistic student academic dataset (350+ records)
    with genuine correlations between study hours, attendance, internal assessments,
    and final marks. Also provides optional anomaly injection for testing
    data cleaning pipelines.
=============================================================================
"""

import os
import random
import numpy as np
import pandas as pd

# Set fixed random seed for reproducible benchmark datasets
np.random.seed(42)
random.seed(42)

FIRST_NAMES = [
    "Aarav", "Aditi", "Akash", "Ananya", "Arjun", "Bhavna", "Chaitanya", "Deepak",
    "Divya", "Esha", "Gaurav", "Harsh", "Ishaan", "Jaya", "Karan", "Kavya",
    "Manish", "Meera", "Neha", "Nikhil", "Pooja", "Pranav", "Priya", "Rahul",
    "Rhea", "Rohan", "Roshni", "Sahil", "Sakshi", "Sameer", "Sanjay", "Shreya",
    "Siddharth", "Sneha", "Tanvi", "Tarun", "Varun", "Vidya", "Vikram", "Yash"
]

LAST_NAMES = [
    "Sharma", "Verma", "Gupta", "Malhotra", "Mehta", "Patel", "Reddy", "Nair",
    "Iyer", "Choudhury", "Bose", "Das", "Singh", "Kumar", "Mishra", "Joshi",
    "Rao", "Pandey", "Saxena", "Deshmukh", "Kulkarni", "Bhat", "Pillai", "Menon"
]

DEPARTMENTS = [
    "Computer Science",
    "Information Technology",
    "Data Science & AI",
    "Electronics & Comm",
    "Mechanical Engineering"
]

SUBJECTS = [
    "Data Structures & Algorithms",
    "Database Management Systems",
    "Machine Learning",
    "Computer Networks",
    "Operating Systems",
    "Software Engineering",
    "Cloud Computing"
]

GENDERS = ["Male", "Female"]


def generate_student_dataset(n_records: int = 350, inject_anomalies: bool = True) -> pd.DataFrame:
    """
    Generate realistic student academic data with genuine domain correlations.

    Parameters:
        n_records (int): Number of student records (minimum 200).
        inject_anomalies (bool): Whether to inject a controlled percentage of
                                 missing values, duplicates, and outliers for
                                 demonstrating data cleaning pipelines.

    Returns:
        pd.DataFrame: Raw or semi-raw student performance dataset.
    """
    n = max(n_records, 200)

    # 1. Base Demographic Attributes
    student_ids = [f"STU{1000 + i}" for i in range(1, n + 1)]
    names = [f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}" for _ in range(n)]
    genders = [random.choice(GENDERS) for _ in range(n)]
    ages = np.random.choice([18, 19, 20, 21, 22, 23], size=n, p=[0.10, 0.25, 0.35, 0.20, 0.08, 0.02])
    departments = np.random.choice(DEPARTMENTS, size=n, p=[0.30, 0.25, 0.20, 0.15, 0.10])
    semesters = np.random.choice(list(range(1, 9)), size=n, p=[0.10, 0.12, 0.15, 0.15, 0.18, 0.12, 0.10, 0.08])
    subjects = np.random.choice(SUBJECTS, size=n)

    # 2. Behavioral & Academic Drivers (correlated realistic distribution)
    # Study hours: Log-normal distribution centered around 4.5 hours/day
    study_hours_raw = np.random.lognormal(mean=1.35, sigma=0.45, size=n)
    study_hours = np.clip(np.round(study_hours_raw, 1), 1.0, 12.0)

    # Attendance: Beta distribution biased toward 75-90%
    attendance_raw = np.random.beta(a=7, b=2, size=n) * 100
    attendance = np.clip(np.round(attendance_raw, 1), 40.0, 100.0)

    # Assignment Scores (out of 100): correlated with study hours and attendance
    assignment_latent = 0.45 * (study_hours / 12.0 * 100) + 0.40 * attendance + np.random.normal(0, 8, size=n)
    assignment_scores = np.clip(np.round(assignment_latent, 1), 25.0, 100.0)

    # Internal Marks (out of 50): based on attendance and assignments
    internal_latent = 0.50 * (assignment_scores / 2.0) + 0.35 * (attendance / 2.0) + np.random.normal(0, 3, size=n)
    internal_marks = np.clip(np.round(internal_latent, 1), 10.0, 50.0)

    # Final Marks (out of 100): Weighted academic formula reflecting genuine pedagogy
    # 35% Internal Marks (scaled to 100), 25% Study Hours impact, 25% Attendance, 15% Assignment Scores + Noise
    final_latent = (
        0.35 * (internal_marks * 2.0) +
        0.25 * (study_hours / 10.0 * 100) +
        0.20 * attendance +
        0.15 * assignment_scores +
        np.random.normal(0, 6.0, size=n)
    )
    final_marks = np.clip(np.round(final_latent, 1), 18.0, 100.0)

    # Construct the primary DataFrame
    df = pd.DataFrame({
        "student_id": student_ids,
        "name": names,
        "gender": genders,
        "age": ages,
        "department": departments,
        "semester": semesters,
        "subject": subjects,
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "assignment_scores": assignment_scores,
        "internal_marks": internal_marks,
        "final_marks": final_marks
    })

    # 3. Controlled Anomaly Injection (for Real-World Data Cleaning Lifecycle Demonstration)
    if inject_anomalies:
        # A) Inject a few missing values (approx 2-3% of rows)
        nan_indices_study = np.random.choice(df.index, size=int(n * 0.025), replace=False)
        df.loc[nan_indices_study, "study_hours"] = np.nan

        nan_indices_att = np.random.choice(df.index, size=int(n * 0.02), replace=False)
        df.loc[nan_indices_att, "attendance_percentage"] = np.nan

        nan_indices_assign = np.random.choice(df.index, size=int(n * 0.02), replace=False)
        df.loc[nan_indices_assign, "assignment_scores"] = np.nan

        # B) Inject a few realistic duplicates (e.g., 4 duplicate rows)
        duplicate_rows = df.sample(n=4, random_state=42)
        df = pd.concat([df, duplicate_rows], ignore_index=True)

        # C) Inject controlled edge outliers / validation violations to test audit
        # e.g. Attendance mistakenly logged as 108% or study hours logged as 24 hrs
        outlier_idx_1 = random.choice(df.index)
        df.loc[outlier_idx_1, "attendance_percentage"] = 112.5

        outlier_idx_2 = random.choice(df.index)
        df.loc[outlier_idx_2, "study_hours"] = 26.0

        outlier_idx_3 = random.choice(df.index)
        df.loc[outlier_idx_3, "final_marks"] = -5.0

    return df


def save_default_datasets(data_dir: str = None):
    """
    Ensure the data directory has standard sample datasets ready in CSV and Excel.
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, "student_performance_sample.csv")
    excel_path = os.path.join(data_dir, "student_performance_sample.xlsx")
    clean_csv_path = os.path.join(data_dir, "student_performance_benchmark_clean.csv")

    # Generate benchmark dataset
    raw_df = generate_student_dataset(n_records=360, inject_anomalies=True)
    raw_df.to_csv(csv_path, index=False)
    raw_df.to_excel(excel_path, index=False)

    # Generate purely clean dataset as reference
    clean_df = generate_student_dataset(n_records=360, inject_anomalies=False)
    clean_df.to_csv(clean_csv_path, index=False)

    return {
        "sample_csv": csv_path,
        "sample_excel": excel_path,
        "clean_csv": clean_csv_path,
        "record_count": len(raw_df)
    }


if __name__ == "__main__":
    results = save_default_datasets()
    print(f"Sample datasets generated successfully: {results}")
