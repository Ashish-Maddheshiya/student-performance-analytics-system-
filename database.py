"""
=============================================================================
Student Performance Data Analytics System - SQLite Database Module
=============================================================================
Purpose:
    Provides automatic SQLite database initialization, storage of imported
    raw student data and cleaned analytical tables, tracking of historical
    analysis batches, and reliable connection handling without external dependencies.
=============================================================================
"""

import os
import sqlite3
import datetime
from typing import Optional, List, Dict, Any
import pandas as pd

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "student_analytics.db")


def get_db_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """
    Create a thread-safe connection to the SQLite database.
    """
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """
    Automatically creates all required tables and indexes if they do not exist.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    # 1. Raw Datasets Audit Log
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS dataset_batches (
            batch_id INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_name TEXT NOT NULL,
            source_type TEXT NOT NULL,
            record_count INTEGER NOT NULL,
            imported_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            notes TEXT
        );
    """)

    # 2. Cleaned Student Performance Data Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cleaned_students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_id INTEGER,
            student_id TEXT NOT NULL,
            name TEXT NOT NULL,
            gender TEXT NOT NULL,
            age INTEGER NOT NULL,
            department TEXT NOT NULL,
            semester INTEGER NOT NULL,
            subject TEXT NOT NULL,
            study_hours REAL NOT NULL,
            attendance_percentage REAL NOT NULL,
            assignment_scores REAL NOT NULL,
            internal_marks REAL NOT NULL,
            final_marks REAL NOT NULL,
            result_status TEXT NOT NULL,
            grade TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (batch_id) REFERENCES dataset_batches (batch_id)
        );
    """)

    # 3. Analytical KPI Summaries Cache
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS analytical_summaries (
            summary_id INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_id INTEGER,
            total_students INTEGER,
            average_marks REAL,
            pass_percentage REAL,
            average_attendance REAL,
            highest_marks REAL,
            lowest_marks REAL,
            top_department TEXT,
            top_subject TEXT,
            recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (batch_id) REFERENCES dataset_batches (batch_id)
        );
    """)

    # Indexes for high performance querying
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_stud_dept ON cleaned_students (department);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_stud_sub ON cleaned_students (subject);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_stud_status ON cleaned_students (result_status);")

    conn.commit()
    conn.close()


def save_cleaned_data_to_db(df: pd.DataFrame, batch_name: str = "Batch_1",
                            source_type: str = "Uploaded/Generated",
                            db_path: str = DEFAULT_DB_PATH) -> int:
    """
    Saves a cleaned DataFrame into the SQLite database under a tracked batch.

    Parameters:
        df (pd.DataFrame): Cleaned student data.
        batch_name (str): Human-readable name for this analytical batch.
        source_type (str): 'CSV', 'Excel', 'Sample Generator', etc.
        db_path (str): Path to SQLite DB.

    Returns:
        int: Created batch_id.
    """
    init_db(db_path)
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    # Insert batch record
    cursor.execute("""
        INSERT INTO dataset_batches (batch_name, source_type, record_count, imported_at)
        VALUES (?, ?, ?, ?)
    """, (batch_name, source_type, len(df), datetime.datetime.now().isoformat()))
    batch_id = cursor.lastrowid

    # Prepare DataFrame for insertion
    df_to_save = df.copy()
    df_to_save["batch_id"] = batch_id
    if "result_status" not in df_to_save.columns:
        df_to_save["result_status"] = df_to_save["final_marks"].apply(lambda m: "Pass" if m >= 40.0 else "Fail")
    if "grade" not in df_to_save.columns:
        def assign_grade(m):
            if m >= 85: return "A+"
            elif m >= 75: return "A"
            elif m >= 65: return "B"
            elif m >= 50: return "C"
            elif m >= 40: return "D"
            else: return "F"
        df_to_save["grade"] = df_to_save["final_marks"].apply(assign_grade)

    # Required columns in target table
    cols = [
        "batch_id", "student_id", "name", "gender", "age", "department",
        "semester", "subject", "study_hours", "attendance_percentage",
        "assignment_scores", "internal_marks", "final_marks",
        "result_status", "grade"
    ]
    # Filter columns that exist
    valid_cols = [c for c in cols if c in df_to_save.columns]
    df_subset = df_to_save[valid_cols]

    df_subset.to_sql("cleaned_students", conn, if_exists="append", index=False)

    # Store analytical summary for this batch
    avg_marks = float(df_to_save["final_marks"].mean())
    pass_pct = float((df_to_save["final_marks"] >= 40.0).mean() * 100)
    avg_att = float(df_to_save["attendance_percentage"].mean())
    max_m = float(df_to_save["final_marks"].max())
    min_m = float(df_to_save["final_marks"].min())
    top_dept = str(df_to_save.groupby("department")["final_marks"].mean().idxmax()) if "department" in df_to_save else "N/A"
    top_sub = str(df_to_save.groupby("subject")["final_marks"].mean().idxmax()) if "subject" in df_to_save else "N/A"

    cursor.execute("""
        INSERT INTO analytical_summaries (
            batch_id, total_students, average_marks, pass_percentage,
            average_attendance, highest_marks, lowest_marks, top_department, top_subject
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (batch_id, len(df), avg_marks, pass_pct, avg_att, max_m, min_m, top_dept, top_sub))

    conn.commit()
    conn.close()
    return batch_id


def load_latest_cleaned_data(db_path: str = DEFAULT_DB_PATH) -> Optional[pd.DataFrame]:
    """
    Retrieve the most recently saved student analytical dataset from SQLite.
    """
    init_db(db_path)
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT batch_id FROM dataset_batches ORDER BY batch_id DESC LIMIT 1")
    row = cursor.fetchone()
    if not row:
        conn.close()
        return None

    latest_batch_id = row["batch_id"]
    query = "SELECT * FROM cleaned_students WHERE batch_id = ?"
    df = pd.read_sql_query(query, conn, params=(latest_batch_id,))
    conn.close()
    return df


def get_all_batches(db_path: str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """
    Return history of imported and processed datasets.
    """
    init_db(db_path)
    conn = get_db_connection(db_path)
    df = pd.read_sql_query("SELECT * FROM dataset_batches ORDER BY batch_id DESC", conn)
    conn.close()
    return df


def get_historical_kpis(db_path: str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """
    Retrieve KPI history across saved batches.
    """
    init_db(db_path)
    conn = get_db_connection(db_path)
    df = pd.read_sql_query("""
        SELECT s.*, b.batch_name, b.imported_at
        FROM analytical_summaries s
        JOIN dataset_batches b ON s.batch_id = b.batch_id
        ORDER BY s.summary_id DESC
    """, conn)
    conn.close()
    return df
